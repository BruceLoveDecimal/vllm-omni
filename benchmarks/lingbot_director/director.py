# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Director pipeline: the world keeps generating while every other model works beside it.

Three kinds of work overlap here:

* The world streams chunks without waiting for anything else.
* The sensor (SAM) reads each finished chunk on its own task.
* The VLM and the image editor run only on events, as background tasks.

A rollout is a list of segments. Restarting from a state frame opens a new
segment. With two world endpoints the old segment keeps streaming until the
new one delivers its first chunk, so restarts leave no gap in the output. With
one endpoint the old segment has to stop first, because a stepwise world serves
one session at a time.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from benchmarks.lingbot_director.clients import EditClient, SegmentClient, VlmClient, WorldSession
from benchmarks.lingbot_director.state_machine import DirectorStateMachine
from benchmarks.lingbot_director.tasks import Task, mask_span

ARMS = ("fixed", "fsm", "rebase", "grounded")
KEYFRAME_CANDIDATES = 4
EDIT_ATTEMPTS = 3

WRITE_PROMPTS = (
    "You write video-generation prompts. The image is the first frame of a generated video. Return only a JSON "
    "object with two keys. execute_prompt: one self-contained English prompt of at most 100 words describing the "
    "goal happening now. hold_prompt: one self-contained English prompt of at most 60 words describing the goal "
    "state already reached and remaining unchanged. Keep the same subject identity, a stationary camera, no scene"
    " cuts and no new objects."
)

SELECT_FRAME = (
    "Image 1 is the initial reference. The remaining images are candidate frames in chronological order. "
    "{criterion} Return only a JSON object with two keys. best: the 1-based position among the candidates, "
    "excluding image 1, of the frame that shows the goal most clearly with the fewest artifacts. scene_preserved:"
    " true if that frame keeps the same subject, surroundings and camera as the reference."
)

WRITE_EDIT = (
    "You write image-editing prompts. The image is the current frame. Describe the desired result in natural "
    "photographic language, in at most 80 words, changing only what the goal requires and preserving background, "
    "lighting, camera and unchanged objects. Return only a JSON object with one key, edit_prompt."
)

CHECK_EDIT = (
    "Image 1 is the original frame and image 2 is its edited version. {criterion} Return only a JSON object with "
    "three keys. state_reached: true if image 2 shows the goal. scene_preserved: true if everything the goal does"
    " not require stayed the same. feedback: one short sentence on what to fix, empty when both are true."
)


@dataclass
class ChunkRecord:
    index: int
    segment: int
    arrived_s: float
    frames: int
    prompt: str
    state: str = ""
    ratio: float | None = None
    achieved: bool | None = None
    observed_s: float | None = None


@dataclass
class Segment:
    number: int
    url: str
    # The prompt this session is currently generating from.
    prompt: str
    session: Any = None
    task: asyncio.Task[None] | None = None
    stopped: bool = False


@dataclass
class Trace:
    task: str
    arm: str
    seed: int
    prompts: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "summary": summarize(self.chunks)}


def longest_run(flags: Sequence[bool | None]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize(chunks: Sequence[ChunkRecord]) -> dict[str, Any]:
    observed = [chunk for chunk in chunks if chunk.observed_s is not None]
    segments = sorted({chunk.segment for chunk in chunks})
    gaps = [later.arrived_s - earlier.arrived_s for earlier, later in zip(chunks, chunks[1:])]
    lag = [chunk.observed_s - chunk.arrived_s for chunk in observed]
    longest_by_segment = {
        segment: longest_run([chunk.achieved for chunk in observed if chunk.segment == segment]) for segment in segments
    }
    return {
        "chunks": len(chunks),
        "segments": len(segments),
        "held_ratio": sum(bool(chunk.achieved) for chunk in observed) / len(observed) if observed else 0.0,
        # A restart can fake a long run, so the headline number never crosses a segment boundary.
        "longest_held_in_one_segment": max(longest_by_segment.values(), default=0),
        "longest_held_by_segment": longest_by_segment,
        "max_chunk_gap_s": max(gaps, default=0.0),
        "median_sensor_lag_s": float(np.median(lag)) if lag else 0.0,
    }


class Director:
    def __init__(
        self,
        *,
        task: Task,
        arm: str,
        seed: int,
        total_chunks: int,
        world_urls: Sequence[str],
        open_world: Callable[[str, np.ndarray, str, int], WorldSession],
        vlm: VlmClient,
        segmenter: SegmentClient,
        editor: EditClient | None,
        max_rebases: int = 1,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}.")
        if arm == "grounded" and editor is None:
            raise ValueError("The grounded arm needs an image-edit endpoint.")
        self.task = task
        self.arm = arm
        self.seed = seed
        self.total_chunks = total_chunks
        self.world_urls = list(world_urls)
        self.open_world = open_world
        self.vlm = vlm
        self.segmenter = segmenter
        self.editor = editor
        self.max_rebases = max_rebases

        self.trace = Trace(task=task.name, arm=arm, seed=seed)
        self.chunk_frames: list[list[np.ndarray]] = []
        self.keyframe: np.ndarray | None = None
        self.keyframes: list[np.ndarray] = []
        self.machine: DirectorStateMachine | None = None
        self.prompt = ""
        self.reference_span = 0.0
        self.rebases = 0
        self.rebase_running = False
        self.active: Segment | None = None
        self.standby: Segment | None = None
        self.next_segment = 0
        # Chunks of older segments are still observed, but they no longer drive the state machine.
        self.control_segment = 0
        self.prompt_events = 0
        self.observations: asyncio.Queue[int] = asyncio.Queue()
        self.group: asyncio.TaskGroup | None = None
        self.started = 0.0

    def _now(self) -> float:
        return time.perf_counter() - self.started

    def _event(self, kind: str, **details: Any) -> None:
        self.trace.events.append({"kind": kind, "at_s": self._now(), "chunk": len(self.chunk_frames), **details})

    # ── planning ─────────────────────────────────────────────────────────

    async def run(self, reference: np.ndarray) -> Trace:
        self.started = time.perf_counter()
        self.reference_span = mask_span(await self.segmenter.best_mask(reference, self.task.concept), self.task.axis)
        if self.reference_span == 0.0:
            raise RuntimeError(f"Concept {self.task.concept!r} was not found in the reference frame.")
        reply = await self.vlm.ask_json(WRITE_PROMPTS, [reference], self.task.goal)
        execute_prompt = reply["execute_prompt"]
        hold_prompt = reply["hold_prompt"]
        self.trace.prompts = {"execute": execute_prompt, "hold": hold_prompt}
        self.machine = DirectorStateMachine(execute_prompt, hold_prompt)
        self.prompt = execute_prompt

        async with asyncio.TaskGroup() as group:
            self.group = group
            group.create_task(self._observe_chunks(reference))
            self._start_segment(reference, execute_prompt, url=self.world_urls[0], as_standby=False)
            if self.arm == "grounded":
                self.rebase_running = True
                group.create_task(self._ground_goal_frame(reference))
        return self.trace

    # ── world segments ───────────────────────────────────────────────────

    def _start_segment(self, image: np.ndarray, prompt: str, *, url: str, as_standby: bool) -> None:
        segment = Segment(number=self.next_segment, url=url, prompt=prompt)
        self.next_segment += 1
        if as_standby:
            self.standby = segment
        else:
            self.active = segment
        budget = self.total_chunks - len(self.chunk_frames)
        assert self.group is not None
        segment.task = self.group.create_task(self._pump(segment, image, prompt, budget))

    async def _pump(self, segment: Segment, image: np.ndarray, prompt: str, budget: int) -> None:
        async with self.open_world(segment.url, image, prompt, budget) as session:
            segment.session = session
            await self._sync_prompt(segment)
            async for chunk in session.chunks():
                if len(self.chunk_frames) >= self.total_chunks:
                    await self._stop(segment)
                    continue
                if segment is self.standby:
                    await self._promote(segment)
                if segment is not self.active:
                    continue
                index = len(self.chunk_frames)
                self.chunk_frames.append(chunk.frames)
                self.trace.chunks.append(
                    ChunkRecord(
                        index=index,
                        segment=segment.number,
                        arrived_s=self._now(),
                        frames=len(chunk.frames),
                        prompt=self.prompt,
                    )
                )
                self.observations.put_nowait(index)
                if len(self.chunk_frames) >= self.total_chunks:
                    await self._stop(segment)

    async def _stop(self, segment: Segment) -> None:
        if not segment.stopped:
            segment.stopped = True
            await segment.session.stop()

    async def _sync_prompt(self, segment: Segment) -> None:
        """Bring a connected active session to the current prompt."""
        if segment is not self.active or segment.session is None or segment.prompt == self.prompt:
            return
        segment.prompt = self.prompt
        self.prompt_events += 1
        await segment.session.set_prompt(f"prompt-{self.prompt_events}", self.prompt)

    async def _promote(self, segment: Segment) -> None:
        """The standby has its first chunk: only now does the old segment stop."""
        previous = self.active
        self.active = segment
        self.standby = None
        self._event("segment_switched", segment=segment.number)
        assert previous is not None
        await self._stop(previous)
        await self._sync_prompt(segment)

    async def _restart(self, frame: np.ndarray) -> None:
        if len(self.chunk_frames) >= self.total_chunks:
            return
        self.keyframe = frame
        self.keyframes.append(frame)
        self.rebases += 1
        self.machine.hold()
        self.prompt = self.machine.prompt
        self.control_segment = self.next_segment
        self._event("restart", overlapped=len(self.world_urls) > 1)
        current = self.active
        assert current is not None
        if len(self.world_urls) > 1:
            url = next(url for url in self.world_urls if url != current.url)
            self._start_segment(frame, self.prompt, url=url, as_standby=True)
            return
        await self._stop(current)
        await current.task
        self._start_segment(frame, self.prompt, url=current.url, as_standby=False)

    # ── sensing ──────────────────────────────────────────────────────────

    async def _observe_chunks(self, reference: np.ndarray) -> None:
        for _ in range(self.total_chunks):
            index = await self.observations.get()
            record = self.trace.chunks[index]
            mask = await self.segmenter.best_mask(self.chunk_frames[index][-1], self.task.concept)
            record.ratio = mask_span(mask, self.task.axis) / self.reference_span
            record.achieved = self.task.achieved(record.ratio)
            record.observed_s = self._now()
            if self.arm != "fixed" and record.segment >= self.control_segment:
                await self._advance(index, record.achieved, reference)
            record.state = self.machine.state

    async def _advance(self, index: int, achieved: bool, reference: np.ndarray) -> None:
        before = self.machine.state
        self.machine.observe(achieved)
        state = self.machine.state
        if state != before:
            self._event("state", state=state)
        self.prompt = self.machine.prompt
        await self._sync_prompt(self.active)
        if self.arm == "fsm" or self.rebase_running or self.rebases >= self.max_rebases:
            return
        if state == "HOLD" and before != "HOLD" and self.arm == "rebase":
            self.rebase_running = True
            self.group.create_task(self._restart_from_history(index, reference))
        elif state == "RECOVER" and before != "RECOVER" and self.keyframe is not None:
            self.rebase_running = True
            self.group.create_task(self._restart_from_keyframe())

    # ── event work, off the chunk path ───────────────────────────────────

    async def _restart_from_history(self, index: int, reference: np.ndarray) -> None:
        chunks = [chunk for chunk in self.trace.chunks[: index + 1] if chunk.achieved][-KEYFRAME_CANDIDATES:]
        candidates = [self.chunk_frames[chunk.index][-1] for chunk in chunks]
        system = SELECT_FRAME.format(criterion=self.task.criterion)
        reply = await self.vlm.ask_json(system, [reference, *candidates], self.task.goal)
        self._event("keyframe_selected", best=reply["best"], scene_preserved=reply["scene_preserved"])
        if reply["scene_preserved"]:
            await self._restart(candidates[reply["best"] - 1])
        self.rebase_running = False

    async def _restart_from_keyframe(self) -> None:
        await self._restart(self.keyframe)
        self.rebase_running = False

    async def _ground_goal_frame(self, reference: np.ndarray) -> None:
        """Edit the goal into the first frame while the world is already playing it forward."""
        reply = await self.vlm.ask_json(WRITE_EDIT, [reference], self.task.goal)
        edit_prompt = reply["edit_prompt"]
        system = CHECK_EDIT.format(criterion=self.task.criterion)
        for attempt in range(EDIT_ATTEMPTS):
            edited = await self.editor.edit(reference, edit_prompt, seed=self.seed + attempt)
            check = await self.vlm.ask_json(system, [reference, edited], self.task.goal)
            self._event("edit_checked", attempt=attempt, **check)
            if check["state_reached"] and check["scene_preserved"]:
                await self._restart(edited)
                break
            edit_prompt = f"{edit_prompt} {check['feedback']}"
        self.rebase_running = False
