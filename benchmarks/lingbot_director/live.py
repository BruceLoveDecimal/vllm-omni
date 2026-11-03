# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The live agent loop: user input and the Director feed one event scheduler that steers the world.

    user keys, clicks, text ──┐                ┌── Director (VLM): discover / resolve
                              ▼                ▼
                     event scheduler + camera control
                              │ prompt and camera at chunk boundaries
                              ▼
                     world (LingBot, WS /v1/realtime/video)
                              │ chunks
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
     viewers (frames)   SAM tracker (cards)   Director feedback

The world never waits for the Director or the tracker: both read the newest
chunk on their own tasks and skip chunks they are too slow for. A segment
that reaches its frame budget continues from its last frame, with the
camera held and the current prompt, so a session can run indefinitely.

An event's duration starts on the chunk the server reports its prompt as
applied. When it has played out the Director checks the latest frames, and
the scheduler keeps the facts, retries on a stronger rung, or drops the event
(see ``events.RUNGS``). Re-anchoring restarts the world from the current
frame, edited toward the event when an image editor is configured.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from benchmarks.lingbot_director.agent import DirectorAgent, Discovery
from benchmarks.lingbot_director.clients import EditClient, WorldSession
from benchmarks.lingbot_director.events import CameraControl, EventScheduler, WorldEvent
from benchmarks.lingbot_director.tracker import ObjectTracker

HOTKEYS = ("u", "o", "f", "g")
SLOTS = tuple("123456789")
RECENT_FRAMES = 3
# Mean absolute grayscale difference, on a 0-1 scale, that counts as a new scene.
SCENE_CHANGE = 0.2


def scene_difference(a: np.ndarray, b: np.ndarray) -> float:
    def small_gray(frame: np.ndarray) -> np.ndarray:
        return frame[::8, ::8].astype(np.float32).mean(axis=2) / 255.0

    return float(np.abs(small_gray(a) - small_gray(b)).mean())


class LiveSession:
    def __init__(
        self,
        *,
        open_world: Callable[[np.ndarray, str, int], WorldSession],
        agent: DirectorAgent,
        tracker: ObjectTracker,
        publish: Callable[[dict[str, Any], list[np.ndarray]], Awaitable[None]],
        segment_chunks: int = 40,
        discover_every: int = 8,
        prompt_transition_chunks: int = 1,
        editor: EditClient | None = None,
        edit_seed: int = 0,
    ) -> None:
        self.open_world = open_world
        self.agent = agent
        self.tracker = tracker
        self.publish = publish
        self.segment_chunks = segment_chunks
        self.discover_every = discover_every
        self.prompt_transition_chunks = prompt_transition_chunks
        self.editor = editor
        self.edit_seed = edit_seed

        self.scheduler = EventScheduler(scene="")
        self.camera = CameraControl()
        self.camera_payload: dict[str, Any] | None = None
        self.proposals: dict[str, str] = {}
        self.slots: dict[str, str] = {}
        self.pending: list[dict[str, Any]] = []
        self.notices: deque[dict[str, Any]] = deque(maxlen=8)
        self.trace: list[dict[str, Any]] = []

        self.session: WorldSession | None = None
        self.sent_prompt = ""
        self.chunk = 0
        self.latest: np.ndarray | None = None
        self.recent: deque[np.ndarray] = deque(maxlen=RECENT_FRAMES)
        self.discovered_frame: np.ndarray | None = None
        self.discovered_chunk = 0
        # The chunk whose last frame the current object boxes were measured on.
        self.tracked_chunk = 0
        self.track_wakeup = asyncio.Event()
        self.discover_wakeup = asyncio.Event()
        self.stopping = False
        self.event_counter = 0
        self.started = 0.0

    def _now(self) -> float:
        return time.perf_counter() - self.started

    def _log(self, kind: str, **details: Any) -> None:
        self.trace.append({"kind": kind, "at_s": round(self._now(), 3), "chunk": self.chunk, **details})

    def _notice(self, level: str, text: str) -> None:
        self.notices.append({"level": level, "text": text, "chunk": self.chunk})

    def describe(self) -> dict[str, Any]:
        return {
            "type": "state",
            "chunk": self.chunk,
            "scheduler": self.scheduler.describe(),
            "objects": self.tracker.describe(),
            "objects_chunk": self.tracked_chunk,
            "proposals": self.proposals,
            "slots": self.slots,
            "camera": sorted(self.camera.held),
            "pending": self.pending,
            "notices": list(self.notices),
        }

    # ── main loop ────────────────────────────────────────────────────────

    async def run(self, reference: np.ndarray, inputs: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
        self.started = time.perf_counter()
        # The opening scene description becomes the background prompt, so it is the one wait at start.
        await self._discover([reference], adopt_scene=True)
        async with asyncio.TaskGroup() as group:
            self.group = group
            world = group.create_task(self._run_world(reference))
            trackers = group.create_task(self._track_loop())
            directors = group.create_task(self._discover_loop())
            listener = group.create_task(self._input_loop(inputs))
            await world
            for task in (trackers, directors, listener):
                task.cancel()
        return self.trace

    async def stop(self) -> None:
        self.stopping = True
        if self.session is not None:
            await self.session.stop()

    def _restart_pending(self) -> bool:
        return self.scheduler.active is not None and self.scheduler.active.restart_pending

    async def _run_world(self, image: np.ndarray) -> None:
        while not self.stopping:
            if self._restart_pending():
                event = self.scheduler.active
                event.restart_pending = False
                image = await self._reanchor_image(image, event)
            prompt = self.scheduler.prompt()
            start_id = f"segment-{self._next_id()}"
            async with self.open_world(image, prompt, self.segment_chunks) as session:
                self.session = session
                self.sent_prompt = prompt
                # The prompt a segment opens with applies from its first chunk.
                self.scheduler.sent(start_id)
                started_ids = [start_id]
                self._log("segment_started", prompt=prompt)
                if self.camera_payload is not None:
                    # Keys still held carry over into the continued segment.
                    await session.interact(f"camera-{self._next_id()}", camera=self.camera_payload)
                async for chunk in session.chunks():
                    self.chunk += 1
                    self.latest = chunk.frames[-1]
                    self.recent.append(chunk.frames[-1])
                    played_out = self.scheduler.on_chunk(self.chunk, started_ids + chunk.started_event_ids)
                    started_ids = []
                    if played_out is not None:
                        self.group.create_task(self._verify(played_out))
                    await self._sync_prompt()
                    await self.publish({**self.describe(), "frames_chunk": self.chunk}, chunk.frames)
                    self.track_wakeup.set()
                    self.discover_wakeup.set()
                    if self._restart_pending():
                        self._log("restart", event_id=self.scheduler.active.event_id)
                        await session.stop()
                        break
            self.session = None
            image = self.latest

    async def _reanchor_image(self, image: np.ndarray, event: WorldEvent) -> np.ndarray:
        if self.editor is None or not event.edit_instruction:
            return image
        started = self._now()
        edited = await self.editor.edit(image, event.edit_instruction, seed=self.edit_seed)
        self._log("reanchor_edited", event_id=event.event_id, seconds=round(self._now() - started, 3))
        return edited

    async def _sync_prompt(self) -> None:
        if self.session is None or self._restart_pending():
            # A restart carries the prompt in the new segment's start instead.
            return
        prompt = self.scheduler.prompt()
        # An event still waiting for its start resends an unchanged prompt so the server can confirm it.
        unconfirmed = self.scheduler.awaiting_apply() and not self.scheduler.active.carriers
        if prompt == self.sent_prompt and not unconfirmed:
            return
        self.sent_prompt = prompt
        interaction_id = f"prompt-{self._next_id()}"
        self._log("prompt_sent", prompt=prompt, interaction_id=interaction_id)
        self.scheduler.sent(interaction_id)
        await self.session.interact(interaction_id, prompt=prompt, transition_chunks=self.prompt_transition_chunks)

    def _next_id(self) -> int:
        self.event_counter += 1
        return self.event_counter

    # ── SAM: keep event cards on their objects ───────────────────────────

    async def _track_loop(self) -> None:
        while True:
            await self.track_wakeup.wait()
            self.track_wakeup.clear()
            chunk = self.chunk
            await self.tracker.update(self.latest)
            self.tracked_chunk = chunk
            await self.publish(self.describe(), [])

    # ── Director: discover on start, on a scene change, and every few chunks ──

    async def _discover_loop(self) -> None:
        while True:
            await self.discover_wakeup.wait()
            self.discover_wakeup.clear()
            changed = scene_difference(self.latest, self.discovered_frame) >= SCENE_CHANGE
            due = self.chunk - self.discovered_chunk >= self.discover_every
            if changed or due:
                await self._discover(list(self.recent), adopt_scene=changed)

    async def _discover(self, frames: list[np.ndarray], *, adopt_scene: bool) -> None:
        started = self._now()
        discovery: Discovery = await self.agent.discover(frames)
        self.discovered_frame = frames[-1]
        self.discovered_chunk = self.chunk
        if adopt_scene:
            # Only a new scene replaces the background; refreshes keep it stable.
            self.scheduler.scene = discovery.scene
        self.tracker.merge(discovery.objects)
        actions = discovery.character_actions[:2] + discovery.environment_events[:2]
        self.proposals = dict(zip(HOTKEYS, actions))
        self._log("discovered", seconds=round(self._now() - started, 3), adopt_scene=adopt_scene, **vars(discovery))
        await self._sync_prompt()
        await self.publish(self.describe(), [])

    # ── user input ───────────────────────────────────────────────────────

    async def _input_loop(self, inputs: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            message = await inputs.get()
            await self.handle(message)

    async def handle(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        self._log("input", message=message)
        if kind == "key":
            payload = self.camera.key(message["key"], bool(message["down"]))
            if payload is not None:
                # Kept even between segments, so the next one starts with the keys still held.
                self.camera_payload = payload
                if self.session is not None:
                    await self.session.interact(f"camera-{self._next_id()}", camera=payload)
        elif kind == "hotkey" and message["key"] in self.proposals:
            self._trigger("hotkey", self.proposals[message["key"]], target=None)
        elif kind == "slot_set":
            self.slots[message["slot"]] = message["text"]
        elif kind == "slot" and message["slot"] in self.slots:
            self._trigger("slot", self.slots[message["slot"]], target=None)
        elif kind == "object_action" and message["object_id"] in self.tracker.objects:
            tracked = self.tracker.objects[message["object_id"]]
            self._trigger("object", f"{message['action']} the {tracked.name}", target=tracked.name)
        elif kind == "text" and message["text"].strip():
            self._trigger("text", message["text"].strip(), target=None)
        elif kind == "stop":
            await self.stop()
        await self.publish(self.describe(), [])

    def _trigger(self, source: str, request: str, *, target: str | None) -> None:
        # Resolution waits on the VLM, so it runs beside the input loop.
        self.group.create_task(self._resolve(source, request, target))

    async def _resolve(self, source: str, request: str, target: str | None) -> None:
        entry = {"source": source, "request": request}
        self.pending.append(entry)
        await self.publish(self.describe(), [])
        started = self._now()
        resolution = await self.agent.resolve(
            self.latest, request, scene=self.scheduler.scene, facts=self.scheduler.facts
        )
        self.pending.remove(entry)
        self._log(
            "resolved",
            source=source,
            request=request,
            seconds=round(self._now() - started, 3),
            resolution=vars(resolution),
        )
        failed = resolution.failed_preconditions
        if failed:
            self._notice("rejected", f"{request}: {', '.join(failed)}")
        else:
            event = WorldEvent(
                event_id=f"event-{self._next_id()}",
                source=source,
                request=request,
                prompt=resolution.visual_event,
                duration_chunks=resolution.duration_chunks,
                facts_after=resolution.facts_after,
                target=target or resolution.target or None,
                kind=resolution.kind,
                success_check=resolution.success_check,
                scene_after=resolution.scene_after,
                edit_instruction=resolution.edit_instruction,
            )
            self.scheduler.submit(event, chunk=self.chunk)
            await self._sync_prompt()
        await self.publish(self.describe(), [])

    # ── Director: check an event once it has played out ─────────────────

    async def _verify(self, event: WorldEvent) -> None:
        started = self._now()
        verdict = await self.agent.verify(list(self.recent), event.success_check)
        outcome = self.scheduler.settle(event, verdict.verdict, chunk=self.chunk)
        self._log(
            "verified",
            event_id=event.event_id,
            question=event.success_check,
            verdict=verdict.verdict,
            observed=verdict.observed,
            outcome=outcome,
            seconds=round(self._now() - started, 3),
        )
        if outcome == "failed":
            self._notice("failed", f"{event.request}: {verdict.observed}")
        elif outcome not in ("succeeded", "stale"):
            self._notice("retry", f"{event.request}: {outcome} ({verdict.observed})")
        await self._sync_prompt()
        await self.publish(self.describe(), [])
