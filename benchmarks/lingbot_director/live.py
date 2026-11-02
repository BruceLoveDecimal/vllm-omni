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
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from benchmarks.lingbot_director.agent import DirectorAgent, Discovery
from benchmarks.lingbot_director.clients import WorldSession
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
    ) -> None:
        self.open_world = open_world
        self.agent = agent
        self.tracker = tracker
        self.publish = publish
        self.segment_chunks = segment_chunks
        self.discover_every = discover_every
        self.prompt_transition_chunks = prompt_transition_chunks

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

    async def _run_world(self, image: np.ndarray) -> None:
        while not self.stopping:
            prompt = self.scheduler.prompt()
            async with self.open_world(image, prompt, self.segment_chunks) as session:
                self.session = session
                self.sent_prompt = prompt
                self._log("segment_started", prompt=prompt)
                if self.camera_payload is not None:
                    # Keys still held carry over into the continued segment.
                    await session.interact(f"camera-{self._next_id()}", camera=self.camera_payload)
                async for chunk in session.chunks():
                    self.chunk += 1
                    self.latest = chunk.frames[-1]
                    self.recent.append(chunk.frames[-1])
                    self.scheduler.on_chunk(self.chunk)
                    await self._sync_prompt()
                    await self.publish({**self.describe(), "frames_chunk": self.chunk}, chunk.frames)
                    self.track_wakeup.set()
                    self.discover_wakeup.set()
            self.session = None
            image = self.latest

    async def _sync_prompt(self) -> None:
        prompt = self.scheduler.prompt()
        if self.session is None or prompt == self.sent_prompt:
            return
        self.sent_prompt = prompt
        self._log("prompt_sent", prompt=prompt)
        await self.session.interact(
            f"prompt-{self._next_id()}", prompt=prompt, transition_chunks=self.prompt_transition_chunks
        )

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
            "resolved", source=source, request=request, seconds=round(self._now() - started, 3), **vars(resolution)
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
            )
            self.scheduler.submit(event, chunk=self.chunk)
            await self._sync_prompt()
        await self.publish(self.describe(), [])
