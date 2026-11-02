# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Interaction orchestration: an event queue that turns triggers into chunk prompts and camera motion."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CAMERA_UTILS = Path(__file__).resolve().parents[2] / "examples/online_serving/streaming_video_generation"
if str(_CAMERA_UTILS) not in sys.path:
    sys.path.insert(0, str(_CAMERA_UTILS))

from camera_utils import wasd_to_camera_payload  # noqa: E402

CAMERA_KEYS = frozenset("wasdijkl")
# A user typing an event outranks a hotkey, which outranks the Director acting on its own.
PRIORITY = {"text": 3, "object": 2, "hotkey": 2, "slot": 2, "director": 1}
MAX_FACTS = 8


@dataclass
class WorldEvent:
    event_id: str
    source: str
    # What the user or the Director asked for, before causal resolution.
    request: str
    # The self-contained prompt that renders the consequence.
    prompt: str
    duration_chunks: int
    # Persistent facts once the event has played out, as resolved by the Director.
    facts_after: list[str] = field(default_factory=list)
    target: str | None = None
    remaining: int = 0

    @property
    def priority(self) -> int:
        return PRIORITY[self.source]

    def describe(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "request": self.request,
            "prompt": self.prompt,
            "target": self.target,
            "remaining": self.remaining,
            "duration_chunks": self.duration_chunks,
        }


class EventScheduler:
    """Composes the chunk prompt from a stable scene, persistent facts and at most one active event.

    LingBot trains on a background prompt plus chunk-wise prompts; the served
    world takes one prompt, so the three parts are joined in that order.
    """

    def __init__(self, scene: str) -> None:
        self.scene = scene
        self.facts: list[str] = []
        self.active: WorldEvent | None = None
        self.queue: list[WorldEvent] = []
        self.history: list[dict[str, Any]] = []

    def prompt(self) -> str:
        parts = [self.scene, *self.facts]
        if self.active is not None:
            parts.append(self.active.prompt)
        return " ".join(part.strip() for part in parts if part.strip())

    def submit(self, event: WorldEvent, *, chunk: int) -> None:
        """Queue an event; one of higher priority replaces the active event at once."""
        if self.active is not None and event.priority > self.active.priority:
            self._log("preempted", self.active, chunk)
            self.active = None
        self.queue.append(event)
        # Stable sort: equal priorities keep their arrival order.
        self.queue.sort(key=lambda queued: -queued.priority)
        if self.active is None:
            self._activate_next(chunk)

    def on_chunk(self, chunk: int) -> None:
        """One more chunk has been generated."""
        if self.active is None:
            return
        self.active.remaining -= 1
        if self.active.remaining > 0:
            return
        if self.active.facts_after:
            self.facts = self.active.facts_after[-MAX_FACTS:]
        self._log("finished", self.active, chunk)
        self.active = None
        self._activate_next(chunk)

    def _activate_next(self, chunk: int) -> None:
        if not self.queue:
            return
        self.active = self.queue.pop(0)
        self.active.remaining = self.active.duration_chunks
        self._log("started", self.active, chunk)

    def _log(self, what: str, event: WorldEvent, chunk: int) -> None:
        self.history.append({"what": what, "chunk": chunk, **event.describe()})

    def describe(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "facts": list(self.facts),
            "active": self.active.describe() if self.active is not None else None,
            "queue": [event.describe() for event in self.queue],
            "prompt": self.prompt(),
        }


class CameraControl:
    """Held WASD/IJKL keys become one velocity command, sent only when the held set changes."""

    def __init__(self) -> None:
        self.held: set[str] = set()

    def key(self, key: str, down: bool) -> dict[str, Any] | None:
        key = key.lower()
        if key not in CAMERA_KEYS:
            return None
        before = set(self.held)
        if down:
            self.held.add(key)
        else:
            self.held.discard(key)
        if self.held == before:
            return None
        return wasd_to_camera_payload(sorted(self.held), mode="velocity")
