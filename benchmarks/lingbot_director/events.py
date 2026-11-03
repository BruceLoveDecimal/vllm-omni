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
# How an event is rendered, from cheapest to strongest. A failed check moves the event one rung up.
#   prompt:   scene + facts + the event's visual description, switched in at a chunk boundary.
#   rewrite:  the Director's scene_after and facts_after replace the old scene and facts, so nothing in
#             the prompt still contradicts the event (a sunset scene asked to turn into a snowy night).
#   reanchor: the world restarts from the current frame, edited toward the event when an editor is set.
#             The restart is an ordinary new session with that frame as its reference image, the same
#             path a segment takes when it reaches its frame budget; the model and its cache are untouched.
RUNGS = ("prompt", "rewrite", "reanchor")


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
    # "action": something a subject or object does. "appearance": weather, light, time of day or another
    # change to how the whole scene looks, which a prompt alone does not reliably change mid-rollout.
    kind: str = "action"
    # A yes/no question about the frame that holds once the event has happened.
    success_check: str = ""
    # The background scene sentence rewritten to agree with the result.
    scene_after: str = ""
    # An image-edit instruction that turns the current frame into the result, for re-anchoring.
    edit_instruction: str = ""
    # queued -> sent -> playing -> verifying -> succeeded | failed.
    status: str = "queued"
    rung: str = "prompt"
    remaining: int = 0
    extended: bool = False
    # The world must restart from the current frame before this event can play.
    restart_pending: bool = False
    # Interaction ids (or segment starts) whose prompt carried this event; the first chunk the server
    # reports one of them as started is the chunk the event begins on.
    carriers: set[str] = field(default_factory=set)
    applied_chunk: int | None = None

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
            "kind": self.kind,
            "status": self.status,
            "rung": self.rung,
            "remaining": self.remaining,
            "duration_chunks": self.duration_chunks,
            "applied_chunk": self.applied_chunk,
        }


class EventScheduler:
    """Composes the chunk prompt from a stable scene, persistent facts and at most one active event.

    LingBot trains on a background prompt plus chunk-wise prompts; the served
    world takes one prompt, so the three parts are joined in that order.

    An event's duration counts from the chunk the server reports its prompt as
    applied, not from when the prompt was sent. When it has played out, the
    Director checks the frames; only a confirmed event writes its facts. A failed
    check retries on the next rung of ``RUNGS``, and an event that fails on the
    last rung is dropped without touching the facts.
    """

    def __init__(self, scene: str) -> None:
        self.scene = scene
        self.facts: list[str] = []
        self.active: WorldEvent | None = None
        self.queue: list[WorldEvent] = []
        self.history: list[dict[str, Any]] = []

    def prompt(self) -> str:
        if self.active is not None and self.active.rung != "prompt":
            parts = [self.active.scene_after or self.scene, *self.active.facts_after, self.active.prompt]
        elif self.active is not None:
            parts = [self.scene, *self.facts, self.active.prompt]
        else:
            parts = [self.scene, *self.facts]
        return " ".join(part.strip() for part in parts if part.strip())

    def awaiting_apply(self) -> bool:
        """The active event's prompt has to reach the world before its duration starts."""
        return self.active is not None and self.active.status == "sent"

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

    def sent(self, carrier_id: str) -> None:
        """A prompt interaction or a new segment now carries the active event's prompt."""
        if self.awaiting_apply():
            self.active.carriers.add(carrier_id)

    def on_chunk(self, chunk: int, started_ids: list[str]) -> WorldEvent | None:
        """One more chunk arrived. Returns the event that has just played out and needs a check."""
        event = self.active
        if event is None:
            return None
        if event.status == "sent" and event.carriers.intersection(started_ids):
            event.status = "playing"
            event.applied_chunk = chunk
            self._log("applied", event, chunk)
        if event.status != "playing":
            return None
        event.remaining -= 1
        if event.remaining > 0:
            return None
        event.status = "verifying"
        self._log("verifying", event, chunk)
        return event

    def settle(self, event: WorldEvent, verdict: str, *, chunk: int) -> str:
        """Apply the Director's check of a played-out event and say what happens next.

        ``verdict`` is "yes", "partial" or "no". Returns "succeeded", "extended",
        the next rung's name, "failed", or "stale" when the event is no longer active.
        """
        if event is not self.active or event.status != "verifying":
            return "stale"
        if verdict == "yes":
            if event.facts_after:
                self.facts = event.facts_after[-MAX_FACTS:]
            if event.scene_after:
                self.scene = event.scene_after
            return self._finish(event, "succeeded", chunk)
        if verdict == "partial" and not event.extended:
            # Under way but not done: give it one more duration on the same rung.
            event.extended = True
            event.status = "playing"
            event.remaining = event.duration_chunks
            self._log("extended", event, chunk)
            return "extended"
        rung = RUNGS.index(event.rung) + 1
        if rung == len(RUNGS):
            return self._finish(event, "failed", chunk)
        self._start_rung(event, RUNGS[rung])
        self._log("escalated", event, chunk)
        return event.rung

    def _finish(self, event: WorldEvent, outcome: str, chunk: int) -> str:
        event.status = outcome
        self._log(outcome, event, chunk)
        self.active = None
        self._activate_next(chunk)
        return outcome

    def _start_rung(self, event: WorldEvent, rung: str) -> None:
        event.rung = rung
        event.status = "sent"
        event.remaining = event.duration_chunks
        event.extended = False
        event.carriers.clear()
        event.applied_chunk = None
        event.restart_pending = rung == "reanchor"

    def _activate_next(self, chunk: int) -> None:
        if not self.queue:
            return
        self.active = self.queue.pop(0)
        # A change to the whole look goes straight to a restart; actions start with the prompt alone.
        self._start_rung(self.active, "reanchor" if self.active.kind == "appearance" else "prompt")
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
