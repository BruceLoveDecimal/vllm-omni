# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Director's VLM roles in the live loop: discover what can happen, resolve what a trigger causes,
and check that it happened.

The Director never runs per frame. It discovers on scene start or a clear
scene change, refreshes its suggestions every few chunks, resolves one
trigger at a time when the user acts, and verifies each event once it has
played out, before its facts are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.lingbot_director.clients import VlmClient

MAX_OBJECTS = 4
MAX_DURATION_CHUNKS = 6

DISCOVER = (
    "You are the Director of an interactive world rendered as video. The images are recent frames in time order; "
    "the last one is now. Return only a JSON object with these keys. scene: one sentence describing the stable "
    "scene, the subject, the viewpoint and the visual style, usable as the background of a video-generation "
    "prompt. objects: up to 4 visible, non-human objects a user could interact with, each {name: a short noun "
    "phrase usable as a segmentation prompt, box: [x0, y0, x1, y1] on the last image with coordinates from 0 to "
    "1000, actions: up to 3 short imperative phrases}. character_actions: 2 short actions the main subject could "
    "do now. environment_events: 2 plausible changes to the environment. Never list objects you cannot see."
)

RESOLVE = (
    "You are the Director of an interactive world rendered as video. The image is the current frame. The user "
    "triggers the request below. Reason about its immediate, physically logical consequence in this scene. Return "
    "only a JSON object with these keys. target: the object or entity acted on, or empty. action: a short verb "
    "phrase. preconditions: a list of {condition: a short phrase, holds: true or false judged from the frame}. "
    "visual_event: one self-contained English video-generation prompt of at most 60 words that shows how the "
    "scene evolves from now into the result, step by step, not just the end state; keep the scene, subject "
    "identity and camera, and place any new entity at a plausible entry point. duration_chunks: an integer from 1 "
    "to 6, how many chunks of about 0.75 s the event needs. facts_after: the complete updated list of short "
    "sentences that must stay true afterwards, starting from the current facts; drop facts the event makes false. "
    'kind: "appearance" if the event changes how the whole scene looks (weather, lighting, time of day, season), '
    'otherwise "action". success_check: one yes/no question about a single frame that is true only once the '
    "event has visibly happened. scene_after: the scene sentence rewritten to agree with the result, with anything "
    "the event makes false removed. edit_instruction: one image-edit instruction that turns the current frame into "
    "the result."
)

VERIFY = (
    "You check whether an event happened in a video world. The images are recent frames in time order; the last "
    "one is now. Answer the question about the last frame. Return only a JSON object with these keys. verdict: "
    '"yes" if it is clearly true, "partial" if the change has visibly started but is not complete, "no" '
    "otherwise. observed: one short sentence on what is actually visible that decides the answer."
)
VERDICTS = ("yes", "partial", "no")


@dataclass
class Discovery:
    scene: str
    objects: list[dict[str, Any]] = field(default_factory=list)
    character_actions: list[str] = field(default_factory=list)
    environment_events: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    target: str
    action: str
    preconditions: list[dict[str, Any]]
    visual_event: str
    duration_chunks: int
    facts_after: list[str]
    kind: str
    success_check: str
    scene_after: str
    edit_instruction: str

    @property
    def failed_preconditions(self) -> list[str]:
        return [item["condition"] for item in self.preconditions if not item["holds"]]


@dataclass
class Verdict:
    verdict: str
    observed: str


class DirectorAgent:
    def __init__(self, vlm: VlmClient) -> None:
        self.vlm = vlm

    async def discover(self, frames: list[np.ndarray]) -> Discovery:
        reply = await self.vlm.ask_json(DISCOVER, frames, "Describe what can happen next.")
        height, width = frames[-1].shape[:2]
        objects = []
        for item in reply["objects"][:MAX_OBJECTS]:
            x0, y0, x1, y1 = item["box"]
            box = [x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000]
            objects.append({"name": item["name"], "box": box, "actions": list(item["actions"])})
        return Discovery(
            scene=reply["scene"],
            objects=objects,
            character_actions=list(reply["character_actions"]),
            environment_events=list(reply["environment_events"]),
        )

    async def resolve(self, frame: np.ndarray, request: str, *, scene: str, facts: list[str]) -> Resolution:
        text = f"Scene: {scene}\nCurrent facts: {facts}\nRequest: {request}"
        reply = await self.vlm.ask_json(RESOLVE, [frame], text)
        return Resolution(
            target=reply["target"],
            action=reply["action"],
            preconditions=list(reply["preconditions"]),
            visual_event=reply["visual_event"],
            duration_chunks=min(max(int(reply["duration_chunks"]), 1), MAX_DURATION_CHUNKS),
            facts_after=list(reply["facts_after"]),
            kind="appearance" if reply["kind"] == "appearance" else "action",
            success_check=reply["success_check"],
            scene_after=reply["scene_after"],
            edit_instruction=reply["edit_instruction"],
        )

    async def verify(self, frames: list[np.ndarray], question: str) -> Verdict:
        reply = await self.vlm.ask_json(VERIFY, frames, f"Question: {question}")
        verdict = str(reply["verdict"]).lower()
        if verdict not in VERDICTS:
            raise ValueError(f"Unexpected verdict {reply['verdict']!r}; expected one of {VERDICTS}.")
        return Verdict(verdict=verdict, observed=reply["observed"])
