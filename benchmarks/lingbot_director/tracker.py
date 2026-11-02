# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Object tracking for event cards: the Director names objects, SAM keeps their boxes current."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.lingbot_director.clients import SegmentClient

Box = tuple[float, float, float, float]
# Discoveries closer than this to a tracked box are the same object.
SAME_OBJECT_IOU = 0.3
# A tracked instance must still overlap its last box this much.
FOLLOW_IOU = 0.1
MAX_MISSES = 3


def iou(a: Box, b: Box) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackedObject:
    object_id: str
    name: str
    box: Box
    actions: list[str] = field(default_factory=list)
    misses: int = 0

    def describe(self) -> dict[str, Any]:
        return {"object_id": self.object_id, "name": self.name, "box": list(self.box), "actions": self.actions}


class ObjectTracker:
    def __init__(self, segmenter: SegmentClient) -> None:
        self.segmenter = segmenter
        self.objects: dict[str, TrackedObject] = {}
        self._next_id = 0

    def merge(self, discovered: list[dict[str, Any]]) -> None:
        """Adopt the Director's objects; refresh actions of ones already tracked."""
        for item in discovered:
            box = tuple(float(value) for value in item["box"])
            match = None
            for tracked in self.objects.values():
                if tracked.name == item["name"] or iou(tracked.box, box) >= SAME_OBJECT_IOU:
                    match = tracked
                    break
            if match is not None:
                match.actions = list(item["actions"])
                continue
            object_id = f"obj-{self._next_id}"
            self._next_id += 1
            self.objects[object_id] = TrackedObject(object_id, item["name"], box, list(item["actions"]))

    async def update(self, frame: np.ndarray) -> None:
        """Follow every object into ``frame``; drop the ones SAM loses several times in a row."""
        for tracked in list(self.objects.values()):
            instances = await self.segmenter.instances(frame, tracked.name, boxes=[list(tracked.box)])
            best = max(instances, key=lambda instance: iou(instance["box"], tracked.box), default=None)
            if best is not None and iou(best["box"], tracked.box) >= FOLLOW_IOU:
                tracked.box = tuple(float(value) for value in best["box"])
                tracked.misses = 0
                continue
            tracked.misses += 1
            if tracked.misses >= MAX_MISSES:
                del self.objects[tracked.object_id]

    def hit(self, x: float, y: float) -> TrackedObject | None:
        """The smallest tracked box under a click, so a handle wins over the door around it."""
        under = [
            obj for obj in self.objects.values() if obj.box[0] <= x <= obj.box[2] and obj.box[1] <= y <= obj.box[3]
        ]
        return min(under, key=lambda obj: (obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1]), default=None)

    def describe(self) -> list[dict[str, Any]]:
        return [tracked.describe() for tracked in self.objects.values()]
