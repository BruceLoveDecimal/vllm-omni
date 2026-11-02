# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scene tasks and the mask measurement that decides whether a goal is reached."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Task:
    name: str
    goal: str
    criterion: str
    # Text concept for the segmentation server.
    concept: str
    # 1 measures width, 0 measures height.
    axis: int
    # Span relative to the reference frame that counts as reached.
    ratio_limit: float
    # True: reached at or below the limit. False: reached at or above it.
    shrinks: bool

    def achieved(self, ratio: float) -> bool:
        if self.shrinks:
            return ratio <= self.ratio_limit
        return ratio >= self.ratio_limit


TASKS = {
    "door-open": Task(
        name="door-open",
        goal=(
            "Open the central white paneled door with the silver lever handle wide enough to clearly see through its "
            "doorway, and keep it open until the video ends. Keep the camera stationary. Preserve this same door, its "
            "white frame, surrounding wooden walls, handle and lighting. Do not remove the door to fake an opening."
        ),
        criterion=(
            "Reached means the same door leaf is visibly swung open with a clear doorway aperture. A narrow ajar "
            "door is not reached."
        ),
        concept="white paneled door",
        axis=1,
        ratio_limit=0.70,
        shrinks=True,
    ),
    "beach-stand": Task(
        name="beach-stand",
        goal=(
            "The seated woman stands up from the sand and stays standing upright until the video ends. The dog stays "
            "beside her. Keep the camera stationary. Preserve the same woman, dog, beach and composition."
        ),
        criterion="Reached means the same woman is upright on her feet. Kneeling or rising is not reached.",
        concept="woman",
        axis=0,
        ratio_limit=1.30,
        shrinks=False,
    ),
}


def mask_span(mask: np.ndarray | None, axis: int) -> float:
    """Robust extent of a mask in pixels. A missing mask spans nothing."""
    if mask is None:
        return 0.0
    pixels = np.nonzero(mask)[axis]
    if len(pixels) == 0:
        return 0.0
    low, high = np.percentile(pixels, (2, 98))
    return float(high - low)
