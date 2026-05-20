# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Camera-control helpers for SANA-WM.

The released model expects camera-to-world trajectories that are converted to
raymap / Plucker features before DiT block injection. The numerical projection
is intentionally left unimplemented until the NVlabs reference formula is ported
and covered by parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SANA_WM_CAMERA_ERROR = (
    "Sana-WM camera raymap / Plucker projection is not implemented yet. "
    "Port the NVlabs reference formula before enabling camera-conditioned forward."
)


@dataclass(frozen=True)
class SanaWmCameraCondition:
    poses: Any
    intrinsics: Any | None = None
    action: str | None = None
    num_frames: int | None = None


def build_plucker_condition(condition: SanaWmCameraCondition) -> Any:
    del condition
    raise NotImplementedError(SANA_WM_CAMERA_ERROR)
