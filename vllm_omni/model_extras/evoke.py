# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

# Request-level Evoke inputs routed from `extra_body` into
# OmniDiffusionSamplingParams.extra_args. The sampling recipe (3-step pyramid,
# no CFG, warp noise) is pinned by the checkpoint and is not request-tunable.
EVOKE_EXTRA_BODY_PARAMS = frozenset(
    {
        "image",
        # OpenCV camera-to-world matrices, one per 24 fps output frame.
        "camera_c2ws",
        # [fx, fy, cx, cy]; defaults to a 70 degree horizontal FOV on 1280x720.
        "camera_intrinsics",
        "camera_hfov_degrees",
        # False disables the geometry warp for i2v (no camera control).
        "geometry",
    }
)
EVOKE_EXTRA_OUTPUT_PARAMS = frozenset()
