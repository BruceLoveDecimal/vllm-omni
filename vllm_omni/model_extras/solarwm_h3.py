# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

# ``camera_c2w``: one absolute 4x4 camera-to-world pose per output frame, as a
# nested list or a path to a .npy/.npz/.json file. ``kv_cache_on_device``
# keeps the six-chunk window cache in accelerator memory instead of pinned host
# memory.
SOLARWM_H3_EXTRA_BODY_PARAMS = frozenset({"camera_c2w", "kv_cache_on_device"})


def get_solarwm_h3_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    del extra_body
    return VideoGenerationDefaults(
        width=1344,
        height=768,
        num_frames=158,
        num_inference_steps=4,
        fps=24.0,
        flow_shift=12.0,
        dimension_multiple=32,
        output="solarwm_h3_output.mp4",
        default_negative_prompt=None,
    )
