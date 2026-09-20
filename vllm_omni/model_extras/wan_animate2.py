# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

# Request-level knobs the Wan2.2-Animate-2 pipeline reads from
# ``sampling_params.extra_args``:
#   prompt_ref            text prompt for the reference-extraction pass
#   segment_frame_length  frames per autoregressive segment (4k+1, default 81)
#   max_driving_frames    cap on how much driving video one request may decode
WAN_ANIMATE2_EXTRA_BODY_PARAMS: frozenset[str] = frozenset({"prompt_ref", "segment_frame_length", "max_driving_frames"})

# Official demo defaults (`infer/wan_animate_2_demo.py`) for the base DiT. The
# distilled release runs 10 steps without CFG; its recipe passes those flags
# explicitly because both releases resolve to the same pipeline class.
_WAN_ANIMATE2_DEFAULTS = VideoGenerationDefaults(
    width=720,
    height=1280,
    num_frames=81,
    num_inference_steps=40,
    fps=24,
    guidance_scale=3.0,
    flow_shift=5.0,
    dimension_multiple=16,
    default_negative_prompt=None,
)


def get_wan_animate2_video_generation_defaults(extra_body: Mapping[str, object] | None) -> VideoGenerationDefaults:
    """Defaults for the shared video examples; ``extra_body`` carries no overrides."""
    del extra_body
    return _WAN_ANIMATE2_DEFAULTS
