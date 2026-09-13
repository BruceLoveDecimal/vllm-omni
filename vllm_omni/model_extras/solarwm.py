# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM defaults for the shared image-to-video entrypoint."""

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

SOLARWM_EXTRA_BODY_PARAMS = frozenset({"camera_path"})


def solarwm_preserves_reference_image_size(*, model, revision=None):
    return True


def get_solarwm_video_generation_defaults(extra_body=None):
    return VideoGenerationDefaults(
        width=864,
        height=480,
        num_frames=81,
        num_inference_steps=4,
        fps=16,
        guidance_scale=1.0,
        dimension_multiple=32,
        default_negative_prompt=None,
        output="solarwm_output.mp4",
    )
