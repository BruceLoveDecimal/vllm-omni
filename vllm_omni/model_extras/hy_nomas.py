# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Default geometry for the shared video example and video serving path."""

from collections.abc import Mapping

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults


def get_hy_nomas_video_generation_defaults(extra_body: Mapping[str, object] | None = None) -> VideoGenerationDefaults:
    return VideoGenerationDefaults(
        width=1280,
        height=704,
        num_frames=120,
        num_inference_steps=8,
        fps=24,
        duration_seconds=5.0,
        output="hy_nomas_output.mp4",
        default_negative_prompt=None,
    )
