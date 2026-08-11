# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# Wan-Animate drives a reference character image with two pre-processed videos.
# The reference image and the pose video travel the standard media path
# (`image_reference` + `video_reference`, which the API allows together), but
# there is no HTTP field for a second video, so the face video is passed as an
# `extra_body` field and read back from
# ``sampling_params.extra_args["face_video"]`` in the pre-process function.
#
# The segment knobs are declared for the same reason: they are model-specific
# and only meaningful to this pipeline.
WAN_ANIMATE_EXTRA_BODY_PARAMS = frozenset(
    {
        "face_video",
        "segment_frame_length",
        "prev_segment_conditioning_frames",
    }
)
WAN_ANIMATE_EXTRA_OUTPUT_PARAMS = frozenset()
