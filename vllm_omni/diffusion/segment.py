# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Internal stop-and-wait protocol for streamed diffusion stage handoffs."""

SEGMENT_KEY = "diffusion_segment"


def segment_envelope(output):
    custom = getattr(output, "custom_output", None) or {}
    marker = custom.get(SEGMENT_KEY)
    if marker is None:
        return None
    if (
        not isinstance(marker, dict)
        or type(marker.get("index")) is not int
        or marker["index"] < 0
        or type(marker.get("finished")) is not bool
        or ("ack" in marker and marker["ack"] is not True)
    ):
        raise ValueError("Invalid diffusion segment envelope")
    return marker
