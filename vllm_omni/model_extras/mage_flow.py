# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# Request-scoped controls consumed by MageFlowPipeline from
# OmniDiffusionSamplingParams.extra_args.
MAGE_FLOW_EXTRA_BODY_PARAMS = frozenset(
    {
        "mage_enable_safety_check",
        "mage_enable_watermark",
        "mage_vision_long_edge",
    }
)
MAGE_FLOW_EXTRA_OUTPUT_PARAMS = frozenset()
