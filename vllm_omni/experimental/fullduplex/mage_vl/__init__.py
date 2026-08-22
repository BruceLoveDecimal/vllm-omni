# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proactive video streaming for Mage-VL, on the core duplex contracts."""

from vllm_omni.experimental.fullduplex.mage_vl.adapter import MageVLDuplexAdapter
from vllm_omni.experimental.fullduplex.mage_vl.policy import (
    Decision,
    MageVLGatePolicy,
    MageVLStreamConfig,
)

__all__ = [
    "Decision",
    "MageVLDuplexAdapter",
    "MageVLGatePolicy",
    "MageVLStreamConfig",
]
