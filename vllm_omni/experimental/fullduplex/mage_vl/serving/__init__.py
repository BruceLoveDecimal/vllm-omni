# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned WebSocket serving for Mage-VL's proactive video streaming."""

from vllm_omni.experimental.fullduplex.mage_vl.serving.server import (
    MageVLDuplexServer,
    MageVLServingConfig,
    SessionsFull,
    create_app,
    create_router,
)

__all__ = [
    "MageVLDuplexServer",
    "MageVLServingConfig",
    "SessionsFull",
    "create_app",
    "create_router",
]
