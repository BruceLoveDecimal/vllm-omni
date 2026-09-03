# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VDN-H3: OpenVDN's hybrid window-softmax / linear-attention MiniMax-H3.

The released checkpoint (``OpenVDN/vdn-minimax-h3``) is the dense H3 backbone
plus a per-block linear-attention branch and two LoRA adapters, so it is served
by ``MiniMaxH3Pipeline`` with a hybrid branch switched on inside each attention
rather than by a pipeline or an attention backend of its own.
"""

from .config import (
    MiniMaxH3HybridAttentionConfig,
    MiniMaxH3HybridGeometry,
    VdnConfigError,
)

__all__ = [
    "MiniMaxH3HybridAttentionConfig",
    "MiniMaxH3HybridGeometry",
    "VdnConfigError",
]
