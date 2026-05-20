# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local Bidirectional Gated DeltaNet scaffold for SANA-WM."""

from __future__ import annotations

from typing import Any

from torch import nn

SANA_WM_GDN_ERROR = (
    "Sana-WM Bidirectional Gated DeltaNet is not implemented yet. "
    "Port the NVlabs Triton recurrence and validate dtype parity before enabling Stage-1 forward."
)


class BidirectionalGatedDeltaNetTriton(nn.Module):
    """Placeholder for SANA-WM's model-local GDN operator.

    This intentionally stays outside the generic diffusion attention backend
    registry because the released operator carries state-space parameters and
    a K-only depthwise convolution that do not fit a plain Q/K/V softmax API.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(SANA_WM_GDN_ERROR)
