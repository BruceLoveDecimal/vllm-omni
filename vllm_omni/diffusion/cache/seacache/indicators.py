# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-specific SeaCache indicators for extractor-driven models.

An indicator turns a transformer's forward kwargs into the SEA-filtered noisy
latents the gate compares across steps, or ``None`` to run the step in full.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.sea_filter import apply_sea_filter


def minimax_h3_indicator(module: torch.nn.Module, config: SeaCacheConfig, **kwargs: Any) -> list[torch.Tensor] | None:
    """Filtered video target grids and denoised audio rows of one packed H3 request.

    Video rows come from ``video_token_layout`` reshaped to ``(t, h, w, C)``
    (Ref2VA publishes spans, of which only ``target`` ones evolve); audio rows
    are ``(channels, t, C)`` and the denoised ones sit at the lowest audio
    timestep, since reference-audio anchors stay pinned at the condition
    timestep. Each modality is filtered at its own ``sigma = 1 - t`` because H3
    shifts the two schedules separately. Text, visual-condition, and padding
    rows never enter the indicator.
    """
    from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import MINIMAX_H3_AUDIO_COND_CHANNELS

    layout = kwargs.get("video_token_layout")
    psp = kwargs.get("packed_seq_params")
    if not isinstance(layout, VideoTokenLayout) or int(module._psp_optional(psp, "num_requests", 1)) != 1:
        return None
    x, audio_x = kwargs["x"], kwargs["audio_x"]
    device = x.device
    inverse_indices = kwargs["inverse_indices"].view(-1).to(device=device, dtype=torch.long)
    row_timesteps = kwargs["unique_timesteps"].view(-1).to(device=device, dtype=torch.float32)[inverse_indices]

    if layout.video_spans:
        spans = [(span.start, span.latent_grid) for span in layout.video_spans if span.role == "target"]
    else:
        spans = [(layout.prefix_len, layout.latent_grid)]
    indicator = []
    for start, grid in spans:
        rows = x[0].narrow(0, start, math.prod(grid))
        sigma = 1.0 - float(row_timesteps[start].item())
        indicator.append(apply_sea_filter(rows.reshape(*grid, rows.shape[-1]), sigma, config.power_exp))

    audio_pos = module._pos_ids(kwargs["audio_pos_info"], "audio_pos_info").view(-1).to(device=device, dtype=torch.long)
    audio_timesteps = row_timesteps[audio_pos]
    t_audio = audio_timesteps.min()
    rows = audio_x[0].index_select(0, audio_pos[audio_timesteps == t_audio])
    channels = MINIMAX_H3_AUDIO_COND_CHANNELS if rows.shape[0] % MINIMAX_H3_AUDIO_COND_CHANNELS == 0 else 1
    rows = rows.reshape(channels, -1, rows.shape[-1])
    indicator.append(apply_sea_filter(rows, 1.0 - float(t_audio.item()), config.power_exp, dims=(1,)))
    return indicator
