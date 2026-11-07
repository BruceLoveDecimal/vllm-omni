# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned clean-estimate masks for experimental PreSTP execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PreSTPConfig:
    spatial_threshold: float = 0.97
    temporal_threshold: float = 0.997
    reuse_threshold: float = 0.5
    reanchor_period: int = 3

    def __post_init__(self) -> None:
        for value in (self.spatial_threshold, self.temporal_threshold, self.reuse_threshold):
            if not 0 <= value <= 1:
                raise ValueError("PreSTP thresholds must be between zero and one.")
        if isinstance(self.reanchor_period, bool) or not isinstance(self.reanchor_period, int):
            raise ValueError("PreSTP reanchor_period must be an integer.")
        if self.reanchor_period < 1:
            raise ValueError("PreSTP reanchor_period must be positive.")


class PreSTPState:
    """One rollout's masks and last-computed clean patches.

    Patches use [batch, latent frames, spatial patches, patch channels].
    The first chunk is dense. A probe compares adjacent frame estimates,
    using the previous chunk's final frame at the chunk boundary. Reused
    masks are refreshed at chunk boundaries, preserving block-causal sampling.
    """

    def __init__(self, config: PreSTPConfig) -> None:
        self.config = config
        self.chunk_index = 0
        self.reference: torch.Tensor | None = None
        self.static_mask: torch.Tensor | None = None
        self.frozen_mask: torch.Tensor | None = None
        self.clean: torch.Tensor | None = None
        self.previous_clean: torch.Tensor | None = None
        self.probe = True
        self.layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def begin_chunk(self, patches: torch.Tensor, *, refresh: bool = False) -> None:
        if patches.ndim != 4 or patches.shape[0] != 1:
            raise ValueError("PreSTP requires one batch of frame/spatial patches.")
        shape = patches.shape[:-1]
        reusable = (
            not refresh
            and self.static_mask is not None
            and self.static_mask.shape == shape
            and self.reference is not None
            and self.reference.shape == patches[:, 0].shape
            and self.chunk_index > 1
            and (self.chunk_index - 1) % self.config.reanchor_period != 0
            and self.static_mask.float().mean().item() >= self.config.reuse_threshold
        )
        self.probe = not reusable
        if self.probe:
            self.static_mask = torch.zeros(shape, dtype=torch.bool, device=patches.device)
        self.frozen_mask = torch.zeros(shape, dtype=torch.bool, device=patches.device)
        self.previous_clean = None
        self.clean = None
        if reusable:
            assert self.reference is not None
            self.clean = self.reference.unsqueeze(1).expand_as(patches).clone()
            spatial_tokens = patches.shape[2]
            for layer, (key, value) in self.layer_kv.items():
                self.layer_kv[layer] = (
                    key[:, -spatial_tokens:].repeat(1, patches.shape[1], 1, 1),
                    value[:, -spatial_tokens:].repeat(1, patches.shape[1], 1, 1),
                )
        else:
            self.layer_kv.clear()

    def update_kv(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        keep_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Store keys before RoPE so cross-chunk reuse receives current positions.
        if keep_indices is None:
            self.layer_kv[layer] = (key, value)
        else:
            cached_key, cached_value = self.layer_kv[layer]
            cached_key.index_copy_(1, keep_indices, key)
            cached_value.index_copy_(1, keep_indices, value)
        return self.layer_kv[layer]

    def keep_indices(self) -> torch.Tensor:
        if self.static_mask is None or self.frozen_mask is None:
            raise RuntimeError("begin_chunk must precede token selection.")
        return (~(self.static_mask | self.frozen_mask)).flatten().nonzero().flatten()

    def update(self, estimate: torch.Tensor, *, step_index: int) -> torch.Tensor:
        """Restore skipped clean patches and select tokens for the next step."""
        if self.static_mask is None or self.frozen_mask is None:
            raise RuntimeError("begin_chunk must precede clean estimation.")
        active = ~(self.static_mask | self.frozen_mask)
        if self.clean is None:
            self.clean = estimate.clone()
        else:
            self.clean = torch.where(active.unsqueeze(-1), estimate, self.clean)
        if self.chunk_index > 0:
            if step_index == 0 and self.probe:
                if self.reference is None:
                    raise RuntimeError("A completed preceding chunk is required.")
                reference = torch.cat((self.reference.unsqueeze(1), self.clean[:, :-1]), dim=1)
                self.static_mask = F.cosine_similarity(self.clean, reference, dim=-1) > self.config.spatial_threshold
            elif self.previous_clean is not None:
                converged = (
                    F.cosine_similarity(self.clean, self.previous_clean, dim=-1) > self.config.temporal_threshold
                )
                self.frozen_mask |= active & converged
        self.previous_clean = self.clean
        return self.clean

    def finish_chunk(self) -> None:
        if self.clean is None:
            raise RuntimeError("Cannot finish a chunk without a clean estimate.")
        self.reference = self.clean[:, -1].clone()
        self.chunk_index += 1

    @staticmethod
    def patchify(latent: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = latent.shape
        if height % 2 or width % 2:
            raise ValueError("PreSTP requires even latent spatial dimensions.")
        return (
            latent.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
            .permute(0, 2, 3, 5, 4, 6, 1)
            .reshape(batch, frames, (height // 2) * (width // 2), 4 * channels)
        )

    @staticmethod
    def unpatchify(patches: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        batch, channels, frames, height, width = shape
        return (
            patches.reshape(batch, frames, height // 2, width // 2, 2, 2, channels)
            .permute(0, 6, 1, 2, 4, 3, 5)
            .reshape(shape)
        )
