# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, transactional video KV memory for eager autoregressive inference.

Keys in storage always retain their original temporal phase. Attention views
may rebase that phase, but never write it back into canonical history.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VideoMemoryConfig:
    window_frames: int = 18
    sink_frames: int = 9
    rebase_sink: bool = True

    def __post_init__(self) -> None:
        for name in ("window_frames", "sink_frames"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"video_memory.{name} must be a non-negative integer")
        if self.window_frames <= self.sink_frames:
            raise ValueError("video_memory.window_frames must exceed sink_frames")
        if type(self.rebase_sink) is not bool:
            raise ValueError("video_memory.rebase_sink must be a bool")

    @classmethod
    def parse(cls, value: Mapping[str, object] | None) -> VideoMemoryConfig | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("model_config.video_memory must be a mapping")
        return cls(**value)

    def reservation_bytes(self, layers: int, tokens_per_frame: int, heads: int, dim: int, itemsize: int) -> int:
        # Canonical, pending, replacement and attention views can coexist during
        # publication. Include per-token int64 positions, not just K and V.
        return 4 * layers * self.window_frames * tokens_per_frame * (2 * heads * dim * itemsize + 8)


def rebase_temporal_keys(key: torch.Tensor, delta: torch.Tensor, temporal_dim: int) -> torch.Tensor:
    """Rotate adjacent real/imaginary pairs on the time axis only.

    ``key`` is [B, T, H, D], ``delta`` is [T]. Computation uses fp32 for
    low-precision caches and fp64 for fp64 references. Spatial channels are
    copied exactly. The input may be shared by many denoising forwards.
    """
    if key.ndim != 4 or delta.shape != (key.shape[1],):
        raise ValueError("Expected keys [B,T,H,D] and one temporal delta per token")
    if temporal_dim <= 0 or temporal_dim % 2 or temporal_dim > key.shape[-1]:
        raise ValueError("temporal_dim must be positive, even and no larger than the head dimension")
    dtype = torch.float64 if key.dtype == torch.float64 else torch.float32
    frequency = 10000.0 ** (-torch.arange(0, temporal_dim, 2, device=key.device, dtype=dtype) / temporal_dim)
    angle = delta.to(device=key.device, dtype=dtype)[:, None] * frequency
    cosine, sine = angle.cos()[None, :, None], angle.sin()[None, :, None]
    pairs = key[..., :temporal_dim].to(dtype).unflatten(-1, (-1, 2))
    real, imag = pairs.unbind(-1)
    rotated = torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), -1).flatten(-2)
    return torch.cat((rotated.to(key.dtype), key[..., temporal_dim:]), dim=-1)


@dataclass(frozen=True)
class MemoryRows:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor

    def select(self, indices: torch.Tensor) -> MemoryRows:
        return MemoryRows(self.key[:, indices], self.value[:, indices], self.positions[indices])


class VideoMemory:
    """One request/branch's clean history, published only after all layers succeed.

    Single-sequence eager execution is intentional. Per-layer selections are
    shared across heads; head-sharded execution requires a distributed scoring
    reduction before enabling compression.
    """

    def __init__(self, config: VideoMemoryConfig, *, num_layers: int, tokens_per_frame: int, temporal_dim: int):
        if num_layers <= 0 or tokens_per_frame <= 0 or temporal_dim <= 0 or temporal_dim % 2:
            raise ValueError("Invalid video memory geometry")
        self.config = config
        self.num_layers = num_layers
        self.tokens_per_frame = tokens_per_frame
        self.temporal_dim = temporal_dim
        self.history: list[MemoryRows | None] = [None] * num_layers
        self.next_frame = 0
        self._active = False
        self._closed = False
        self._pending: dict[int, MemoryRows] = {}
        self._seen: set[int] = set()

    @contextmanager
    def forward(self, start_frame: int, frames: int, *, commit: bool) -> Iterator[None]:
        if self._closed or self._active:
            raise RuntimeError("Video memory is closed or already in a forward")
        if start_frame != self.next_frame or frames <= 0:
            raise ValueError("Video memory chunks must be contiguous and nonempty")
        if frames + self.config.sink_frames > self.config.window_frames:
            raise ValueError("Video memory must fit sink and the complete current chunk")
        self._active = True
        self._frames = frames
        self._commit = commit
        try:
            yield
            if len(self._seen) != self.num_layers:
                raise RuntimeError("Video memory forward did not execute all layers")
            if commit:
                replacement = [self._retained(i, self._pending[i]) for i in range(self.num_layers)]
                self.history = replacement
                self.next_frame += frames
        finally:
            self._pending.clear()
            self._seen.clear()
            self._active = False

    def _retained(self, layer: int, current: MemoryRows) -> MemoryRows:
        previous = self.history[layer]
        if previous is None:
            return current
        rows = MemoryRows(
            torch.cat((previous.key, current.key), 1),
            torch.cat((previous.value, current.value), 1),
            torch.cat((previous.positions, current.positions)),
        )
        sink = self.config.sink_frames * self.tokens_per_frame
        tail = (self.config.window_frames - self.config.sink_frames) * self.tokens_per_frame
        indices = torch.arange(rows.positions.numel(), device=rows.key.device)
        keep = (indices < sink) | (indices >= indices.numel() - tail)
        return rows.select(indices[keep])

    def assemble(
        self, layer: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._active or layer in self._seen or not 0 <= layer < self.num_layers:
            raise RuntimeError("Each video memory layer must execute exactly once per forward")
        if key.ndim != 4 or key.shape != value.shape or query.shape != key.shape or key.shape[0] != 1:
            raise ValueError("Video memory requires matching single-sequence Q/K/V [1,T,H,D]")
        if key.shape[1] != self._frames * self.tokens_per_frame:
            raise ValueError("Current token count does not match video memory geometry")
        self._seen.add(layer)
        positions = torch.arange(self.next_frame, self.next_frame + self._frames, device=key.device)
        positions = positions.repeat_interleave(self.tokens_per_frame)
        if self._commit:
            # These slices must own storage: QKV projections otherwise keep Q
            # and the unselected tokens alive after a compression/commit.
            self._pending[layer] = MemoryRows(key.detach().clone(), value.detach().clone(), positions)
        previous = self.history[layer]
        if previous is None:
            return key, value
        sink = previous.positions < self.config.sink_frames
        tail_start = max(
            self.config.sink_frames,
            self.next_frame + self._frames - self.config.window_frames + self.config.sink_frames,
        )
        keep = sink | (previous.positions >= tail_start)
        indices = torch.where(keep)[0]
        rows = previous.select(indices)
        if self.config.rebase_sink:
            delta = torch.where(rows.positions < self.config.sink_frames, tail_start - self.config.sink_frames, 0)
            history_key = rebase_temporal_keys(rows.key, delta, self.temporal_dim)
        else:
            history_key = rows.key
        return torch.cat((history_key, key), 1), torch.cat((rows.value, value), 1)

    def close(self) -> None:
        self.history = [None] * self.num_layers
        self._pending.clear()
        self._seen.clear()
        self._closed = True
