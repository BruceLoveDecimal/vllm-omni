# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deep Forcing sink rebasing and participative compression for eager LingBot inference.

Deep Sink temporal rebasing and Participative Compression (PC) are independent
opt-in policies. Keys in storage always retain their original temporal phase.
Attention views may rebase that phase, but never write it back into canonical
history.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepForcingConfig:
    window_frames: int = 18
    sink_frames: int = 0
    rebase_sink: bool = False
    pc_budget_frames: int = 0
    pc_recent_frames: int = 3
    rebase_pc: bool = False

    def __post_init__(self) -> None:
        for name in ("window_frames", "sink_frames", "pc_budget_frames", "pc_recent_frames"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"deep_forcing.{name} must be a non-negative integer")
        if self.window_frames <= self.sink_frames:
            raise ValueError("deep_forcing.window_frames must exceed sink_frames")
        if self.pc_budget_frames:
            if not self.sink_frames + self.pc_recent_frames <= self.pc_budget_frames < self.window_frames:
                raise ValueError("PC budget must fit sink and recent, and be smaller than the window")
        for name, dependency in (("rebase_sink", "sink_frames"), ("rebase_pc", "pc_budget_frames")):
            value = getattr(self, name)
            if type(value) is not bool:
                raise ValueError(f"deep_forcing.{name} must be a bool")
            if value and not getattr(self, dependency):
                raise ValueError(f"deep_forcing.{name} requires positive {dependency}")

    @classmethod
    def parse(cls, value: Mapping[str, object] | None) -> DeepForcingConfig | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("model_config.deep_forcing must be a mapping")
        # Dataclass construction rejects unknown keys; __post_init__ validates
        # each untyped configuration value before it reaches model execution.
        return cls(**value)  # type: ignore[arg-type]

    def reservation_bytes(
        self,
        layers: int,
        tokens_per_frame: int,
        heads: int,
        dim: int,
        itemsize: int,
        *,
        chunk_frames: int | None = None,
    ) -> int:
        # All-layer owners: canonical + transactional replacement + pending
        # rows, and old/new/pending query history. Attention assembly,
        # selection, rebasing and scoring scratch exist one layer at a time;
        # do not multiply those views by the transformer layer count.
        chunk = chunk_frames if chunk_frames is not None else self.window_frames - self.sink_frames
        if self.pc_budget_frames:
            chunk = min(chunk, self.pc_recent_frames)
        capacity = self.window_frames
        history = capacity
        publication = 2 * capacity
        if self.pc_budget_frames:
            history = capacity - 1
            # At saturation PC publishes its smaller budget; otherwise the
            # joined history is strictly below capacity. An unspecified chunk
            # size must also cover one-frame forwards.
            smallest_chunk = chunk if chunk_frames is not None else 1
            publication = max(2 * capacity - smallest_chunk - 2, history + self.pc_budget_frames)
        kv = 2 * heads * dim * itemsize
        row = kv + 16
        noisy_or_clean = layers * tokens_per_frame * (history + chunk) * row
        publishing = layers * tokens_per_frame * publication * row
        if self.pc_budget_frames:
            noisy_or_clean += layers * tokens_per_frame * (self.pc_recent_frames + chunk) * (kv // 2)
            publishing += layers * tokens_per_frame * self.pc_recent_frames * kv
        resident = max(noisy_or_clean, publishing)
        if self.pc_budget_frames:
            resident += layers * history * tokens_per_frame * 8  # frozen indices
        # Four full-width fp32/fp64 K/V windows bound one layer's gathered
        # rows, rotation/scoring intermediates and concatenated attention view.
        # Model activations/backend workspace are charged by the runner.
        scratch = 4 * self.window_frames * tokens_per_frame * (2 * heads * dim * max(4, itemsize) + 16)
        return resident + scratch


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
    token_ids: torch.Tensor

    def select(self, indices: torch.Tensor) -> MemoryRows:
        return MemoryRows(
            self.key[:, indices], self.value[:, indices], self.positions[indices], self.token_ids[indices]
        )


def participative_indices(
    query: torch.Tensor, rows: MemoryRows, *, sink_frames: int, recent_start: int, capacity: int
) -> torch.Tensor:
    """Top-C using sum of QK logits, with mandatory sink/recent preservation."""
    positions = rows.positions
    indices = torch.arange(positions.numel(), device=positions.device)
    protected = (positions < sink_frames) | (positions >= recent_start)
    forced = indices[protected]
    if forced.numel() > capacity:
        raise ValueError("PC capacity cannot discard protected sink/recent tokens")
    candidates = indices[~protected]
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    q_sum = query[0].to(dtype).flatten(1).sum(0)
    scores = rows.key[0, candidates].to(dtype).flatten(1) @ q_sum
    chosen = torch.argsort(scores, descending=True, stable=True)[: capacity - forced.numel()]
    return torch.cat((forced, candidates[chosen])).sort().values


class DeepForcing:
    """One request/branch's clean history, published only after all layers succeed.

    Single-sequence eager execution is intentional. PC selections are shared
    across heads; head-sharded execution requires a distributed scoring
    reduction before enabling compression.
    """

    def __init__(self, config: DeepForcingConfig, *, num_layers: int, tokens_per_frame: int, temporal_dim: int):
        if num_layers <= 0 or tokens_per_frame <= 0 or temporal_dim <= 0 or temporal_dim % 2:
            raise ValueError("Invalid Deep Forcing geometry")
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
        self._selection: dict[int, torch.Tensor] = {}
        self._query_history: dict[int, torch.Tensor] = {}
        self._pending_queries: dict[int, torch.Tensor] = {}

    @contextmanager
    def forward(self, start_frame: int, frames: int, *, commit: bool) -> Iterator[None]:
        if self._closed or self._active:
            raise RuntimeError("Deep Forcing is closed or already in a forward")
        if start_frame != self.next_frame or frames <= 0:
            raise ValueError("Deep Forcing chunks must be contiguous and nonempty")
        if frames + self.config.sink_frames > self.config.window_frames:
            raise ValueError("Deep Forcing must fit sink and the complete current chunk")
        if self.config.pc_budget_frames and self.config.pc_recent_frames < frames:
            raise ValueError("PC recent frames must protect the complete current chunk")
        original_selection = dict(self._selection)
        success = False
        self._active = True
        self._frames = frames
        self._commit = commit
        try:
            yield
            if len(self._seen) != self.num_layers:
                raise RuntimeError("Deep Forcing forward did not execute all layers")
            if commit:
                # No attention consumer remains after the output head succeeds.
                # Canonical history is assigned only at the end, so a failed
                # replacement leaves the previous history intact.
                replacement: list[MemoryRows | None] = [
                    self._retained(i, self._pending.pop(i)) for i in range(self.num_layers)
                ]
                queries = dict(self._query_history)
                for layer in tuple(self._pending_queries):
                    query = self._pending_queries.pop(layer)
                    old = queries.get(layer)
                    joined = query if old is None else torch.cat((old, query), 1)
                    queries[layer] = joined[:, -self.config.pc_recent_frames * self.tokens_per_frame :].clone()
                self.history = replacement
                self._query_history = queries
                self._selection.clear()
                self.next_frame += frames
            success = True
        finally:
            if not success:
                self._selection = original_selection
            self._pending_queries.clear()
            self._pending.clear()
            self._seen.clear()
            self._active = False

    def _previous(self, layer: int) -> MemoryRows | None:
        previous = self.history[layer]
        if previous is not None and layer in self._selection:
            return previous.select(self._selection[layer])
        return previous

    def _retained(self, layer: int, current: MemoryRows) -> MemoryRows:
        previous = self._previous(layer)
        if previous is None:
            return current
        capacity = self.config.window_frames * self.tokens_per_frame
        previous_capacity = capacity - current.positions.numel()
        count = previous.positions.numel()
        if count <= previous_capacity:
            slices = [slice(None)]
        else:
            sink = min(self.config.sink_frames * self.tokens_per_frame, count)
            tail = previous_capacity - sink
            slices = [slice(0, sink)]
            if tail:
                slices.append(slice(count - tail, count))
        # Slice before concatenation: never allocate a full old+current window
        # only to copy it again during truncation. All slices are temporary
        # views, and the one concatenation per tensor owns exactly retained KV.
        return MemoryRows(
            torch.cat([*(previous.key[:, part] for part in slices), current.key], 1),
            torch.cat([*(previous.value[:, part] for part in slices), current.value], 1),
            torch.cat([*(previous.positions[part] for part in slices), current.positions]),
            torch.cat([*(previous.token_ids[part] for part in slices), current.token_ids]),
        )

    def assemble(
        self, layer: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._active or layer in self._seen or not 0 <= layer < self.num_layers:
            raise RuntimeError("Each Deep Forcing layer must execute exactly once per forward")
        if key.ndim != 4 or key.shape != value.shape or query.shape != key.shape or key.shape[0] != 1:
            raise ValueError("Deep Forcing requires matching single-sequence Q/K/V [1,T,H,D]")
        if key.shape[1] != self._frames * self.tokens_per_frame:
            raise ValueError("Current token count does not match Deep Forcing geometry")
        self._seen.add(layer)
        positions = torch.arange(self.next_frame, self.next_frame + self._frames, device=key.device)
        positions = positions.repeat_interleave(self.tokens_per_frame)
        if self._commit:
            # These slices must own storage: QKV projections otherwise keep Q
            # and the unselected tokens alive after a compression/commit.
            token_ids = torch.arange(
                self.next_frame * self.tokens_per_frame,
                (self.next_frame + self._frames) * self.tokens_per_frame,
                device=key.device,
            )
            self._pending[layer] = MemoryRows(key.detach().clone(), value.detach().clone(), positions, token_ids)
        previous = self._previous(layer)
        if self._commit and self.config.pc_budget_frames:
            self._pending_queries[layer] = query.detach().clone()
        if previous is None:
            return key, value
        if (
            self.config.pc_budget_frames
            and layer not in self._selection
            and previous.positions.numel() + key.shape[1] >= self.config.window_frames * self.tokens_per_frame
        ):
            older_queries = self._query_history.get(layer)
            recent_queries = query if older_queries is None else torch.cat((older_queries, query), 1)
            recent_queries = recent_queries[:, -self.config.pc_recent_frames * self.tokens_per_frame :]
            selected = participative_indices(
                recent_queries,
                previous,
                sink_frames=self.config.sink_frames,
                recent_start=self.next_frame + self._frames - self.config.pc_recent_frames,
                capacity=self.config.pc_budget_frames * self.tokens_per_frame - key.shape[1],
            )
            previous = previous.select(selected)
            self._selection[layer] = selected

        sink = previous.positions < self.config.sink_frames
        tail_start = max(
            self.config.sink_frames,
            self.next_frame + self._frames - self.config.window_frames + self.config.sink_frames,
        )
        if self.config.pc_budget_frames:
            tail_start = max(self.config.sink_frames, self.next_frame + self._frames - self.config.pc_recent_frames)
            keep = torch.ones_like(sink)
        else:
            keep = sink | (previous.positions >= tail_start)
        rows = previous if self.config.pc_budget_frames else previous.select(torch.where(keep)[0])
        middle_frames = 0
        if self.config.rebase_sink or self.config.rebase_pc:
            delta = torch.zeros_like(rows.positions)
            if self.config.rebase_pc:
                middle = torch.where((rows.positions >= self.config.sink_frames) & (rows.positions < tail_start))[0]
                middle_frames = math.ceil(middle.numel() / self.tokens_per_frame)
                target = (
                    tail_start
                    - middle_frames
                    + torch.arange(middle.numel(), device=key.device) // self.tokens_per_frame
                )
                delta[middle] = target - rows.positions[middle]
            if self.config.rebase_sink:
                sink_shift = max(0, tail_start - middle_frames - self.config.sink_frames)
                delta = torch.where(rows.positions < self.config.sink_frames, sink_shift, delta)
            history_key = rebase_temporal_keys(rows.key, delta, self.temporal_dim)
        else:
            history_key = rows.key
        return torch.cat((history_key, key), 1), torch.cat((rows.value, value), 1)

    def close(self) -> None:
        self.history = [None] * self.num_layers
        self._pending.clear()
        self._seen.clear()
        self._selection.clear()
        self._query_history.clear()
        self._pending_queries.clear()
        self._closed = True
