# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, transactional video KV memory for eager autoregressive inference.

Keys in storage always retain their original temporal phase. Attention views
may rebase that phase, but never write it back into canonical history.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VideoMemoryConfig:
    window_frames: int = 18
    sink_frames: int = 9
    rebase_sink: bool = True
    retrieval_frames: int = 0
    bank_bytes: int = 0
    rebase_retrieval: bool = False
    world_keep_ratio: float = 1.0
    world_score: str = "pairwise_mean"
    pc_budget_frames: int = 0
    pc_recent_frames: int = 3

    def __post_init__(self) -> None:
        for name in (
            "window_frames",
            "sink_frames",
            "retrieval_frames",
            "bank_bytes",
            "pc_budget_frames",
            "pc_recent_frames",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"video_memory.{name} must be a non-negative integer")
        if self.window_frames <= self.sink_frames:
            raise ValueError("video_memory.window_frames must exceed sink_frames")
        if (
            isinstance(self.world_keep_ratio, bool)
            or not isinstance(self.world_keep_ratio, (int, float))
            or not 0 <= self.world_keep_ratio <= 1
        ):
            raise ValueError("video_memory.world_keep_ratio must be in [0,1]")
        if self.world_score not in ("pairwise_mean", "centroid"):
            raise ValueError("video_memory.world_score must be pairwise_mean or centroid")
        if self.world_keep_ratio != 1 and not self.retrieval_frames:
            raise ValueError("World compression requires retrieval")
        if self.pc_budget_frames:
            if not self.sink_frames + self.pc_recent_frames <= self.pc_budget_frames < self.window_frames:
                raise ValueError("PC budget must fit sink and recent, and be smaller than the window")
            if self.retrieval_frames:
                raise ValueError("Combined retrieval and PC is not enabled")
        if self.retrieval_frames and not self.bank_bytes:
            raise ValueError("video_memory retrieval requires a positive bank_bytes limit")
        if self.sink_frames + self.retrieval_frames >= self.window_frames:
            raise ValueError("video_memory window must leave room for current frames")
        if type(self.rebase_retrieval) is not bool:
            raise ValueError("video_memory.rebase_retrieval must be a bool")
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


def world_compress(rows: MemoryRows, tokens_per_frame: int, keep_ratio: float, score: str) -> MemoryRows:
    """Anchor + pooled non-anchor selection, independently for each layer.

    pairwise_mean exactly factors mean cosine similarity as normalized keys
    dotted with the mean *normalized* anchor. centroid reproduces the released
    code's normalized raw-anchor centroid. Neither materializes T x T scores.
    """
    count = rows.positions.numel()
    if count <= tokens_per_frame or keep_ratio == 1:
        return rows
    dtype = torch.float64 if rows.key.dtype == torch.float64 else torch.float32
    flat = rows.key[0].to(dtype).flatten(1)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1)

    anchor, candidates = flat[:tokens_per_frame], flat[tokens_per_frame:]
    center = normalize(anchor).mean(0) if score == "pairwise_mean" else normalize(anchor.mean(0))
    scores = normalize(candidates) @ center
    keep = math.ceil((count - tokens_per_frame) * keep_ratio)
    selected = torch.argsort(scores, stable=True)[:keep].sort().values + tokens_per_frame
    indices = torch.cat((torch.arange(tokens_per_frame, device=flat.device), selected))
    return rows.select(indices)


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


@dataclass(frozen=True)
class BankChunk:
    start: int
    frames: int
    pose: torch.Tensor
    layers: tuple[MemoryRows, ...]

    @property
    def nbytes(self) -> int:
        return self.pose.numel() * self.pose.element_size() + sum(
            tensor.numel() * tensor.element_size()
            for rows in self.layers
            for tensor in (rows.key, rows.value, rows.positions)
        )


def camera_distances(current: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """WorldKV translation/rotation ranking, with finite zero-distance handling."""
    translation = (candidates[:, :3, 3] - current[:3, 3]).square().sum(-1)
    relative = candidates[:, :3, :3].transpose(-1, -2) @ current[:3, :3]
    rotation = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    return translation / translation.max().clamp_min(1e-12) + rotation / rotation.max().clamp_min(1e-12)


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
        self.bank: dict[int, BankChunk] = {}
        self.pose: torch.Tensor | None = None
        self._retrieved: tuple[BankChunk, ...] = ()
        self._active = False
        self._closed = False
        self._pending: dict[int, MemoryRows] = {}
        self._seen: set[int] = set()
        self._selection: dict[int, MemoryRows] = {}
        self._query_history: dict[int, torch.Tensor] = {}
        self._pending_queries: dict[int, torch.Tensor] = {}

    def set_pose(self, pose: torch.Tensor) -> None:
        """Supply a representative absolute C2W pose in the session coordinate frame."""
        if self._active or self._closed:
            raise RuntimeError("Set camera pose only between video memory forwards")
        pose = pose.detach().to(device="cpu", dtype=torch.float64, copy=True)
        if pose.shape != (4, 4) or not torch.isfinite(pose).all():
            raise ValueError("Video memory requires a finite 4x4 camera pose")
        rotation = pose[:3, :3]
        if (
            not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=pose.dtype), atol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0, dtype=pose.dtype), atol=1e-4)
            or not torch.allclose(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pose.dtype))
        ):
            raise ValueError("Video memory camera pose must be a rigid C2W transform")
        self.pose = pose

    @property
    def bank_nbytes(self) -> int:
        return sum(chunk.nbytes for chunk in self.bank.values())

    def _retrieve(self, frames: int) -> tuple[BankChunk, ...]:
        if not self.config.retrieval_frames:
            return ()
        if self.pose is None:
            raise ValueError("World retrieval requires an absolute camera pose for every chunk")
        recent_start = max(
            self.config.sink_frames,
            self.next_frame
            + frames
            - self.config.window_frames
            + self.config.sink_frames
            + self.config.retrieval_frames,
        )
        candidates = [
            chunk
            for chunk in self.bank.values()
            if chunk.start >= self.config.sink_frames and chunk.start + chunk.frames <= recent_start
        ]
        if not candidates:
            return ()
        distances = camera_distances(self.pose, torch.stack([chunk.pose for chunk in candidates]))
        ranked = torch.argsort(distances, stable=True).tolist()
        budget = self.config.retrieval_frames * self.tokens_per_frame
        selected = []
        for index in ranked:
            chunk = candidates[index]
            cost = max(rows.positions.numel() for rows in chunk.layers)
            if cost <= budget:
                selected.append(chunk)
                budget -= cost
        return tuple(sorted(selected, key=lambda chunk: chunk.start))

    def _publish_bank(self) -> dict[int, BankChunk]:
        bank = dict(self.bank)
        if not self.config.retrieval_frames or self.next_frame < self.config.sink_frames:
            return bank
        assert self.pose is not None
        layers = tuple(
            MemoryRows(
                rows.key.to(device="cpu", copy=True),
                rows.value.to(device="cpu", copy=True),
                rows.positions.to(device="cpu", copy=True),
            )
            for rows in (
                world_compress(
                    self._pending[i], self.tokens_per_frame, self.config.world_keep_ratio, self.config.world_score
                )
                for i in range(self.num_layers)
            )
        )
        chunk = BankChunk(self.next_frame, self._frames, self.pose.clone(), layers)
        if chunk.nbytes > self.config.bank_bytes:
            # Do not destroy useful history to insert an entry that cannot fit.
            return bank
        bank[chunk.start] = chunk
        size = sum(entry.nbytes for entry in bank.values())
        while size > self.config.bank_bytes:
            oldest = next(iter(bank))
            size -= bank.pop(oldest).nbytes
        return bank

    @contextmanager
    def forward(self, start_frame: int, frames: int, *, commit: bool) -> Iterator[None]:
        if self._closed or self._active:
            raise RuntimeError("Video memory is closed or already in a forward")
        if start_frame != self.next_frame or frames <= 0:
            raise ValueError("Video memory chunks must be contiguous and nonempty")
        if frames + self.config.sink_frames + self.config.retrieval_frames > self.config.window_frames:
            raise ValueError("Video memory must fit sink and the complete current chunk")
        if self.config.pc_budget_frames and self.config.pc_recent_frames < frames:
            raise ValueError("PC recent frames must protect the complete current chunk")
        original_selection = dict(self._selection)
        success = False
        self._retrieved = self._retrieve(frames)
        self._active = True
        self._frames = frames
        self._commit = commit
        try:
            yield
            if len(self._seen) != self.num_layers:
                raise RuntimeError("Video memory forward did not execute all layers")
            if commit:
                replacement = [self._retained(i, self._pending[i]) for i in range(self.num_layers)]
                queries = dict(self._query_history)
                for layer, query in self._pending_queries.items():
                    old = queries.get(layer)
                    joined = query if old is None else torch.cat((old, query), 1)
                    queries[layer] = joined[:, -self.config.pc_recent_frames * self.tokens_per_frame :].clone()
                bank = self._publish_bank()
                self.history, self.bank = replacement, bank
                self._query_history = queries
                self._selection.clear()
                self.next_frame += frames
                self.pose = None
            success = True
        finally:
            if not success:
                self._selection = original_selection
            self._pending_queries.clear()
            self._pending.clear()
            self._seen.clear()
            self._active = False
            self._retrieved = ()

    def _retained(self, layer: int, current: MemoryRows) -> MemoryRows:
        previous = self._selection.get(layer, self.history[layer])
        if previous is None:
            return current
        rows = MemoryRows(
            torch.cat((previous.key, current.key), 1),
            torch.cat((previous.value, current.value), 1),
            torch.cat((previous.positions, current.positions)),
        )
        sink = self.config.sink_frames * self.tokens_per_frame
        tail = (
            self.config.window_frames - self.config.sink_frames - self.config.retrieval_frames
        ) * self.tokens_per_frame
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
        previous = self._selection.get(layer, self.history[layer])
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
            self._selection[layer] = previous

        sink = previous.positions < self.config.sink_frames
        tail_start = max(
            self.config.sink_frames,
            self.next_frame
            + self._frames
            - self.config.window_frames
            + self.config.sink_frames
            + self.config.retrieval_frames,
        )
        if self.config.pc_budget_frames:
            tail_start = max(self.config.sink_frames, self.next_frame + self._frames - self.config.pc_recent_frames)
            keep = torch.ones_like(sink)
        else:
            keep = sink | (previous.positions >= tail_start)
        indices = torch.where(keep)[0]
        rows = previous.select(indices)
        if self.config.rebase_sink:
            delta = torch.where(rows.positions < self.config.sink_frames, tail_start - self.config.sink_frames, 0)
            if self.config.pc_budget_frames:
                middle = torch.where((rows.positions >= self.config.sink_frames) & (rows.positions < tail_start))[0]
                middle_frames = math.ceil(middle.numel() / self.tokens_per_frame)
                sink_shift = max(0, tail_start - middle_frames - self.config.sink_frames)
                delta = torch.where(rows.positions < self.config.sink_frames, sink_shift, 0)
                target = (
                    tail_start
                    - middle_frames
                    + torch.arange(middle.numel(), device=key.device) // self.tokens_per_frame
                )
                delta[middle] = target - rows.positions[middle]
            history_key = rebase_temporal_keys(rows.key, delta, self.temporal_dim)
        else:
            history_key = rows.key
        sink_count = int((rows.positions < self.config.sink_frames).sum())
        keys, values = [history_key[:, :sink_count]], [rows.value[:, :sink_count]]
        retrieved_frames = sum(chunk.frames for chunk in self._retrieved)
        virtual_start = tail_start - retrieved_frames
        for chunk in self._retrieved:
            retrieved = chunk.layers[layer]
            k = retrieved.key.to(key.device)
            if self.config.rebase_retrieval:
                delta = torch.full_like(retrieved.positions, virtual_start - chunk.start)
                k = rebase_temporal_keys(k, delta, self.temporal_dim)
            keys.append(k)
            values.append(retrieved.value.to(value.device))
            virtual_start += chunk.frames
        keys.extend((history_key[:, sink_count:], key))
        values.extend((rows.value[:, sink_count:], value))
        return torch.cat(keys, 1), torch.cat(values, 1)

    def close(self) -> None:
        self.history = [None] * self.num_layers
        self._pending.clear()
        self._seen.clear()
        self._selection.clear()
        self._query_history.clear()
        self._pending_queries.clear()
        self.bank.clear()
        self.pose = None
        self._retrieved = ()
        self._closed = True
