# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed PixelUMM local attention on vLLM-Omni's packed varlen attention path.

After the decoder's tile-major permutation every query tile is a contiguous
token range, so one varlen call covers all tiles instead of one call per tile.
Each tile's 3x3 spatial neighbourhood is gathered once into contiguous K/V, in
the order of the eager concatenation, and the layer's own attention backend
runs the call from ``AttentionMetadata.extra`` cu_seqlens. Only backends that
keep N-document packed boundaries (FlashAttention on CUDA/ROCm/MUSA) qualify;
others keep the per-tile path.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

Neighborhoods = Sequence[tuple[tuple[int, int], Sequence[tuple[int, int]]]]


@dataclass(frozen=True)
class PackedLocalAttention:
    """Packed varlen plan for every query tile of a ``[batch, tokens]`` sequence.

    Segment ``s`` is one query tile. It reads the gathered tokens
    ``kv_index[cu_seqlens_k[s] : cu_seqlens_k[s + 1]]`` of the flattened batch.
    """

    kv_index: torch.Tensor
    metadata: AttentionMetadata

    def __call__(
        self, attention: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """``[B, N, heads, D]`` queries over ``[B, N, kv_heads, D]`` K/V -> ``[B, N, heads, D]``."""
        batch, count, heads, dim = query.shape
        kv_heads = key.shape[-2]
        # The packed path takes one sequence; batch elements are separate segments.
        query = query.reshape(1, batch * count, heads, dim)
        key = key.reshape(-1, kv_heads, dim)
        value = value.reshape(-1, kv_heads, dim)
        key = key.index_select(0, self.kv_index).unsqueeze(0)
        value = value.index_select(0, self.kv_index).unsqueeze(0)
        output = attention(query, key, value, self.metadata)
        return output.view(batch, count, heads, dim)


def plan_local_attention(
    neighborhoods: Neighborhoods, batch: int, tokens: int, device: torch.device
) -> PackedLocalAttention:
    """Build the packed plan for the tiles of ``local_attention_layout``.

    Tiles are consecutive ranges covering ``[0, tokens)``; batch ``b`` is
    offset by ``b * tokens`` in the packed sequence.
    """
    query_ends = [0]
    query_lengths = []
    kv_ends = [0]
    kv_lengths = []
    run_firsts = []
    run_lengths = []
    for index in range(batch):
        offset = index * tokens
        for (start, end), neighbors in neighborhoods:
            query_ends.append(offset + end)
            query_lengths.append(end - start)
            kv_length = 0
            for first, last in neighbors:
                run_firsts.append(offset + first)
                run_lengths.append(last - first)
                kv_length += last - first
            kv_ends.append(kv_ends[-1] + kv_length)
            kv_lengths.append(kv_length)
    # Consecutive runs expand to the tokens each segment reads, in neighbour order.
    total = kv_ends[-1]
    lengths = torch.tensor(run_lengths, device=device)
    run_starts = torch.cumsum(lengths, 0) - lengths
    shift = torch.tensor(run_firsts, device=device) - run_starts
    kv_index = torch.arange(total, device=device) + shift.repeat_interleave(lengths, output_size=total)
    extra = {
        "cu_seqlens_q": torch.tensor(query_ends, dtype=torch.int32, device=device),
        "cu_seqlens_k": torch.tensor(kv_ends, dtype=torch.int32, device=device),
        "max_seqlen_q": max(query_lengths),
        "max_seqlen_k": max(kv_lengths),
    }
    return PackedLocalAttention(kv_index=kv_index, metadata=AttentionMetadata(extra=extra))


def packed_local_attention(
    attention: torch.nn.Module, neighborhoods: Neighborhoods, hidden_states: torch.Tensor
) -> PackedLocalAttention | None:
    """Plan one varlen call for all local-attention tiles, or decline when the backend cannot isolate tiles."""
    backend = attention.attn_backend
    if (
        backend is None
        or not backend.supports_multi_doc_packed_varlen()
        or hidden_states.device.type == "cpu"
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or torch.is_grad_enabled()
    ):
        return None
    batch, tokens = hidden_states.shape[:2]
    return plan_local_attention(neighborhoods, batch, tokens, hidden_states.device)


def choose_halo_grid(tiles_h: int, tiles_w: int, world_size: int) -> tuple[int, int] | None:
    """Choose a 2D processor grid (gh, gw) minimizing halo boundary tiles."""
    if world_size <= 1 or (tiles_h <= 3 and tiles_w <= 3):
        return None
    best: tuple[tuple[int, float, int], tuple[int, int]] | None = None
    for gh in range(1, min(tiles_h, world_size) + 1):
        if world_size % gh:
            continue
        gw = world_size // gh
        if gw > tiles_w:
            continue
        halo_cost = (gh - 1) * tiles_w + (gw - 1) * tiles_h
        aspect_gap = abs(tiles_h / gh - tiles_w / gw)
        score = (halo_cost, aspect_gap, gh)
        if best is None or score < best[0]:
            best = (score, (gh, gw))
    return None if best is None else best[1]


def _partition_tiles(total: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(total, parts)
    ranges = []
    start = 0
    for idx in range(parts):
        end = start + base + (1 if idx < extra else 0)
        ranges.append((start, end))
        start = end
    return ranges


@dataclass(frozen=True)
class HaloShardBox:
    """Spatial bounds owned by one rank in latent-cell and token coordinates."""

    rank: int
    latent_h0: int
    latent_h1: int
    latent_w0: int
    latent_w1: int
    token_h0: int
    token_h1: int
    token_w0: int
    token_w1: int


@dataclass(frozen=True)
class HaloPeerSend:
    """Precomputed flat token indices to send to one neighbor rank."""

    rank: int
    token_index: torch.Tensor


@dataclass(frozen=True)
class HaloPeerRecv:
    """Receive buffer geometry from one neighbor rank."""

    rank: int
    tokens_per_batch: int


@dataclass(frozen=True)
class HaloLocalAttentionPlan:
    """Per-layer exact 3x3 local attention with async K/V halo exchange and interior overlap."""

    n_interior: int
    n_boundary: int
    send_peers: tuple[HaloPeerSend, ...]
    recv_peers: tuple[HaloPeerRecv, ...]
    interior_neighborhoods: tuple[tuple[tuple[int, int], tuple[tuple[int, int], ...]], ...]
    boundary_neighborhoods: tuple[tuple[tuple[int, int], tuple[tuple[int, int, int], ...]], ...]
    interior_packed: PackedLocalAttention | None
    boundary_packed: PackedLocalAttention | None

    def __call__(
        self,
        attention: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        process_group: dist.ProcessGroup,
        on_launch: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        batch, count, heads, dim = query.shape
        kv_heads = key.shape[2]
        k_flat = key.reshape(batch * count, kv_heads, dim)
        v_flat = value.reshape(batch * count, kv_heads, dim)

        # 1. Launch async P2P K/V halo exchange for boundary tiles.
        ops: list[dist.P2POp] = []
        send_bufs: list[torch.Tensor] = []
        recv_bufs: list[torch.Tensor] = []
        for peer in self.send_peers:
            send_kv = torch.stack(
                (k_flat.index_select(0, peer.token_index), v_flat.index_select(0, peer.token_index)), dim=0
            ).contiguous()
            send_bufs.append(send_kv)
            ops.append(
                dist.P2POp(dist.isend, send_kv, dist.get_global_rank(process_group, peer.rank), group=process_group)
            )
        for peer in self.recv_peers:
            recv_kv = torch.empty((2, batch * peer.tokens_per_batch, kv_heads, dim), dtype=key.dtype, device=key.device)
            recv_bufs.append(recv_kv)
            ops.append(
                dist.P2POp(dist.irecv, recv_kv, dist.get_global_rank(process_group, peer.rank), group=process_group)
            )
        reqs = dist.batch_isend_irecv(ops) if ops else []
        if on_launch is not None:
            on_launch()
        try:
            return self._attend(attention, query, key, value, k_flat, v_flat, reqs, send_bufs, recv_bufs)
        except BaseException:
            # Complete this layer's exchange even when local attention fails,
            # so peers do not wait on transfers that never finish.
            for req in reqs:
                req.wait()
            raise

    def exchange_placeholder(
        self,
        batch: int,
        kv_heads: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
        process_group: dist.ProcessGroup,
    ) -> None:
        """Perform one layer's halo exchange with zero K/V after a local failure.

        The sizes match ``__call__``; the received values are discarded.
        """
        ops: list[dist.P2POp] = []
        for peer in self.send_peers:
            send_kv = torch.zeros((2, peer.token_index.numel(), kv_heads, dim), dtype=dtype, device=device)
            ops.append(
                dist.P2POp(dist.isend, send_kv, dist.get_global_rank(process_group, peer.rank), group=process_group)
            )
        for peer in self.recv_peers:
            recv_kv = torch.empty((2, batch * peer.tokens_per_batch, kv_heads, dim), dtype=dtype, device=device)
            ops.append(
                dist.P2POp(dist.irecv, recv_kv, dist.get_global_rank(process_group, peer.rank), group=process_group)
            )
        for req in dist.batch_isend_irecv(ops) if ops else []:
            req.wait()

    def _attend(
        self,
        attention: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        reqs: list,
        send_bufs: list[torch.Tensor],
        recv_bufs: list[torch.Tensor],
    ) -> torch.Tensor:
        batch, _, _, dim = query.shape
        kv_heads = key.shape[2]
        # 2. Compute interior neighborhoods while halo K/V is in flight.
        interior_out: torch.Tensor | None = None
        if self.n_interior > 0:
            if self.interior_packed is not None:
                interior_out = self.interior_packed(attention, query[:, : self.n_interior].contiguous(), key, value)
            else:
                parts = []
                for (start, end), neighbors in self.interior_neighborhoods:
                    local_k = torch.cat([key[:, first:last] for first, last in neighbors], dim=1)
                    local_v = torch.cat([value[:, first:last] for first, last in neighbors], dim=1)
                    parts.append(attention(query[:, start:end].contiguous(), local_k, local_v))
                interior_out = torch.cat(parts, dim=1)

        # 3. Wait for halo K/V and compute boundary neighborhoods.
        for req in reqs:
            req.wait()
        del send_bufs

        boundary_out: torch.Tensor | None = None
        if self.n_boundary > 0:
            if self.boundary_packed is not None:
                all_k = torch.cat([k_flat, *(buf[0] for buf in recv_bufs)], dim=0)
                all_v = torch.cat([v_flat, *(buf[1] for buf in recv_bufs)], dim=0)
                boundary_out = self.boundary_packed(attention, query[:, self.n_interior :].contiguous(), all_k, all_v)
            else:
                recv_k = [
                    buf[0].view(batch, peer.tokens_per_batch, kv_heads, dim)
                    for buf, peer in zip(recv_bufs, self.recv_peers)
                ]
                recv_v = [
                    buf[1].view(batch, peer.tokens_per_batch, kv_heads, dim)
                    for buf, peer in zip(recv_bufs, self.recv_peers)
                ]
                parts = []
                for (start, end), neighbors in self.boundary_neighborhoods:
                    local_k = torch.cat(
                        [
                            key[:, first:last] if src < 0 else recv_k[src][:, first:last]
                            for src, first, last in neighbors
                        ],
                        dim=1,
                    )
                    local_v = torch.cat(
                        [
                            value[:, first:last] if src < 0 else recv_v[src][:, first:last]
                            for src, first, last in neighbors
                        ],
                        dim=1,
                    )
                    parts.append(attention(query[:, start:end].contiguous(), local_k, local_v))
                boundary_out = torch.cat(parts, dim=1)

        if interior_out is None:
            assert boundary_out is not None
            return boundary_out
        if boundary_out is None:
            return interior_out
        return torch.cat((interior_out, boundary_out), dim=1)


def _build_packed_subset(
    query_ranges: Sequence[tuple[int, int]],
    kv_runs_per_segment: Sequence[Sequence[tuple[int, int]]],
    query_base_offset: int,
    query_stride_per_batch: int,
    batch: int,
    device: torch.device,
) -> PackedLocalAttention:
    query_ends = [0]
    query_lengths = []
    kv_ends = [0]
    kv_lengths = []
    run_firsts = []
    run_lengths = []
    for b in range(batch):
        q_offset = b * query_stride_per_batch
        for seg_idx, (q_start, q_end) in enumerate(query_ranges):
            query_ends.append(q_offset + (q_end - query_base_offset))
            query_lengths.append(q_end - q_start)
            kv_len = 0
            for first, last in kv_runs_per_segment[b * len(query_ranges) + seg_idx]:
                run_firsts.append(first)
                run_lengths.append(last - first)
                kv_len += last - first
            kv_ends.append(kv_ends[-1] + kv_len)
            kv_lengths.append(kv_len)
    total = kv_ends[-1]
    lengths = torch.tensor(run_lengths, device=device)
    run_starts = torch.cumsum(lengths, 0) - lengths
    shift = torch.tensor(run_firsts, device=device) - run_starts
    kv_index = torch.arange(total, device=device) + shift.repeat_interleave(lengths, output_size=total)
    extra = {
        "cu_seqlens_q": torch.tensor(query_ends, dtype=torch.int32, device=device),
        "cu_seqlens_k": torch.tensor(kv_ends, dtype=torch.int32, device=device),
        "max_seqlen_q": max(query_lengths),
        "max_seqlen_k": max(kv_lengths),
    }
    return PackedLocalAttention(kv_index=kv_index, metadata=AttentionMetadata(extra=extra))


def plan_halo_local_attention(
    grid: tuple[int, int, int],
    tile: int,
    latent_token_grid: int,
    world_size: int,
    rank: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    attention: torch.nn.Module,
) -> tuple[HaloShardBox, tuple[HaloShardBox, ...], torch.Tensor, torch.Tensor, HaloLocalAttentionPlan] | None:
    """Build the spatial-tile Halo-SP partition, local permutation, and attention plan."""
    frames, height, width = grid
    if tile <= 0 or tile % latent_token_grid != 0:
        return None
    tiles_h = math.ceil(height / tile)
    tiles_w = math.ceil(width / tile)
    proc_grid = choose_halo_grid(tiles_h, tiles_w, world_size)
    if proc_grid is None:
        return None
    gh, gw = proc_grid
    h_ranges = _partition_tiles(tiles_h, gh)
    w_ranges = _partition_tiles(tiles_w, gw)

    # Map every global tile (th, tw) to its owning rank and build shard boxes.
    tile_owner: dict[tuple[int, int], int] = {}
    boxes: list[HaloShardBox] = []
    for r in range(world_size):
        rh, rw = divmod(r, gw)
        th0, th1 = h_ranges[rh]
        tw0, tw1 = w_ranges[rw]
        tok_h0 = th0 * tile
        tok_h1 = min(th1 * tile, height)
        tok_w0 = tw0 * tile
        tok_w1 = min(tw1 * tile, width)
        boxes.append(
            HaloShardBox(
                rank=r,
                latent_h0=tok_h0 // latent_token_grid,
                latent_h1=tok_h1 // latent_token_grid,
                latent_w0=tok_w0 // latent_token_grid,
                latent_w1=tok_w1 // latent_token_grid,
                token_h0=tok_h0,
                token_h1=tok_h1,
                token_w0=tok_w0,
                token_w1=tok_w1,
            )
        )
        for th in range(th0, th1):
            for tw in range(tw0, tw1):
                tile_owner[th, tw] = r

    my_box = boxes[rank]
    my_rh, my_rw = divmod(rank, gw)
    my_th0, my_th1 = h_ranges[my_rh]
    my_tw0, my_tw1 = w_ranges[my_rw]

    def _tile_neighbors(row_idx: int, col_idx: int) -> tuple[int, int, list[tuple[int, int]]]:
        center_h = min(max(row_idx, 1), max(tiles_h - 2, 1))
        center_w = min(max(col_idx, 1), max(tiles_w - 2, 1))
        nbrs = [
            (nh, nw)
            for nh in range(max(0, center_h - 1), min(tiles_h, center_h + 2))
            for nw in range(max(0, center_w - 1), min(tiles_w, center_w + 2))
        ]
        return center_h, center_w, nbrs

    def _tile_token_count(th: int, tw: int) -> int:
        return frames * min(tile, height - th * tile) * min(tile, width - tw * tile)

    # Order local tiles as (is_boundary, center_h, center_w, th, tw) so interior
    # queries form a contiguous prefix [0, n_interior) and boundary queries follow.
    local_tiles = []
    for th in range(my_th0, my_th1):
        for tw in range(my_tw0, my_tw1):
            center_h, center_w, nbrs = _tile_neighbors(th, tw)
            is_boundary = int(any(tile_owner[nh, nw] != rank for nh, nw in nbrs))
            local_tiles.append((is_boundary, center_h, center_w, th, tw, nbrs))
    local_tiles.sort(key=lambda item: item[:5])

    local_tiles_h = my_th1 - my_th0
    local_tiles_w = my_tw1 - my_tw0
    tile_ranks = [0] * (local_tiles_h * local_tiles_w)
    local_ranges: dict[tuple[int, int], tuple[int, int]] = {}
    offset = 0
    n_interior = 0
    for idx, (is_boundary, _, _, th, tw, _) in enumerate(local_tiles):
        tile_ranks[(th - my_th0) * local_tiles_w + (tw - my_tw0)] = idx
        count = _tile_token_count(th, tw)
        local_ranges[th, tw] = (offset, offset + count)
        offset += count
        if is_boundary == 0:
            n_interior = offset
    n_local = offset
    n_boundary = n_local - n_interior

    local_grid = (frames, my_box.token_h1 - my_box.token_h0, my_box.token_w1 - my_box.token_w0)
    axes = [torch.arange(size, device=device) for size in local_grid]
    time, row, column = (axis.flatten() for axis in torch.meshgrid(*axes, indexing="ij"))
    tile_rank_t = torch.tensor(tile_ranks, device=device)[(row // tile) * local_tiles_w + column // tile]
    order = tile_rank_t * frames + time
    order = order * tile + row % tile
    order = order * tile + column % tile
    perm = torch.argsort(order)
    inverse = torch.argsort(perm)

    # Determine remote tiles we receive from each peer and local tiles we send to each peer.
    recv_tiles_by_peer: dict[int, set[tuple[int, int]]] = {}
    for _, _, _, _, _, nbrs in local_tiles:
        for nh, nw in nbrs:
            owner = tile_owner[nh, nw]
            if owner != rank:
                recv_tiles_by_peer.setdefault(owner, set()).add((nh, nw))

    send_tiles_by_peer: dict[int, set[tuple[int, int]]] = {}
    for peer in range(world_size):
        if peer == rank:
            continue
        prh, prw = divmod(peer, gw)
        pth0, pth1 = h_ranges[prh]
        ptw0, ptw1 = w_ranges[prw]
        for th in range(pth0, pth1):
            for tw in range(ptw0, ptw1):
                _, _, nbrs = _tile_neighbors(th, tw)
                for nh, nw in nbrs:
                    if tile_owner[nh, nw] == rank:
                        send_tiles_by_peer.setdefault(peer, set()).add((nh, nw))

    # Build deterministic send indices (expanded across batch).
    send_peers: list[HaloPeerSend] = []
    for peer in sorted(send_tiles_by_peer):
        sorted_tiles = sorted(send_tiles_by_peer[peer])
        ranges = [local_ranges[t] for t in sorted_tiles]
        single_indices = torch.cat([torch.arange(first, last, device=device) for first, last in ranges])
        if batch == 1:
            batched_indices = single_indices
        else:
            batched_indices = torch.cat([single_indices + b * n_local for b in range(batch)])
        send_peers.append(HaloPeerSend(rank=peer, token_index=batched_indices))

    # Build deterministic receive layout per peer.
    recv_peers: list[HaloPeerRecv] = []
    remote_tile_loc: dict[tuple[int, int], tuple[int, int, int]] = {}
    peer_halo_base: list[int] = []
    halo_cursor = batch * n_local
    for peer_idx, peer in enumerate(sorted(recv_tiles_by_peer)):
        sorted_tiles = sorted(recv_tiles_by_peer[peer])
        peer_offset = 0
        for t in sorted_tiles:
            cnt = _tile_token_count(*t)
            remote_tile_loc[t] = (peer_idx, peer_offset, peer_offset + cnt)
            peer_offset += cnt
        recv_peers.append(HaloPeerRecv(rank=peer, tokens_per_batch=peer_offset))
        peer_halo_base.append(halo_cursor)
        halo_cursor += batch * peer_offset

    # Group local tiles into interior and boundary neighborhoods.
    interior_neighborhoods: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    boundary_neighborhoods: list[tuple[tuple[int, int], tuple[tuple[int, int, int], ...]]] = []
    first = 0
    while first < len(local_tiles):
        is_bnd, center_h, center_w = local_tiles[first][:3]
        last = first
        while last + 1 < len(local_tiles) and local_tiles[last + 1][:3] == (is_bnd, center_h, center_w):
            last += 1
        q_start = local_ranges[local_tiles[first][3], local_tiles[first][4]][0]
        q_end = local_ranges[local_tiles[last][3], local_tiles[last][4]][1]
        nbrs = local_tiles[first][5]
        if is_bnd == 0:
            interior_neighborhoods.append(((q_start, q_end), tuple(local_ranges[n] for n in nbrs)))
        else:
            refs = []
            for n in nbrs:
                if tile_owner[n] == rank:
                    r0, r1 = local_ranges[n]
                    refs.append((-1, r0, r1))
                else:
                    refs.append(remote_tile_loc[n])
            boundary_neighborhoods.append(((q_start, q_end), tuple(refs)))
        first = last + 1

    # Build packed varlen plans when the backend supports multi-doc packed varlen.
    backend = getattr(attention, "attn_backend", None)
    use_packed = (
        backend is not None
        and backend.supports_multi_doc_packed_varlen()
        and device.type != "cpu"
        and dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
    )
    interior_packed: PackedLocalAttention | None = None
    boundary_packed: PackedLocalAttention | None = None
    if use_packed:
        if interior_neighborhoods:
            q_ranges = [seg[0] for seg in interior_neighborhoods]
            kv_runs = []
            for b in range(batch):
                base = b * n_local
                for _, nbr_ranges in interior_neighborhoods:
                    kv_runs.append([(base + r0, base + r1) for r0, r1 in nbr_ranges])
            interior_packed = _build_packed_subset(q_ranges, kv_runs, 0, n_interior, batch, device)
        if boundary_neighborhoods:
            q_ranges = [seg[0] for seg in boundary_neighborhoods]
            kv_runs = []
            for b in range(batch):
                local_base = b * n_local
                for _, nbr_refs in boundary_neighborhoods:
                    runs = []
                    for src, r0, r1 in nbr_refs:
                        if src < 0:
                            runs.append((local_base + r0, local_base + r1))
                        else:
                            p_base = peer_halo_base[src] + b * recv_peers[src].tokens_per_batch
                            runs.append((p_base + r0, p_base + r1))
                    kv_runs.append(runs)
            boundary_packed = _build_packed_subset(q_ranges, kv_runs, n_interior, n_boundary, batch, device)

    plan = HaloLocalAttentionPlan(
        n_interior=n_interior,
        n_boundary=n_boundary,
        send_peers=tuple(send_peers),
        recv_peers=tuple(recv_peers),
        interior_neighborhoods=tuple(interior_neighborhoods),
        boundary_neighborhoods=tuple(boundary_neighborhoods),
        interior_packed=interior_packed,
        boundary_packed=boundary_packed,
    )
    return my_box, tuple(boxes), perm, inverse, plan
