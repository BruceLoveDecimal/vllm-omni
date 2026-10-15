# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Suffix-only transformer evaluation with full chunk-causal prefix attention."""

import torch

from .flow_prefix_cache import FlowPrefixCache, FlowPrefixEntry, FlowPrefixStep
from .packed_dit import (
    PackedDiT,
    PackedRows,
    RaggedRowAttention,
    attn_norm_preserving_eager,
    conv_pos_embed_tensor_geometry,
    final_norm_preserving_eager,
    layer_norm_preserving_eager,
    mark_packed_compile_metadata,
    pack_rows,
    ragged_attention_tensor_geometry,
    rope_tensor_geometry,
    rotate_in_place,
)


def forward_prefix(
    estimator: PackedDiT,
    x: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    rows: PackedRows,
    attention: RaggedRowAttention,
    query_indices: torch.Tensor,
    prefix_indices: torch.Tensor,
    prefix_kv: tuple[torch.Tensor, ...],
    block_modulation: torch.Tensor,
    final_modulation: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    save_indices: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    # The cheap input convolutions still see the complete trajectory, preserving
    # their left context. All transformer projections/MLPs run on suffixes only.
    dit = estimator.dit
    h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
    h = (conv_pos_embed_tensor_geometry(estimator, h, rows, attention) + h)[:, query_indices]
    rope = (rope[0][:, query_indices], rope[1][:, query_indices])
    residual = h
    kv = []
    for index, block in enumerate(dit.transformer_blocks):
        norm, gate, shift, scale, ff_gate = attn_norm_preserving_eager(
            block, h, None, block_modulation[index : index + 1]
        )
        norm = norm.to(block.attn.to_q.weight.dtype)
        query, key, value = block.attn.to_q(norm), block.attn.to_k(norm), block.attn.to_v(norm)
        rotate_in_place(query, *rope)
        rotate_in_place(key, *rope)
        full_key = key.new_zeros((1, x.shape[1], key.shape[-1]))
        full_value = value.new_zeros((1, x.shape[1], value.shape[-1]))
        full_key[:, prefix_indices] = prefix_kv[2 * index]
        full_value[:, prefix_indices] = prefix_kv[2 * index + 1]
        full_key[:, query_indices] = key
        full_value[:, query_indices] = value
        kv.extend((full_key[:, save_indices], full_value[:, save_indices]))
        attended = ragged_attention_tensor_geometry(query, full_key, full_value, attention, attention.max_seqlen_q)
        attended = block.attn.to_out[1](block.attn.to_out[0](attended.to(query.dtype)))
        h = h + gate.unsqueeze(1) * attended
        norm = layer_norm_preserving_eager(block.ff_norm, h)
        h = h + ff_gate.unsqueeze(1) * block.ff(norm * (1 + scale[:, None]) + shift[:, None])
    if dit.long_skip_connection is not None:
        h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
    return dit.proj_out(final_norm_preserving_eager(dit.norm_out, h, None, final_modulation)), tuple(kv)


def _indices(rows: PackedRows, prefixes: tuple[int, ...], device):
    starts = rows.starts_host.tolist()
    prefix = torch.cat([torch.arange(start, start + p, device=device) for start, p in zip(starts, prefixes)])
    query = torch.cat(
        [
            torch.arange(start + p, start + length, device=device)
            for start, p, length in zip(starts, prefixes, rows.lengths)
        ]
    )
    return prefix.long(), query.long()


def solve_flow_euler_prefix(
    estimator: PackedDiT,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    rows: PackedRows,
    *,
    cfg_rate: float,
    modulation_key: tuple,
    max_bytes: int,
) -> torch.Tensor:
    cache = estimator.prefix_cache
    if cache is None or cache.max_bytes != max_bytes:
        cache = estimator.prefix_cache = FlowPrefixCache(max_bytes)
    chunk_size = int(estimator.dit.static_chunk_size)
    cache.reset((modulation_key, cfg_rate, chunk_size, noise.dtype, noise.device))
    starts = rows.starts_host.tolist()
    signatures = [
        torch.cat((noise[0, a:b], mu[0, a:b], cond[0, a:b], spks[i].expand(b - a, -1)), dim=-1)
        for i, (a, b) in enumerate(zip(starts, starts[1:]))
    ]
    # Bound newly retained state as well as the LRU; split admission across rows.
    steps = len(time_span) - 1
    dims = [block.attn.inner_dim for block in estimator.dit.transformer_blocks]
    bytes_per_frame = signatures[0].shape[-1] * signatures[0].element_size()
    bytes_per_frame += steps * 2 * (2 * sum(dims) + noise.shape[-1]) * noise.element_size()
    limit = max_bytes // (len(rows.lengths) * bytes_per_frame)
    keep = [min(length, limit) // chunk_size * chunk_size for length in rows.lengths]
    matches = cache.lookup(signatures, chunk_size)
    prefixes = tuple(length for _, length in matches)
    # A hit already owns these tensors: do not copy and reinsert the same
    # multi-step prefix on every streaming chunk once the budget is reached.
    keep = [n if n > hit else 0 for n, hit in zip(keep, prefixes)]
    if not any(keep) and not any(prefixes):
        from .packed_dit import solve_flow_euler_packed

        return solve_flow_euler_packed(
            estimator,
            noise,
            time_span,
            mu,
            spks,
            cond,
            rows,
            cfg_rate=cfg_rate,
            streaming=True,
            modulation_key=modulation_key,
        )
    twin_rows = pack_rows(rows.lengths * 2, noise.device)
    attention = RaggedRowAttention(
        twin_rows,
        heads=estimator.dit.transformer_blocks[0].attn.heads,
        head_dim=estimator.dit.transformer_blocks[0].attn.dim_head,
        chunk_size=chunk_size,
        prefix_lengths=prefixes * 2,
    )
    mark_packed_compile_metadata(twin_rows, attention)
    prefix_indices, query_indices = _indices(twin_rows, prefixes * 2, noise.device)
    save_indices, _ = _indices(twin_rows, tuple(keep) * 2, noise.device)
    torch._dynamo.mark_dynamic(save_indices, 0)
    torch._dynamo.mark_dynamic(prefix_indices, 0)
    torch._dynamo.mark_dynamic(query_indices, 0)
    rope = rope_tensor_geometry(estimator, twin_rows, attention)
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)))[twin_rows.row_ids].unsqueeze(0)
    blocks, finals = estimator.step_modulations(time_span, modulation_key)
    if estimator.compiled_prefix_forward is None:
        estimator.compiled_prefix_forward = torch.compile(
            forward_prefix, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True}
        )
    saved = [FlowPrefixEntry(signature[:n].clone(), []) for signature, n in zip(signatures, keep)]
    total = rows.total
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(steps):
        old_kv = tuple(
            torch.cat(
                [
                    entry.steps[step].kv[layer][branch : branch + 1, :length]
                    for branch in range(2)
                    for entry, length in matches
                    if length
                ],
                dim=1,
            )
            if any(prefixes)
            else noise.new_empty((1, 0, dim))
            for layer, dim in enumerate([dim for dim in dims for _ in range(2)])
        )
        velocity, kv = estimator.compiled_prefix_forward(
            estimator,
            torch.cat((x, x), dim=1),
            mu_cfg,
            spks_cfg,
            cond_cfg,
            twin_rows,
            attention,
            query_indices,
            prefix_indices,
            old_kv,
            blocks[step],
            finals[step],
            rope,
            save_indices,
        )
        full_velocity = velocity.new_zeros((1, 2 * total, velocity.shape[-1]))
        full_velocity[:, query_indices] = velocity
        if any(prefixes):
            full_velocity[:, prefix_indices] = torch.cat(
                [
                    entry.steps[step].velocity[branch : branch + 1, :length]
                    for branch in range(2)
                    for entry, length in matches
                    if length
                ],
                dim=1,
            )
        saved_start = 0
        saved_total = sum(keep)
        for row, (a, n) in enumerate(zip(starts, keep)):
            if n:
                saved_kv = tuple(
                    torch.cat(
                        (
                            k[:, saved_start : saved_start + n],
                            k[:, saved_total + saved_start : saved_total + saved_start + n],
                        ),
                        dim=0,
                    )
                    for k in kv
                )
                saved_velocity = torch.cat(
                    (
                        full_velocity[:, a : a + n],
                        full_velocity[:, total + a : total + a + n],
                    ),
                    dim=0,
                )
                saved[row].steps.append(FlowPrefixStep(saved_kv, saved_velocity))
            saved_start += n
        x = x + dt * ((1 + cfg_rate) * full_velocity[:, :total] - cfg_rate * full_velocity[:, total:])
        t = t + dt
        if step < steps - 1:
            dt = time_span[step + 2] - t
    for entry, n, (old, hit) in zip(saved, keep, matches):
        if n:
            replaces = old if old is not None and min(n, hit) >= old.signature.shape[0] else None
            cache.put(entry, replaces=replaces)
    return x.float()
