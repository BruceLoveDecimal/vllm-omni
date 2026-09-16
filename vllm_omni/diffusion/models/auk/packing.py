# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed variable-length layout for the AuK DiT.

A batch of AuK requests differs in reference, target and text length. The
padded ``[rows, max_len]`` layout spends attention and feed-forward work on
padding and forces the attention kernel through a boolean mask. Packing keeps
only the real tokens of every row back to back, ``[total_tokens, dim]``, and
describes the row boundaries with cumulative offsets so FlashAttention's
varlen kernel attends within each row and nowhere else.

:class:`PackPlan` holds the index tensors that move data between the two
layouts. Every tensor has a shape fixed by the plan's *capacities*, so a plan
built for a token bucket can be copied into static buffers and replayed under
a CUDA graph; filler tokens beyond the real count form one trailing segment of
their own and read from a scratch slot, so they never touch a real row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import torch

__all__ = ["PackPlan", "build_pack_plan", "flash_attn_varlen", "packed_attention_available"]


def _resolve_flash_attn_varlen() -> Callable[..., torch.Tensor] | None:
    """Find a FlashAttention varlen kernel, preferring the repo's platform resolver."""
    try:
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func

        if flash_attn_varlen_func is not None:
            return flash_attn_varlen_func
    except Exception:  # noqa: BLE001 - any import trouble means "not this source"
        pass
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func as vllm_varlen

        return vllm_varlen
    except Exception:  # noqa: BLE001
        return None


flash_attn_varlen = _resolve_flash_attn_varlen()


def packed_attention_available(device: torch.device | str) -> bool:
    """Packed attention needs a CUDA device and a varlen kernel."""
    return flash_attn_varlen is not None and torch.device(device).type == "cuda"


@dataclass
class PackPlan:
    """Index tensors that pack padded ``[rows, len]`` streams into varlen sequences.

    ``audio`` covers the ``[ref | target]`` stream and ``text`` the projected
    text stream. ``*_idx`` gather from the flattened padded stream (a scratch
    slot appended at the end serves the filler tokens), ``*_seg`` give each
    packed token's row, ``*_pos`` its rotary position within the row. The
    joint order interleaves each row's audio then text tokens; the single
    order interleaves text then audio, matching the padded forward. ``*_cu``
    are int32 cumulative offsets with a trailing filler segment.
    """

    rows: int
    audio_len: int
    text_len: int
    audio_real: int
    text_real: int
    audio_idx: torch.Tensor
    audio_seg: torch.Tensor
    audio_pos: torch.Tensor
    text_idx: torch.Tensor
    text_seg: torch.Tensor
    text_pos: torch.Tensor
    joint_perm: torch.Tensor
    joint_inv: torch.Tensor
    joint_cu: torch.Tensor
    joint_max: int
    single_perm: torch.Tensor
    single_inv: torch.Tensor
    single_pos: torch.Tensor
    single_cu: torch.Tensor
    single_max: int

    @property
    def audio_capacity(self) -> int:
        return int(self.audio_idx.shape[0])

    @property
    def text_capacity(self) -> int:
        return int(self.text_idx.shape[0])

    def tensors(self) -> dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self) if torch.is_tensor(getattr(self, f.name))}

    def copy_into(self, static: PackPlan) -> None:
        """Copy this plan's tensors into a same-capacity static plan for graph replay."""
        for name, value in self.tensors().items():
            getattr(static, name).copy_(value)

    def clone(self) -> PackPlan:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        for name, value in self.tensors().items():
            values[name] = value.clone()
        return PackPlan(**values)

    def to(self, device: torch.device | str) -> PackPlan:
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        for name, value in self.tensors().items():
            values[name] = value.to(device, non_blocking=True)
        return PackPlan(**values)


def _lengths_mask(lengths: list[int], width: int) -> torch.Tensor:
    return torch.arange(width)[None, :] < torch.tensor(lengths, dtype=torch.long)[:, None]


def build_pack_plan_from_lengths(
    ref_lens: list[int],
    target_lens: list[int],
    text_lens: list[int],
    *,
    ref_len: int,
    target_len: int,
    text_len: int,
    audio_capacity: int | None = None,
    text_capacity: int | None = None,
    device: torch.device | str = "cpu",
) -> PackPlan:
    """Build the plan from host-side row lengths, without touching the device.

    The rows are laid out the way :class:`AuKTransformer` pads them: the audio
    stream is ``[ref (padded to ref_len) | target (padded to target_len)]``
    and the text stream is padded to ``text_len``. Building on the CPU from
    lengths the caller already knows avoids the host synchronisations a
    mask-based build needs, which matters when a plan is rebuilt per Euler
    step under CUDA graph replay.
    """
    if not (len(ref_lens) == len(target_lens) == len(text_lens)):
        raise ValueError("ref, target and text length lists must have one entry per row")
    target_mask = _lengths_mask(list(target_lens), target_len)
    audio_mask = torch.cat([_lengths_mask(list(ref_lens), ref_len), target_mask], dim=1) if ref_len > 0 else target_mask
    text_mask = _lengths_mask(list(text_lens), text_len)
    plan = build_pack_plan(audio_mask, text_mask, audio_capacity=audio_capacity, text_capacity=text_capacity)
    return plan.to(device)


def _pack_stream(
    mask: torch.Tensor, capacity: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack one ``[rows, len]`` mask: gather indices, row ids, positions, lengths."""
    rows, length = mask.shape
    flat = mask.reshape(-1)
    real_idx = flat.nonzero(as_tuple=False).squeeze(1)
    real = int(real_idx.numel())
    capacity = real if capacity is None else capacity
    if capacity < real:
        raise ValueError(f"Pack capacity {capacity} is smaller than the {real} real tokens.")
    scratch = rows * length
    idx = torch.full((capacity,), scratch, dtype=torch.long, device=mask.device)
    idx[:real] = real_idx
    seg = torch.full((capacity,), rows, dtype=torch.long, device=mask.device)
    seg[:real] = real_idx // length
    lengths = mask.sum(dim=1)
    starts = torch.cumsum(lengths, dim=0) - lengths
    pos = torch.zeros(capacity, dtype=torch.long, device=mask.device)
    pos[:real] = torch.arange(real, device=mask.device) - starts[seg[:real]]
    return idx, seg, pos, lengths, real


def _interleave(
    first_seg: torch.Tensor,
    second_seg: torch.Tensor,
    first_lens: torch.Tensor,
    second_lens: torch.Tensor,
    rows: int,
    capacity: int,
    real: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Order ``cat(first, second)`` as ``[first_0, second_0, first_1, ...]`` with filler last."""
    keys = torch.cat([first_seg * 2, second_seg * 2 + 1])
    perm = torch.argsort(keys, stable=True)
    inv = torch.argsort(perm)
    lens = first_lens + second_lens
    # The filler segment only exists when the capacity leaves room for it: an
    # empty trailing segment would be passed to the kernel for nothing.
    segments = rows + 1 if capacity > real else rows
    cu = torch.zeros(segments + 1, dtype=torch.int32, device=keys.device)
    cu[1 : rows + 1] = torch.cumsum(lens, dim=0).to(torch.int32)
    cu[segments] = capacity
    longest = max(int(lens.max().item()) if rows else 0, capacity - real)
    return perm, inv, cu, longest


@torch.no_grad()
def build_pack_plan(
    audio_mask: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    audio_capacity: int | None = None,
    text_capacity: int | None = None,
) -> PackPlan:
    """Build the plan for padded audio ``[rows, na]`` and text ``[rows, nt]`` masks.

    Capacities default to the real token counts (the eager path); a CUDA graph
    passes its token buckets so the plan's shapes stay static.
    """
    if audio_mask.shape[0] != text_mask.shape[0]:
        raise ValueError("audio and text masks must have the same number of rows")
    rows = int(audio_mask.shape[0])
    audio_idx, audio_seg, audio_pos, audio_lens, audio_real = _pack_stream(audio_mask, audio_capacity)
    text_idx, text_seg, text_pos, text_lens, text_real = _pack_stream(text_mask, text_capacity)
    capacity = int(audio_idx.shape[0] + text_idx.shape[0])
    real = audio_real + text_real
    # Filler tokens carry row id ``rows`` so the interleave sorts them last.
    joint_perm, joint_inv, joint_cu, joint_max = _interleave(
        audio_seg, text_seg, audio_lens, text_lens, rows, capacity, real
    )
    single_perm, single_inv, single_cu, single_max = _interleave(
        text_seg, audio_seg, text_lens, audio_lens, rows, capacity, real
    )
    # In the single stream a row's audio tokens follow its text tokens, so
    # their rotary positions continue after the text.
    single_pos = torch.cat([text_pos, audio_pos + text_lens[audio_seg.clamp(max=rows - 1)]])[single_perm]
    return PackPlan(
        rows=rows,
        audio_len=int(audio_mask.shape[1]),
        text_len=int(text_mask.shape[1]),
        audio_real=audio_real,
        text_real=text_real,
        audio_idx=audio_idx,
        audio_seg=audio_seg,
        audio_pos=audio_pos,
        text_idx=text_idx,
        text_seg=text_seg,
        text_pos=text_pos,
        joint_perm=joint_perm,
        joint_inv=joint_inv,
        joint_cu=joint_cu,
        joint_max=joint_max,
        single_perm=single_perm,
        single_inv=single_inv,
        single_pos=single_pos,
        single_cu=single_cu,
        single_max=single_max,
    )
