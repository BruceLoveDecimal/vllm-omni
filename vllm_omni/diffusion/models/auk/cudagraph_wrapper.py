# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-shape CUDA graph for one AuK DiT denoise step."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer
from vllm_omni.diffusion.models.auk.packing import PackPlan, build_pack_plan_from_lengths

logger = init_logger(__name__)


def _next_power_of_2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_x_mask: torch.Tensor
    static_text: torch.Tensor
    static_c_mask: torch.Tensor
    static_ref: torch.Tensor
    static_ref_mask: torch.Tensor
    static_timestep: torch.Tensor
    static_cfg: torch.Tensor | None
    static_out: torch.Tensor
    # Packed-layout plan captured with the graph; its index tensors are
    # refreshed before every replay, its capacities fix the packed shapes.
    static_plan: PackPlan | None = None
    # Identity of the row lengths last copied into static_plan, so consecutive
    # Euler steps of one batch skip the copy.
    plan_key: tuple | None = None


class AuKCUDAGraphWrapper:
    """Replay one DiT denoise step and leave Euler scheduling to the caller.

    The graph is keyed by the bucketed batch size, the target, text and
    reference sequence lengths, plus whether the CFG branch is enabled.
    Timestep and the CFG strength are mutable scalar buffers, so all Euler
    steps and CFG values within one path reuse one graph.

    Batch rows are independent requests; each row's real length is carried by
    the padding masks, so a batch of mixed-length requests shares the graph of
    its longest member.
    """

    _TARGET_ALIGNMENT = 64
    _TEXT_ALIGNMENT = 64
    _REF_ALIGNMENT = 50

    def __init__(self, dit: AuKTransformer, *, enabled: bool = True, max_graphs: int = 32) -> None:
        self.dit = dit
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._plans: OrderedDict[tuple, PackPlan] = OrderedDict()
        self._pool_handle: int | None = None

    @classmethod
    def _bucket_inputs(
        cls,
        x: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        x_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad every axis to its bucket: batch to a power of two, lengths to fixed steps.

        Padded batch rows keep their first target frame and text token valid so
        no attention row is fully masked; their output is discarded.
        """
        batch = x.shape[0]
        batch_bucket = _next_power_of_2(batch)
        target_bucket = round_up(x.shape[1], cls._TARGET_ALIGNMENT)
        text_bucket = round_up(text.shape[1], cls._TEXT_ALIGNMENT)
        ref_bucket = round_up(ref.shape[1], cls._REF_ALIGNMENT)
        if x_mask is None:
            x_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        pad_rows = batch_bucket - batch
        x_mask = F.pad(x_mask, (0, target_bucket - x_mask.shape[1], 0, pad_rows), value=False)
        c_mask = F.pad(c_mask, (0, text_bucket - c_mask.shape[1], 0, pad_rows), value=False)
        if pad_rows:
            x_mask[batch:, 0] = True
            c_mask[batch:, 0] = True
        return (
            F.pad(x, (0, 0, 0, target_bucket - x.shape[1], 0, pad_rows)),
            x_mask,
            F.pad(text, (0, 0, 0, text_bucket - text.shape[1], 0, pad_rows)),
            c_mask,
            F.pad(ref, (0, 0, 0, ref_bucket - ref.shape[1], 0, pad_rows)),
            F.pad(ref_mask, (0, ref_bucket - ref_mask.shape[1], 0, pad_rows), value=False),
        )

    @staticmethod
    def _key(
        x: torch.Tensor,
        text: torch.Tensor,
        ref: torch.Tensor,
        uses_cfg: bool,
        timestep_rank: int = 0,
    ) -> tuple[int, int, int, int, bool, int]:
        """Graph key: bucketed batch and lengths, CFG branch, and whether the timestep is per row."""
        return (x.shape[0], x.shape[1], text.shape[1], ref.shape[1], uses_cfg, timestep_rank)

    @torch.no_grad()
    def __call__(
        self,
        *,
        x: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
        cfg_strength: float,
        mask: torch.Tensor | None = None,
        row_lengths: tuple[Sequence[int], Sequence[int], Sequence[int]] | None = None,
    ) -> torch.Tensor:
        uses_cfg = cfg_strength >= 1e-5
        cfg_strength = torch.tensor(cfg_strength, device=x.device, dtype=torch.float32)
        if not self.enabled or x.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            if uses_cfg:
                return self._run_cfg(x, mask, text, c_mask, ref, ref_mask, timestep, cfg_strength=cfg_strength)
            return self._run(x, mask, text, c_mask, ref, ref_mask, timestep)

        batch, target_frames = x.shape[:2]
        x, x_mask, text, c_mask, ref, ref_mask = self._bucket_inputs(x, text, c_mask, ref, ref_mask, mask)
        if timestep.ndim == 1:
            # Step execution gives every row its own timestep; the filler
            # rows the batch bucket adds can carry any value.
            timestep = torch.cat([timestep, timestep[-1:].expand(x.shape[0] - timestep.shape[0])])
        inputs = (x, x_mask, text, c_mask, ref, ref_mask, timestep)
        key = self._key(x, text, ref, uses_cfg, timestep.ndim)
        plan = plan_key = None
        if self.dit.packed_attention:
            plan, plan_key = self._packed_plan(x_mask, c_mask, ref_mask, uses_cfg, row_lengths)
            key = key + (plan.audio_capacity, plan.text_capacity)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(*inputs, cfg_strength=cfg_strength, uses_cfg=uses_cfg, plan=plan)
            entry.plan_key = plan_key
            if len(self._cache) >= self.max_graphs:
                self._cache.popitem(last=False)
            self._cache[key] = entry
        else:
            self._cache.move_to_end(key)

        entry.static_x.copy_(x)
        entry.static_x_mask.copy_(x_mask)
        entry.static_text.copy_(text)
        entry.static_c_mask.copy_(c_mask)
        entry.static_ref.copy_(ref)
        entry.static_ref_mask.copy_(ref_mask)
        entry.static_timestep.copy_(timestep)
        if entry.static_cfg is not None:
            entry.static_cfg.copy_(cfg_strength)
        if plan is not None and entry.plan_key != plan_key:
            assert entry.static_plan is not None
            plan.copy_into(entry.static_plan)
            entry.plan_key = plan_key
        entry.graph.replay()
        return entry.static_out[:batch, :target_frames].clone()

    _PACK_ALIGNMENT = 64
    _MAX_PLANS = 128

    def _packed_plan(
        self,
        x_mask: torch.Tensor,
        c_mask: torch.Tensor,
        ref_mask: torch.Tensor,
        uses_cfg: bool,
        row_lengths: tuple[Sequence[int], Sequence[int], Sequence[int]] | None,
    ) -> tuple[PackPlan, tuple]:
        """Return the packed plan for the bucketed rows, built on the host and cached.

        The rows are the ones the DiT will see: ``[ref | target]`` audio and
        the text, doubled under CFG. ``row_lengths`` are the real rows'
        (reference, target, text) lengths when the caller knows them, which
        avoids the device synchronisation of reading them off the masks;
        filler rows added by the batch bucket keep one target frame and one
        text token (see :meth:`_bucket_inputs`). Capacities round the real
        token counts up so the plan's shapes, and hence the graph, are shared
        by nearby batch compositions; the extra room always leaves a filler
        segment, and the kernel's max-segment bound is the bucket total so any
        composition that fits the bucket replays correctly. Plans are cached
        by their lengths, so an ODE loop pays for one build per batch.
        """
        rows = int(x_mask.shape[0])
        if row_lengths is None:
            ref_lens = [int(n) for n in ref_mask.sum(dim=1).tolist()]
            target_lens = [int(n) for n in x_mask.sum(dim=1).tolist()]
            text_lens = [int(n) for n in c_mask.sum(dim=1).tolist()]
        else:
            ref_lens, target_lens, text_lens = ([int(n) for n in lens] for lens in row_lengths)
            filler = rows - len(target_lens)
            ref_lens += [0] * filler
            target_lens += [1] * filler
            text_lens += [1] * filler
        if uses_cfg:
            ref_lens, target_lens, text_lens = ref_lens * 2, target_lens * 2, text_lens * 2
        shape = (int(ref_mask.shape[1]), int(x_mask.shape[1]), int(c_mask.shape[1]))
        plan_key = (tuple(ref_lens), tuple(target_lens), tuple(text_lens), shape)
        plan = self._plans.get(plan_key)
        if plan is None:
            audio_capacity = round_up(sum(ref_lens) + sum(target_lens) + 1, self._PACK_ALIGNMENT)
            text_capacity = round_up(sum(text_lens) + 1, self._PACK_ALIGNMENT)
            plan = build_pack_plan_from_lengths(
                ref_lens,
                target_lens,
                text_lens,
                ref_len=shape[0],
                target_len=shape[1],
                text_len=shape[2],
                audio_capacity=audio_capacity,
                text_capacity=text_capacity,
                device=x_mask.device,
            )
            plan.joint_max = plan.single_max = audio_capacity + text_capacity
            if len(self._plans) >= self._MAX_PLANS:
                self._plans.popitem(last=False)
            self._plans[plan_key] = plan
        else:
            self._plans.move_to_end(plan_key)
        return plan, plan_key

    def _run(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
        plan: PackPlan | None = None,
    ) -> torch.Tensor:
        return self.dit(x, text, timestep, mask=x_mask, c_mask=c_mask, ref=ref, ref_mask=ref_mask, plan=plan)

    def _run_cfg(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
        *,
        cfg_strength: torch.Tensor,
        plan: PackPlan | None = None,
    ) -> torch.Tensor:
        pred = self.dit(
            x,
            text,
            timestep,
            mask=x_mask,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            cfg_infer=True,
            cache=True,
            plan=plan,
        )
        conditional, unconditional = pred.chunk(2, dim=0)
        return conditional + (conditional - unconditional) * cfg_strength

    def _capture(
        self,
        *inputs: torch.Tensor,
        cfg_strength: torch.Tensor,
        uses_cfg: bool,
        plan: PackPlan | None = None,
    ) -> _GraphEntry:
        static_inputs = tuple(value.clone() for value in inputs)
        static_cfg = cfg_strength.clone() if uses_cfg else None
        static_plan = plan.clone() if plan is not None else None
        try:
            for _ in range(3):
                if uses_cfg:
                    assert static_cfg is not None
                    self._run_cfg(*static_inputs, cfg_strength=static_cfg, plan=static_plan)
                else:
                    self._run(*static_inputs, plan=static_plan)
            # CFG warm-up populates AuKTransformer's Python-side projected-text
            # cache. Clear it before capture so the graph includes
            # project_text(static_text), rather than closing over the first
            # request's projection and ignoring later static_text updates.
            self.dit.clear_cache()
            if static_cfg is not None:
                static_cfg.copy_(cfg_strength)
            if self._pool_handle is None:
                self._pool_handle = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool_handle):
                # Capture the actual CFG path. The scalar value remains a
                # mutable buffer, so cfg=2 and cfg=3 share this graph.
                if uses_cfg:
                    assert static_cfg is not None
                    static_out = self._run_cfg(*static_inputs, cfg_strength=static_cfg, plan=static_plan)
                else:
                    static_out = self._run(*static_inputs, plan=static_plan)
        finally:
            self.dit.clear_cache()

        logger.info(
            "Captured AuK DiT single-step CUDA graph: batch=%d target_frames=%d text_tokens=%d ref_frames=%d cfg=%s",
            inputs[0].shape[0],
            inputs[0].shape[1],
            inputs[2].shape[1],
            inputs[4].shape[1],
            cfg_strength,
        )
        return _GraphEntry(
            graph=graph,
            static_x=static_inputs[0],
            static_x_mask=static_inputs[1],
            static_text=static_inputs[2],
            static_c_mask=static_inputs[3],
            static_ref=static_inputs[4],
            static_ref_mask=static_inputs[5],
            static_timestep=static_inputs[6],
            static_cfg=static_cfg,
            static_out=static_out,
            static_plan=static_plan,
        )


__all__ = ["AuKCUDAGraphWrapper"]
