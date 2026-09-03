# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fusing a low-rank adapter into the MiniMax-H3 checkpoint stream.

Two releases need exactly the same arithmetic here and disagree only about
which files carry it: FastVideo's FastH3 four-step student (``fasth3.py``) and
OpenVDN's VDN-H3 hybrid-attention checkpoint (``vdn/weight_fusion.py``). Both
are written in the diffusers namespace (``transformer_blocks.0.attn.to_q``)
while vLLM-Omni loads H3's native one (``blocks.0.attn.qkv_proj``), whose
attention and MLP projections are fused, so both have to place a delta into a
grouped QKV matrix and swap the halves of a fused gate/up matrix rather than
adding it as it comes.

That placement is the part that is easy to get wrong and impossible to notice
afterwards - a transposed or mis-slotted delta still loads, still generates, and
is simply not the model the release describes - so it lives here once. Every
mapping below was verified tensor by tensor against the released FastH3 full
checkpoint (see ``fasth3.py``); VDN reuses the same targets through its own
``.attn.orig.`` rerooting.

A release-specific front end reads its own files, decides whether it claims
them, and hands this module ``{native parameter: ParamPatch}``; the fusion then
rebuilds each delta on the accelerator as the checkpoint streams past.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import torch

from vllm_omni.platforms import current_omni_platform


class MiniMaxH3FusionError(ValueError):
    """A low-rank artifact for MiniMax-H3 cannot be applied as one."""


# How a delta enters its native parameter. H3 stores attention as one grouped
# QKV matrix and the MLP as one fused gate/up matrix, so those need placing.
PLAIN, QKV_Q, QKV_K, QKV_V, SWAP_HALVES = "plain", "q", "k", "v", "swap_halves"
QKV_SLOTS = (QKV_Q, QKV_K, QKV_V)

# Adapter module prefix -> the native parameter it edits, minus the
# ``.weight``/``.bias`` suffix.
MODEL_LEVEL_TARGETS = {
    "proj_in": "video_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_in": "audio_patch_proj",
    "audio_proj_out": "final_layer.audio_out",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "norm_out.norm": "final_layer.norm",
}

# Per-block adapter suffix -> (native suffix, layout).
BLOCK_TARGETS = {
    "attn.to_q": ("attn.qkv_proj", QKV_Q),
    "attn.to_k": ("attn.qkv_proj", QKV_K),
    "attn.to_v": ("attn.qkv_proj", QKV_V),
    "attn.to_out.0": ("attn.out_proj", PLAIN),
    "attn.to_gate_compress": ("attn.to_gate_compress", PLAIN),
    "ff.net.0.proj": ("mlp.fc1", SWAP_HALVES),
    "ff.net.2": ("mlp.fc2", PLAIN),
    "adaln_proj.linear": ("adaln_proj.linear", PLAIN),
    "norm1": ("norm1", PLAIN),
    "norm2": ("norm2", PLAIN),
}

# Adapter block prefix -> native block prefix.
BLOCK_PREFIXES = (
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("transformer_blocks.", "blocks."),
)


@dataclass
class LowRankFactors:
    """One ``scale * lora_B @ lora_A`` contribution to one native parameter."""

    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    scale: float = 1.0


@dataclass
class ParamPatch:
    """Everything one or more adapters contribute to one native parameter.

    ``low_rank`` maps a layout to the contributions aimed at it, keyed by
    whichever artifact contributed each. A grouped QKV parameter collects three
    layouts; a parameter edited by two adapters collects two contributions per
    layout, which are summed rather than composed - stacked LoRAs add their
    deltas.
    """

    low_rank: dict[str, dict[str, LowRankFactors]] = field(default_factory=dict)
    diff: torch.Tensor | None = None
    layout: str = PLAIN

    def factors(self, layout: str, *, key: str = "") -> LowRankFactors:
        """The contribution ``key`` aims at ``layout``, created on demand.

        Keyed rather than positional: the A and B halves of one pair arrive as
        separate tensors and must land in the same slot, a second adapter
        editing the same parameter must not overwrite the first, and an adapter
        that edits a parameter its neighbour does not must not leave an empty
        slot behind for the pairing check to trip over.
        """
        slots = self.low_rank.setdefault(layout, {})
        if key not in slots:
            slots[key] = LowRankFactors()
        return slots[key]


def swap_halves(tensor: torch.Tensor, *, error_cls: type[Exception] = MiniMaxH3FusionError) -> torch.Tensor:
    """Exchange the two halves of a fused gate/up matrix.

    The diffusers export stores the feed-forward projection value-first while
    H3's native ``mlp.fc1`` is gate-first, so a delta computed in the diffusers
    layout has to be swapped before it can be added to the native parameter.
    """
    if tensor.shape[0] % 2:
        raise error_cls(f"fused gate/up delta must split evenly, got {tuple(tensor.shape)}")
    first, second = tensor.chunk(2, dim=0)
    return torch.cat((second, first), dim=0)


def place_in_grouped_qkv(
    deltas: Mapping[str, torch.Tensor],
    *,
    head_dim: int,
    error_cls: type[Exception] = MiniMaxH3FusionError,
) -> torch.Tensor:
    """Interleave per-projection deltas into H3's grouped QKV layout.

    The checkpoint stores one head group at a time as ``[q, k, v]``, which is
    what ``_reorder_grouped_qkv_to_qkv`` unpacks on the way in. A delta built
    from the separate diffusers projections has to be folded back into that
    order.
    """
    missing = sorted(set(QKV_SLOTS) - set(deltas))
    if missing:
        raise error_cls(f"grouped QKV delta is missing its {missing} projections")
    parts = []
    for slot in QKV_SLOTS:
        delta = deltas[slot]
        if delta.shape[0] % head_dim:
            raise error_cls(f"QKV {slot} delta rows {delta.shape[0]} are not a multiple of head_dim {head_dim}")
        parts.append(delta.reshape(delta.shape[0] // head_dim, head_dim, *delta.shape[1:]))
    groups = parts[0].shape[0]
    if any(part.shape[0] != groups for part in parts):
        raise error_cls("QKV projections disagree on the number of head groups")
    return torch.cat(parts, dim=1).reshape(groups * 3 * head_dim, *parts[0].shape[2:])


def resolve_native_target(module: str) -> tuple[str, str, tuple[str, int] | None] | None:
    """Map an adapter module path to ``(native path, layout, block)``.

    ``block`` is the ``(native block prefix, index)`` the module sits in, or
    ``None`` for a model-level one. Coverage is checked per block, so the index
    has to survive the mapping rather than being folded into the path.
    """
    native = MODEL_LEVEL_TARGETS.get(module)
    if native is not None:
        return native, PLAIN, None
    for adapter_prefix, native_prefix in BLOCK_PREFIXES:
        if not module.startswith(adapter_prefix):
            continue
        remainder = module[len(adapter_prefix) :]
        index, _, suffix = remainder.partition(".")
        if not index.isdigit():
            return None
        target = BLOCK_TARGETS.get(suffix)
        if target is None:
            return None
        native_suffix, layout = target
        return f"{native_prefix}{index}.{native_suffix}", layout, (native_prefix, int(index))
    return None


def check_block_coverage(
    seen: Mapping[str, set[int]],
    *,
    expected: Mapping[str, int],
    source: object,
    error_cls: type[Exception] = MiniMaxH3FusionError,
) -> None:
    """Every block of the model this artifact is loaded against must be edited.

    A release drops tensors training left unchanged, so per-parameter coverage
    is legitimately sparse - but both artifacts that reach here edit every
    block, so a block with no edits at all means the artifact does not match
    this model.
    """
    for prefix, count in expected.items():
        indices = seen.get(prefix, set())
        wanted = set(range(count))
        if indices == wanted:
            continue
        missing = sorted(wanted - indices)
        extra = sorted(indices - wanted)
        raise error_cls(
            f"{source} edits {len(indices)} of the model's {count} {prefix}* blocks "
            f"(missing={missing[:5]}, unknown={extra[:5]}); it is not an adapter for this checkpoint"
        )


class LowRankWeightFusion:
    """Fuse low-rank (and full-rank) deltas into the H3 checkpoint stream."""

    #: Raised for every failure this fusion detects. Subclasses narrow it so a
    #: caller can catch one release's problems without catching the other's.
    error_cls: type[Exception] = MiniMaxH3FusionError
    #: How the artifact names itself in an error message.
    label: str = "MiniMax-H3 adapter"

    def __init__(
        self,
        *,
        source: object,
        patches: Mapping[str, ParamPatch],
        head_dim: int,
        injections: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self._source = source
        self._patches = dict(patches)
        self._head_dim = head_dim
        # Parameters the base checkpoint does not carry, assigned into the
        # stream instead of fused onto an existing weight.
        self._injections = dict(injections or {})
        self._injected: set[str] = set()
        self._applied: set[str] = set()
        self._device: torch.device | None = None

    @property
    def source(self) -> object:
        return self._source

    def validate_pairs(self) -> None:
        """Every low-rank contribution must carry both of its factors.

        A half-read pair would otherwise be skipped silently by ``fuse`` and
        the parameter would keep its base value under a release that says it
        does not.
        """
        for native_param, patch in self._patches.items():
            for slot, contributions in patch.low_rank.items():
                for factors in contributions.values():
                    if factors.a is None or factors.b is None:
                        raise self.error_cls(f"{self.label} has an unpaired factor for {native_param} slot {slot!r}")
            # Only the low-rank factors are placed into H3's fused QKV and
            # gate/up layouts; a full-rank delta is added as it comes, so one
            # aimed at a fused parameter would silently land transposed.
            if patch.diff is not None and patch.layout != PLAIN:
                raise self.error_cls(
                    f"{self.label} carries a full-rank delta for {native_param}, which H3 stores in the "
                    f"{patch.layout!r} fused layout; this loader can only place low-rank factors there"
                )

    def _compute_device(self, weight: torch.Tensor) -> torch.device:
        """Where to reconstruct a delta.

        H3's per-block modulation projection is 96768x2688, so rebuilding all
        of a release's patched parameters is a few TFLOP of low-rank products.
        On CPU that adds minutes to a load that already has a startup deadline,
        so the accelerator does the arithmetic whenever there is one.
        """
        if weight.device.type != "cpu":
            return weight.device
        if self._device is None:
            # Ask the platform rather than PyTorch's global accelerator
            # registry, so an out-of-tree backend controls its own placement.
            self._device = current_omni_platform.get_torch_device()
        return self._device

    @staticmethod
    def _widen(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move to ``device``, then widen to float32.

        Asking ``Tensor.to`` for a device and a dtype at once converts on the
        host and ships twice the bytes; splitting it moves bfloat16 and widens
        on the accelerator.
        """
        return tensor.to(device, non_blocking=True).to(torch.float32)

    def _low_rank_delta(self, factors: LowRankFactors, layout: str, device: torch.device) -> torch.Tensor:
        a, b = factors.a, factors.b
        assert a is not None and b is not None  # validate_pairs ran first
        if layout == SWAP_HALVES:
            # Permuting the rows of B permutes the rows of the product, so swap
            # the low-rank factor instead of the full delta.
            b = swap_halves(b, error_cls=self.error_cls)
        delta = self._widen(b, device) @ self._widen(a, device)
        return delta if factors.scale == 1.0 else delta * factors.scale

    def fuse(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """Return ``weight`` with every contribution to it added."""
        patch = self._patches.get(name)
        if patch is None:
            return weight
        self._applied.add(name)

        device = self._compute_device(weight)

        delta: torch.Tensor | None = None
        if patch.low_rank:
            if patch.layout in QKV_SLOTS:
                # Each projection's contributions sum before placement: the
                # placement is a permutation, so summing on either side agrees.
                per_slot = {
                    slot: _sum_deltas(self._low_rank_delta(factors, slot, device) for factors in contributions.values())
                    for slot, contributions in patch.low_rank.items()
                }
                delta = place_in_grouped_qkv(per_slot, head_dim=self._head_dim, error_cls=self.error_cls)
            else:
                delta = _sum_deltas(
                    self._low_rank_delta(factors, patch.layout, device)
                    for factors in patch.low_rank[patch.layout].values()
                )
        if patch.diff is not None:
            diff = self._widen(patch.diff, device)
            delta = diff if delta is None else delta + diff
        if delta is None:
            return weight
        if delta.shape != weight.shape:
            raise self.error_cls(
                f"{self.label} delta for {name} has shape {tuple(delta.shape)}, parameter is {tuple(weight.shape)}"
            )
        # Leave the result on the compute device. These weights are bound for
        # the accelerator anyway, so returning them to host memory would pay a
        # device-to-host copy of the whole checkpoint only for the loader to
        # send it straight back: measured at 152s against 15s for 60 GiB of
        # patched projections, against 17s for the unavoidable upload alone.
        # Fold the base weight into the freshly built delta in place. Promoting
        # the weight to float32 on its own would allocate two more buffers the
        # size of the parameter, and H3's largest patched projection is 0.5 GiB.
        return delta.add_(weight.to(device, non_blocking=True)).to(weight.dtype)

    @property
    def injection_names(self) -> set[str]:
        """Names of the parameters this artifact assigns rather than fuses.

        Kept separate from the payloads so a subclass can stream several
        gigabytes of them off disk while still stating up front what it will
        contribute - which is what the collision check below needs.
        """
        return set(self._injections)

    def iter_injections(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Parameters the base checkpoint does not carry.

        Yielded after the checkpoint's own stream. Subclasses that read these
        lazily override this; the default holds them in memory.
        """
        yield from self._injections.items()

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Fuse every streamed checkpoint tensor on its way into the model."""
        if self._applied or self._injected:
            # ``validate_fully_applied`` released the deltas, so a second stream
            # would fuse nothing and then pass its own completeness check: the
            # server would serve base H3 weights under the artifact's contract.
            raise self.error_cls(f"{self._source} has already been fused into this checkpoint")
        injected_names = self.injection_names
        for name, weight in weights:
            if name in injected_names:
                raise self.error_cls(
                    f"the checkpoint already provides {name}, which this artifact assigns; "
                    "assigning it would discard the checkpoint's own weight"
                )
            yield name, self.fuse(name, weight)
        # Assigned parameters have no counterpart in the base checkpoint, so
        # they join the stream after it rather than being folded into a tensor.
        for name, weight in self.iter_injections():
            self._injected.add(name)
            yield name, weight

    def validate_fully_applied(self, loaded: Iterable[str] | None = None) -> None:
        """Close the fusion: every edit must have met its parameter.

        A silently unapplied delta is the failure mode that matters here: the
        model would load and generate, just not as the release describes. The
        weights are loaded once, so the mapped payloads are dropped afterwards
        rather than held for the life of the process.

        ``loaded`` is the set of parameter names ``load_weights`` actually
        consumed. An assigned parameter lands on a module the base transformer
        may not have; if that module was never built, ``load_weights`` only logs
        a skip and the server would serve an uninitialized parameter. Yielding a
        tensor is not evidence it arrived, so the injections are closed against
        that set when it is available.
        """
        missing = sorted(set(self._patches) - self._applied)
        if missing:
            raise self.error_cls(
                f"{self.label} edits {len(missing)} parameters the checkpoint never provided: {missing[:5]}"
            )
        arrived = self._injected if loaded is None else set(loaded)
        uninjected = sorted(self.injection_names - arrived)
        if uninjected:
            raise self.error_cls(
                f"{self.label} assigns {len(uninjected)} parameters that never reached the model: {uninjected[:5]}"
            )
        self.release()

    def release(self) -> None:
        """Drop every mapped payload once the weights have been loaded."""
        for patch in self._patches.values():
            patch.low_rank.clear()
            patch.diff = None
        self._injections.clear()


def _sum_deltas(deltas: Iterable[torch.Tensor]) -> torch.Tensor:
    """Sum reconstructed deltas in place after the first."""
    total: torch.Tensor | None = None
    for delta in deltas:
        total = delta if total is None else total.add_(delta)
    if total is None:
        raise MiniMaxH3FusionError("no low-rank contribution to sum")
    return total


__all__ = [
    "BLOCK_PREFIXES",
    "BLOCK_TARGETS",
    "MODEL_LEVEL_TARGETS",
    "PLAIN",
    "QKV_K",
    "QKV_Q",
    "QKV_SLOTS",
    "QKV_V",
    "SWAP_HALVES",
    "LowRankFactors",
    "LowRankWeightFusion",
    "MiniMaxH3FusionError",
    "ParamPatch",
    "check_block_coverage",
    "place_in_grouped_qkv",
    "resolve_native_target",
    "swap_halves",
]
