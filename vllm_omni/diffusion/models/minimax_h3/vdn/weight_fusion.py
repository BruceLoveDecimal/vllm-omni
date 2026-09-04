# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Folding a VDN-H3 release into the H3 checkpoint stream.

VDN publishes three things over the dense backbone: a per-block linear-attention
branch (800 tensors that H3 has no parameter for, so they are *assigned*), a
Stage-B LoRA over the attention projections, and - in the 8-step release - a
second ``turbo`` LoRA over rather more. Both LoRAs are part of the architecture
rather than a request-time choice: without Stage-B the branch reads projections
it was never aligned to, so there is nothing to switch off, and they are merged
into the weights instead of being managed by the PEFT runtime.

Two adapters editing the same projection add their deltas. That is exactly what
VDN's own inference does (``merge_lora_state`` runs once per adapter over the
same weights), and it is what "merge" means for LoRA: the composition happens in
the activations, not in the factors.

The diffusers-to-native placement lives next door in ``lowrank_fusion.py``;
what this module adds is the ``.attn.orig.`` rerooting its wrapper
introduced, the per-adapter ``lora_A.<name>.`` infix, and the branch's own key
translation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.logger import init_logger

from .checkpoint import VdnAdapter, VdnCheckpoint, VdnCheckpointError
from .lowrank_fusion import (
    QKV_SLOTS,
    LowRankWeightFusion,
    ParamPatch,
    check_block_coverage,
    resolve_native_target,
)

logger = init_logger(__name__)

#: VDN wraps the original attention as ``attn.orig`` so the teacher stays
#: recoverable; the projections underneath are H3's own.
_HYBRID_WRAPPER = ".attn.orig."
_ATTENTION = ".attn."

#: Where the branch's parameters live on our side of the name divide.
_BRANCH_BLOCK_PREFIX = "transformer_blocks."
_NATIVE_BLOCK_PREFIX = "blocks."
_BRANCH_ATTENTION = ".attn."
_NATIVE_BRANCH_ROOT = "attn.hybrid."
_BRANCH_SUBMODULE_RENAMES = (("linear_attention.", "linear."),)

#: Exactly what one converted block contributes. Checked as a set so a release
#: that adds or drops a tensor is refused rather than half-loaded: the branch
#: has no base weights to fall back on, so a missing tensor is an uninitialised
#: parameter in the middle of every attention.
BRANCH_BLOCK_SUFFIXES = frozenset(
    {
        "linear_attention.alpha.A_log",
        "linear_attention.alpha.dt_bias",
        "linear_attention.alpha.down.weight",
        "linear_attention.alpha.up.weight",
        "linear_attention.beta_proj.weight",
        "linear_attention.norm.weight",
        "linear_attention.output_gate.down.weight",
        "linear_attention.output_gate.up.weight",
        "linear_attention.output_gate.up.bias",
        "linear_attention.short_conv.k_sp.weight",
        "linear_attention.short_conv.k_tm.weight",
        "linear_attention.short_conv.v_sp.weight",
        "linear_attention.short_conv.v_tm.weight",
        "softmax_gate.up.weight",
        "softmax_gate.up.bias",
        "to_out_linear.weight",
    }
)


class VdnWeightFusion(LowRankWeightFusion):
    """Assign VDN's branch and merge its adapters as the checkpoint streams."""

    error_cls = VdnCheckpointError
    label = "VDN-H3 checkpoint"

    def __init__(
        self,
        *,
        source: Path,
        patches: Mapping[str, ParamPatch],
        head_dim: int,
        branch_path: Path,
        branch_keys: Mapping[str, str],
        adapter_names: tuple[str, ...],
    ) -> None:
        super().__init__(source=source, patches=patches, head_dim=head_dim)
        self._branch_path = branch_path
        # native parameter name -> its key in the branch safetensors. The
        # tensors themselves stay on disk until the stream reaches them: the
        # branch is 4.3 GiB and the loader is already holding a checkpoint.
        self._branch_keys = dict(branch_keys)
        self.adapter_names = adapter_names

    @property
    def injection_names(self) -> set[str]:
        return set(self._branch_keys)

    def iter_injections(self) -> Iterator[tuple[str, torch.Tensor]]:
        with safe_open(self._branch_path, framework="pt", device="cpu") as branch:
            for native_name, source_key in self._branch_keys.items():
                yield native_name, branch.get_tensor(source_key)

    def release(self) -> None:
        super().release()
        self._branch_keys.clear()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: VdnCheckpoint,
        *,
        head_dim: int,
        num_blocks: int,
        num_refiner_blocks: int,
    ) -> VdnWeightFusion:
        """Read every declared file and build the fusion, or refuse the release."""
        branch_keys = _read_branch_keys(checkpoint.branch_path, num_blocks=num_blocks)
        patches: dict[str, ParamPatch] = {}
        for adapter in checkpoint.adapters:
            _read_adapter(
                adapter,
                patches=patches,
                num_blocks=num_blocks,
                num_refiner_blocks=num_refiner_blocks,
            )

        fusion = cls(
            source=checkpoint.root,
            patches=patches,
            head_dim=head_dim,
            branch_path=checkpoint.branch_path,
            branch_keys=branch_keys,
            adapter_names=tuple(adapter.name for adapter in checkpoint.adapters),
        )
        fusion.validate_pairs()
        logger.info(
            "VDN-H3 fusion: %d branch tensors assigned, %d parameters patched by adapters %s",
            len(branch_keys),
            len(patches),
            list(fusion.adapter_names),
        )
        return fusion


def _read_branch_keys(branch_path: Path, *, num_blocks: int) -> dict[str, str]:
    """Map every branch tensor to the parameter it becomes.

    Reads names only. Coverage is per block and exact in both directions,
    because the branch is the part of the architecture the base checkpoint
    cannot supply a fallback for.
    """
    with safe_open(branch_path, framework="pt", device="cpu") as branch:
        keys = list(branch.keys())

    mapped: dict[str, str] = {}
    per_block: dict[int, set[str]] = {}
    unmapped: list[str] = []
    for key in keys:
        resolved = _resolve_branch_key(key)
        if resolved is None:
            unmapped.append(key)
            continue
        block_index, suffix, native_name = resolved
        if native_name in mapped:
            raise VdnCheckpointError(f"{branch_path} carries two tensors for {native_name}")
        mapped[native_name] = key
        per_block.setdefault(block_index, set()).add(suffix)

    if unmapped:
        raise VdnCheckpointError(
            f"{branch_path} has {len(unmapped)} tensors that name no known VDN branch parameter: {sorted(unmapped)[:5]}"
        )
    for block_index in range(num_blocks):
        suffixes = per_block.get(block_index, set())
        if suffixes != BRANCH_BLOCK_SUFFIXES:
            missing = sorted(BRANCH_BLOCK_SUFFIXES - suffixes)
            extra = sorted(suffixes - BRANCH_BLOCK_SUFFIXES)
            raise VdnCheckpointError(
                f"{branch_path} block {block_index} carries {len(suffixes)} of "
                f"{len(BRANCH_BLOCK_SUFFIXES)} branch tensors (missing={missing[:5]}, unknown={extra[:5]})"
            )
    unknown_blocks = sorted(set(per_block) - set(range(num_blocks)))
    if unknown_blocks:
        raise VdnCheckpointError(
            f"{branch_path} carries branch tensors for blocks {unknown_blocks[:5]}, but this model has {num_blocks}"
        )
    return mapped


def _resolve_branch_key(key: str) -> tuple[int, str, str] | None:
    """``transformer_blocks.N.attn.<suffix>`` -> ``(N, suffix, native name)``."""
    if not key.startswith(_BRANCH_BLOCK_PREFIX):
        return None
    remainder = key[len(_BRANCH_BLOCK_PREFIX) :]
    index, _, rest = remainder.partition(".")
    if not index.isdigit() or not rest.startswith(_BRANCH_ATTENTION[1:]):
        return None
    suffix = rest[len(_BRANCH_ATTENTION) - 1 :]
    if suffix not in BRANCH_BLOCK_SUFFIXES:
        return None
    native_suffix = suffix
    for old, new in _BRANCH_SUBMODULE_RENAMES:
        if native_suffix.startswith(old):
            native_suffix = new + native_suffix[len(old) :]
            break
    return int(index), suffix, f"{_NATIVE_BLOCK_PREFIX}{index}.{_NATIVE_BRANCH_ROOT}{native_suffix}"


def _read_adapter(
    adapter: VdnAdapter,
    *,
    patches: dict[str, ParamPatch],
    num_blocks: int,
    num_refiner_blocks: int,
) -> None:
    """Fold one adapter's factors into ``patches``, checking its own contract."""
    unmapped: list[str] = []
    seen_modules: set[str] = set()
    blocks_seen: dict[str, set[int]] = {}

    with safe_open(adapter.weights_path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            split = _split_adapter_key(key, adapter.name)
            if split is None:
                unmapped.append(key)
                continue
            module, role = split
            target = resolve_native_target(_reroot(module))
            if target is None:
                unmapped.append(key)
                continue
            native_module, layout, block = target
            seen_modules.add(module)
            if block is not None:
                blocks_seen.setdefault(block[0], set()).add(block[1])
            native_param = f"{native_module}.weight"
            patch = patches.setdefault(native_param, ParamPatch(layout=layout))
            # The three QKV projections legitimately aim different layouts at
            # one grouped parameter; anything else disagreeing about how a delta
            # enters would place one of them wrongly.
            grouped = patch.layout in QKV_SLOTS and layout in QKV_SLOTS
            if patch.layout != layout and not grouped:
                raise VdnCheckpointError(
                    f"adapter {adapter.name!r} aims a {layout!r} delta at {native_param}, which another "
                    f"adapter already edits as {patch.layout!r}"
                )
            factors = patch.factors(layout, key=adapter.name)
            # The scale is per module: the turbo adapter puts rank 16 on the
            # AdaLN projections and 64 elsewhere.
            factors.scale = adapter.scale_for(module)
            setattr(factors, "a" if role == "lora_a" else "b", checkpoint.get_tensor(key))

    if unmapped:
        raise VdnCheckpointError(
            f"adapter {adapter.name!r} at {adapter.weights_path} has {len(unmapped)} tensors that name no "
            f"known H3 parameter: {sorted(unmapped)[:5]}"
        )
    if adapter.exact_targets:
        # An ``exact_targets`` adapter lists every module it edits, so the two
        # sets have to agree exactly: a target with no tensors is a truncated
        # file, and a tensor with no target is an adapter for another model.
        declared = set(adapter.targets)
        missing = sorted(declared - seen_modules)
        extra = sorted(seen_modules - declared)
        if missing or extra:
            raise VdnCheckpointError(
                f"adapter {adapter.name!r} declares {len(declared)} targets but carries "
                f"{len(seen_modules)} (missing={missing[:5]}, unexpected={extra[:5]})"
            )
    else:
        # The Stage-B LoRA states its targets as suffix patterns, so the
        # meaningful check is that it reached every block of this model.
        check_block_coverage(
            blocks_seen,
            expected={"blocks.": num_blocks, "token_refiner.blocks.": num_refiner_blocks},
            source=f"adapter {adapter.name!r} at {adapter.weights_path}",
            error_cls=VdnCheckpointError,
        )


def _split_adapter_key(key: str, adapter_name: str) -> tuple[str, str] | None:
    """``<module>.lora_A.<name>.weight`` -> ``(module, role)``.

    The adapter name is held to the directory it came from: a file whose
    tensors are named for another adapter would merge at this one's scales.
    """
    for marker, role in ((".lora_A.", "lora_a"), (".lora_B.", "lora_b")):
        module, found, rest = key.partition(marker)
        if not found:
            continue
        if rest != f"{adapter_name}.weight":
            return None
        return module, role
    return None


def _reroot(module: str) -> str:
    """Undo VDN's ``HybridAttention`` wrapper in an adapter target path."""
    return module.replace(_HYBRID_WRAPPER, _ATTENTION, 1) if _HYBRID_WRAPPER in module else module


def resolve_vdn_fusion(
    checkpoint: VdnCheckpoint | None,
    *,
    head_dim: int,
    num_blocks: int,
    num_refiner_blocks: int,
) -> VdnWeightFusion | None:
    if checkpoint is None:
        return None
    return VdnWeightFusion.from_checkpoint(
        checkpoint,
        head_dim=head_dim,
        num_blocks=num_blocks,
        num_refiner_blocks=num_refiner_blocks,
    )


def branch_parameter_names(num_blocks: int) -> Iterable[str]:
    """Every parameter the branch assigns, for tests and diagnostics."""
    for block in range(num_blocks):
        for suffix in sorted(BRANCH_BLOCK_SUFFIXES):
            resolved = _resolve_branch_key(f"{_BRANCH_BLOCK_PREFIX}{block}.attn.{suffix}")
            assert resolved is not None
            yield resolved[2]


__all__ = [
    "BRANCH_BLOCK_SUFFIXES",
    "VdnWeightFusion",
    "branch_parameter_names",
    "resolve_vdn_fusion",
]
