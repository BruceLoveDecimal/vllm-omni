# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Locating and reading a VDN-H3 checkpoint.

The release is an exploded directory rather than a single file::

    stage-dmd-step-250/
      model_spec.json                    architecture + adapter declarations
      metadata.json                      artifact kind and format version
      linear_branch/model.safetensors    the branch's own 800 tensors
      adapters/default/…                 Stage-B LoRA (rank 64) over q/k/v/o
      adapters/turbo/…                   the 8-step DMD LoRA

Nothing here touches the model: it resolves a local path or a Hub id into that
layout, checks the artifact is the format this server reads, and reports what
the checkpoint declares. Applying it is ``weight_fusion.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

from .config import (
    HYBRID_TRANSFORM_TYPE,
    HYBRID_TRANSFORM_VERSION,
    SPEC_FORMAT_VERSION,
    VdnConfigError,
)

logger = init_logger(__name__)

SPEC_FILE = "model_spec.json"
METADATA_FILE = "metadata.json"
BRANCH_DIR = "linear_branch"
BRANCH_WEIGHTS = "model.safetensors"
ADAPTERS_DIR = "adapters"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
#: The adapter peft names when a spec entry does not name itself. Position is
#: the fallback, exactly as VDN's own exporter resolves it.
DEFAULT_ADAPTER_NAME = "default"
TURBO_ADAPTER_NAME = "turbo"

#: The 8-step ladder the ``turbo`` adapter was distilled on. VDN's scheduler
#: builds ``linspace(1, 0, num_steps + 1)`` and drives one forward per interval,
#: so eight evaluations are these nine uniform positions; H3's per-modality
#: shifts (12 video / 3 audio) are applied on top, as they are for every other
#: pinned schedule here.
VDN_TURBO_DENOISE_STEPS = 8
VDN_TURBO_BASE_SCHEDULE = DMD2SigmaSchedule.from_positions(
    tuple(1.0 - index / VDN_TURBO_DENOISE_STEPS for index in range(VDN_TURBO_DENOISE_STEPS + 1))
)
#: The shifts the released checkpoints were trained and benchmarked at.
VDN_VIDEO_SHIFT = 12.0
VDN_AUDIO_SHIFT = 3.0

VDN_SUPPORTED_TASKS = frozenset({"t2va"})

_SPEC_KEYS = frozenset(
    {"checkpoint", "subdir", "revision", "adapters", "window_group_batch", "window_impl", "linear_attention_enabled"}
)


class VdnCheckpointError(VdnConfigError):
    """The artifact at the configured path is not a VDN-H3 checkpoint."""


@dataclass(frozen=True)
class VdnAdapter:
    """One LoRA the checkpoint declares, and the scale each target merges at."""

    name: str
    weights_path: Path
    rank: int
    alpha: float
    targets: tuple[str, ...]
    exact_targets: bool
    rank_pattern: Mapping[str, int] = field(default_factory=dict)
    alpha_pattern: Mapping[str, float] = field(default_factory=dict)

    def scale_for(self, module: str) -> float:
        """``alpha / rank`` for one adapter module path.

        The turbo adapter puts rank 16 on the AdaLN projections and rank 64
        everywhere else, with alpha tracking rank, so every pair merges at
        exactly 1.0 - but the scale is read from the declaration rather than
        assumed, because an adapter whose alpha does not track its rank would
        otherwise merge far too strongly with nothing to show for it.
        """
        rank = int(self.rank_pattern.get(module, self.rank))
        alpha = float(self.alpha_pattern.get(module, self.alpha))
        if rank <= 0:
            raise VdnCheckpointError(f"adapter {self.name!r} declares rank {rank} for {module!r}")
        return alpha / rank


@dataclass(frozen=True)
class VdnCheckpoint:
    """A resolved VDN release: where its files are and what it declares."""

    root: Path
    transform_config: Mapping[str, Any]
    branch_path: Path
    adapters: tuple[VdnAdapter, ...]
    base_source: str | None = None
    base_revision: str | None = None

    @property
    def has_turbo(self) -> bool:
        return any(adapter.name == TURBO_ADAPTER_NAME for adapter in self.adapters)

    @property
    def base_schedule(self) -> tuple[float, ...] | None:
        """The rectified-flow positions this checkpoint samples on.

        Only the DMD variant pins one. The 50-step Stage-B checkpoint is not
        distilled, so it keeps the uniform ladder derived from the request.
        """
        return VDN_TURBO_BASE_SCHEDULE.base_schedule if self.has_turbo else None


@dataclass(frozen=True)
class VdnSpec:
    """The ``model_config['vdn']`` block, parsed."""

    checkpoint: str
    subdir: str | None = None
    revision: str | None = None
    adapters: tuple[str, ...] | None = None
    window_group_batch: int = 4
    window_impl: str = "auto"
    linear_attention_enabled: bool = True

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any]) -> VdnSpec:
        unknown = sorted(set(spec) - _SPEC_KEYS)
        if unknown:
            raise VdnCheckpointError(f"unknown vdn config keys {unknown}; supported: {sorted(_SPEC_KEYS)}")
        checkpoint = spec.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise VdnCheckpointError("vdn.checkpoint must name a local directory or a Hugging Face repository")
        adapters = spec.get("adapters")
        if adapters is not None:
            if isinstance(adapters, str) or not isinstance(adapters, Sequence):
                raise VdnCheckpointError("vdn.adapters must be a list of adapter names")
            adapters = tuple(str(name) for name in adapters)
        subdir = spec.get("subdir")
        revision = spec.get("revision")
        return cls(
            checkpoint=checkpoint,
            subdir=None if subdir is None else str(subdir),
            revision=None if revision is None else str(revision),
            adapters=adapters,
            window_group_batch=int(spec.get("window_group_batch", 4)),
            window_impl=str(spec.get("window_impl", "auto")),
            linear_attention_enabled=bool(spec.get("linear_attention_enabled", True)),
        )


def resolve_vdn_checkpoint(spec: VdnSpec) -> VdnCheckpoint:
    """Resolve ``spec`` to a readable checkpoint directory and read it."""
    root = _resolve_root(spec)
    spec_path = root / SPEC_FILE
    if not spec_path.is_file():
        raise VdnCheckpointError(
            f"{root} carries no {SPEC_FILE}; point vdn.checkpoint at a VDN release directory "
            "(for a Hub id, name the release with vdn.subdir, e.g. stage-dmd-step-250)"
        )
    model_spec = _read_json(spec_path)
    _check_metadata(root)

    if model_spec.get("format_version") != SPEC_FORMAT_VERSION:
        raise VdnCheckpointError(
            f"{spec_path} states format_version={model_spec.get('format_version')!r}; "
            f"this server reads {SPEC_FORMAT_VERSION}"
        )
    transform_config = _read_transform(model_spec, spec_path)

    branch_path = root / BRANCH_DIR / BRANCH_WEIGHTS
    if not branch_path.is_file():
        raise VdnCheckpointError(f"{root} carries no {BRANCH_DIR}/{BRANCH_WEIGHTS}; the linear branch is missing")

    adapters = _read_adapters(root, model_spec, selected=spec.adapters)
    base = model_spec.get("base") or {}

    logger.info(
        "VDN-H3 checkpoint %s: window chunk=%s radius=%s, adapters=%s",
        root,
        transform_config.get("softmax_attention", {}).get("chunk"),
        transform_config.get("softmax_attention", {}).get("radius"),
        [adapter.name for adapter in adapters],
    )
    return VdnCheckpoint(
        root=root,
        transform_config=transform_config,
        branch_path=branch_path,
        adapters=adapters,
        base_source=base.get("source"),
        base_revision=base.get("revision"),
    )


def _resolve_root(spec: VdnSpec) -> Path:
    local = Path(spec.checkpoint)
    if local.is_dir():
        root = local / spec.subdir if spec.subdir else local
        if not root.is_dir():
            raise VdnCheckpointError(f"{root} does not exist under the configured vdn.checkpoint")
        return root
    if not spec.subdir:
        raise VdnCheckpointError(
            f"vdn.checkpoint={spec.checkpoint!r} is not a local directory, so it is read as a Hugging Face "
            "repository and vdn.subdir must name the release inside it (e.g. stage-dmd-step-250)"
        )
    # Imported here so a local-directory checkpoint never reaches the Hub code.
    from vllm_omni.model_executor.model_loader.weight_utils import (
        download_weights_from_hf_specific,
    )

    downloaded = download_weights_from_hf_specific(
        model_name_or_path=spec.checkpoint,
        cache_dir=None,
        allow_patterns=[f"{spec.subdir}/**"],
        revision=spec.revision,
        require_all=True,
    )
    return Path(downloaded) / spec.subdir


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VdnCheckpointError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VdnCheckpointError(f"{path} must hold a JSON object")
    return payload


def _check_metadata(root: Path) -> None:
    """Refuse anything but a finished ``weights`` artifact.

    A trainer's ``train_state`` carries optimizer moments and a smoke-test
    export carries truncated blocks; both would load far enough to look fine.
    """
    metadata_path = root / METADATA_FILE
    if not metadata_path.is_file():
        raise VdnCheckpointError(f"{root} carries no {METADATA_FILE}")
    metadata = _read_json(metadata_path)
    if metadata.get("checkpoint_format_version") != SPEC_FORMAT_VERSION:
        raise VdnCheckpointError(
            f"{metadata_path} states checkpoint_format_version="
            f"{metadata.get('checkpoint_format_version')!r}; this server reads {SPEC_FORMAT_VERSION}"
        )
    if metadata.get("kind") != "weights":
        raise VdnCheckpointError(f"{metadata_path} is a {metadata.get('kind')!r} artifact, not released weights")
    inner = metadata.get("metadata") or {}
    if inner.get("truncated_blocks"):
        raise VdnCheckpointError(f"{root} is a truncated smoke-test artifact")


def _read_transform(model_spec: Mapping[str, Any], spec_path: Path) -> Mapping[str, Any]:
    transforms = model_spec.get("transforms") or []
    hybrid = [entry for entry in transforms if entry.get("type") == HYBRID_TRANSFORM_TYPE]
    if len(transforms) != 1 or len(hybrid) != 1:
        kinds = [entry.get("type") for entry in transforms]
        raise VdnCheckpointError(
            f"{spec_path} declares transforms {kinds}; this server serves exactly one "
            f"{HYBRID_TRANSFORM_TYPE!r} transform"
        )
    entry = hybrid[0]
    if entry.get("version") != HYBRID_TRANSFORM_VERSION:
        raise VdnCheckpointError(
            f"{spec_path} declares {HYBRID_TRANSFORM_TYPE} version {entry.get('version')!r}; "
            f"this server reads version {HYBRID_TRANSFORM_VERSION}"
        )
    config = entry.get("config")
    if not isinstance(config, Mapping):
        raise VdnCheckpointError(f"{spec_path} declares a {HYBRID_TRANSFORM_TYPE} transform with no config")
    return config


def _read_adapters(
    root: Path,
    model_spec: Mapping[str, Any],
    *,
    selected: tuple[str, ...] | None,
) -> tuple[VdnAdapter, ...]:
    """Read the adapters the spec declares, in declaration order.

    Order is load-bearing only in that both adapters' deltas are summed, which
    commutes - but the checkpoint's own order is what the release describes and
    what an operator reads in the log, so it is preserved rather than sorted.
    """
    declared = model_spec.get("adapters") or []
    if not declared:
        raise VdnCheckpointError(f"{root}: a VDN checkpoint declares at least the Stage-B LoRA")

    adapters: list[VdnAdapter] = []
    for index, entry in enumerate(declared):
        if entry.get("type") != "lora":
            raise VdnCheckpointError(f"{root}: unknown adapter type {entry.get('type')!r}")
        config = entry.get("config") or {}
        # VDN's exporter names an adapter by its config, falling back to peft's
        # own default for the first, unnamed entry.
        name = str(config.get("name") or (DEFAULT_ADAPTER_NAME if index == 0 else f"adapter_{index}"))
        if selected is not None and name not in selected:
            logger.info("VDN-H3: skipping declared adapter %r (not in vdn.adapters)", name)
            continue
        weights_path = root / ADAPTERS_DIR / name / ADAPTER_WEIGHTS
        if not weights_path.is_file():
            raise VdnCheckpointError(
                f"{root} declares adapter {name!r} but carries no {weights_path.relative_to(root)}"
            )
        rank = config.get("rank")
        alpha = config.get("alpha")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise VdnCheckpointError(f"adapter {name!r} declares rank={rank!r}")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise VdnCheckpointError(f"adapter {name!r} declares alpha={alpha!r}")
        adapters.append(
            VdnAdapter(
                name=name,
                weights_path=weights_path,
                rank=int(rank),
                alpha=float(alpha),
                targets=tuple(str(target) for target in config.get("targets") or ()),
                exact_targets=bool(config.get("exact_targets", False)),
                rank_pattern=dict(config.get("rank_pattern") or {}),
                alpha_pattern=dict(config.get("alpha_pattern") or {}),
            )
        )

    if selected is not None:
        missing = sorted(set(selected) - {adapter.name for adapter in adapters})
        if missing:
            raise VdnCheckpointError(f"{root} declares no adapter named {missing}")
    if not adapters:
        raise VdnCheckpointError(f"{root}: vdn.adapters selected none of the declared adapters")
    return tuple(adapters)


__all__ = [
    "VDN_AUDIO_SHIFT",
    "VDN_SUPPORTED_TASKS",
    "VDN_TURBO_BASE_SCHEDULE",
    "VDN_TURBO_DENOISE_STEPS",
    "VDN_VIDEO_SHIFT",
    "VdnAdapter",
    "VdnCheckpoint",
    "VdnCheckpointError",
    "VdnSpec",
    "resolve_vdn_checkpoint",
]
