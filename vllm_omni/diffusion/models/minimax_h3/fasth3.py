# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FastVideo FastH3: a four-step DMD2 student of MiniMax-H3.

FastH3 replaces H3's 49 denoiser evaluations with four. It ships as an adapter
over the base checkpoint rather than as a full release, so it reuses H3's text
encoder, video VAE, audio VAE, tokenizers and schedulers unchanged.

The artifact is *not* a PEFT LoRA, and it is not request-switchable. Its own
metadata states the reconstruction as::

    W = W_base + lora_B @ lora_A; then .diff/.diff_b added and .set_weight assigned

so besides rank-64 factors it carries full-rank ``.diff``/``.diff_b`` deltas for
RMSNorm weights, biases, patch projections and the final layer - none of which a
LoRA layer can express - and the VSA variants add ``.set_weight`` tensors for
compression gates that do not exist in the base transformer at all. The adapter
is therefore fused into the checkpoint stream at load time, before the weights
are sharded, which is also what the release's model card requires.

The low-rank factors carry no alpha: the reconstruction adds ``lora_B @ lora_A``
directly, i.e. a scale of exactly 1.

Two checkpoint spellings meet here. The adapter is written in the diffusers
namespace (``transformer_blocks.0.attn.to_q``) while vLLM-Omni loads H3's native
one (``blocks.0.attn.qkv_proj``), whose attention and MLP projections are fused.
Every mapping and layout convention in ``lowrank_fusion.py`` was verified tensor
by tensor against the released full checkpoint
(``FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree``): ``W_base + delta``
reproduces it to bf16 rounding. This module owns what is FastH3's alone - the
release identity, the declared tensor counts, the four-step ladder and the
serving contract; the fusion arithmetic is shared with VDN-H3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from vllm.logger import init_logger

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError

from .lowrank_fusion import (
    LowRankWeightFusion,
    MiniMaxH3FusionError,
    ParamPatch,
    check_block_coverage,
    resolve_native_target,
)

if TYPE_CHECKING:
    from .minimax_h3_transformer import MiniMaxH3DiTModel

logger = init_logger(__name__)

# FastVideo's generic adapter container, not a FastH3 marker: their tools emit
# it for ordinary H3 adapters too.
FASTH3_FORMAT = "fastvideo-lora-v2"
FASTH3_MANIFEST = "adapter_manifest.json"
# The release identity: the distilled student the adapter came from, over the
# base it edits. The name is held to FastVideo's own namespace as well as the
# student's name, so an unrelated adapter that merely mentions FastH3 is not
# claimed.
FASTH3_BASE_MODEL = "MiniMaxAI/MiniMax-H3"
_FASTH3_IDENTITY_KEY = "finetuned_model"
_FASTH3_IDENTITY_NAMESPACE = "fastvideo/"
_FASTH3_IDENTITY_MARKER = "fasth3"

# The rectified-flow positions the student was distilled at, and the ladder the
# server samples on. The release states them as `dmd_denoising_steps`
# [999, 749, 500, 250] (`sampling.base_timesteps` in the bundle manifest):
# timestep indices out of 1000, i.e. pre-shift positions, closed here with the
# terminal 0.0 every rectified-flow schedule ends on. The opening rung is 0.999
# rather than 1.0 because training capped the noise level there
# (`max_timestep_ratio`).
#
# These are positions, not final sigmas: H3's schedulers apply their own
# per-modality shift on top (12 for video, 3 for audio), which is what
# reproduces the levels the student saw. Nothing here overrides those shifts.
FASTH3_BASE_SCHEDULE = DMD2SigmaSchedule.from_positions((0.999, 0.749, 0.5, 0.25, 0.0))
# Its five points bound four transformer forwards, one per sigma interval. That
# count is the one a request states, the one Cache-DiT is refreshed with and the
# one step execution admits a request on - the same interval contract H3's
# pinned checkpoint schedules and native LoRAs already use.
FASTH3_DENOISE_STEPS = FASTH3_BASE_SCHEDULE.num_inference_steps
# Preview v1 distills the text-to-video-and-audio path only.
FASTH3_SUPPORTED_TASKS = frozenset({"t2va"})

_LORA_A = ".lora_A.weight"
_LORA_B = ".lora_B.weight"
_DIFF = ".diff"
_DIFF_B = ".diff_b"
_SET_WEIGHT = ".set_weight"

# The attention role MiniMaxH3Attention gives its 50 DiT blocks. The
# compression gates live on exactly these layers, so this is the role whose
# resolved backend decides whether the artifact runs sparse.
_H3_DIT_ATTENTION_ROLE = "self"


def _resolve_dit_attention_backend(od_config: Any) -> str:
    """The backend the 50-block H3 DiT will actually resolve to.

    The DiT's attention layers carry role ``"self"``, so a ``per_role`` entry
    overrides the default for exactly the layers the compression gates live on.
    Reading only the default would accept a config that runs the sparse student
    dense, and reject a per-role-only config that is correct.
    """
    attention_config = getattr(od_config, "diffusion_attention_config", None)
    per_role = getattr(attention_config, "per_role", None) or {}
    spec = per_role.get(_H3_DIT_ATTENTION_ROLE)
    if spec is not None:
        return str(getattr(spec, "backend", "") or "").upper()
    backend = str(getattr(od_config, "diffusion_attention_backend", "") or "").upper()
    if backend:
        return backend
    default_spec = getattr(attention_config, "default", None)
    return str(getattr(default_spec, "backend", "") or "").upper()


class FastH3AdapterError(MiniMaxH3FusionError):
    """The artifact is a FastH3 adapter, but it cannot be applied as one."""


def _split_adapter_key(name: str) -> tuple[str, str] | None:
    """Split an adapter tensor name into ``(module path, role)``."""
    for marker, role in (
        (_LORA_A, "lora_a"),
        (_LORA_B, "lora_b"),
        (_DIFF_B, "diff_b"),
        (_DIFF, "diff"),
        (_SET_WEIGHT, "set_weight"),
    ):
        if name.endswith(marker):
            return name[: -len(marker)], role
    return None


def _is_fasth3_release(metadata: Mapping[str, str]) -> bool:
    """Whether this is a FastH3 release rather than merely FastVideo's format.

    Claiming on the container alone would fuse an ordinary H3 adapter into the
    checkpoint and put every request on a four-step schedule it was never
    distilled for, and so would a bare ``fasth3`` substring: an adapter someone
    else named after the student is not the student.
    """
    if metadata.get("format") != FASTH3_FORMAT:
        return False
    identity = metadata.get(_FASTH3_IDENTITY_KEY, "").lower()
    if not identity.startswith(_FASTH3_IDENTITY_NAMESPACE) or _FASTH3_IDENTITY_MARKER not in identity:
        return False
    # Only held against a file that states it.
    base_model = metadata.get("base_model")
    return base_model is None or base_model.casefold() == FASTH3_BASE_MODEL.casefold()


def _check_declared_counts(
    metadata: Mapping[str, str],
    counted: Mapping[str, int],
    *,
    weights_path: Path,
) -> None:
    """Hold the artifact to the tensor counts its own metadata declares.

    The writer of the ``fastvideo-lora-v2`` format records how many tensors of
    each kind it emitted, which is the one statement in the file about its own
    completeness. A truncated or partially re-exported artifact is otherwise
    indistinguishable from a small one, so a claimed file has to carry all of
    them rather than opt out by omission.
    """
    for key, seen in counted.items():
        declared = metadata.get(key)
        if declared is None:
            raise FastH3AdapterError(
                f"{weights_path} declares the FastH3 identity but omits {key}; "
                "a claimed adapter has to state what it emitted"
            )
        try:
            expected = int(declared)
        except (TypeError, ValueError) as exc:
            raise FastH3AdapterError(f"{weights_path} declares a non-numeric {key}={declared!r}") from exc
        if expected != seen:
            raise FastH3AdapterError(
                f"{weights_path} declares {key}={expected} but carries {seen}; the adapter is incomplete "
                "and would leave most of the transformer on base H3 weights"
            )


class FastH3WeightFusion(LowRankWeightFusion):
    """Fuse a FastH3 adapter into the H3 checkpoint stream as it is loaded."""

    error_cls = FastH3AdapterError
    label = "FastH3 adapter"

    def __init__(
        self,
        *,
        source: Path,
        patches: Mapping[str, ParamPatch],
        head_dim: int,
        requires_vsa: bool,
        injections: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(source=source, patches=patches, head_dim=head_dim, injections=injections)
        self.requires_vsa = requires_vsa

    @property
    def source(self) -> Path:
        source = self._source
        assert isinstance(source, Path)
        return source

    @property
    def base_schedule(self) -> tuple[float, ...]:
        """The rectified-flow positions this student samples on.

        The fused checkpoint is a four-step student, so the ladder comes from
        the release rather than from the many-step teacher's metadata or from
        the uniform one ``num_inference_steps`` would otherwise derive.
        """
        return FASTH3_BASE_SCHEDULE.base_schedule

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        head_dim: int,
        num_blocks: int,
        num_refiner_blocks: int,
    ) -> FastH3WeightFusion | None:
        """Build a fusion from an adapter file or directory, else ``None``.

        Returning ``None`` keeps every other ``--lora-path`` artifact on the
        dynamic LoRA route; only a file carrying the FastH3 release identity is
        claimed here.

        The block counts are the model's, and the artifact has to cover them:
        claiming a partial adapter would switch the server onto the four-step
        contract while most of the transformer still held base H3 weights.
        """
        weights_path = _resolve_adapter_file(path)
        if weights_path is None:
            return None

        patches: dict[str, ParamPatch] = {}
        gate_tensors: list[str] = []
        injections: dict[str, torch.Tensor] = {}
        unmapped: list[str] = []
        blocks_seen: dict[str, set[int]] = {}
        counted = {"low_rank_tensors": 0, "diff_tensors": 0}
        with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if not _is_fasth3_release(metadata):
                return None
            for name in checkpoint.keys():
                split = _split_adapter_key(name)
                if split is None:
                    unmapped.append(name)
                    continue
                module, role = split
                target = resolve_native_target(module)
                if role == "set_weight":
                    # A VSA compression gate. The base transformer has no such
                    # parameter, so this is assigned into the stream rather than
                    # fused onto an existing weight.
                    if target is None:
                        unmapped.append(name)
                        continue
                    native_module, _, block = target
                    if block is not None:
                        blocks_seen.setdefault(block[0], set()).add(block[1])
                    gate_tensors.append(name)
                    injections[f"{native_module}.weight"] = checkpoint.get_tensor(name)
                    continue
                if target is None:
                    unmapped.append(name)
                    continue
                native_module, layout, block = target
                if block is not None:
                    blocks_seen.setdefault(block[0], set()).add(block[1])
                counted["diff_tensors" if role in ("diff", "diff_b") else "low_rank_tensors"] += 1
                native_param = f"{native_module}.{'bias' if role == 'diff_b' else 'weight'}"
                patch = patches.setdefault(native_param, ParamPatch(layout=layout))
                tensor = checkpoint.get_tensor(name)
                if role in ("diff", "diff_b"):
                    if patch.diff is not None:
                        raise FastH3AdapterError(f"duplicate {role} for {native_param}")
                    patch.diff = tensor
                else:
                    # The reconstruction adds ``lora_B @ lora_A`` directly, so
                    # every pair carries a scale of exactly 1.
                    factors = patch.factors(layout)
                    setattr(factors, "a" if role == "lora_a" else "b", tensor)

        if unmapped:
            raise FastH3AdapterError(
                f"FastH3 adapter at {weights_path} has {len(unmapped)} tensors that name no known "
                f"H3 parameter: {sorted(unmapped)[:5]}"
            )
        _check_declared_counts(
            metadata,
            {**counted, "set_weight_tensors": len(gate_tensors)},
            weights_path=weights_path,
        )
        check_block_coverage(
            blocks_seen,
            expected={"blocks.": num_blocks, "token_refiner.blocks.": num_refiner_blocks},
            source=weights_path,
            error_cls=FastH3AdapterError,
        )

        fusion = cls(
            source=weights_path,
            patches=patches,
            head_dim=head_dim,
            requires_vsa=bool(gate_tensors),
            injections=injections,
        )
        fusion.validate_pairs()
        logger.info(
            "FastH3 adapter %s: rank=%s, parameters patched=%d, low-rank=%s, diff=%s, set_weight=%d",
            weights_path,
            metadata.get("rank", "?"),
            len(patches),
            metadata.get("low_rank_tensors", "?"),
            metadata.get("diff_tensors", "?"),
            len(gate_tensors),
        )
        return fusion

    def check_serving_contract(
        self,
        *,
        partition: str,
        od_config: Any,
        video_shift: float,
        audio_shift: float,
    ) -> None:
        """Hold a starting server to the ladder this student was trained on."""
        if partition == "ref2va":
            raise ValueError("FastH3 preview v1 distills T2VA only, so it cannot serve a Ref2VA partition")
        offloads = [
            flag
            for flag in ("enable_cpu_offload", "enable_layerwise_offload", "enable_distributed_layerwise_offload")
            if getattr(od_config, flag, False)
        ]
        if offloads:
            # A host-weight plan installs the transformer without going through
            # load_weights(), which is where the fusion and its completeness
            # check live. Serving base H3 weights under a four-step schedule
            # would otherwise degrade output with nothing to signal it.
            raise ValueError(
                f"FastH3 is fused while the checkpoint streams in, so it cannot be combined with "
                f"{sorted(offloads)}. Serve it without offload."
            )
        if self.requires_vsa:
            backend = _resolve_dit_attention_backend(od_config)
            if backend != "FASTVIDEO_VSA":
                raise ValueError(
                    f"{self.source} is a Video Sparse Attention variant of FastH3. Its compression "
                    "gates only mean anything to the VSA kernel, and any other backend would run it "
                    f"as dense attention on a student distilled for 90% sparsity (got {backend or 'default'}). "
                    "Serve it with --diffusion-attention-backend FASTVIDEO_VSA."
                )
            parallel_config = getattr(od_config, "parallel_config", None)
            ring_degree = int(getattr(parallel_config, "ring_degree", 1) or 1)
            allgather_degree = int(getattr(parallel_config, "allgather_degree", 1) or 1)
            if ring_degree != 1 or allgather_degree != 1:
                raise ValueError(
                    "FastH3 VSA supports local attention or pure Ulysses sequence parallelism; "
                    "ring/all-gather SP does not give the block-sparse kernel the complete packed sequence."
                )
        logger.info(
            "FastH3 adapter active: sigma points %s for %d transformer forwards, "
            "flow_shift=%g, audio_flow_shift=%g, tasks=%s",
            list(self.base_schedule),
            FASTH3_DENOISE_STEPS,
            video_shift,
            audio_shift,
            sorted(FASTH3_SUPPORTED_TASKS),
        )

    def check_task(self, task: str) -> None:
        """Refuse a task this preview never distilled."""
        if task not in FASTH3_SUPPORTED_TASKS:
            raise OmniClientError(
                f"FastH3 preview v1 distills {sorted(FASTH3_SUPPORTED_TASKS)} only, got task={task!r}"
            )

    def check_request(self, sampling: Any, *, video_shift: float, audio_shift: float) -> None:
        """Refuse a request that would sample the student off its rungs."""
        if sampling.lora_request is not None:
            # The adapter is already in the weights and the dynamic LoRA manager
            # is skipped, so nothing would apply the requested one. Serving the
            # request anyway would quietly ignore it.
            raise OmniClientError(
                f"this server fused {self.source} into the checkpoint at startup, so per-request "
                "lora is unavailable; drop the lora field"
            )
        if int(sampling.num_inference_steps or 0) != FASTH3_DENOISE_STEPS:
            raise OmniClientError(
                f"FastH3 is a four-step student and requires num_inference_steps={FASTH3_DENOISE_STEPS} "
                "(one transformer forward per sigma interval)"
            )
        # The checkpoint's per-modality shifts turn the release's positions into
        # the noise levels the student was distilled at, so a request that moves
        # them samples where it was never trained.
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", video_shift), ("audio_flow_shift", audio_shift)):
            try:
                requested = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"FastH3 requires {key}={expected:g}") from exc
            if not math.isclose(requested, expected):
                raise OmniClientError(f"FastH3 requires {key}={expected:g}, got {requested:g}")


def resolve_fasth3_fusion(od_config: Any, transformer: MiniMaxH3DiTModel) -> FastH3WeightFusion | None:
    """Claim ``--lora-path`` when it points at a FastH3 adapter.

    FastH3 rewrites RMSNorm weights and biases, so it cannot be expressed as a
    request-switchable LoRA and is fused into the checkpoint instead. Any other
    artifact returns None here and stays on the dynamic LoRA route.
    """
    lora_path = getattr(od_config, "lora_path", None)
    if isinstance(lora_path, (list, tuple)):
        if len(lora_path) != 1:
            return None
        lora_path = lora_path[0]
    if not lora_path:
        return None
    # Placing a delta into the fused QKV parameter needs the head size, and
    # completeness is judged against the model's depth, so the architecture is
    # only read once an adapter is actually configured.
    arch = transformer.arch
    return FastH3WeightFusion.from_path(
        lora_path,
        head_dim=arch.attention_head_dim,
        num_blocks=arch.num_layers,
        num_refiner_blocks=arch.token_refiner_num_layers,
    )


def _safetensors_metadata(path: Path) -> Mapping[str, str]:
    """The header metadata of a safetensors file, or ``{}`` if unreadable."""
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            return checkpoint.metadata() or {}
    except Exception:  # noqa: BLE001 - a path that will not open is simply not ours
        return {}


def _resolve_adapter_file(path: str | Path) -> Path | None:
    """Find the single adapter file at ``path``, or ``None``."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate if candidate.suffix == ".safetensors" else None
    if not candidate.is_dir():
        return None
    named = candidate / "adapter_model.safetensors"
    if named.is_file():
        return named
    files = sorted(candidate.glob("*.safetensors"))
    if len(files) == 1:
        return files[0]
    # The published repository bundles four variants under one root, marked by an
    # adapter_manifest.json. Such a bundle is ambiguous rather than loadable - but
    # only hold that against a directory that actually carries FastH3 artifacts;
    # an unrelated multi-shard LoRA stays on the dynamic route via ``None``.
    if (candidate / FASTH3_MANIFEST).is_file() or any(
        _is_fasth3_release(_safetensors_metadata(file)) for file in files
    ):
        raise FastH3AdapterError(
            f"{candidate} holds several FastH3 adapters; point --lora-path at one variant "
            "(for example dense-datafree/adapter_model.safetensors)"
        )
    return None


__all__ = [
    "FASTH3_BASE_MODEL",
    "FASTH3_BASE_SCHEDULE",
    "FASTH3_DENOISE_STEPS",
    "FASTH3_FORMAT",
    "FASTH3_SUPPORTED_TASKS",
    "FastH3AdapterError",
    "FastH3WeightFusion",
    "resolve_fasth3_fusion",
]
