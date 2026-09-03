# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""What a VDN-H3 server will and will not answer.

The checkpoint is a hybrid architecture rather than a decoding option, so the
things it cannot do are startup errors, not runtime surprises. Each refusal
below exists because the alternative is a server that runs and produces a
quietly different sample: a task VDN never trained, a schedule its distillation
never saw, or a parallel or offload mode that would route around the branch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.errors import OmniClientError

from .checkpoint import (
    VDN_AUDIO_SHIFT,
    VDN_SUPPORTED_TASKS,
    VDN_TURBO_DENOISE_STEPS,
    VDN_VIDEO_SHIFT,
    VdnCheckpoint,
    VdnSpec,
    resolve_vdn_checkpoint,
)
from .config import MiniMaxH3HybridAttentionConfig

if TYPE_CHECKING:
    from .weight_fusion import VdnWeightFusion

logger = init_logger(__name__)

_OFFLOAD_FLAGS = (
    "enable_cpu_offload",
    "enable_layerwise_offload",
    "enable_distributed_layerwise_offload",
)


class VdnServingContract:
    """The startup and per-request checks a VDN checkpoint imposes."""

    def __init__(self, checkpoint: VdnCheckpoint, config: MiniMaxH3HybridAttentionConfig) -> None:
        self.checkpoint = checkpoint
        self.config = config

    @property
    def base_schedule(self) -> tuple[float, ...] | None:
        return self.checkpoint.base_schedule

    def build_fusion(self, *, head_dim: int, num_blocks: int, num_refiner_blocks: int) -> VdnWeightFusion:
        """Read the release's tensors, ready to fold into the weight stream.

        Names and adapter factors are read now; the branch's several gigabytes
        stay on disk until the stream reaches them.
        """
        from .weight_fusion import VdnWeightFusion

        return VdnWeightFusion.from_checkpoint(
            self.checkpoint,
            head_dim=head_dim,
            num_blocks=num_blocks,
            num_refiner_blocks=num_refiner_blocks,
        )

    def check_serving_contract(self, *, partition: str, od_config: Any, lora_path: Any) -> None:
        """Hold a starting server to what this checkpoint can serve."""
        if partition != "fl2va":
            # The FL2VA partition is the one that serves T2VA. A Ref2VA
            # partition has no T2VA path at all, and ``combined`` would load a
            # second 66 GiB transformer that the branch can never be used with.
            raise ValueError(
                f"VDN-H3 distills {sorted(VDN_SUPPORTED_TASKS)} only, which the FL2VA partition serves; "
                f"this server started the {partition!r} partition. Start it with --task-type t2va."
            )
        if lora_path:
            # Both of VDN's adapters are already in the weights, and its
            # schedule is pinned; a second artifact would either be ignored or
            # would move the sample off the rungs the student was distilled at.
            raise ValueError("VDN-H3 merges its own adapters at load time, so --lora-path cannot be combined with it")

        offloads = [flag for flag in _OFFLOAD_FLAGS if getattr(od_config, flag, False)]
        if offloads:
            # A host-weight plan installs the transformer without going through
            # load_weights(), which is where the branch is assigned and its
            # completeness checked. The server would then run dense H3 weights
            # under a hybrid architecture, with nothing to signal it.
            raise ValueError(
                f"VDN-H3 is fused while the checkpoint streams in, so it cannot be combined with "
                f"{sorted(offloads)}. Serve it on more GPUs instead (--tensor-parallel-size 2)."
            )

        parallel = getattr(od_config, "parallel_config", None)
        ring = int(getattr(parallel, "ring_degree", 1) or 1)
        allgather = int(getattr(parallel, "allgather_degree", 1) or 1)
        if ring != 1 or allgather != 1:
            # Ring attention dispatches through its own kernels, which never see
            # the decomposed window; all-gather-KV shards keys rather than rows,
            # which the frame recurrence cannot use.
            raise ValueError(
                "VDN-H3 supports local execution, tensor parallelism and strict Ulysses sequence "
                f"parallelism; got ring_degree={ring}, allgather_degree={allgather}."
            )
        mode = str(getattr(parallel, "ulysses_mode", "strict") or "strict")
        if mode != "strict":
            # The branch exchanges its own packed buffer, which assumes equal
            # row shards and an exact head split rather than padded heads.
            raise ValueError(f"VDN-H3 requires ulysses_mode='strict', got {mode!r}")

        logger.info(
            "VDN-H3 active: window chunk=%d radius=%d, %s, %d denoise steps, shifts %g/%g",
            self.config.chunk,
            self.config.radius,
            "linear branch enabled" if self.config.linear_attention_enabled else "LINEAR BRANCH DISABLED (ablation)",
            VDN_TURBO_DENOISE_STEPS if self.checkpoint.has_turbo else 0,
            VDN_VIDEO_SHIFT,
            VDN_AUDIO_SHIFT,
        )

    def check_task(self, task: str) -> None:
        if task not in VDN_SUPPORTED_TASKS:
            raise OmniClientError(f"VDN-H3 serves {sorted(VDN_SUPPORTED_TASKS)} only, got task={task!r}")

    def check_request(self, sampling: Any, *, video_shift: float, audio_shift: float) -> None:
        """Refuse a request that would sample this checkpoint off its rungs."""
        if sampling.lora_request is not None:
            raise OmniClientError(
                "this server merged a VDN-H3 checkpoint at startup, so per-request lora is unavailable; "
                "drop the lora field"
            )
        # The window and the branch were trained together at these shifts, and
        # the distilled ladder's positions only become the noise levels the
        # student saw once they are applied.
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", video_shift), ("audio_flow_shift", audio_shift)):
            try:
                requested = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"VDN-H3 requires {key}={expected:g}") from exc
            if not math.isclose(requested, expected):
                raise OmniClientError(f"VDN-H3 requires {key}={expected:g}, got {requested:g}")


def resolve_vdn_serving(
    od_config: Any,
    release: Mapping[str, Any],
    model_path: Path,
) -> VdnServingContract | None:
    """Claim a VDN-H3 checkpoint, from the server flag or the release itself.

    ``--model-config '{"vdn": {...}}'`` names one explicitly; a packaged release
    can instead declare it in ``model_index.json`` under ``_minimax_h3.vdn``, so
    ``vllm-omni serve <dir>`` needs no flags. The flag wins when both appear.

    Returns ``None`` for an ordinary dense H3, which is every other checkpoint -
    that is what keeps this whole package out of the dense path.
    """
    model_config = getattr(od_config, "model_config", None) or {}
    raw = model_config.get("vdn") or release.get("vdn")
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("model_config['vdn'] must be an object naming a VDN-H3 checkpoint")
    spec = VdnSpec.from_mapping(raw)
    candidate = Path(spec.checkpoint)
    if not candidate.is_absolute() and not candidate.is_dir():
        # A release that declares its own hybrid names it relative to itself.
        packaged = model_path / spec.checkpoint
        if packaged.is_dir():
            spec = replace(spec, checkpoint=str(packaged))
    checkpoint = resolve_vdn_checkpoint(spec)

    # The head dim comes from the transformer config the server already read,
    # so the architecture is checked against the model it will actually run.
    tf_config = getattr(od_config, "tf_model_config", None) or {}
    mapping = tf_config.to_dict() if hasattr(tf_config, "to_dict") else dict(tf_config)
    head_dim = mapping.get("attention_head_dim")
    if not isinstance(head_dim, int):
        raise ValueError(f"the transformer config states no attention_head_dim, got {head_dim!r}")
    config = MiniMaxH3HybridAttentionConfig.from_transform_config(
        checkpoint.transform_config,
        attention_head_dim=head_dim,
        window_group_batch=spec.window_group_batch,
        window_impl=spec.window_impl,
        linear_attention_enabled=spec.linear_attention_enabled,
    )
    return VdnServingContract(checkpoint, config)


__all__ = ["VdnServingContract", "resolve_vdn_serving"]
