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
from typing import Any

from vllm.logger import init_logger

from vllm_omni.errors import OmniClientError

from .checkpoint import (
    VDN_AUDIO_SHIFT,
    VDN_SUPPORTED_TASKS,
    VDN_TURBO_DENOISE_STEPS,
    VDN_VIDEO_SHIFT,
    VdnCheckpoint,
)
from .config import MiniMaxH3HybridAttentionConfig

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


__all__ = ["VdnServingContract"]
