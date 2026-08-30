# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.platforms import current_omni_platform

from .runtime_config import _VoxCPM2RuntimeConfig


class VoxCPM2OmniARAsyncScheduler(OmniARAsyncScheduler):
    """VoxCPM2 scheduler variant for full unified decode graph serving.

    VoxCPM2's full unified decode graph only applies to pure decode batches.
    At saturation, waiting requests are deferred to keep full decode batches on
    the CUDA Graph path. Below capacity, normal mixed admission resumes so a
    sparse arrival does not wait for every active decode request to finish.
    """

    def _unified_decode_graph_enabled(self) -> bool:
        runtime_config = _VoxCPM2RuntimeConfig.from_vllm_config(self.vllm_config)
        return runtime_config.unified_decode_graph_available(use_cuda_graph=current_omni_platform.is_cuda())

    def _should_defer_waiting_for_unified_decode_graph(self) -> bool:
        if not self._unified_decode_graph_enabled():
            return False
        if not self.waiting or not self.running:
            return False

        has_decode_ready = any(
            getattr(request, "status", None) == RequestStatus.RUNNING
            and not request.is_finished()
            and self._get_confirmed_num_computed_tokens(request) >= request.num_prompt_tokens
            for request in self.running
        )
        if not has_decode_ready:
            return False
        return True

    def _unified_decode_graph_capacity_saturated(self) -> bool:
        max_num_seqs = int(getattr(self.scheduler_config, "max_num_seqs", 1))
        # Async scheduling moves requests out of ``running`` while their model
        # batch is in flight. ``requests`` remains the authoritative lifecycle
        # registry, so include it when observing the active high-water mark.
        queued_and_running = len(self.running) + len(self.waiting)
        active_requests = max(queued_and_running, len(getattr(self, "requests", ())))
        if active_requests >= max_num_seqs:
            self._unified_decode_graph_saturated = True
        elif active_requests == 0:
            self._unified_decode_graph_saturated = False
        return bool(getattr(self, "_unified_decode_graph_saturated", False))

    def _should_defer_waiting_admission(self) -> bool:
        saturated = self._unified_decode_graph_capacity_saturated()
        if not self._should_defer_waiting_for_unified_decode_graph():
            return False
        runtime_config = _VoxCPM2RuntimeConfig.from_vllm_config(self.vllm_config)
        return saturated if runtime_config.unified_decode_graph_admit_when_unsaturated else True
