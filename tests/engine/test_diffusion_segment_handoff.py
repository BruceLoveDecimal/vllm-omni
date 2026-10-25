# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ordered diffusion handoffs retain parent lifetime and cancel both stages."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_omni.diffusion.segment import SEGMENT_KEY
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def harness():
    orch = object.__new__(Orchestrator)
    orch.stage_pools = [
        SimpleNamespace(
            num_replicas=1, get_bound_replica_id=lambda rid: 0, collective_rpc=AsyncMock(return_value=None)
        ),
        SimpleNamespace(num_replicas=1),
    ]
    orch._forward_to_next_stage = AsyncMock()
    state = OrchestratorRequestState(request_id="r", final_stage_id=1)
    return orch, state


def output(index, *, ack=False, finished=False):
    marker = {"index": index, "finished": finished}
    if ack:
        marker["ack"] = True
    return SimpleNamespace(request_id="r", finished=finished or ack, custom_output={SEGMENT_KEY: marker})


@pytest.mark.asyncio
async def test_ordered_segments_ack_and_terminal_output():
    orch, state = harness()
    assert await orch._route_diffusion_segment(0, 0, output(0), state)
    assert state.diffusion_segment_pending
    first = output(0, ack=True)
    assert not await orch._route_diffusion_segment(1, 0, first, state)
    assert not first.finished
    orch.stage_pools[0].collective_rpc.assert_awaited_once_with(0, "acknowledge_segment", timeout=60, args=("r", 0))
    assert await orch._route_diffusion_segment(0, 0, output(1, finished=True), state)
    last = output(1, ack=True, finished=True)
    assert not await orch._route_diffusion_segment(1, 0, last, state)
    assert last.finished
    assert state.finished_stage_ids == {0}
    assert orch.stage_pools[0].collective_rpc.await_count == 1
    assert orch._forward_to_next_stage.await_count == 2


@pytest.mark.asyncio
async def test_duplicate_or_missing_segment_is_rejected():
    orch, state = harness()
    with pytest.raises(ValueError, match="Out-of-order"):
        await orch._route_diffusion_segment(0, 0, output(1), state)
    with pytest.raises(ValueError, match="Unexpected"):
        await orch._route_diffusion_segment(1, 0, output(0, ack=True), state)
    await orch._route_diffusion_segment(0, 0, output(0), state)
    with pytest.raises(ValueError, match="credit"):
        await orch._route_diffusion_segment(0, 0, output(0), state)


@pytest.mark.asyncio
async def test_abort_fans_out_to_both_stages_even_between_decode_calls():
    orch, _ = harness()
    for pool in orch.stage_pools:
        pool.abort_requests = AsyncMock(return_value=[])
        pool.release_bindings = Mock()
        pool.final_output = False
    assert await orch._abort_request_ids(["r"]) == []
    for pool in orch.stage_pools:
        pool.abort_requests.assert_awaited_once_with(["r"])
        pool.release_bindings.assert_called_once_with(["r"])


@pytest.mark.asyncio
async def test_ack_failure_uses_request_error_cleanup():
    orch, state = harness()
    state.diffusion_segment_pending = True
    orch.stage_pools[0].collective_rpc.return_value = {"supported": False, "error": "worker lost"}
    orch._handle_stage_error = AsyncMock()
    ack = output(0, ack=True)
    await orch._route_output(1, 0, ack, state, None)
    orch._handle_stage_error.assert_awaited_once()
    assert "worker lost" in orch._handle_stage_error.call_args.kwargs["error"]
