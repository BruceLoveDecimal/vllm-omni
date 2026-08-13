# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import janus
import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _OrchestratorDuplexStagePort,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.errors import StageInputProcessingError
from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexStageRequestContext,
    DuplexStageSubmission,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeStageClient:
    def __init__(
        self,
        *,
        next_inputs: list[dict[str, Any]] | None = None,
        final_output: bool = False,
        process_errors: list[Exception] | None = None,
    ) -> None:
        self.stage_id = 0
        self.replica_id = 0
        self.stage_type = "llm"
        self.final_output = final_output
        self.final_output_type = "text"
        self.default_sampling_params = SamplingParams(max_tokens=1)
        self.requires_multimodal_data = False
        self.engine_input_source = [0]
        self.is_comprehension = False
        self.model_stage = None
        self.custom_process_input_func = None
        self.next_inputs = list(next_inputs or [])
        self.process_errors = list(process_errors or [])
        self.add_request_calls: list[tuple[Any, ...]] = []
        self.decoded_source_tokens: str | None = None
        self._engine_core_outputs = queue.Queue()

    async def add_request_async(self, *args, **_kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        try:
            return self._engine_core_outputs.get_nowait()
        except queue.Empty:
            return SimpleNamespace(outputs=[])

    def process_engine_inputs(self, _source_outputs, prompt=None, streaming_context=None):
        if self.process_errors:
            raise self.process_errors.pop(0)
        decoder = getattr(streaming_context, "source_token_decoder", None)
        if callable(decoder):
            self.decoded_source_tokens = decoder([11, 12], skip_special_tokens=True)
        return list(self.next_inputs)

    async def abort_requests_async(self, _request_ids: list[str]) -> None:
        return None

    def set_engine_outputs(self, _outputs) -> None:
        return None

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class FakeOutputProcessor:
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer

    def add_request(self, *args, **kwargs) -> None:
        return None


class FakeInputProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process_inputs(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            prompt_token_ids=[101, 102],
            prompt_embeds=None,
            external_req_id=None,
        )


class FakePrewarmPool:
    stage_type = "llm"

    def __init__(self, role: str) -> None:
        self.stage_vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=64,
                stage_connector_config={"extra": {"role": role}},
            )
        )
        self.submitted: list[Any] = []

    async def submit_initial(self, _request_id, _req_state, request, prompt_text=None):
        self.submitted.append(request)
        return 0

    def get_bound_replica_id(self, _request_id):
        return 0


def _duplex_stage_port_submission():
    stage_pools = []
    for stage_id in range(3):
        pool = SimpleNamespace(
            stage_client=SimpleNamespace(default_sampling_params=SamplingParams(max_tokens=1)),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            submit_initial=AsyncMock(return_value=stage_id + 10),
            submit_update=AsyncMock(return_value=stage_id + 20),
        )
        stage_pools.append(pool)
    request_states: dict[str, OrchestratorRequestState] = {}
    prewarm = AsyncMock()
    port = _OrchestratorDuplexStagePort(
        stage_pools=stage_pools,
        request_states=request_states,
        running_counter=None,
        cleanup_request_ids=AsyncMock(),
        async_chunk=True,
        prewarm_async_chunk_stages=prewarm,
    )
    context = DuplexStageRequestContext(
        request_id="req-duplex",
        session_id="session-duplex",
        fence=DuplexFence("session-duplex"),
        stage_id=0,
        final_stage_id=2,
        config_generation=0,
        sampling_params=tuple(SamplingParams(max_tokens=1) for _ in range(3)),
    )
    port.ensure_request(context)
    submission = DuplexStageSubmission(
        context=context,
        prompt={"prompt_token_ids": [1, 2]},
        already_submitted=False,
    )
    return port, stage_pools, request_states, prewarm, submission


def _request_output(request_id: str) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text="transcript",
        token_ids=[11, 12],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
        metrics=None,
        lora_request=None,
    )


@pytest.mark.asyncio
async def test_forward_text_prompt_uses_target_stage_input_processor() -> None:
    class SourceTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ":".join(str(token_id) for token_id in token_ids)

    stage0 = FakeStageClient(final_output=True)
    stage1 = FakeStageClient(
        final_output=True,
        next_inputs=[{"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}],
    )
    stage_pools = [
        StagePool(
            0,
            [stage0],
            output_processor=FakeOutputProcessor(tokenizer=SourceTokenizer()),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
        StagePool(
            1,
            [stage1],
            output_processor=FakeOutputProcessor(),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
    ]
    request_q = janus.Queue()
    output_q = janus.Queue()
    rpc_q = janus.Queue()
    orchestrator = Orchestrator(
        request_async_queue=request_q.async_q,
        output_async_queue=output_q.async_q,
        rpc_async_queue=rpc_q.async_q,
        stage_pools=stage_pools,
        async_chunk=False,
    )
    input_processor = FakeInputProcessor()
    orchestrator._stage_input_processors[1] = input_processor
    req_state = OrchestratorRequestState(
        request_id="req-text",
        prompt={"prompt": "original"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=1,
    )

    await orchestrator._forward_to_next_stage("req-text", 0, _request_output("req-text"), req_state)

    assert input_processor.calls
    assert input_processor.calls[0]["prompt"] == {"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}
    assert stage1.decoded_source_tokens == "11:12"
    assert req_state.streaming.source_token_decoder is None
    assert stage1.add_request_calls
    submitted_request = stage1.add_request_calls[0][0]
    assert submitted_request.prompt_token_ids == [101, 102]
    assert submitted_request.external_req_id == "req-text"


def _stage_bridge_orchestrator(
    stage1: FakeStageClient,
) -> tuple[Orchestrator, FakeInputProcessor, asyncio.Queue[Any]]:
    stage_pools = [
        StagePool(
            0,
            [FakeStageClient(final_output=True)],
            output_processor=FakeOutputProcessor(),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
        StagePool(
            1,
            [stage1],
            output_processor=FakeOutputProcessor(),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
    ]
    output_q: asyncio.Queue[Any] = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=output_q,
        rpc_async_queue=asyncio.Queue(),
        stage_pools=stage_pools,
        async_chunk=False,
    )
    input_processor = FakeInputProcessor()
    orchestrator._stage_input_processors[1] = input_processor
    return orchestrator, input_processor, output_q


def _bridge_request_state(request_id: str) -> OrchestratorRequestState:
    return OrchestratorRequestState(
        request_id=request_id,
        prompt={"prompt": "original"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=1,
    )


@pytest.mark.asyncio
async def test_stage_input_error_is_request_scoped_without_duplex_control_plane() -> None:
    stage1 = FakeStageClient(
        final_output=True,
        next_inputs=[{"prompt": "hello"}],
        process_errors=[
            StageInputProcessingError("No latent or hidden_states found in thinker output"),
        ],
    )
    orchestrator, input_processor, output_q = _stage_bridge_orchestrator(stage1)
    assert orchestrator.duplex_control_plane is None

    failed_state = _bridge_request_state("req-failed")
    orchestrator.request_states[failed_state.request_id] = failed_state
    await orchestrator._forward_to_next_stage(
        failed_state.request_id,
        0,
        _request_output(failed_state.request_id),
        failed_state,
    )

    error = output_q.get_nowait()
    assert isinstance(error, ErrorMessage)
    assert error.request_id == failed_state.request_id
    assert error.stage_id == 1
    assert error.fatal is False
    assert "stage-0->stage-1" in error.error
    assert "No latent or hidden_states" in error.error
    assert failed_state.request_id not in orchestrator.request_states

    # The typed failure is isolated to one request. The same orchestrator and
    # stage can still bridge and submit a later request.
    next_state = _bridge_request_state("req-next")
    orchestrator.request_states[next_state.request_id] = next_state
    await orchestrator._forward_to_next_stage(
        next_state.request_id,
        0,
        _request_output(next_state.request_id),
        next_state,
    )

    assert input_processor.calls
    assert stage1.add_request_calls
    assert stage1.add_request_calls[-1][0].external_req_id == next_state.request_id
    assert next_state.request_id in orchestrator.request_states
    assert output_q.empty()


@pytest.mark.asyncio
async def test_unexpected_stage_input_error_still_propagates() -> None:
    stage1 = FakeStageClient(
        final_output=True,
        process_errors=[RuntimeError("unexpected processor bug")],
    )
    orchestrator, _, output_q = _stage_bridge_orchestrator(stage1)
    req_state = _bridge_request_state("req-bug")

    with pytest.raises(RuntimeError, match="unexpected processor bug"):
        await orchestrator._forward_to_next_stage(
            req_state.request_id,
            0,
            _request_output(req_state.request_id),
            req_state,
        )

    assert output_q.empty()


@pytest.mark.asyncio
async def test_async_prewarm_skips_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("sender")
    stage2 = FakePrewarmPool("receiver")
    orchestrator.stage_pools = [stage0, stage1, stage2]
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    orchestrator._record_duplex_stage_submission = MagicMock()
    req_state = OrchestratorRequestState(
        request_id="req-prewarm",
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
        duplex_identity=SimpleNamespace(),
    )

    await orchestrator._prewarm_async_chunk_stages(
        "req-prewarm",
        SimpleNamespace(prompt_token_ids=[1, 2], resumable=True),
        req_state,
    )

    assert stage1.submitted == []
    assert len(stage2.submitted) == 1
    assert 1 not in req_state.stage_submit_ts
    assert 2 in req_state.stage_submit_ts
    orchestrator._record_duplex_stage_submission.assert_called_once_with(
        2,
        "req-prewarm",
        0,
        req_state,
    )


@pytest.mark.asyncio
async def test_duplex_prewarm_runs_after_first_stage0_submission() -> None:
    port, stage_pools, request_states, prewarm, submission = _duplex_stage_port_submission()

    result = await port.submit(submission)

    assert result.stage_id == 0
    stage_pools[0].submit_initial.assert_awaited_once()
    prewarm.assert_awaited_once_with("req-duplex", ANY, request_states["req-duplex"])


@pytest.mark.asyncio
async def test_async_route_forwards_to_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
    )
    stage0 = SimpleNamespace(final_output=False)
    stage1 = FakePrewarmPool("sender")
    orchestrator.stage_pools = [stage0, stage1]
    orchestrator._forward_to_next_stage = AsyncMock()
    req_state = OrchestratorRequestState(
        request_id="req-route",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(2)],
        final_stage_id=1,
    )
    req_state.stage_submit_ts[0] = 1.0
    output = SimpleNamespace(request_id="req-route", finished=True)

    await orchestrator._route_output(0, 0, output, req_state, None)

    orchestrator._forward_to_next_stage.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_segment_does_not_complete_final_output_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
        cleanup_parent=lambda _request_id: [],
    )
    orchestrator.stage_pools = [SimpleNamespace(final_output=True)]
    orchestrator.output_async_queue = asyncio.Queue()
    orchestrator._cleanup_request_ids = AsyncMock()

    req_state = OrchestratorRequestState(
        request_id="req-segment-final-output",
        sampling_params_list=[SamplingParams(max_tokens=1)],
        final_stage_id=0,
        final_output_stage_ids={0},
    )
    req_state.streaming.enabled = True
    req_state.streaming.segment_finished = True
    output = SimpleNamespace(
        request_id=req_state.request_id,
        finished=True,
    )

    await orchestrator._route_output(0, 0, output, req_state, None)

    assert req_state.finished_final_output_stage_ids == set()
    orchestrator._cleanup_request_ids.assert_not_awaited()
    routed = orchestrator.output_async_queue.get_nowait()
    assert routed.finished is False
