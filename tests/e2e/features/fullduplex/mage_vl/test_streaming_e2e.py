# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mage-VL proactive streaming, end to end: real video, real gate, real engine.

``test_adapter.py`` pins the streaming contract with the model stubbed. This runs the
production wiring -- the vision tower and cognition gate scoring each segment, `AsyncLLM`
generating the answer -- so that what is asserted is the behaviour a deployment would show:
silent stretches produce nothing, an interesting segment makes the model speak on its own,
and an interruption stops the answer without leaking output.

Needs the checkpoint (``MAGE_VL_PATH``) and a GPU; skipped otherwise. The engine is built
once for the module because its startup dominates the runtime of everything here.
"""

import asyncio
import os

import pytest

torch = pytest.importorskip("torch")

from tests.helpers.mark import hardware_test  # noqa: E402
from vllm_omni.experimental.fullduplex.core import protocol as ev  # noqa: E402
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime  # noqa: E402
from vllm_omni.experimental.fullduplex.core.session import (  # noqa: E402
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.mage_vl import (  # noqa: E402
    MageVLDuplexAdapter,
    MageVLSegmentScorer,
    MageVLStreamConfig,
    build_text_stream,
    sample_segments,
)

MODEL = os.environ.get("MAGE_VL_PATH", "")
VIDEO = os.environ.get(
    "MAGE_VIDEO", os.path.join(MODEL, "examples", "soccer-broadcast.mp4") if MODEL else ""
)

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.skipif(not MODEL or not os.path.isdir(MODEL), reason="set MAGE_VL_PATH"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


def _build_vision_stack():
    """Vision tower + gate, loaded straight from the checkpoint files."""
    import json

    from safetensors import safe_open
    from safetensors.torch import load_file
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_omni.model_executor.models.mage_vl.gate import MageVLCognitionGate
    from vllm_omni.model_executor.models.mage_vl.vision import MageVLVisionTransformer
    from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig
    from vllm_omni.transformers_utils.processors.mage_vl import MageVLProcessor

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29613")
    context = set_current_vllm_config(VllmConfig())
    context.__enter__()
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    config = MageVLConfig.from_pretrained(MODEL)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        tower = MageVLVisionTransformer(config.vision_config, prefix="visual")
    tower.eval()
    for layer in tower.encoder.layers:
        layer.self_attn.attn.attn_backend = AttentionBackendEnum.TORCH_SDPA
        layer.self_attn.attn.is_flash_attn_backend = False

    with open(f"{MODEL}/model.safetensors.index.json") as handle:
        weight_map = json.load(handle)["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key in [k for k in weight_map if k.startswith("model.visual.")]:
        by_file.setdefault(weight_map[key], []).append(key)

    def weights():
        for filename, keys in by_file.items():
            with safe_open(f"{MODEL}/{filename}", framework="pt", device="cpu") as handle:
                for key in keys:
                    yield key[len("model.visual.") :], handle.get_tensor(key).cuda().bfloat16()

    tower.load_weights(weights())
    torch.set_default_dtype(torch.float32)

    gate = MageVLCognitionGate()
    gate.load_weights(load_file(f"{MODEL}/streammind_gate.safetensors").items())
    gate = gate.cuda().float().eval()

    return MageVLProcessor.from_pretrained(MODEL), tower, gate


@pytest.fixture(scope="module")
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def stack():
    return _build_vision_stack()


@pytest.fixture(scope="module")
def engine(loop):
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    async def build():
        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=MODEL,
                dtype="bfloat16",
                max_model_len=8192,
                limit_mm_per_prompt={"image": 8},
                gpu_memory_utilization=0.45,
                enforce_eager=True,
            )
        )

    async_llm = loop.run_until_complete(build())
    yield async_llm
    async_llm.shutdown()
    # Let the engine's background output handler finish on the loop it was created with;
    # otherwise it wakes up after the loop is gone and logs an error at interpreter exit.
    loop.run_until_complete(asyncio.sleep(0))


@pytest.fixture(scope="module")
def segments():
    found = sample_segments(VIDEO, segment_seconds=8.0, frames_per_segment=2)
    assert found, f"no segments sampled from {VIDEO}"
    return found[:4]


def _run(loop, adapter, events):
    session = DuplexSession(
        session_id="mage-e2e",
        config=DuplexSessionConfig(input_modalities=("video",), proactive=True),
    )
    runtime = DuplexRuntime(session, adapter)

    async def stream():
        for event in events:
            await asyncio.sleep(0.02)
            yield event

    emitted: list[dict] = []

    async def emit(event):
        emitted.append(event)

    loop.run_until_complete(runtime.run(stream(), emit))
    return emitted, session


@hardware_test(res={"cuda": "L4"})
def test_gate_decisions_are_stable_and_drive_the_stream(loop, stack, engine, segments):
    """The gate's own numbers decide when to speak, and repeat runs agree.

    Judged on the trigger set rather than on generated text: the decision is what this
    path owns, and a single boolean assertion on a streamed response is exactly the kind
    of check that goes flaky.
    """
    processor, tower, gate = stack
    scores_per_run = []

    for _ in range(3):
        scorer = MageVLSegmentScorer(processor, tower, gate, device="cuda")
        scores_per_run.append([scorer(segment) for segment in segments])

    assert all(run == scores_per_run[0] for run in scores_per_run), (
        f"gate scores drifted between runs: {scores_per_run}"
    )
    assert all(0.0 <= score <= 1.0 for score in scores_per_run[0])

    # A threshold below every observed score forces a trigger; one above forces silence.
    scores = scores_per_run[0]
    quiet = MageVLStreamConfig(speak_threshold=min(1.0, max(scores) + 1e-6))
    loud = MageVLStreamConfig(speak_threshold=max(0.0, min(scores) - 1e-6), cooldown_segments=99)

    events = [
        {"type": ev.INPUT_APPEND, "modality": "video", "data": segment} for segment in segments
    ] + [{"type": ev.CLOSE}]

    quiet_out, _ = _run(
        loop,
        MageVLDuplexAdapter(
            MageVLSegmentScorer(processor, tower, gate, device="cuda"),
            build_text_stream(engine),
            config=quiet,
        ),
        events,
    )
    assert [e["type"] for e in quiet_out] == [], "below threshold the model must stay silent"

    loud_out, _ = _run(
        loop,
        MageVLDuplexAdapter(
            MageVLSegmentScorer(processor, tower, gate, device="cuda"),
            build_text_stream(engine),
            config=loud,
        ),
        events,
    )
    types = [e["type"] for e in loud_out]
    assert types.count(ev.RESPONSE_CREATED) == 1, "the cooldown holds it to one answer"
    assert types.count(ev.RESPONSE_DONE) == 1
    text = "".join(e["data"] for e in loud_out if e["type"] == ev.RESPONSE_DELTA)
    assert text.strip(), "a triggered response must carry text"


@hardware_test(res={"cuda": "L4"})
def test_barge_in_stops_generation_without_leaking_output(loop, stack, engine, segments):
    """Interrupting mid-answer: no delta after the cancel, and the engine survives it."""
    processor, tower, gate = stack
    always = MageVLStreamConfig(speak_threshold=0.0, cooldown_segments=0)
    adapter = MageVLDuplexAdapter(
        MageVLSegmentScorer(processor, tower, gate, device="cuda"),
        build_text_stream(engine, max_tokens=256),
        config=always,
    )

    emitted, session = _run(
        loop,
        adapter,
        [
            {"type": ev.INPUT_APPEND, "modality": "video", "data": segments[0]},
            {"type": ev.RESPONSE_CANCEL},
            {"type": ev.CLOSE},
        ],
    )

    types = [e["type"] for e in emitted]
    assert ev.RESPONSE_CANCELLED in types
    assert ev.RESPONSE_DONE not in types, "a cancelled answer must not report done"
    cancel_at = types.index(ev.RESPONSE_CANCELLED)
    assert ev.RESPONSE_DELTA not in types[cancel_at:], "no output after the interruption"
    assert session.epoch == 1

    # The engine must still serve after an aborted request.
    after, _ = _run(
        loop,
        MageVLDuplexAdapter(
            MageVLSegmentScorer(processor, tower, gate, device="cuda"),
            build_text_stream(engine),
            config=always,
        ),
        [
            {"type": ev.INPUT_APPEND, "modality": "video", "data": segments[0]},
            {"type": ev.CLOSE},
        ],
    )
    assert "".join(e["data"] for e in after if e["type"] == ev.RESPONSE_DELTA).strip()
