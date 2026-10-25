# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.diffusion.models.test_hy_nomas_pipeline import _sampling
from vllm_omni.diffusion.models.hy_nomas.stages import (
    PAYLOAD_KEY,
    HyNomasDecodePipeline,
    HyNomasDenoisePipeline,
    validate_latent_segments,
)
from vllm_omni.model_executor.stage_input_processors.diffusion_disagg import diffusion_stage_handoff

pytest_plugins = ["tests.diffusion.models.test_hy_nomas_pipeline"]

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def payload():
    return [
        dict(
            version=1,
            index=0,
            total=2,
            latent_offset=0,
            pixel_frames=8,
            finished=False,
            latents=torch.zeros(1, 32, 2, 1, 1),
        ),
        dict(
            version=1,
            index=1,
            total=2,
            latent_offset=2,
            pixel_frames=3,
            finished=True,
            latents=torch.ones(1, 32, 1, 1, 1),
        ),
    ]


def batch(prompt):
    return SimpleNamespace(
        requests=[SimpleNamespace(request_id="world", prompt=prompt, sampling_params=SimpleNamespace(output_type="pt"))]
    )


def test_producer_emits_all_segments_and_cleans_session(pipeline):
    pipeline.__class__ = HyNomasDenoisePipeline
    sampling = _sampling()
    sampling.output_type = "pt"  # Producer must not try to construct a decoder.
    request = SimpleNamespace(request_id="world", prompt="landscape", sampling_params=sampling)
    result = pipeline(SimpleNamespace(requests=[request]))
    assert not hasattr(pipeline, "decoder")
    segments = result.custom_output[PAYLOAD_KEY]
    assert [s["index"] for s in segments] == [0, 1]
    assert [s["latent_offset"] for s in segments] == [0, 1]
    assert [s["finished"] for s in segments] == [False, True]
    assert [s["pixel_frames"] for s in segments] == [4, 4]
    assert all(s["latents"].device.type == "cpu" for s in segments)
    assert not pipeline._sessions
    assert sum(call[2] for call in pipeline.transformer.calls) == 2
    handed = diffusion_stage_handoff([result], {"prompt": "landscape"})[0]
    assert handed[PAYLOAD_KEY] is segments


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", 0),
        ("latent_offset", 0),
        ("total", 3),
        ("version", 2),
        ("pixel_frames", 0),
        ("finished", False),
        ("latents", torch.zeros(1, 16, 1, 1, 1)),
    ],
)
def test_invalid_segment_rejected(field, value):
    segments = payload()
    segments[1][field] = value
    with pytest.raises(ValueError):
        validate_latent_segments(segments)


def test_incomplete_request_rejected():
    with pytest.raises(ValueError):
        validate_latent_segments(payload()[:1])


class FakeDecoder(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.created = 0
        self.finished = 0

    def create_streaming_decode_session(self, **kwargs):
        assert kwargs == {"to_cpu": True, "retain_history": False}
        self.created += 1
        owner = self

        class Session:
            offset = 0

            def push(self, latent, *, pixel_frames):
                # Stateful output exposes accidental reinitialization per segment.
                result = torch.full((1, 3, pixel_frames, 1, 1), self.offset)
                self.offset += latent.shape[2]
                return result

            def finish(self):
                owner.finished += 1
                return torch.empty(0)

        return Session()


def test_decode_uses_one_local_session_and_finishes_once():
    model = HyNomasDecodePipeline.__new__(HyNomasDecodePipeline)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.decoder = FakeDecoder()
    first = model(batch({PAYLOAD_KEY: payload()})).output
    second = model(batch({PAYLOAD_KEY: payload()})).output
    assert torch.equal(first, second)
    assert first.shape == (1, 3, 11, 1, 1)
    assert first[:, :, :8].eq(0).all() and first[:, :, 8:].eq(2).all()
    assert model.decoder.created == model.decoder.finished == 2
    malformed = payload()
    malformed[1]["latent_offset"] = 7
    with pytest.raises(ValueError):
        model(batch({PAYLOAD_KEY: malformed}))
    assert model.decoder.created == 2  # Validate the whole handoff before GPU work.


def test_producer_failure_releases_kv(pipeline, monkeypatch):
    pipeline.__class__ = HyNomasDenoisePipeline

    def fail(*args, **kwargs):
        raise RuntimeError("injected denoise failure")

    monkeypatch.setattr(pipeline, "denoise_step", fail)
    request = SimpleNamespace(request_id="world", prompt="landscape", sampling_params=_sampling())
    with pytest.raises(RuntimeError, match="injected"):
        pipeline(SimpleNamespace(requests=[request]))
    assert not pipeline._sessions


def test_real_decoder_preserves_overlap_and_partial_segment(monkeypatch):
    from tests.diffusion.models.test_hy_nomas_decoder import _SDPA, tiny_decoder
    from vllm_omni.diffusion.models.hy_nomas import decoder as decoder_module

    monkeypatch.setattr(decoder_module, "Attention", _SDPA)
    model = HyNomasDecodePipeline.__new__(HyNomasDecodePipeline)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.decoder = tiny_decoder()
    model._decode_sessions = {}
    segments = payload()
    segments[0].update(latents=torch.randn(1, 32, 4, 1, 1), pixel_frames=15)
    segments[1].update(latent_offset=4, latents=torch.randn(1, 32, 1, 1, 1), pixel_frames=4)
    with torch.inference_mode():
        whole = model.decoder.decode(torch.cat([s["latents"] for s in segments], dim=2), return_dict=False)[0]
        expected = torch.cat([whole[:, :, :15], whole[:, :, 16:20]], dim=2)
        actual = model(batch({PAYLOAD_KEY: segments})).output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    streamed = torch.cat([model(stream_batch(segment)).output for segment in segments], dim=2)
    torch.testing.assert_close(streamed, expected, rtol=0, atol=0)
    assert not model._decode_sessions


def streaming_model():
    model = HyNomasDecodePipeline.__new__(HyNomasDecodePipeline)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.decoder = FakeDecoder()
    model._decode_sessions = {}
    return model


def stream_batch(segment):
    from vllm_omni.diffusion.segment import SEGMENT_KEY

    return batch({PAYLOAD_KEY: [segment], SEGMENT_KEY: {"index": segment["index"], "finished": segment["finished"]}})


def test_incremental_decode_matches_full_request_and_releases_history():
    model = streaming_model()
    segments = payload()
    first = model(stream_batch(segments[0]))
    assert first.finished  # One terminal engine invocation, parent remains active.
    assert len(model._decode_sessions) == 1
    second = model(stream_batch(segments[1]))
    assert not model._decode_sessions
    assert model.decoder.created == model.decoder.finished == 1
    actual = torch.cat([first.output, second.output], dim=2)
    expected = model(batch({PAYLOAD_KEY: payload()})).output
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("operation", ["duplicate", "skip", "cancel", "failure"])
def test_stream_session_fails_closed(operation, monkeypatch):
    model = streaming_model()
    segments = payload()
    if operation == "skip":
        with pytest.raises(ValueError):
            model(stream_batch(segments[1]))
    else:
        model(stream_batch(segments[0]))
        if operation == "cancel":
            model.release_stream_request("world")
            model.release_stream_request("world")
        elif operation == "duplicate":
            with pytest.raises(ValueError):
                model(stream_batch(segments[0]))
        else:

            def fail(*args, **kwargs):
                raise RuntimeError("decode failure")

            monkeypatch.setattr(model._decode_sessions["world"].session, "push", fail)
            with pytest.raises(RuntimeError, match="decode failure"):
                model(stream_batch(segments[1]))
    assert not model._decode_sessions
    # A new rollout can start after cancellation or failure.
    model(stream_batch(segments[0]))
    model(stream_batch(segments[1]))
    assert not model._decode_sessions


def test_buffered_decoder_segment_still_acknowledges_without_media(monkeypatch):
    from vllm_omni.diffusion.segment import SEGMENT_KEY

    model = streaming_model()
    create = model.decoder.create_streaming_decode_session

    def buffering_session(**kwargs):
        session = create(**kwargs)
        session.push = lambda *args, **kw: torch.empty(0)
        return session

    monkeypatch.setattr(model.decoder, "create_streaming_decode_session", buffering_session)
    result = model(stream_batch(payload()[0]))
    assert result.output is None
    assert result.custom_output[SEGMENT_KEY] == {"index": 0, "finished": False, "ack": True}
    assert "world" in model._decode_sessions
