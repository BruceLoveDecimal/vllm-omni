# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Segment sequencing tests with tiny components, not released-model validation."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn

from vllm_omni.diffusion.models.hy_nomas import pipeline as pipeline_module
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=1)
        self.double_blocks = [None]
        self.dtype = torch.float32
        self.calls = []
        self.conditions = []

    def forward(self, hidden_states, timestep, update_cache=False, frame_offset=0, **kwargs):
        self.calls.append((hidden_states.clone(), timestep.clone(), update_cache, frame_offset))
        self.conditions.append(kwargs)
        return (torch.ones_like(hidden_states),)


@pytest.fixture
def pipeline(monkeypatch):
    result = pipeline_module.HyNomasPipeline.__new__(pipeline_module.HyNomasPipeline)
    nn.Module.__init__(result)
    result.device = torch.device("cpu")
    result.height = result.width = 32
    result.segment_frames = 1
    result.max_segments = 4
    result.max_cache_segments = None
    result.max_text_length = 1280
    result._sessions = {}
    result._bound_session = None
    result.transformer = _TinyTransformer()
    result.text_encoder = result.tokenizer = None
    result.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    encoded = pipeline_module.EncodedPrompt(torch.zeros(1, 2, 1), torch.ones(1, 2, dtype=torch.bool), None, None)
    monkeypatch.setattr(pipeline_module, "encode_segment_prompt", lambda *args: encoded)
    return result


def _sampling():
    return SimpleNamespace(
        height=32,
        width=32,
        guidance_scale=1.0,
        num_outputs_per_prompt=1,
        num_inference_steps=2,
        fps=24,
        num_frames=8,
        extra_args={},
        max_sequence_length=None,
        generator=torch.Generator("cpu").manual_seed(42),
        seed=42,
        output_type="latent",
    )


def test_segment_scheduler_commits_once_after_denoising(pipeline):
    state = StepRequestState(request_id="world", prompt="A landscape", sampling=_sampling())
    pipeline.prepare_encode(state)
    initial = state.extra["initial_noise"].clone()
    for index in range(2):
        pipeline.prepare_next_chunk(state)
        assert not state.chunk_denoise_completed
        while not state.chunk_denoise_completed:
            velocity = pipeline.denoise_step(None, states=[state])
            pipeline.step_scheduler(state, velocity)
        output = pipeline.post_decode(state)
        torch.testing.assert_close(output.output, initial[:, :, index : index + 1] - 1, atol=1e-6, rtol=1e-6)
        assert state.chunk_index == index + 1
        assert pipeline._sessions["world"].frame_offset == index + 1
    assert state.request_denoise_completed
    assert [call[2] for call in pipeline.transformer.calls] == [False, False, True] * 2
    assert [call[3] for call in pipeline.transformer.calls] == [0] * 3 + [1] * 3
    expected_generator = torch.Generator("cpu").manual_seed(42)
    torch.randn(initial.shape, generator=expected_generator)
    for index in range(2):
        noise = torch.randn((1, 1, 1, 1, 1), generator=expected_generator)
        expected = (1 - 0.206897) * (initial[:, :, index : index + 1] - 1) + 0.206897 * noise
        actual, time, _, _ = pipeline.transformer.calls[index * 3 + 2]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(time, torch.tensor([206.897]))


def test_forward_failure_releases_session(pipeline, monkeypatch):
    request = SimpleNamespace(request_id="failed", prompt="A landscape", sampling_params=_sampling())
    batch = SimpleNamespace(requests=[request])

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate denoise failure")

    monkeypatch.setattr(pipeline, "_predict", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        pipeline(batch)
    assert "failed" not in pipeline._sessions


def test_bound_session_isolation_and_release(pipeline):
    first = StepRequestState(request_id="first", prompt="A landscape", sampling=_sampling())
    second = StepRequestState(request_id="second", prompt="A landscape", sampling=_sampling())
    pipeline.prepare_encode(first)
    pipeline.prepare_encode(second)
    with pipeline.bind_ar_diffusion_state("first", SimpleNamespace(session_id="first")):
        with pytest.raises(ValueError, match="match"):
            pipeline.prepare_next_chunk(second)
    assert pipeline._bound_session is None
    pipeline.close_ar_diffusion_session("first")
    assert "first" not in pipeline._sessions
    assert "second" in pipeline._sessions


@pytest.mark.parametrize("segment_frames, num_frames, rejected", [(1, 8, True), (1, 4, False), (10, 44, False)])
def test_only_nonfinal_short_video_segments_are_rejected(pipeline, segment_frames, num_frames, rejected):
    pipeline.segment_frames = segment_frames
    pipeline.decoder = SimpleNamespace(
        temporal_window_latents=10,
        create_streaming_decode_session=lambda **kwargs: object(),
    )
    sampling = _sampling()
    sampling.output_type = "np"
    sampling.num_frames = num_frames
    state = StepRequestState(request_id="video", prompt="A landscape", sampling=sampling)
    if rejected:
        with pytest.raises(ValueError, match="PixelUMM window"):
            pipeline.prepare_encode(state)
        assert "video" not in pipeline._sessions
    else:
        pipeline.prepare_encode(state)
        assert state.extra["lengths"][-1] == 1


@pytest.mark.parametrize("explicit_generator, explicit_device", [(False, None), (True, None), (False, "cuda")])
def test_preprocess_cpu_default_preserves_explicit_rng(explicit_generator, explicit_device):
    generator = torch.Generator("cpu").manual_seed(15) if explicit_generator else None
    sampling = SimpleNamespace(generator=generator, generator_device=explicit_device)
    request = SimpleNamespace(sampling_params=sampling)
    processed = pipeline_module.get_hy_nomas_pre_process_func(SimpleNamespace())(request)
    assert processed is request
    assert sampling.generator is generator
    expected_device = explicit_device if explicit_generator or explicit_device is not None else "cpu"
    assert sampling.generator_device == expected_device
    if generator is not None:
        assert generator.initial_seed() == 15


def test_prompt_boundary_replaces_embeddings_mask_and_action_ranges(pipeline, monkeypatch):
    initial = pipeline_module.EncodedPrompt(torch.zeros(1, 2, 1), torch.tensor([[True, False]]), None, None)
    updated = pipeline_module.EncodedPrompt(torch.full((1, 3, 1), 7.0), torch.tensor([[True, True, False]]), [1, 2], 1)
    encoded_prompts = []

    def encode(prompt, *args):
        encoded_prompts.append(prompt)
        return updated if prompt == "new action" else initial

    monkeypatch.setattr(pipeline_module, "encode_segment_prompt", encode)
    monkeypatch.setattr(
        pipeline_module, "_action_windows", lambda prompt, *args: [(0, 1, 0)] if prompt == "new action" else None
    )
    state = StepRequestState(request_id="interactive", prompt="initial", sampling=_sampling())
    pipeline.prepare_encode(state)
    pipeline.prepare_next_chunk(state)
    assert state.prompt_embeds is initial.embeds
    while not state.chunk_denoise_completed:
        pipeline.step_scheduler(state, pipeline.denoise_step(None, states=[state]))
    pipeline.post_decode(state)
    state.interaction_sessions["prompt"] = pipeline_module.HyNomasPromptSession(prompt="new action")
    pipeline.prepare_next_chunk(state)
    pipeline.denoise_step(None, states=[state])
    assert encoded_prompts == ["initial", "new action"]
    assert state.prompt_embeds is updated.embeds
    assert state.prompt_embeds_mask is updated.mask
    assert state.extra["encoded"] is updated
    condition = pipeline.transformer.conditions[-1]
    torch.testing.assert_close(condition["encoder_hidden_states"], updated.embeds)
    torch.testing.assert_close(condition["encoder_attention_mask"], updated.mask)
    assert condition["action_text_offsets"] == [[1, 2]]
    assert condition["action_frame_windows"] == [(0, 1, 0)]
    assert condition["cache_action_start"] == 1


def test_nonstepwise_bound_ar_forward_is_rejected(pipeline):
    request = SimpleNamespace(request_id="request", prompt="A landscape", sampling_params=_sampling())
    with pipeline.bind_ar_diffusion_state("session", SimpleNamespace(session_id="session")):
        with pytest.raises(ValueError, match="step_execution"):
            pipeline(SimpleNamespace(requests=[request]))
    assert pipeline._sessions == {}


def test_live_camera_keeps_native_first_latent_stationary(pipeline):
    pipeline.segment_frames = 3
    sampling = _sampling()
    sampling.num_frames = 12
    state = StepRequestState(request_id="camera", prompt="A landscape", sampling=sampling)
    pipeline.prepare_encode(state)
    poses = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
    poses[:, 2, 3] = torch.tensor([0.0, 1.0, 2.0])
    state.interaction_sessions["camera"] = pipeline_module.CameraSession(
        has_received_input=True, last_absolute_poses=poses
    )
    pipeline.prepare_next_chunk(state)
    pipeline.denoise_step(None, states=[state])
    camera = pipeline.transformer.conditions[-1]["camera_extrinsics"]
    assert camera.shape == (1, 9, 4, 4)
    # The native transformer front-pads these 4k+1 poses. All four poses in
    # the first latent are identity, and later latent frames contain motion.
    packed = torch.cat([camera[:, :1].repeat(1, 3, 1, 1), camera], dim=1).reshape(1, 3, 4, 4, 4)
    torch.testing.assert_close(packed[0, 0], torch.eye(4).repeat(4, 1, 1))
    torch.testing.assert_close(packed[0, 1:, :, 2, 3], torch.ones(2, 4))


def test_generator_list_is_rejected_without_replacing_caller_rng(pipeline):
    sampling = _sampling()
    generators = [torch.Generator("cpu").manual_seed(123)]
    sampling.generator = generators
    state = StepRequestState(request_id="rng", prompt="A landscape", sampling=sampling)
    with pytest.raises(ValueError, match="one torch.Generator"):
        pipeline.prepare_encode(state)
    assert sampling.generator is generators
    assert "rng" not in pipeline._sessions


@pytest.mark.parametrize("limit", [None, 1, 2])
def test_segment_history_eviction_matches_native_commit_order(pipeline, monkeypatch, limit):
    pipeline.max_cache_segments = limit
    sampling = _sampling()
    sampling.num_frames = 16
    state = StepRequestState(request_id="history", prompt="A landscape", sampling=sampling)
    pipeline.prepare_encode(state)
    session = pipeline._sessions["history"]
    session.kv_cache = [{}, {}]
    observations = []

    def predict(state, latents, timestep, *, commit=False):
        counts = [layer.get("mask", torch.empty(1, 0)).shape[1] for layer in session.kv_cache]
        observations.append((session.frame_offset, commit, counts))
        if commit:
            for index, layer in enumerate(session.kv_cache):
                for name in ("key", "value", "mask"):
                    value = session.frame_offset + 10 * index
                    new = torch.full((1, 3), value, dtype=torch.float32)
                    old = layer.get(name, torch.empty(1, 0))
                    layer[name] = torch.cat((old, new), dim=1)
        return torch.zeros_like(latents)

    monkeypatch.setattr(pipeline, "_predict", predict)
    for segment in range(4):
        pipeline.prepare_next_chunk(state)
        pipeline.denoise_step(None, states=[state])
        pipeline.post_decode(state)
        retained = segment + 1 if limit is None else min(segment + 1, limit)
        assert session.frame_offset == segment + 1
        assert session.segment_token_lengths == [3] * retained
        for index, layer in enumerate(session.kv_cache):
            expected = torch.arange(segment + 1 - retained, segment + 1).repeat_interleave(3) + 10 * index
            for name in ("key", "value", "mask"):
                torch.testing.assert_close(layer[name][0], expected.float())
                assert layer[name].untyped_storage().nbytes() == retained * 3 * 4
    for segment in range(4):
        before = segment if limit is None else min(segment, limit)
        commit = segment if limit is None else min(segment, limit - 1)
        assert observations[2 * segment] == (segment, False, [3 * before] * 2)
        assert observations[2 * segment + 1] == (segment, True, [3 * commit] * 2)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_invalid_history_limit_rejected_before_loading(value, monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        quantization_config=None,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        model_config={"hy_nomas_max_cache_segments": value},
    )
    with pytest.raises(ValueError, match="hy_nomas_max_cache_segments"):
        pipeline_module.HyNomasPipeline(od_config=config)


def test_history_budget_only_bounds_kv(pipeline):
    pipeline.transformer.weight = nn.Parameter(torch.zeros(1))
    pipeline.transformer.config = SimpleNamespace(
        num_layers=2,
        num_attention_heads=1,
        attention_head_dim=2,
        in_channels=1,
        text_embed_dim=1,
    )
    pipeline.decoder = SimpleNamespace(temporal_window_latents=10)
    full = pipeline.ar_diffusion_kv_cache_spec().model_owned_state_bytes_per_session
    pipeline.max_cache_segments = 2
    bounded = pipeline.ar_diffusion_kv_cache_spec().model_owned_state_bytes_per_session
    per_segment_kv = 2 * 2 * (1 + 1280) * (2 * 1 * 2 * 4 + 1)
    assert full - bounded == 2 * per_segment_kv
    pipeline.max_cache_segments = 5
    assert pipeline.ar_diffusion_kv_cache_spec().model_owned_state_bytes_per_session == full
