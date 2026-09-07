# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-execution contract of ``SanaWmStreamingPipeline`` with the heavy
components (transformer, text encoder, VAE) replaced by stand-ins.

Drives ``prepare_encode -> denoise_step -> step_scheduler -> post_decode``
exactly the way ``DiffusionModelRunner`` does in streaming mode and pins the
chunk schedule, the per-chunk frame indices, the cache save pass, the
overlap-decode frame counts and the request-validation errors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm_streaming import (
    SANA_WM_STREAMING_DEFAULT_NUM_FRAMES,
    SanaWmStreamingPipeline,
)
from vllm_omni.diffusion.models.sana_wm.self_forcing import SanaWmSelfForcingSchedule
from vllm_omni.diffusion.models.sana_wm.streaming_cache import SanaWmStreamingCache
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

LATENT_CHANNELS = 4
HEIGHT = 64
WIDTH = 96
LATENT_H, LATENT_W = HEIGHT // 32, WIDTH // 32
NUM_BLOCKS = 2


class _FakeTransformer:
    """Records every streaming forward and returns a deterministic velocity."""

    def __init__(self) -> None:
        self.blocks = [object() for _ in range(NUM_BLOCKS)]
        self.calls: list[dict] = []

    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        frame_index: torch.Tensor,
        cache: SanaWmStreamingCache,
        save_cache: bool,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
        spatial_raymap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del encoder_attention_mask, spatial_raymap
        assert encoder_hidden_states is not None
        assert raymap is not None and plucker is not None
        self.calls.append(
            {
                "frames": int(hidden_states.shape[2]),
                "timestep": timestep.detach().clone(),
                "frame_index": frame_index.detach().cpu().tolist(),
                "save_cache": save_cache,
                "chunk_index": cache.chunks_committed,
                "raymap_frames": int(raymap.shape[0]),
                "plucker_frames": int(plucker.shape[1]),
            }
        )
        # v = 0.5 * x keeps the Euler trajectory easy to reproduce by hand.
        return hidden_states * 0.5


def _image():
    from PIL import Image

    return Image.new("RGB", (WIDTH, HEIGHT), (10, 20, 30))


def _prompt(num_frames: int, action: str | None = None):
    return {
        "prompt": "a test prompt",
        "multi_modal_data": {"image": _image()},
        "sana_wm": {
            "action": action or f"w-{num_frames - 1}",
            "num_frames": num_frames,
            "height": HEIGHT,
            "width": WIDTH,
            "intrinsics": {"fx": 48.0, "fy": 48.0, "cx": 48.0, "cy": 32.0},
        },
    }


def _sampling(**overrides) -> OmniDiffusionSamplingParams:
    params = dict(height=HEIGHT, width=WIDTH, num_frames=49, seed=7, output_type="np")
    params.update(overrides)
    return OmniDiffusionSamplingParams(**params)


def _pipeline(monkeypatch, *, config: SanaWmConfig | None = None) -> SanaWmStreamingPipeline:
    pipeline = object.__new__(SanaWmStreamingPipeline)
    pipeline.od_config = None
    pipeline.sana_wm_config = config or SanaWmConfig(streaming=True, chunk_size=3)
    pipeline.self_forcing_schedule = SanaWmSelfForcingSchedule.from_config(pipeline.sana_wm_config)
    pipeline.transformer = _FakeTransformer()
    pipeline.device = torch.device("cpu")
    pipeline._last_prompt_attention_mask = None
    decoded: list[int] = []
    pipeline._decoded_latent_frames = decoded

    monkeypatch.setattr(pipeline, "_runtime_device_dtype", lambda: (torch.device("cpu"), torch.float32))
    monkeypatch.setattr(
        pipeline,
        "_native_prompt_embeds",
        lambda prompt, *, device, dtype: torch.ones(1, 6, 8, dtype=dtype),
    )

    def fake_encode_first_frame(image, *, height, width, latent_height, latent_width, device, dtype):
        return torch.full((1, LATENT_CHANNELS, 1, latent_height, latent_width), 2.0, dtype=dtype)

    monkeypatch.setattr(pipeline, "_vae_encode_first_frame", fake_encode_first_frame)

    def fake_decode(latents, *, output_type, device, dtype):
        if output_type == "latent":
            return latents
        frames = 1 + 8 * (latents.shape[2] - 1)
        decoded.append(latents.shape[2])
        video = np.zeros((1, frames, HEIGHT, WIDTH, 3), dtype=np.float32)
        # Stamp the source latent frame id into each pixel frame for tracing.
        video[0, :, 0, 0, 0] = np.arange(frames)
        return video

    monkeypatch.setattr(pipeline, "_decode_native_latents", fake_decode)
    return pipeline


def _state(prompt, sampling: OmniDiffusionSamplingParams) -> StepRequestState:
    return StepRequestState(request_id="req-0", sampling=sampling, prompt=prompt)


def _input_batch(state: StepRequestState) -> InputBatch:
    """Single-request step batch, shaped the way the runner assembles it."""
    return InputBatch(
        request_ids=[state.request_id],
        num_reqs=1,
        num_reqs_after_padding=1,
        idx_mapping=torch.tensor([0]),
        idx_mapping_np=np.array([0]),
        latents=state.latents,
        timesteps=state.timesteps,
        prompt_embeds=state.prompt_embeds,
        prompt_embeds_mask=state.prompt_embeds_mask,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        states=(state,),
    )


def _drive_chunk(pipeline: SanaWmStreamingPipeline, state: StepRequestState):
    """Run one chunk exactly like the runner's streaming loop."""
    while not state.chunk_denoise_completed:
        noise_pred = pipeline.denoise_step(_input_batch(state), states=[state])
        pipeline.step_scheduler(state, noise_pred)
    return pipeline.post_decode(state)


def test_stepwise_rollout_chunks_frames_and_outputs(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    num_frames = 49  # 7 latent frames: chunk 0 = [0, 4), chunk 1 = [4, 7)
    state = pipeline.prepare_encode(_state(_prompt(num_frames), _sampling(num_frames=num_frames)))

    assert state.total_chunks == 2
    assert state.chunk_num_steps == 4
    assert state.timesteps.tolist() == [1000.0, 960.0, 889.0, 727.0]
    assert state.latents.shape == (1, LATENT_CHANNELS, 3, LATENT_H, LATENT_W)
    assert state.do_true_cfg is False
    assert state.extra["boundaries"] == [0, 4, 7]
    assert state.extra["frame_index"].tolist() == [0, 1, 2, 3]
    assert state.extra["gen_range"] == (1, 4)
    assert state.extra["camera_full"]["raymap"].shape == (7, 20)
    noise = state.extra["noise_full"]
    torch.testing.assert_close(state.latents, noise[:, :, 1:4])

    outputs = []
    while not state.request_denoise_completed:
        outputs.append(_drive_chunk(pipeline, state))

    calls = pipeline.transformer.calls
    # 4 denoise forwards + 1 save pass per chunk.
    assert len(calls) == 2 * 5
    chunk0, chunk1 = calls[:5], calls[5:]
    assert [c["frames"] for c in chunk0] == [4] * 5
    assert [c["frames"] for c in chunk1] == [3] * 5
    assert [c["save_cache"] for c in chunk0] == [False] * 4 + [True]
    assert [c["save_cache"] for c in chunk1] == [False] * 4 + [True]
    assert all(c["frame_index"] == [0, 1, 2, 3] for c in chunk0)
    assert all(c["frame_index"] == [4, 5, 6] for c in chunk1)
    assert all(c["raymap_frames"] == c["frames"] == c["plucker_frames"] for c in calls)
    # Conditioning frame held at t = 0 inside chunk 0; other frames carry the
    # schedule; the save pass runs at t = 0 everywhere.
    assert chunk0[0]["timestep"].tolist() == [[[0.0, 1000.0, 1000.0, 1000.0]]]
    assert chunk0[3]["timestep"].tolist() == [[[0.0, 727.0, 727.0, 727.0]]]
    assert chunk0[4]["timestep"].tolist() == [[[0.0, 0.0, 0.0, 0.0]]]
    assert chunk1[0]["timestep"].tolist() == [[[1000.0, 1000.0, 1000.0]]]
    assert chunk1[4]["timestep"].tolist() == [[[0.0, 0.0, 0.0]]]
    assert [c["chunk_index"] for c in chunk0] == [0] * 5
    assert [c["chunk_index"] for c in chunk1] == [1] * 5

    # Overlap decode: chunk 0 decodes 4 latent frames -> 25 pixel frames, chunk
    # 1 decodes [3, 7) -> 25 frames and drops the overlap frame -> 24.
    assert pipeline._decoded_latent_frames == [4, 4]
    videos = [out.output["payload"]["video"] for out in outputs]
    assert videos[0].shape == (1, 25, HEIGHT, WIDTH, 3)
    assert videos[1].shape == (1, 24, HEIGHT, WIDTH, 3)
    assert videos[1][0, 0, 0, 0, 0] == 1  # first emitted frame of chunk 1 is decoded frame 1
    assert sum(v.shape[1] for v in videos) == num_frames
    assert [out.chunk_index for out in outputs] == [0, 1]
    assert [out.total_chunks for out in outputs] == [2, 2]
    assert [out.finished for out in outputs] == [False, True]
    for out in outputs:
        meta = out.output["metadata"]["sana_wm"]
        assert meta["backend"] == "native_gdn_streaming"
        assert meta["total_chunks"] == 2
    assert outputs[0].output["metadata"]["sana_wm"]["frame_index"] == [0, 1, 2, 3]
    assert outputs[1].output["metadata"]["sana_wm"]["frame_index"] == [4, 5, 6]
    assert outputs[1].output["metadata"]["sana_wm"]["cached_latent_frames"] == 7

    # Request-scoped state is released on the last chunk.
    for key in ("cache", "camera_full", "history_latents", "noise_full", "first_latent"):
        assert key not in state.extra


def test_euler_trajectory_matches_hand_computation(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    state = pipeline.prepare_encode(_state(_prompt(25), _sampling(num_frames=25)))
    assert state.total_chunks == 1
    x = state.latents.clone()
    schedule = pipeline.self_forcing_schedule
    for step in range(schedule.num_steps):
        sigma, sigma_next = schedule.sigma_pair(step)
        noise_pred = pipeline.denoise_step(_input_batch(state), states=[state])
        torch.testing.assert_close(noise_pred, x * 0.5)
        pipeline.step_scheduler(state, noise_pred)
        x = x - (sigma - sigma_next) * (x * 0.5)
        torch.testing.assert_close(state.latents, x)
        assert state.step_in_chunk == step + 1 and state.step_index == step + 1
    output = pipeline.post_decode(state)
    assert output.finished is True and output.chunk_index == 0


def test_latent_output_type_returns_generated_frames(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    state = pipeline.prepare_encode(_state(_prompt(49), _sampling(num_frames=49, output_type="latent")))
    first = _drive_chunk(pipeline, state)
    latents = first.output["payload"]["latents"]
    assert latents.shape == (1, LATENT_CHANNELS, 3, LATENT_H, LATENT_W)
    assert pipeline._decoded_latent_frames == []


@pytest.mark.parametrize("num_frames", [17, 33, 41])
def test_rejects_num_frames_off_the_chunk_grid(monkeypatch, num_frames) -> None:
    pipeline = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="24k\\+1"):
        pipeline.prepare_encode(_state(_prompt(num_frames), _sampling(num_frames=num_frames)))


def test_rejects_guidance_and_step_count_mismatch(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="guidance_scale must be <= 1.0"):
        pipeline.prepare_encode(
            _state(_prompt(49), _sampling(num_frames=49, guidance_scale=5.0, guidance_scale_provided=True))
        )
    with pytest.raises(ValueError, match="num_inference_steps must be 4"):
        pipeline.prepare_encode(_state(_prompt(49), _sampling(num_frames=49, num_inference_steps=50)))
    # An explicit matching count and guidance_scale=1.0 are accepted.
    state = pipeline.prepare_encode(
        _state(
            _prompt(49),
            _sampling(num_frames=49, num_inference_steps=4, guidance_scale=1.0, guidance_scale_provided=True),
        )
    )
    assert state.total_chunks == 2


def test_request_mode_forward_is_refused(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    request = OmniDiffusionRequest(prompt=_prompt(49), sampling_params=_sampling(), request_id="req-0")
    with pytest.raises(NotImplementedError, match="diffusion-streaming-output"):
        pipeline.forward(DiffusionRequestBatch(requests=[request]))


def test_default_num_frames_is_on_the_grid() -> None:
    assert (SANA_WM_STREAMING_DEFAULT_NUM_FRAMES - 1) % (8 * 3) == 0


def test_seeded_noise_is_reproducible(monkeypatch) -> None:
    first = _pipeline(monkeypatch).prepare_encode(_state(_prompt(49), _sampling(num_frames=49, seed=11)))
    second = _pipeline(monkeypatch).prepare_encode(_state(_prompt(49), _sampling(num_frames=49, seed=11)))
    torch.testing.assert_close(first.extra["noise_full"], second.extra["noise_full"])
    other = _pipeline(monkeypatch).prepare_encode(_state(_prompt(49), _sampling(num_frames=49, seed=12)))
    assert not torch.allclose(first.extra["noise_full"], other.extra["noise_full"])
