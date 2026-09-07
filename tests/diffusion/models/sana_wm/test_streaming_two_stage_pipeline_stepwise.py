# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-execution contract of ``SanaWmStreamingTwoStagePipeline`` with the
Stage-1 transformer, the refiner, the text encoders and the VAE replaced by
stand-ins.

Pins what the refiner adds on top of the Stage-1 stepwise contract: one
``refine_block`` per chunk on the clean Stage-1 block at the block's absolute
start frame, the sink handed over exactly once, the *refined* history feeding
the overlap decode (frame counts unchanged), refined latents in
``output_type="latent"``, the refiner metadata, and the request-scoped refiner
state freed on the last chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm_streaming_two_stage import (
    SanaWmStreamingTwoStagePipeline,
    resolve_sana_wm_refiner_paths,
)
from vllm_omni.diffusion.models.sana_wm.refiner import SanaWmRefinerKvCache, SanaWmRefinerSchedule
from vllm_omni.diffusion.models.sana_wm.self_forcing import SanaWmSelfForcingSchedule
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

LATENT_CHANNELS = 4
HEIGHT = 64
WIDTH = 96
LATENT_H, LATENT_W = HEIGHT // 32, WIDTH // 32
NUM_BLOCKS = 2
REFINER_LAYERS = 2


class _FakeTransformer:
    def __init__(self) -> None:
        self.blocks = [object() for _ in range(NUM_BLOCKS)]
        self.calls = 0

    def forward_streaming(self, hidden_states, timestep, *, frame_index, cache, save_cache, **kwargs):
        self.calls += 1
        return hidden_states * 0.5


class _FakeRefinerRunner:
    """Records ``refine_block`` calls and returns a recognisable refined block."""

    def __init__(self) -> None:
        self.schedule = SanaWmRefinerSchedule()
        self.num_layers = REFINER_LAYERS
        self.calls: list[dict] = []
        self.sink_frames = 1

    def new_cache(self, *, latent_height, latent_width):
        return SanaWmRefinerKvCache(num_layers=REFINER_LAYERS, tokens_per_frame=latent_height * latent_width)

    def refine_block(
        self,
        cache,
        clean_block,
        *,
        block_start,
        encoder_hidden_states,
        encoder_attention_mask,
        fps,
        generator,
        sink_latents=None,
    ):
        if cache.sink_kv_pre is None:
            assert sink_latents is not None
            cache.sink_kv_pre = [(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))] * REFINER_LAYERS
        self.calls.append(
            {
                "block_start": block_start,
                "frames": int(clean_block.shape[2]),
                "clean": clean_block.detach().clone(),
                "sink": None if sink_latents is None else sink_latents.detach().clone(),
                "fps": fps,
                "prompt_shape": tuple(encoder_hidden_states.shape),
                "generator_seed": generator.initial_seed(),
            }
        )
        tokens = int(clean_block.shape[2]) * clean_block.shape[3] * clean_block.shape[4]
        cache.append_history(
            [(torch.zeros(1, tokens, 2), torch.zeros(1, tokens, 2))] * REFINER_LAYERS, clean_block.shape[2]
        )
        # refined = clean + 100 marks the refined latents unmistakably.
        return clean_block + 100.0


def _image():
    from PIL import Image

    return Image.new("RGB", (WIDTH, HEIGHT), (10, 20, 30))


def _prompt(num_frames: int):
    return {
        "prompt": "a test prompt",
        "multi_modal_data": {"image": _image()},
        "sana_wm": {
            "action": f"w-{num_frames - 1}",
            "num_frames": num_frames,
            "height": HEIGHT,
            "width": WIDTH,
            "intrinsics": {"fx": 48.0, "fy": 48.0, "cx": 48.0, "cy": 32.0},
        },
    }


def _sampling(**overrides):
    params = SimpleNamespace(
        height=HEIGHT,
        width=WIDTH,
        num_frames=overrides.pop("num_frames", 49),
        num_inference_steps=None,
        guidance_scale=None,
        guidance_scale_provided=False,
        seed=7,
        generator=None,
        output_type="np",
        extra_args={},
        resolved_frame_rate=16.0,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def _pipeline(monkeypatch) -> SanaWmStreamingTwoStagePipeline:
    pipeline = object.__new__(SanaWmStreamingTwoStagePipeline)
    pipeline.od_config = None
    pipeline.sana_wm_config = SanaWmConfig(streaming=True, chunk_size=3)
    pipeline.self_forcing_schedule = SanaWmSelfForcingSchedule.from_config(pipeline.sana_wm_config)
    pipeline.refiner_schedule = SanaWmRefinerSchedule()
    pipeline.transformer = _FakeTransformer()
    pipeline._refiner_runner = _FakeRefinerRunner()
    pipeline.device = torch.device("cpu")
    pipeline._last_prompt_attention_mask = None
    decoded: list[torch.Tensor] = []
    pipeline._decoded = decoded

    monkeypatch.setattr(pipeline, "_runtime_device_dtype", lambda: (torch.device("cpu"), torch.float32))
    monkeypatch.setattr(
        pipeline, "_native_prompt_embeds", lambda prompt, *, device, dtype: torch.ones(1, 6, 8, dtype=dtype)
    )
    monkeypatch.setattr(
        pipeline,
        "_encode_refiner_prompt",
        lambda text, *, device, dtype: (torch.full((1, 5, 12), 0.25, dtype=dtype), None),
    )

    def fake_encode_first_frame(image, *, height, width, latent_height, latent_width, device, dtype):
        return torch.full((1, LATENT_CHANNELS, 1, latent_height, latent_width), 2.0, dtype=dtype)

    monkeypatch.setattr(pipeline, "_vae_encode_first_frame", fake_encode_first_frame)

    def fake_decode(latents, *, output_type, device, dtype):
        if output_type == "latent":
            return latents
        decoded.append(latents.detach().clone())
        frames = 1 + 8 * (latents.shape[2] - 1)
        return np.zeros((1, frames, HEIGHT, WIDTH, 3), dtype=np.float32)

    monkeypatch.setattr(pipeline, "_decode_native_latents", fake_decode)
    return pipeline


def _state(prompt, sampling) -> StepRequestState:
    return StepRequestState(request_id="req-0", sampling=sampling, prompt=prompt)


def _drive_chunk(pipeline, state):
    batch = SimpleNamespace(states=[state])
    while not state.chunk_denoise_completed:
        noise_pred = pipeline.denoise_step(batch, states=[state])
        pipeline.step_scheduler(state, noise_pred)
    return pipeline.post_decode(state)


def test_refiner_runs_once_per_chunk_and_feeds_the_decode(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    num_frames = 73  # 10 latent frames: chunks [0,4) [4,7) [7,10)
    state = pipeline.prepare_encode(_state(_prompt(num_frames), _sampling(num_frames=num_frames)))
    assert state.total_chunks == 3
    extra = state.extra
    assert extra["backend"] == "native_gdn_streaming+ltx2_refiner"
    assert extra["refiner_fps"] == 16.0
    assert extra["refiner_seed"] == 7
    assert isinstance(extra["refiner_cache"], SanaWmRefinerKvCache)
    torch.testing.assert_close(extra["refined_history"], extra["first_latent"])

    outputs = []
    stage1_blocks = []
    while not state.request_denoise_completed:
        # The clean Stage-1 block is state.latents right before post_decode.
        batch = SimpleNamespace(states=[state])
        while not state.chunk_denoise_completed:
            pipeline.step_scheduler(state, pipeline.denoise_step(batch, states=[state]))
        stage1_blocks.append(state.latents.clone())
        outputs.append(pipeline.post_decode(state))

    runner = pipeline._refiner_runner
    assert [call["block_start"] for call in runner.calls] == [1, 4, 7]
    assert all(call["frames"] == 3 for call in runner.calls)
    assert runner.calls[0]["sink"] is not None and runner.calls[0]["sink"].shape[2] == 1
    torch.testing.assert_close(runner.calls[0]["sink"], extra_first := stage1_first_latent(pipeline, state))
    assert all(call["sink"] is None for call in runner.calls[1:])
    for call, clean in zip(runner.calls, stage1_blocks):
        torch.testing.assert_close(call["clean"], clean)
    assert all(call["prompt_shape"] == (1, 5, 12) for call in runner.calls)
    assert all(call["generator_seed"] == 7 for call in runner.calls)

    # Decode reads the refined history: the conditioning frame (raw) then
    # refined blocks (clean + 100), always chunk_size + 1 latent frames.
    decoded = pipeline._decoded
    assert [d.shape[2] for d in decoded] == [4, 4, 4]
    torch.testing.assert_close(decoded[0][:, :, :1], extra_first)
    torch.testing.assert_close(decoded[0][:, :, 1:], stage1_blocks[0] + 100.0)
    torch.testing.assert_close(decoded[1][:, :, :1], stage1_blocks[0][:, :, -1:] + 100.0)
    torch.testing.assert_close(decoded[1][:, :, 1:], stage1_blocks[1] + 100.0)
    torch.testing.assert_close(decoded[2][:, :, 1:], stage1_blocks[2] + 100.0)

    videos = [out.output["payload"]["video"] for out in outputs]
    assert [v.shape[1] for v in videos] == [25, 24, 24]
    assert sum(v.shape[1] for v in videos) == num_frames
    assert [out.finished for out in outputs] == [False, False, True]
    for index, out in enumerate(outputs):
        meta = out.output["metadata"]["sana_wm"]
        assert meta["backend"] == "native_gdn_streaming+ltx2_refiner"
        assert meta["refiner_backend"] == "native_ltx2_chunk_causal"
        assert meta["refiner_blocks_refined"] == index + 1
        assert meta["refiner_sigmas"] == [0.909375, 0.725, 0.421875, 0.0]
        assert meta["refiner_seed"] == 7
    assert outputs[0].output["metadata"]["sana_wm"]["refiner_cached_latent_frames"] == 4
    assert outputs[2].output["metadata"]["sana_wm"]["refiner_cached_latent_frames"] == 10

    for key in ("refiner_cache", "refined_history", "refiner_prompt_embeds", "refiner_generator", "cache"):
        assert key not in state.extra


def stage1_first_latent(pipeline, state):
    return torch.full((1, LATENT_CHANNELS, 1, LATENT_H, LATENT_W), 2.0)


def test_latent_output_returns_refined_block(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    state = pipeline.prepare_encode(_state(_prompt(49), _sampling(num_frames=49, output_type="latent")))
    clean = None
    batch = SimpleNamespace(states=[state])
    while not state.chunk_denoise_completed:
        pipeline.step_scheduler(state, pipeline.denoise_step(batch, states=[state]))
    clean = state.latents.clone()
    out = pipeline.post_decode(state)
    torch.testing.assert_close(out.output["payload"]["latents"], clean + 100.0)
    assert pipeline._decoded == []


def test_refiner_seed_and_steps_overrides(monkeypatch) -> None:
    pipeline = _pipeline(monkeypatch)
    state = pipeline.prepare_encode(
        _state(
            _prompt(49), _sampling(num_frames=49, extra_args={"sana_wm_refiner_seed": 99, "sana_wm_refiner_steps": 3})
        )
    )
    assert state.extra["refiner_seed"] == 99
    assert state.extra["refiner_generator"].initial_seed() == 99
    with pytest.raises(ValueError, match="refiner_steps must be 3"):
        _pipeline(monkeypatch).prepare_encode(
            _state(_prompt(49), _sampling(num_frames=49, extra_args={"sana_wm_refiner_steps": 2}))
        )


def test_refiner_paths_resolution(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_SANA_WM_REFINER_ROOT", raising=False)
    monkeypatch.delenv("VLLM_OMNI_SANA_WM_REFINER_TEXT_ENCODER", raising=False)
    with pytest.raises(FileNotFoundError, match="refiner tree"):
        resolve_sana_wm_refiner_paths(tmp_path)
    for sub in ("refiner/transformer", "refiner/connectors", "refiner/text_encoder"):
        (tmp_path / sub).mkdir(parents=True)
        (tmp_path / sub / "config.json").write_text("{}")
    paths = resolve_sana_wm_refiner_paths(tmp_path)
    assert paths.transformer_dir == tmp_path / "refiner" / "transformer"
    assert paths.text_encoder_dir == tmp_path / "refiner" / "text_encoder"

    other = tmp_path / "elsewhere"
    (other / "transformer").mkdir(parents=True)
    (other / "connectors").mkdir()
    (other / "transformer" / "config.json").write_text("{}")
    (other / "connectors" / "config.json").write_text("{}")
    gemma = tmp_path / "gemma"
    gemma.mkdir()
    (gemma / "config.json").write_text("{}")
    monkeypatch.setenv("VLLM_OMNI_SANA_WM_REFINER_ROOT", str(other))
    monkeypatch.setenv("VLLM_OMNI_SANA_WM_REFINER_TEXT_ENCODER", str(gemma))
    paths = resolve_sana_wm_refiner_paths(tmp_path)
    assert paths.root == other and paths.text_encoder_dir == gemma
