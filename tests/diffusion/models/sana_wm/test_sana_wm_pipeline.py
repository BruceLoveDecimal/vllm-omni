# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_stage1_pipeline_declares_components() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    assert SanaWmPipeline._dit_modules == ["transformer"]
    assert SanaWmPipeline._encoder_modules == ["text_encoder", "camera_encoder"]
    assert SanaWmPipeline._vae_modules == ["vae"]


def test_sana_wm_stage1_preprocess_accepts_action() -> None:
    from vllm_omni.diffusion.models.sana_wm import get_sana_wm_pre_process_func

    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "drive forward",
                "multi_modal_data": {"image": object()},
                "sana_wm": {"action": "w-1", "num_frames": 1},
            }
        ],
        sampling_params=SimpleNamespace(height=64, width=64, num_frames=1),
    )

    get_sana_wm_pre_process_func(SimpleNamespace())(request)
    payload = request.prompts[0]["additional_information"]["sana_wm"]

    assert payload["action"] == "w-1"
    assert payload["num_frames"] == 1


def test_sana_wm_native_decode_uses_ltx2_vae_when_requested() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    class FakeVAE(torch.nn.Module):
        dtype = torch.float32
        spatial_compression_ratio = 32
        config = SimpleNamespace(timestep_conditioning=True)

        def __init__(self) -> None:
            super().__init__()
            self.seen_timestep = None
            self.seen_latents_shape = None

        def decode(self, latents, timestep=None, return_dict=False):
            self.seen_timestep = timestep
            self.seen_latents_shape = tuple(latents.shape)
            video = torch.zeros(latents.shape[0], 3, latents.shape[2], latents.shape[3], latents.shape[4])
            return (video,)

    pipeline = SanaWmPipeline(od_config=None)
    fake_vae = FakeVAE()
    pipeline.vae = fake_vae
    latents = torch.randn(1, 128, 1, 2, 2)

    output = pipeline._decode_native_smoke_latents(
        latents,
        output_type="pt",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert fake_vae.seen_latents_shape == (1, 128, 1, 2, 2)
    assert fake_vae.seen_timestep is not None
    assert torch.isfinite(output).all()


def test_sana_wm_native_smoke_scheduler_runs_requested_steps() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmPipeline

    class FakeTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_timesteps: list[float] = []

        def forward(self, hidden_states, timestep, **kwargs):
            del kwargs
            self.seen_timesteps.append(float(timestep.flatten()[0]))
            return torch.ones_like(hidden_states)

    pipeline = SanaWmPipeline(od_config=None)
    pipeline.sana_wm_config = SanaWmConfig(
        num_blocks=1,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
        chunk_plucker_channels=48,
    )
    fake_transformer = FakeTransformer()
    pipeline.transformer = fake_transformer
    req = SimpleNamespace(
        prompts=[
            {
                "prompt": "scheduler smoke",
                "multi_modal_data": {"image": object()},
                "sana_wm": {"action": "w-1", "num_frames": 1, "height": 64, "width": 64},
            }
        ],
        sampling_params=SimpleNamespace(
            height=64,
            width=64,
            num_frames=1,
            num_inference_steps=3,
            seed=0,
            extra_args={"sana_wm_native_smoke": True, "sana_wm_hash_prompt_smoke": True},
        ),
    )

    output = pipeline(req)

    assert output.custom_output["sana_wm_sampling_steps"] == 3
    assert len(fake_transformer.seen_timesteps) == 3
    assert fake_transformer.seen_timesteps == sorted(fake_transformer.seen_timesteps, reverse=True)
