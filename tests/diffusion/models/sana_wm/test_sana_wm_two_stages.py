# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_two_stage_declares_isolated_refiner_components() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmTwoStagesPipeline

    pipe = SanaWmTwoStagesPipeline(od_config=None)

    assert "refiner_text_encoder" in pipe._encoder_modules
    assert "refiner_connectors" in pipe._encoder_modules
    assert pipe._resident_modules == ["refiner_transformer"]
    assert pipe.refiner_transformer is None
    assert pipe.refiner_text_encoder is None
    assert pipe.refiner_connectors is None


def test_sana_wm_two_stage_refiner_loader_requires_checkpoint() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmTwoStagesPipeline

    pipe = SanaWmTwoStagesPipeline(od_config=None)

    with pytest.raises(ValueError, match="checkpoint resolution"):
        pipe.ensure_refiner_components()


def test_sana_wm_two_stage_refiner_prompt_encoder_uses_connectors() -> None:
    import types

    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmTwoStagesPipeline

    class FakeBatch(types.SimpleNamespace):
        def to(self, device):
            self.input_ids = self.input_ids.to(device)
            self.attention_mask = self.attention_mask.to(device)
            return self

    class FakeTokenizer:
        model_max_length = 4
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

        def __call__(self, prompts, **kwargs):
            assert prompts == ["drive forward"]
            assert kwargs["max_length"] == 4
            return FakeBatch(
                input_ids=torch.zeros(1, 4, dtype=torch.long),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
            )

    class FakeTextEncoder(torch.nn.Module):
        def forward(self, **kwargs):
            hidden = torch.ones(1, 4, 2)
            return types.SimpleNamespace(hidden_states=[hidden, hidden * 2])

    class FakeConnectors(torch.nn.Module):
        def forward(self, prompt_embeds, attention_mask, *, padding_side):
            assert prompt_embeds.shape == (1, 4, 4)
            assert padding_side == "left"
            return prompt_embeds, prompt_embeds + 1, attention_mask

    pipe = SanaWmTwoStagesPipeline(od_config=None)
    pipe.refiner_tokenizer = FakeTokenizer()
    pipe.refiner_text_encoder = FakeTextEncoder()
    pipe.refiner_connectors = FakeConnectors()

    prompt, audio_prompt, mask = pipe._encode_refiner_prompt(
        "drive forward",
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_sequence_length=4,
    )

    assert prompt.shape == (1, 4, 4)
    assert torch.equal(audio_prompt, prompt + 1)
    assert mask.shape == (1, 4)


def test_sana_wm_two_stage_inprocess_refiner_step_runs_with_fake_components(monkeypatch) -> None:
    import types

    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmTwoStagesPipeline

    class FakeRope:
        def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps=24.0):
            return torch.zeros(batch_size, 3, num_frames * height * width, 2, device=device)

        def prepare_audio_coords(self, batch_size, num_frames, device):
            return torch.zeros(batch_size, 1, num_frames, 2, device=device)

    class FakeTransformer(torch.nn.Module):
        config = types.SimpleNamespace(patch_size=1, patch_size_t=1, audio_in_channels=4)

        def __init__(self):
            super().__init__()
            self.rope = FakeRope()
            self.audio_rope = FakeRope()
            self.calls = 0

        def forward(self, hidden_states, audio_hidden_states, **kwargs):
            self.calls += 1
            assert kwargs["encoder_hidden_states"].shape[-1] == 4
            return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)

    pipe = SanaWmTwoStagesPipeline(od_config=None)
    pipe.refiner_transformer = FakeTransformer()
    monkeypatch.setattr(pipe, "ensure_refiner_components", lambda **kwargs: None)
    monkeypatch.setattr(
        pipe,
        "_encode_refiner_prompt",
        lambda *args, **kwargs: (
            torch.ones(1, 4, 4),
            torch.ones(1, 4, 4),
            torch.ones(1, 4, dtype=torch.long),
        ),
    )

    latents = torch.ones(1, 4, 1, 2, 2)
    output = pipe._run_inprocess_refiner(
        latents=latents,
        prompt_text="drive forward",
        payload={"fps": 24.0},
        sampling_params=types.SimpleNamespace(
            extra_args={
                "sana_wm_refiner_output_type": "latent",
                "sana_wm_inprocess_refiner_steps": 2,
            }
        ),
    )

    assert output.custom_output["sana_wm_backend"] == "native_inprocess_refiner"
    assert output.custom_output["sana_wm_refiner_steps"] == 2
    assert output.output.shape == latents.shape
    assert pipe.refiner_transformer.calls == 2
