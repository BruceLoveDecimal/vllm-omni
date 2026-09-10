# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity against the pinned upstream AuK implementation.

Set AUK_REFERENCE_PATH to a checkout of Tencent-Hunyuan/AuK at d9f30ffe.
These tests exercise nonzero random weights; a zero-initialized DiT would
otherwise hide attention, masking, and conditioning regressions.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.auk.audio import prepare_audio
from vllm_omni.diffusion.models.auk.auk_conditioner import AuKConditioner
from vllm_omni.diffusion.models.auk.auk_transformer import Flux2Edit
from vllm_omni.diffusion.models.auk.configuration_auk import read_auk_config
from vllm_omni.diffusion.models.auk.request import parse_request
from vllm_omni.diffusion.models.auk.sampling import sample_latents, time_grid
from vllm_omni.diffusion.models.auk.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(scope="module")
def upstream():
    path = os.getenv("AUK_REFERENCE_PATH")
    if not path:
        pytest.skip("Set AUK_REFERENCE_PATH to run upstream parity tests")
    sys.path.insert(0, str(Path(path) / "src"))
    from auk.model.cfm_edit import CFMEdit
    from auk.model.flux2_edit import Flux2Edit as ReferenceDiT
    from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAE as ReferenceVAE
    from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig as ReferenceVAEConfig

    yield SimpleNamespace(cfm=CFMEdit, dit=ReferenceDiT, vae=ReferenceVAE, vae_config=ReferenceVAEConfig)
    sys.path.remove(str(Path(path) / "src"))


def make_dit(cls):
    return cls(
        dim=64,
        heads=4,
        dim_head=16,
        latent_dim=8,
        text_hidden_dim=16,
        num_layers=2,
        num_single_layers=2,
        dropout=0,
        ff_mult=2,
        attn_mask_enabled=True,
    ).eval()


@pytest.mark.parametrize("reference_frames", [0, 5])
@pytest.mark.parametrize("cfg", [False, True])
def test_dit_parity(upstream, reference_frames, cfg):
    torch.manual_seed(17)
    reference = make_dit(upstream.dit)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.2, 0.2)
    port = make_dit(Flux2Edit)
    port.load_state_dict(reference.state_dict(), strict=True)
    kwargs = dict(
        x=torch.randn(2, 7, 8),
        text=torch.randn(2, 4, 16),
        time=torch.tensor(0.37),
        ref=torch.randn(2, reference_frames, 8),
        cfg_infer=cfg,
        ref_mask=torch.ones(2, reference_frames, dtype=torch.bool),
        c_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
        mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
    )
    with torch.inference_mode():
        actual, expected = port(**kwargs), reference(**kwargs)
    assert expected.abs().max() > 1e-4
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class RecordedThinker(nn.Module):
    def __init__(self, states):
        super().__init__()
        self.states = states
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=len(states) - 1))

    def forward(self, **kwargs):
        return SimpleNamespace(hidden_states=self.states)


def test_layer_fusion_parity(upstream):
    torch.manual_seed(3)
    states = tuple(torch.randn(2, 9, 16) * (idx + 1) for idx in range(5))
    thinker = RecordedThinker(states)
    original = upstream.cfm(
        transformer=make_dit(upstream.dit), text_encoder=thinker, text_processor=None, num_channels=8
    )
    port = AuKConditioner(thinker, None)
    weights = torch.tensor([-3.0, 0.1, 2.2, -0.3])
    with torch.no_grad():
        original.layer_weights.copy_(weights)
        original.layer_scale.fill_(1.7)
        port.layer_weights.copy_(weights)
        port.layer_scale.copy_(original.layer_scale)
    inputs = {"input_ids": torch.zeros(2, 9, dtype=torch.long), "attention_mask": torch.ones(2, 9)}
    expected, _ = original.encode_text(inputs, "cpu")
    torch.testing.assert_close(port.fuse(states), expected, rtol=0, atol=0)


@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("reference_frames", [0, 3])
def test_sampler_parity(upstream, flash, reference_frames):
    from torchdiffeq import odeint

    torch.manual_seed(19)
    original = make_dit(upstream.dit)
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.uniform_(-0.2, 0.2)
    port = make_dit(Flux2Edit)
    port.load_state_dict(original.state_dict())
    noise, ref, semantic = torch.randn(1, 8, 8), torch.randn(1, reference_frames, 8), torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    cfg = 0.0 if flash else 2.0
    times = time_grid(flash=flash, steps=8, sway=-1.0, device="cpu")

    def velocity(t, x):
        result = original(
            x=x,
            text=semantic,
            time=t,
            ref=ref,
            ref_mask=torch.ones(1, reference_frames, dtype=torch.bool),
            c_mask=mask,
            cfg_infer=not flash,
            cache=True,
        )
        if flash:
            return result
        cond, uncond = result.chunk(2)
        return cond + (cond - uncond) * cfg

    with torch.inference_mode():
        expected = odeint(velocity, noise, times, method="euler")[-1]
        actual = sample_latents(port, noise, ref, semantic, mask, times, cfg)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert port.text_cond is None and port.text_uncond is None


def test_vae_parity_and_rng_isolation(upstream):
    config = dict(
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        upsample_initial_channel=48,
        downsample_rates=[2, 2],
        downsample_channels=[12, 24, 48],
        latent_dim=8,
        flow_hidden_channels=16,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
    )
    original = upstream.vae(upstream.vae_config.from_dict(config)).eval()
    port = BigVGANFlowVAE(BigVGANFlowVAEConfig.from_dict(config)).eval()
    with torch.no_grad():
        original.global_mean.copy_(torch.arange(8) / 10)
        original.global_log_std.copy_(torch.arange(8) / 10 + 0.5)
    port.load_state_dict(original.state_dict(), strict=True)
    waveform = torch.randn(1, 1, 64)
    lengths = torch.tensor([60])
    with torch.inference_mode():
        torch.manual_seed(42)
        expected, expected_len = original.encoding_and_normalization(waveform, lengths)
        rng_state = torch.random.get_rng_state().clone()
        actual, actual_len = port.encoding_and_normalization(
            waveform, lengths, generator=torch.Generator().manual_seed(42)
        )
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_len, expected_len)
        decoded = port.inference_from_latents(port.denormalize(actual).transpose(1, 2))
        expected_audio = original.inference_from_latents(original.denormalize(expected).transpose(1, 2))
        torch.testing.assert_close(decoded, expected_audio, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("seconds", [None, 0, -1, float("nan"), float("inf"), True, "3"])
def test_invalid_text_duration(seconds):
    with pytest.raises(ValueError):
        parse_request({"input": "hello"}, {"gen_seconds": seconds})


def test_edit_instruction_is_not_spoken():
    audio = ([0.0] * 480, 24000)
    request = parse_request({"instruction": "Remove the background noise.", "ref_audio": audio}, {})
    assert request.instruction == "Remove the background noise."
    assert request.audio is audio and request.seconds is None


def test_tts_prompt_and_no_mutation():
    prompt = {"input": "Hello", "instruct": "Speak softly", "mm_processor_kwargs": {"gen_seconds": 2}}
    result = parse_request(prompt, {"gen_seconds": 3})
    assert "Hello" in result.instruction and "Speak softly" in result.instruction
    assert result.seconds == 3
    assert prompt["mm_processor_kwargs"]["gen_seconds"] == 2


def test_native_yaml_recognition(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  name: AuK-Flash\n  arch: {}\n  vae: {}\n  text_encoder: {}\n")
    assert read_auk_config(str(tmp_path))["model"]["name"] == "AuK-Flash"
    path.write_text("model:\n  name: unrelated\n")
    assert read_auk_config(str(tmp_path)) is None


def test_yaml_interpolation(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  name: AuK\n  arch: {}\n  text_encoder: {}\n"
        "  vae:\n    latent_dim: 64\n    model_init_kwargs:\n      latent_dim: ${model.vae.latent_dim}\n"
    )
    assert read_auk_config(str(tmp_path))["model"]["vae"]["model_init_kwargs"]["latent_dim"] == 64


def test_audio_preprocessing_matches_release(upstream):
    import torchaudio

    waveform = torch.randn(2, 4800)
    original = waveform.clone()
    expected = torchaudio.transforms.Resample(48000, 24000)(waveform.mean(0, keepdim=True))
    torch.testing.assert_close(prepare_audio((waveform, 48000), 24000), expected, rtol=0, atol=0)
    torch.testing.assert_close(waveform, original, rtol=0, atol=0)
    for invalid in [(torch.tensor([float("nan")]), 24000), (torch.zeros(3, 10), 24000), ([], 24000)]:
        with pytest.raises(ValueError):
            prepare_audio(invalid, 24000)


@pytest.mark.parametrize("audio_sr", [None, 16000, 24000, 44100])
def test_processor_parity(upstream, tmp_path, audio_sr):
    """Compare actual token IDs, audio features and masks to build_cond_inputs."""
    import soundfile as sf
    from transformers import Qwen2_5OmniProcessor

    encoder_path = os.getenv("AUK_PROCESSOR_PATH")
    if not encoder_path:
        pytest.skip("Set AUK_PROCESSOR_PATH to the Qwen2.5-Omni-3B processor snapshot")
    processor = Qwen2_5OmniProcessor.from_pretrained(encoder_path)
    instruction = "Say the following with the same voice: Hello."
    audio = None
    content = [{"type": "text", "text": instruction}]
    if audio_sr is None:
        content[0]["text"] += "|<no_prompt_audio>|"
    else:
        audio = torch.sin(torch.arange(audio_sr).float() * (2 * torch.pi * 440 / audio_sr)).unsqueeze(0)
        path = str(tmp_path / "reference.wav")
        sf.write(path, audio.squeeze(0).numpy(), audio_sr, subtype="FLOAT")
        content.append({"type": "audio", "audio": path})
    expected = upstream.cfm.build_cond_inputs([[{"role": "user", "content": content}]], processor)

    class InputRecorder(RecordedThinker):
        def forward(self, **kwargs):
            self.inputs = kwargs
            return super().forward(**kwargs)

    recorder = InputRecorder((torch.ones(1, 1, 8), torch.ones(1, 1, 8)))
    conditioner = AuKConditioner(recorder, processor)
    conditioner(instruction, audio, audio_sr or 24000, "cpu")
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(recorder.inputs[key], value, rtol=1e-6, atol=1e-6, msg=key)


@pytest.mark.parametrize("with_audio", [False, True])
def test_real_thinker_conditioning_parity(upstream, tmp_path, with_audio):
    """Exercise real audio/text towers and layer collection with a tiny Qwen."""
    import soundfile as sf
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration

    encoder_path = os.getenv("AUK_PROCESSOR_PATH")
    if not encoder_path:
        pytest.skip("Set AUK_PROCESSOR_PATH to the Qwen2.5-Omni-3B processor snapshot")
    processor = Qwen2_5OmniProcessor.from_pretrained(encoder_path)
    config = Qwen2_5OmniThinkerConfig(
        vision_start_token_id=151652,
        text_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=152064,
            rope_scaling={"type": "default", "mrope_section": [1, 1, 2]},
        ),
        audio_config=dict(d_model=32, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=64, output_dim=32),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32),
    )
    thinker = Qwen2_5OmniThinkerForConditionalGeneration(config).eval()
    conditioner = AuKConditioner(thinker, processor)
    original = upstream.cfm(
        transformer=make_dit(upstream.dit), text_encoder=thinker, text_processor=processor, num_channels=8
    )
    with torch.no_grad():
        original.layer_weights.copy_(torch.tensor([-1.0, 1.5]))
        original.layer_scale.fill_(0.7)
        conditioner.layer_weights.copy_(original.layer_weights)
        conditioner.layer_scale.copy_(original.layer_scale)
    instruction = "Say hello."
    content = [{"type": "text", "text": instruction}]
    audio = None
    if with_audio:
        audio = torch.sin(torch.arange(24000).float() * (2 * torch.pi * 440 / 24000)).unsqueeze(0)
        path = str(tmp_path / "reference.wav")
        sf.write(path, audio.squeeze().numpy(), 24000, subtype="FLOAT")
        content.append({"type": "audio", "audio": path})
    else:
        content[0]["text"] += "|<no_prompt_audio>|"
    inputs = original.build_cond_inputs([[{"role": "user", "content": content}]], processor)
    with torch.inference_mode():
        expected, expected_mask = original.encode_text(inputs, "cpu")
        actual, actual_mask = conditioner(instruction, audio, 24000, "cpu")
    assert actual.shape[-1] == 32 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_mask, expected_mask)


def test_flash_grid_and_base_validation():
    expected = torch.tensor([0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1])
    torch.testing.assert_close(time_grid(flash=True, steps=32, sway=0, device="cpu"), expected, rtol=0, atol=0)
    for steps, sway in [(0, -1), (True, -1), (4, float("nan")), (4, 2)]:
        with pytest.raises(ValueError):
            time_grid(flash=False, steps=steps, sway=sway, device="cpu")
