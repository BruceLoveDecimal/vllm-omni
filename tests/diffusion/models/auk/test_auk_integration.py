# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AuK's vLLM-Omni registration, attention, loading, and request contracts."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.models.auk.auk_transformer import Flux2Edit
from vllm_omni.diffusion.models.auk.pipeline_auk import AuKPipeline, get_auk_post_process_func
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def test_yaml_discovery_and_registry(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  name: AuK\n  arch:\n    dim: 1536\n  text_encoder: {}\n  vae: {}\n"
    )
    assert is_diffusion_model(str(tmp_path))
    assert resolve_model_class_name(str(tmp_path)) == "AuKPipeline"
    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == "AuKPipeline"
    assert config.tf_model_config.dim == 1536
    assert DiffusionModelRegistry._try_load_model_cls("AuKPipeline") is AuKPipeline


@pytest.mark.parametrize("with_reference", [False, True])
def test_omni_attention_matches_sdpa(with_reference, monkeypatch):
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

    # Select the real CPU backend without depending on accelerator discovery.
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.selector._cached_get_backend_cls",
        lambda *args, **kwargs: SDPABackend,
    )
    # Omni has no CPU platform dispatcher; exercise its unchanged SDPA kernel.
    monkeypatch.setattr(SDPAImpl, "forward", SDPAImpl._forward_impl)
    config = OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})
    kwargs = dict(
        dim=64,
        heads=4,
        dim_head=16,
        latent_dim=8,
        text_hidden_dim=16,
        num_layers=1,
        num_single_layers=1,
        dropout=0,
        attn_mask_enabled=True,
    )
    reference = Flux2Edit(**kwargs, attn_backend="torch").eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.2, 0.2)
    with set_current_diffusion_config(config):
        port = Flux2Edit(**kwargs, attn_backend="omni").eval()
    port.load_state_dict(reference.state_dict(), strict=True)
    data = dict(
        x=torch.randn(2, 5, 8),
        text=torch.randn(2, 3, 16),
        time=torch.tensor(0.4),
        mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        c_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        ref=torch.randn(2, 2 if with_reference else 0, 8),
        cfg_infer=True,
        ref_mask=torch.ones(2, 2 if with_reference else 0, dtype=torch.bool),
    )
    with torch.inference_mode():
        torch.testing.assert_close(port(**data), reference(**data), rtol=1e-5, atol=1e-6)


class Conditioner(nn.Module):
    def forward(self, instruction, audio, sample_rate, device):
        self.last_instruction = instruction
        return torch.ones(1, 2, 16, device=device), torch.ones(1, 2, dtype=torch.bool, device=device)


class Decoder(nn.Module):
    def encoding_and_normalization(self, waveform, sample_lengths, generator):
        length = int(sample_lengths[0]) // 480
        return torch.randn(1, length, 8, generator=generator), torch.tensor([length])

    def denormalize(self, latent):
        return latent

    def inference_from_latents(self, latent):
        return latent.mean(1, keepdim=True).repeat_interleave(480, -1)


def small_pipeline(flash):
    pipeline = AuKPipeline.__new__(AuKPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.compute_dtype = torch.float32
    pipeline.sample_rate, pipeline.hop_size, pipeline.latent_dim = 24000, 480, 8
    pipeline.flash = flash
    pipeline.conditioner = Conditioner()
    pipeline.vae = Decoder()
    pipeline.transformer = Flux2Edit(
        dim=64, heads=4, dim_head=16, latent_dim=8, text_hidden_dim=16, num_layers=1, num_single_layers=1, dropout=0
    ).eval()
    return pipeline.eval()


@pytest.mark.parametrize("flash", [False, True])
def test_pipeline_seed_duration_and_output(flash):
    pipeline = small_pipeline(flash)
    prompt = {"instruction": "Remove noise.", "ref_audio": (torch.zeros(480), 24000), "gen_seconds": 0.045}
    params = OmniDiffusionSamplingParams(seed=7, num_inference_steps=2)
    batch = DiffusionRequestBatch([OmniDiffusionRequest(prompt=prompt, request_id="test", sampling_params=params)])
    rng_state = torch.random.get_rng_state().clone()
    first = pipeline(batch)[0].output
    second = pipeline(batch)[0].output
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(first, second)
    assert first.shape == (1, 3 * 480)
    assert pipeline.conditioner.last_instruction == "Remove noise."
    result = get_auk_post_process_func(None)(first)
    assert result.shape == first.shape and str(result.dtype) == "float32"
    assert AuKPipeline.audio_sample_rate == 24000


def test_pipeline_rejects_overlength_before_encoding():
    pipeline = small_pipeline(False)
    batch = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt={"input": "hello", "gen_seconds": 31},
                request_id="long",
                sampling_params=OmniDiffusionSamplingParams(),
            )
        ]
    )
    with pytest.raises(ValueError, match="30 seconds"):
        pipeline(batch)


def test_checkpoint_rejects_missing_and_unexpected_weights(tmp_path):
    from safetensors.torch import save_file

    pipeline = small_pipeline(False)
    pipeline.model_path = tmp_path
    pipeline.checkpoint_name = "auk_base.safetensors"
    state = {"layer_weights": torch.ones(2), "layer_scale": torch.ones(1), "wrong": torch.ones(1)}
    save_file(state, str(tmp_path / pipeline.checkpoint_name))
    with pytest.raises(ValueError, match="Unexpected AuK checkpoint keys"):
        pipeline.load_weights(())
    del state["wrong"]
    save_file(state, str(tmp_path / pipeline.checkpoint_name))
    with pytest.raises(RuntimeError, match="Missing key"):
        pipeline.load_weights(())
