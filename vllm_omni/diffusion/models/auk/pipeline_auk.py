# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-stage continuous-latent speech generation and editing with AuK."""

import math
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.auk.audio import prepare_audio
from vllm_omni.diffusion.models.auk.auk_conditioner import AuKConditioner
from vllm_omni.diffusion.models.auk.auk_transformer import Flux2Edit
from vllm_omni.diffusion.models.auk.configuration_auk import read_auk_config
from vllm_omni.diffusion.models.auk.request import parse_request
from vllm_omni.diffusion.models.auk.sampling import sample_latents, time_grid
from vllm_omni.diffusion.models.auk.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig
from vllm_omni.diffusion.models.interface import SupportAudioOutput, SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


def get_auk_post_process_func(od_config):
    def post_process(audio, output_type="np"):
        if output_type == "pt":
            return audio
        return audio.float().cpu().numpy()

    return post_process


class AuKPipeline(nn.Module, SupportAudioOutput, SupportsComponentDiscovery):
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 24000
    supports_request_batch = False
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["conditioner.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        if od_config.parallel_config.world_size != 1:
            raise ValueError("AuK currently supports a single GPU; model parallelism is not implemented")
        if od_config.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("AuK supports float32 or bfloat16 autocast")
        from huggingface_hub import snapshot_download
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        self.od_config = od_config
        self.device = get_local_device()
        self.compute_dtype = od_config.dtype
        self.model_path = Path(od_config.model)
        if not self.model_path.is_dir():
            self.model_path = Path(
                snapshot_download(
                    od_config.model,
                    revision=od_config.revision,
                    allow_patterns=["config.yaml", "*.safetensors"],
                )
            )
        config = read_auk_config(str(self.model_path))
        if config is None:
            raise ValueError("Expected an AuK or AuK-Flash config.yaml")
        config = config["model"]
        self.flash = config["name"] == "AuK-Flash"
        self.checkpoint_name = "auk_flash.safetensors" if self.flash else "auk_base.safetensors"
        for filename in (self.checkpoint_name, "vae.safetensors"):
            if not (self.model_path / filename).is_file():
                raise FileNotFoundError(self.model_path / filename)
        vae_config = config["vae"]
        self.sample_rate = int(vae_config["target_sample_rate"])
        self.hop_size = int(vae_config["downsample_rate"])
        self.latent_dim = int(vae_config["latent_dim"])
        if self.sample_rate != 24000 or self.hop_size != 480 or self.latent_dim != 64:
            raise ValueError("Unsupported AuK VAE layout; expected 24 kHz, hop 480, latent dimension 64")
        method = config.get("schedule", {}).get("odeint_kwargs", {}).get("method", "euler")
        if method != "euler":
            raise ValueError(f"AuK supports the released Euler sampler, got {method}")

        # A sibling snapshot is convenient for fully offline deployments.
        encoder_override = od_config.additional_config.get("qwen_path")
        encoder_path = encoder_override or config["text_encoder"]["text_encoder_path"]
        if encoder_override is None and not Path(encoder_path).is_dir():
            sibling = self.model_path.parent / "Qwen2.5-Omni-3B"
            encoder_path = str(sibling) if sibling.is_dir() else "Qwen/Qwen2.5-Omni-3B"
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            encoder_path,
            torch_dtype=torch.float32,
        )
        if getattr(thinker, "visual", None) is not None:
            thinker.visual = None
        self.conditioner = AuKConditioner(thinker, Qwen2_5OmniProcessor.from_pretrained(encoder_path))
        arch = dict(config["arch"])
        arch.update(attn_backend="omni", checkpoint_activations=False)
        self.transformer = Flux2Edit(**arch, latent_dim=self.latent_dim)
        self.vae = BigVGANFlowVAE(BigVGANFlowVAEConfig.from_dict(vae_config.get("model_init_kwargs", {})))
        # The release keeps weights and VAE operations in fp32, using autocast
        # only for the conditioner and the denoising transformer.
        self.float().eval().requires_grad_(False)

    def load_weights(self, weights) -> set[str]:
        from safetensors.torch import load_file

        state = load_file(str(self.model_path / self.checkpoint_name))
        transformer = {k.removeprefix("transformer."): v for k, v in state.items() if k.startswith("transformer.")}
        fusion = {k: state[k] for k in ("layer_weights", "layer_scale")}
        unknown = set(state) - {"transformer." + k for k in transformer} - set(fusion)
        if unknown:
            raise ValueError(f"Unexpected AuK checkpoint keys: {sorted(unknown)[:10]}")
        self.transformer.load_state_dict(transformer, strict=True)
        with torch.no_grad():
            for name, weight in fusion.items():
                parameter = getattr(self.conditioner, name)
                if parameter.shape != weight.shape:
                    raise ValueError(f"AuK {name} shape {weight.shape} does not match encoder {parameter.shape}")
                parameter.copy_(weight)
        self.vae.load_state_dict(load_file(str(self.model_path / "vae.safetensors")), strict=True)
        self.to(self.device)
        return {name for name, _ in self.named_parameters()}

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        params = req.sampling_params
        if params.num_outputs_per_prompt != 1:
            raise ValueError("AuK currently generates one audio output per request")
        extra = dict(params.extra_args or {})
        prompt = req.prompts[0]
        request = parse_request(prompt, extra)
        waveform = prepare_audio(request.audio, self.sample_rate)
        reference_frames = 0 if waveform is None else waveform.shape[-1] // self.hop_size
        if waveform is not None and reference_frames == 0:
            raise ValueError("Reference audio must contain at least one 20 ms latent frame")
        # Preserve the native operation order at floating-point boundaries.
        target_frames = (
            reference_frames
            if request.seconds is None
            else math.ceil(request.seconds * self.sample_rate / self.hop_size)
        )
        # The public ComfyUI recipe bounds source + target to 30 seconds.
        if target_frames + reference_frames > 1500:
            raise ValueError("AuK supports at most 30 seconds of reference + generated audio per request")
        seed = extra.get("seed", params.seed)
        generator = torch.Generator(device=self.device)
        if seed is None:
            generator.seed()
        elif isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        else:
            generator.manual_seed(seed)
        reference = torch.zeros((1, 0, self.latent_dim), device=self.device, dtype=torch.float32)
        if waveform is not None:
            reference, lengths = self.vae.encoding_and_normalization(
                waveform.unsqueeze(0).to(self.device),
                sample_lengths=torch.tensor([reference_frames * self.hop_size], device=self.device),
                generator=generator,
            )
            reference = reference[:, : min(reference_frames, int(lengths[0]))]
        # Keep diffusion noise independent from stochastic VAE encoding.
        if seed is not None:
            generator.manual_seed(seed)
        noise = torch.randn(
            (1, target_frames, self.latent_dim),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        steps = extra.get("nfe", params.num_inference_steps if params.num_inference_steps is not None else 32)
        times = time_grid(
            flash=self.flash,
            steps=steps,
            sway=extra.get("sway_sampling_coef", -1.0),
            device=self.device,
        )
        cfg = 0.0 if self.flash else extra.get("cfg_strength", 2.0)
        with torch.autocast(self.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
            source_sr = self.sample_rate if request.audio is None else request.audio[1]
            semantic_audio = prepare_audio(request.audio, source_sr)
            semantic, semantic_mask = self.conditioner(request.instruction, semantic_audio, source_sr, self.device)
            latent = sample_latents(self.transformer, noise, reference, semantic, semantic_mask, times, cfg)
        audio = self.vae.inference_from_latents(self.vae.denormalize(latent).transpose(1, 2))
        if not torch.isfinite(audio).all():
            raise RuntimeError("AuK produced non-finite audio")
        return [DiffusionOutput(output=audio.squeeze(0).float())]
