# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM-5B Stage2 image/camera-to-video through the Omni request contract."""

import html
import json
import math
from pathlib import Path
from typing import ClassVar

import ftfy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.platforms import current_omni_platform

from .components import load_text_encoder, load_transformer, load_vae
from .transformer import SolarWMCache

logger = init_logger(__name__)


def prepare_camera(path, latent_frames, device):
    """Read absolute per-pixel C2W or latent-aligned relative W2C; default static."""
    views = torch.eye(4, dtype=torch.float32).repeat(latent_frames, 1, 1)
    if path is not None:
        with np.load(path, allow_pickle=False) as data:
            if "viewmats" in data:
                array = data["viewmats"]
                if array.shape != (latent_frames, 4, 4):
                    raise ValueError(f"viewmats must have shape ({latent_frames},4,4)")
                views = torch.from_numpy(array.copy()).float()
            elif "c2w" in data:
                c2w = np.asarray(data["c2w"], dtype=np.float32)
                indices = np.array([0, *(1 + 4 * i for i in range(latent_frames - 1))])
                if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) <= indices[-1]:
                    raise ValueError("c2w must cover the complete generated pixel horizon")
                c2w = c2w[indices]
                # Reference computes C2W rebasing in NumPy, then W2C inversion
                # in torch FP32; preserve that operation order.
                rotation = c2w[0, :3, :3].T
                origin = np.eye(4, dtype=np.float32)
                origin[:3, :3] = rotation
                origin[:3, 3] = -rotation @ c2w[0, :3, 3]
                relative = np.stack([np.eye(4, dtype=np.float32), *(origin @ frame for frame in c2w[1:])])
                relative = torch.from_numpy(relative)
                views[:, :3, :3] = relative[:, :3, :3].transpose(-1, -2)
                views[:, :3, 3] = -torch.einsum("...ij,...j->...i", views[:, :3, :3], relative[:, :3, 3])
            else:
                raise ValueError("Camera NPZ needs c2w or viewmats")
    if not torch.isfinite(views).all():
        raise ValueError("Camera matrices must be finite")
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(latent_frames, 1, 1)
    intrinsics[:, 0, 0] = 969.6969696969696 / (960 * 2)
    intrinsics[:, 1, 1] = 969.6969696969696 / (540 * 2)
    intrinsics[:, :2, 2] = 0.5
    return views.unsqueeze(0).to(device), intrinsics.unsqueeze(0).to(device)


def prepare_image(image, height, width, device):
    if isinstance(image, list) and len(image) == 1:
        image = image[0]
    if not isinstance(image, Image.Image):
        raise ValueError("SolarWM needs one PIL first-frame image")
    pixels = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float()[None] / 255
    source_height, source_width = pixels.shape[-2:]
    scale = max(height / source_height, width / source_width)
    resized = (round(source_height * scale), round(source_width * scale))
    if resized != (source_height, source_width):
        pixels = F.interpolate(pixels, size=resized, mode="bilinear", align_corners=False)
    top, left = (resized[0] - height) // 2, (resized[1] - width) // 2
    return (pixels[:, :, top : top + height, left : left + width] * 2 - 1).unsqueeze(2).to(device)


class SolarWMStage2Pipeline(nn.Module, SupportImageInput, SupportsComponentDiscovery, ProgressBarMixin):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config, prefix=""):
        super().__init__()
        self.device = get_local_device()
        self.dtype = torch.bfloat16
        pc = od_config.parallel_config
        if (
            any(
                getattr(pc, name, 1) != 1
                for name in (
                    "tensor_parallel_size",
                    "ulysses_degree",
                    "ring_degree",
                    "allgather_degree",
                    "cfg_parallel_size",
                    "pipeline_parallel_size",
                    "vae_patch_parallel_size",
                )
            )
            or pc.use_hsdp
            or pc.data_parallel_size not in (None, 1)
        ):
            raise ValueError("SolarWM currently supports single-device execution only")
        if (
            od_config.enable_cpu_offload
            or od_config.enable_layerwise_offload
            or getattr(od_config, "enable_distributed_layerwise_offload", False)
        ):
            raise ValueError("SolarWM currently requires resident components")
        if getattr(od_config, "vae_use_slicing", False) or getattr(od_config, "vae_use_tiling", False):
            raise ValueError("SolarWM currently requires untiled, unsliced temporal VAE decoding")
        if od_config.quantization_config is not None:
            raise ValueError("SolarWM quantization is not supported")
        if od_config.cache_backend is not None and od_config.cache_backend != "none":
            raise ValueError("SolarWM Stage2 does not support diffusion block caching")
        root = Path(od_config.model)
        index = json.loads((root / "model_index.json").read_text())
        base, stage = (root / index["base_path"]).resolve(), (root / index["stage_path"]).resolve()
        self.transformer = load_transformer(base, stage, self.device, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(base / "tokenizer", local_files_only=True)
        self.text_encoder = load_text_encoder(base, self.device)
        self.vae = load_vae(base, self.device)
        self.weights_sources = []

    def load_weights(self, weights):
        # Native checkpoint components were strictly loaded above.
        if next(iter(weights), None) is not None:
            raise ValueError("SolarWM expects only its manifest-selected native weights")
        return {name for name, _ in self.named_parameters()}

    @torch.no_grad()
    def forward(self, req):
        sampling = req.sampling_params
        if len(req.prompts) != 1 or (sampling.num_outputs_per_prompt or 1) != 1:
            raise ValueError("SolarWM supports one video per request")
        if sampling.num_inference_steps != 4:
            raise ValueError("Released SolarWM Stage2 requires four denoising evaluations per chunk")
        if sampling.guidance_scale not in (None, 1.0):
            raise ValueError("SolarWM's distilled student uses guidance_scale=1")
        prompt = req.prompts[0]
        if not isinstance(prompt, dict):
            raise ValueError("SolarWM requires prompt and multi_modal_data.image")
        height = sampling.height if sampling.height is not None else 480
        width = sampling.width if sampling.width is not None else 864
        requested_frames = int(sampling.num_frames if sampling.num_frames is not None else 81)
        if requested_frames < 1 or min(height, width) <= 0 or height % 32 or width % 32:
            raise ValueError("Positive frame count and spatial dimensions divisible by 32 are required")
        latent_frames = math.ceil((requested_frames + 3) / 12) * 3
        views, intrinsics = prepare_camera((sampling.extra_args or {}).get("camera_path"), latent_frames, self.device)
        pixels = prepare_image((prompt.get("multi_modal_data") or {}).get("image"), height, width, self.device)
        text = " ".join(html.unescape(html.unescape(ftfy.fix_text(prompt.get("prompt", "")))).split())
        tokens = self.tokenizer([text], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        context = self.text_encoder(**tokens).last_hidden_state
        context = context.masked_fill(~tokens["attention_mask"].bool().unsqueeze(-1), 0).to(self.dtype)
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device).view(1, 48, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=self.device).view(1, 48, 1, 1, 1)
        first = ((self.vae.encode(pixels).latent_dist.mode() - mean) * (1 / std)).to(self.dtype)
        generator = sampling.generator
        if not isinstance(generator, torch.Generator):
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed or 0)
        if generator.device != self.device:
            # Request transport can reconstruct a CPU generator. The released
            # student draws noise on the accelerator; preserve the request seed.
            generator = torch.Generator(device=self.device).manual_seed(generator.initial_seed())
        # Match the reference's BTCHW random draw order, including future chunks.
        shape = (1, latent_frames, 48, height // 16, width // 16)
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        noise[:, 0] = first[:, :, 0]
        output = torch.empty_like(noise)
        cache = SolarWMCache()
        # Build the reference CPU schedule and retain its original sigmas.
        # Recovering sigma by dividing the raw timestep on CUDA changes BF16
        # rounding ties in re-noising, which compounds over a long rollout.
        grid = torch.linspace(1, 0, 1001, dtype=torch.float32)[:-1]
        grid = 5 * grid / (1 + 4 * grid)
        sigmas = grid[[0, 250, 500, 750]]
        steps = (sigmas * 1000).to(self.device)
        sigmas = sigmas.to(self.device)
        with self.progress_bar(total=latent_frames // 3) as progress:
            for start in range(0, latent_frames, 3):
                latent = noise[:, start : start + 3].clone()
                for index, step in enumerate(steps):
                    times = step.expand(1, 3).clone()
                    if start == 0:
                        times[:, 0] = 0
                    flow = self.transformer(
                        latent.permute(0, 2, 1, 3, 4),
                        times,
                        context,
                        views[:, start : start + 3],
                        intrinsics[:, start : start + 3],
                        cache=cache,
                        start_frame=start,
                    ).permute(0, 2, 1, 3, 4)
                    x0 = (latent.float() - times[..., None, None, None] * flow.float() / 1000).to(self.dtype)
                    if start == 0:
                        x0[:, 0] = first[:, :, 0]
                    latent = x0
                    if index < 3:
                        sigma = sigmas[index + 1].expand(1, 3).clone()
                        if start == 0:
                            sigma[:, 0] = 0
                        renoise = torch.randn(latent.shape, generator=generator, device=self.device, dtype=self.dtype)
                        sigma = sigma[..., None, None, None]
                        latent = ((1 - sigma) * x0.float() + sigma * renoise.float()).to(self.dtype)
                if not torch.isfinite(latent).all():
                    raise RuntimeError(f"Non-finite SolarWM output at latent frame {start}")
                output[:, start : start + 3] = latent
                self.transformer(
                    latent.permute(0, 2, 1, 3, 4),
                    torch.zeros(1, 3, device=self.device),
                    context,
                    views[:, start : start + 3],
                    intrinsics[:, start : start + 3],
                    cache=cache,
                    start_frame=start,
                    commit=True,
                )
                progress.update()
        del cache, noise
        if sampling.output_type == "latent":
            return DiffusionOutput(output=output.permute(0, 2, 1, 3, 4))
        decoded = []

        def consume(chunk):
            if not torch.isfinite(chunk).all():
                raise RuntimeError("Non-finite SolarWM VAE output")
            decoded.append(((chunk.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu())

        with current_omni_platform.create_autocast_context(
            device_type=self.device.type, dtype=self.dtype, enabled=True
        ):
            z = output.permute(0, 2, 1, 3, 4)
            z = z / (1 / std.to(z.dtype)) + mean.to(z.dtype)
            self.vae.decode_with_chunks(z, on_chunk=consume)
        frames = torch.cat(decoded, dim=2)[:, :, :requested_frames].permute(0, 2, 3, 4, 1).contiguous()
        logger.info(
            "SolarWM generated %d new frames (%dx%d), %d latent chunks",
            frames.shape[1],
            width,
            height,
            latent_frames // 3,
        )
        return DiffusionOutput(output=frames)


def get_solarwm_post_process_func(od_config):
    def post_process(video, output_type="np", sampling_params=None):
        output_type = getattr(sampling_params, "output_type", None) or output_type
        if output_type == "latent":
            return video
        if output_type == "pil":
            result = [[Image.fromarray(frame) for frame in clip] for clip in video.numpy()]
        elif output_type == "pt":
            result = video
        else:
            result = video.numpy()
        return {"payload": {"video": result}, "metadata": {}}

    return post_process
