# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-stage SANA-WM pipeline scaffold."""

from __future__ import annotations

import json
import time
from copy import copy
from pathlib import Path
from typing import Any, ClassVar, Iterable

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import (
    SanaWmPipeline,
    get_sana_wm_pre_process_func,
    get_sana_wm_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

SANA_WM_INPROCESS_REFINER_ARG = "sana_wm_inprocess_refiner"
SANA_WM_INPROCESS_REFINER_STEPS_ARG = "sana_wm_inprocess_refiner_steps"


class SanaWmTwoStagesPipeline(SanaWmPipeline):
    """SANA-WM Stage 1 plus the bundled LTX-2 refiner components."""

    include_refiner: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = [
        "text_encoder",
        "camera_encoder",
        "refiner_text_encoder",
        "refiner_connectors",
    ]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["refiner_transformer"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.refiner_transformer: nn.Module | None = None
        self.refiner_text_encoder: nn.Module | None = None
        self.refiner_connectors: nn.Module | None = None
        self.refiner_tokenizer: Any | None = None

    def _local_files_only(self) -> bool:
        if self.release_paths is not None:
            return True
        if self.od_config is None or self.od_config.model is None:
            return False
        return Path(str(self.od_config.model)).expanduser().exists()

    def _ensure_refiner_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_text_encoder is not None and self.refiner_tokenizer is not None:
            self.refiner_text_encoder.to(device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner text encoder requires a resolved checkpoint.")

        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        local_path = str(self.release_paths.refiner_text_encoder_dir)
        self.refiner_tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        with torch.device("cpu"):
            self.refiner_text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device)

    def _ensure_refiner_connectors(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_connectors is not None:
            self.refiner_connectors.to(device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner connectors require a resolved checkpoint.")

        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        self.refiner_connectors = LTX2TextConnectors.from_pretrained(
            str(self.release_paths.root),
            subfolder="refiner/connectors",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)

    def _load_refiner_transformer_config(self) -> dict[str, Any]:
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner transformer requires a resolved checkpoint.")
        with self.release_paths.refiner_transformer_config.open(encoding="utf-8") as f:
            return json.load(f)

    def _ensure_refiner_transformer(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_transformer is not None:
            self.refiner_transformer.to(device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner transformer requires a resolved checkpoint.")

        from safetensors.torch import load_file
        from contextlib import nullcontext

        from vllm.config import (
            CompilationConfig,
            DeviceConfig,
            VllmConfig,
            get_current_vllm_config,
            set_current_vllm_config,
        )
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import create_transformer_from_config

        try:
            get_current_vllm_config()
            vllm_config_context = nullcontext()
        except AssertionError:
            vllm_config = VllmConfig(
                compilation_config=CompilationConfig(),
                device_config=DeviceConfig(device=device),
            )
            vllm_config_context = set_current_vllm_config(vllm_config)

        with vllm_config_context, torch.device("cpu"):
            self.refiner_transformer = create_transformer_from_config(self._load_refiner_transformer_config())
        state_dict = load_file(str(self.release_paths.refiner_transformer_weights), device="cpu")
        if hasattr(self.refiner_transformer, "load_weights"):
            self.refiner_transformer.load_weights(state_dict.items())
        else:
            self.refiner_transformer.load_state_dict(state_dict, strict=False)
        self.refiner_transformer.to(device=device, dtype=dtype)

    def ensure_refiner_components(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Load the LTX-2 19B refiner components from the SANA-WM release tree."""

        runtime_device, runtime_dtype = self._runtime_device_dtype()
        device = device or runtime_device
        dtype = dtype or runtime_dtype
        self._ensure_refiner_text_encoder(device=device, dtype=dtype)
        self._ensure_refiner_connectors(device=device, dtype=dtype)
        self._ensure_refiner_transformer(device=device, dtype=dtype)

    @staticmethod
    def _pack_refiner_latents(
        latents: torch.Tensor,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_refiner_latents(
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    def _refiner_patch_sizes(self) -> tuple[int, int]:
        config = getattr(self.refiner_transformer, "config", None)
        patch_size = int(getattr(config, "patch_size", 1))
        patch_size_t = int(getattr(config, "patch_size_t", 1))
        return patch_size, patch_size_t

    def _refiner_max_sequence_length(self, extra_args: dict[str, Any]) -> int:
        if "sana_wm_refiner_max_sequence_length" in extra_args:
            return int(extra_args["sana_wm_refiner_max_sequence_length"])
        tokenizer_max_length = getattr(self.refiner_tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_length, int) and tokenizer_max_length < 100000:
            return tokenizer_max_length
        encoder_config = getattr(self.refiner_text_encoder, "config", None)
        config_max_len = getattr(encoder_config, "max_position_embeddings", None)
        if config_max_len is None:
            config_max_len = getattr(encoder_config, "max_seq_len", None)
        return int(config_max_len or 1024)

    def _encode_refiner_prompt(
        self,
        prompt_text: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.refiner_tokenizer is None or self.refiner_text_encoder is None or self.refiner_connectors is None:
            raise RuntimeError("SANA-WM refiner prompt encoding requires loaded refiner components.")

        if getattr(self.refiner_tokenizer, "pad_token", None) is None:
            self.refiner_tokenizer.pad_token = self.refiner_tokenizer.eos_token
        self.refiner_tokenizer.padding_side = "left"

        encoded = self.refiner_tokenizer(
            [prompt_text.strip()],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.refiner_text_encoder(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=-1).flatten(2, 3).to(dtype=dtype)
        return self.refiner_connectors(
            hidden_states,
            encoded.attention_mask.to(device),
            padding_side=getattr(self.refiner_tokenizer, "padding_side", "left"),
        )

    @staticmethod
    def _stage2_sigmas(num_steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        try:
            from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

            values = list(STAGE_2_DISTILLED_SIGMA_VALUES)
        except Exception:
            values = [0.7, 0.5, 0.3]
        if not values:
            values = [0.7]
        if num_steps <= len(values):
            values = values[:num_steps]
        else:
            values = values + [values[-1]] * (num_steps - len(values))
        return torch.tensor(values, device=device, dtype=dtype)

    def _run_inprocess_refiner(
        self,
        *,
        latents: torch.Tensor,
        prompt_text: str,
        payload: dict[str, Any],
        sampling_params: Any | None,
    ) -> DiffusionOutput:
        extra_args = self._extra_args(sampling_params)
        device, dtype = self._runtime_device_dtype()
        self.ensure_refiner_components(device=device, dtype=dtype)
        if self.refiner_transformer is None:
            raise RuntimeError("SANA-WM refiner transformer did not initialize.")

        latents = latents.to(device=device, dtype=dtype)
        latent_num_frames = int(latents.shape[2])
        latent_height = int(latents.shape[3])
        latent_width = int(latents.shape[4])
        patch_size, patch_size_t = self._refiner_patch_sizes()
        packed_latents = self._pack_refiner_latents(latents, patch_size=patch_size, patch_size_t=patch_size_t)

        prompt_embeds, audio_prompt_embeds, attention_mask = self._encode_refiner_prompt(
            prompt_text,
            device=device,
            dtype=dtype,
            max_sequence_length=self._refiner_max_sequence_length(extra_args),
        )

        config = getattr(self.refiner_transformer, "config", None)
        audio_channels = int(getattr(config, "audio_in_channels", 128))
        audio_tokens = int(extra_args.get("sana_wm_refiner_audio_tokens", 1))
        audio_latents = torch.zeros(
            packed_latents.shape[0],
            audio_tokens,
            audio_channels,
            device=device,
            dtype=packed_latents.dtype,
        )

        frame_rate = float(payload.get("fps", getattr(sampling_params, "resolved_frame_rate", None) or 24.0))
        video_coords = None
        if hasattr(self.refiner_transformer, "rope"):
            video_coords = self.refiner_transformer.rope.prepare_video_coords(
                packed_latents.shape[0],
                latent_num_frames,
                latent_height,
                latent_width,
                device,
                fps=frame_rate,
            )
        audio_coords = None
        if hasattr(self.refiner_transformer, "audio_rope"):
            audio_coords = self.refiner_transformer.audio_rope.prepare_audio_coords(
                audio_latents.shape[0],
                audio_tokens,
                device,
            )

        num_steps = max(1, int(extra_args.get(SANA_WM_INPROCESS_REFINER_STEPS_ARG, 1)))
        sigmas = self._stage2_sigmas(num_steps, device=device, dtype=packed_latents.dtype)
        for index, sigma in enumerate(sigmas):
            timestep = (sigma * 1000.0).expand(packed_latents.shape[0])
            transformer_output = self.refiner_transformer(
                hidden_states=packed_latents,
                audio_hidden_states=audio_latents,
                encoder_hidden_states=prompt_embeds,
                audio_encoder_hidden_states=audio_prompt_embeds,
                timestep=timestep,
                audio_timestep=timestep,
                encoder_attention_mask=attention_mask,
                audio_encoder_attention_mask=attention_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                fps=frame_rate,
                audio_num_frames=audio_tokens,
                video_coords=video_coords,
                audio_coords=audio_coords,
                return_dict=False,
            )
            if not isinstance(transformer_output, tuple) or len(transformer_output) < 2:
                raise RuntimeError("SANA-WM refiner transformer must return video and audio noise predictions.")
            noise_pred_video, noise_pred_audio = transformer_output[:2]
            next_sigma = sigmas[index + 1] if index + 1 < len(sigmas) else torch.zeros_like(sigma)
            delta = sigma - next_sigma
            packed_latents = packed_latents - delta * noise_pred_video.to(packed_latents.dtype)
            audio_latents = audio_latents - delta * noise_pred_audio.to(audio_latents.dtype)

        refined_latents = self._unpack_refiner_latents(
            packed_latents,
            latent_num_frames,
            latent_height,
            latent_width,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
        )
        output_type = str(extra_args.get("sana_wm_refiner_output_type", extra_args.get("sana_wm_output_type", "np")))
        if output_type != "latent":
            self._ensure_vae(device=device, dtype=dtype)
        output = self._decode_native_smoke_latents(refined_latents, output_type=output_type, device=device, dtype=dtype)
        return DiffusionOutput(
            output=output,
            custom_output={
                "sana_wm_backend": "native_inprocess_refiner",
                "sana_wm_output_space": output_type,
                "sana_wm_refiner_steps": num_steps,
                "sana_wm_refiner_latent_shape": tuple(refined_latents.shape),
            },
        )

    def forward(self, req: OmniDiffusionRequest, *args: Any, **kwargs: Any) -> DiffusionOutput:
        extra_args = self._extra_args(getattr(req, "sampling_params", None))
        if bool(extra_args.get("sana_wm_load_refiner_components", False)):
            device, dtype = self._runtime_device_dtype()
            self.ensure_refiner_components(device=device, dtype=dtype)
        if bool(extra_args.get(SANA_WM_INPROCESS_REFINER_ARG, False)):
            if len(req.prompts) != 1:
                raise ValueError("SANA-WM in-process refiner currently supports exactly one prompt per request.")
            prompt = req.prompts[0]
            if isinstance(prompt, str):
                raise ValueError("SANA-WM in-process refiner requires a mapping prompt.")
            from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import normalize_sana_wm_payload

            normalized_prompt = normalize_sana_wm_payload(prompt)
            payload = normalized_prompt["additional_information"]["sana_wm"]
            start = time.perf_counter()
            stage1_sampling_params = req.sampling_params
            if stage1_sampling_params is not None:
                clone = getattr(stage1_sampling_params, "clone", None)
                stage1_sampling_params = clone() if callable(clone) else copy(stage1_sampling_params)
                stage1_extra_args = dict(getattr(stage1_sampling_params, "extra_args", None) or {})
                stage1_extra_args["sana_wm_output_type"] = "latent"
                stage1_sampling_params.extra_args = stage1_extra_args
            stage1 = self._run_native_smoke_backend(
                prompt=normalized_prompt,
                payload=payload,
                sampling_params=stage1_sampling_params,
            )
            if not isinstance(stage1.output, torch.Tensor):
                raise RuntimeError("SANA-WM in-process refiner requires Stage-1 latent tensor output.")
            refined = self._run_inprocess_refiner(
                latents=stage1.output,
                prompt_text=str(normalized_prompt.get("prompt") or ""),
                payload=payload,
                sampling_params=req.sampling_params,
            )
            refined.custom_output = {
                **(stage1.custom_output or {}),
                **(refined.custom_output or {}),
                "sana_wm_stage1_backend": stage1.custom_output.get("sana_wm_backend")
                if stage1.custom_output
                else "unknown",
            }
            refined.stage_durations = {
                **(stage1.stage_durations or {}),
                "sana_wm_inprocess_refiner_s": time.perf_counter() - start,
            }
            return refined
        return super().forward(req, *args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        return super().load_weights(weights)


__all__ = ["SanaWmTwoStagesPipeline", "get_sana_wm_pre_process_func", "get_sana_wm_post_process_func"]
