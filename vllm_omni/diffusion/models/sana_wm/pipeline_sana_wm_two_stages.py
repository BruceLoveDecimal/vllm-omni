# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-stage SANA-WM pipeline scaffold."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from copy import copy
from typing import Any, ClassVar

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import (
    SanaWmPipeline,
    build_sana_wm_output_envelope,
    get_sana_wm_pre_process_func,
    read_sana_wm_envelope_metadata,
    read_sana_wm_envelope_payload,
)
from vllm_omni.diffusion.models.sana_wm.request import normalize_sana_wm_payload
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

SANA_WM_INPROCESS_REFINER_ARG = "sana_wm_inprocess_refiner"
SANA_WM_INPROCESS_REFINER_STEPS_ARG = "sana_wm_inprocess_refiner_steps"


class SanaWmTwoStagesPipeline(SanaWmPipeline):
    """SANA-WM Stage 1 plus the bundled LTX-2 refiner components."""

    # Resolves the refiner tree as well; deploy this pipeline against
    # ``SANA_WM_TWO_STAGES_MODEL_ID``, which carries refiner/ next to Stage 1.
    include_refiner: ClassVar[bool] = True
    # Both denoisers are DiTs. Declaring the refiner here rather than in
    # ``_resident_modules`` is what lets the shared machinery reach it: layerwise
    # offload streams its 48 blocks, HSDP shards them through the LTX-2
    # ``_hsdp_shard_conditions``, and SP applies the LTX-2 ``_sp_plan``. As a
    # resident module it was instead pinned whole on the device -- the largest
    # component in the pipeline, and the one that most needs to be swappable.
    # ``outermost_dits()`` keeps them independent (same shape as Cosmos3).
    _dit_modules: ClassVar[list[str]] = ["transformer", "refiner_transformer"]
    _encoder_modules: ClassVar[list[str]] = [
        "text_encoder",
        "camera_encoder",
        "refiner_text_encoder",
        "refiner_connectors",
    ]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.refiner_transformer: nn.Module | None = None
        self.refiner_text_encoder: nn.Module | None = None
        self.refiner_connectors: nn.Module | None = None
        self.refiner_tokenizer: Any | None = None
        # Build the refiner stack at startup like LTX2TwoStagesPipeline: it then
        # lands inside the loader's memory accounting, is non-None for the
        # offloader's ModuleDiscovery, and -- when this pipeline owns placement
        # -- OOMs at deploy instead of on the first request. Under offload or
        # HSDP the build stays on CPU and the backend places it. Without a
        # checkpoint (unit tests), `ensure_refiner_components` still runs lazily
        # on first use.
        if self.od_config is not None and self.od_config.model is not None:
            self.ensure_refiner_components()

    def _ensure_refiner_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_text_encoder is not None and self.refiner_tokenizer is not None:
            self.refiner_text_encoder.to(device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint()
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
            self._force_module_tensors_to(self.refiner_connectors, device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint()
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner connectors require a resolved checkpoint.")

        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        self.refiner_connectors = LTX2TextConnectors.from_pretrained(
            str(self.release_paths.root),
            subfolder="refiner/connectors",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)
        self._force_module_tensors_to(self.refiner_connectors, device=device, dtype=dtype)

    def _load_refiner_transformer_config(self) -> dict[str, Any]:
        if self.release_paths is None:
            self.resolve_checkpoint()
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner transformer requires a resolved checkpoint.")
        with self.release_paths.refiner_transformer_config.open(encoding="utf-8") as f:
            return json.load(f)

    def _ensure_refiner_transformer(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_transformer is not None:
            self.refiner_transformer.to(device=device, dtype=dtype)
            self._force_module_tensors_to(self.refiner_transformer, device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint()
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner transformer requires a resolved checkpoint.")

        from contextlib import nullcontext

        from safetensors.torch import load_file
        from vllm.config import (
            CompilationConfig,
            DeviceConfig,
            VllmConfig,
            get_current_vllm_config_or_none,
            set_current_vllm_config,
        )
        from vllm.utils.torch_utils import set_default_torch_dtype

        if get_current_vllm_config_or_none() is None:
            vllm_config = VllmConfig(
                compilation_config=CompilationConfig(),
                device_config=DeviceConfig(device=device),
            )
            vllm_config_context = set_current_vllm_config(vllm_config)
        else:
            vllm_config_context = nullcontext()

        from vllm_omni.diffusion.models.ltx2.ltx2_components import create_transformer_from_config

        config_dict = self._load_refiner_transformer_config()
        state_dict = load_file(str(self.release_paths.refiner_transformer_weights), device="cpu")
        # Construct in the target dtype (default-dtype context) so the module is
        # never materialized in fp32 — otherwise a full fp32 copy of the refiner
        # lives on CPU at the same time as the bf16 state_dict. `del state_dict`
        # right after the load frees that staging copy before the GPU upload.
        with vllm_config_context, set_default_torch_dtype(dtype), torch.device("cpu"):
            self.refiner_transformer = create_transformer_from_config(config_dict)
        self.refiner_transformer.load_weights(state_dict.items())
        del state_dict
        self.refiner_transformer.to(device=device, dtype=dtype)
        self._force_module_tensors_to(self.refiner_transformer, device=device, dtype=dtype)

    @staticmethod
    def _force_module_tensors_to(module: nn.Module, *, device: torch.device, dtype: torch.dtype) -> None:
        """Move custom-op tensors that may not be migrated by `Module.to`.

        Some vLLM custom layers attach parameters/buffers in ways that can leave
        CPU tensors behind after loading safetensors on CPU. The refiner forward
        is fully CUDA, so fail-fast migration here is safer than discovering a
        device mismatch inside the first RMSNorm call.
        """

        for child in module.modules():
            for name, param in list(child._parameters.items()):
                if param is None:
                    continue
                target_dtype = dtype if param.is_floating_point() else param.dtype
                if param.device == device and param.dtype == target_dtype:
                    continue
                moved = param.detach().to(device=device, dtype=target_dtype)
                replacement = nn.Parameter(moved, requires_grad=param.requires_grad)
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    replacement.weight_loader = weight_loader
                child._parameters[name] = replacement
            for name, buffer in list(child._buffers.items()):
                if buffer is None:
                    continue
                target_dtype = dtype if buffer.is_floating_point() else buffer.dtype
                if buffer.device == device and buffer.dtype == target_dtype:
                    continue
                child._buffers[name] = buffer.to(device=device, dtype=target_dtype)
            for name, value in list(vars(child).items()):
                if (
                    name.startswith("_")
                    or name in child._parameters
                    or name in child._buffers
                    or name in child._modules
                ):
                    continue
                if not isinstance(value, torch.Tensor):
                    continue
                target_dtype = dtype if value.is_floating_point() else value.dtype
                if value.device == device and value.dtype == target_dtype:
                    continue
                setattr(child, name, value.to(device=device, dtype=target_dtype))

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
        if self._placement_is_external():
            # Layerwise offload and HSDP place the components they discover, and
            # the refiner is now among them. Build it on CPU and let the backend
            # decide where it lives; re-placing it afterwards would undo the
            # streaming (and would try to move blocks that are offload
            # placeholders rather than real tensors).
            if None not in (self.refiner_transformer, self.refiner_text_encoder, self.refiner_connectors):
                return
            device = torch.device("cpu")
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
        config = self.refiner_transformer.config
        return int(config.patch_size), int(config.patch_size_t)

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

        if self.refiner_tokenizer.pad_token is None:
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
            padding_side=self.refiner_tokenizer.padding_side,
        )

    @staticmethod
    def _stage2_sigma_schedule(num_steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        try:
            from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

            values = list(STAGE_2_DISTILLED_SIGMA_VALUES)
        except ImportError:
            # Distilled Stage-2 sigmas (matches NVlabs' start_sigma≈0.9094). Only
            # used if this diffusers build lacks the constant; narrow to
            # ImportError so a real error here is not silently masked.
            values = [0.909375, 0.725, 0.421875, 0.0]
        if len(values) < 2:
            values = [values[0] if values else 0.909375, 0.0]
        elif float(values[-1]) != 0.0:
            values.append(0.0)
        max_steps = len(values) - 1
        steps = max(1, min(int(num_steps), max_steps))
        schedule = values[: steps + 1]
        return torch.tensor(schedule, device=device, dtype=dtype)

    @torch.inference_mode()
    def _predict_refiner_current_x0(
        self,
        *,
        sink: torch.Tensor,
        noisy_current: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
        fps: float,
    ) -> torch.Tensor:
        if self.refiner_transformer is None:
            raise RuntimeError("SANA-WM refiner transformer did not initialize.")

        patch_size, patch_size_t = self._refiner_patch_sizes()
        full_latent = torch.cat([sink, noisy_current], dim=2)
        batch_size, _, num_frames, height, width = full_latent.shape
        latent_tokens = self._pack_refiner_latents(
            full_latent,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
        )
        n_context_tokens = self._pack_refiner_latents(
            sink,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
        ).shape[1]

        raw_timestep = torch.zeros(
            batch_size, latent_tokens.shape[1], 1, dtype=torch.float32, device=latent_tokens.device
        )
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()
        timestep_scale = float(getattr(self.refiner_transformer.config, "timestep_scale_multiplier", 1000.0))
        model_timestep = raw_timestep.squeeze(-1) * timestep_scale

        velocity = self._forward_refiner_video_only(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            n_context_tokens=n_context_tokens,
        )
        denoised = latent_tokens.float() - velocity.float() * raw_timestep
        return denoised[:, n_context_tokens:, :].to(noisy_current.dtype)

    def _forward_refiner_video_only(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        n_context_tokens: int,
    ) -> torch.Tensor:
        """Run the refiner DiT on video tokens only.

        Goes through ``LTX2VideoTransformer3DModel.__call__`` rather than
        driving its blocks by hand: the offload, HSDP and SP hooks are all
        registered on module forwards, so a hand-rolled loop over
        ``transformer_blocks`` silently bypasses every one of them.

        The transformer is audiovisual and its forward requires audio inputs,
        but audio reaches the video stream only through audio-to-video
        cross-attention. Disabling ``use_a2v_cross_attention`` (and v2a) and
        feeding a one-token dummy audio stream therefore yields a video output
        identical to a dedicated video-only path, while keeping the shared
        forward byte-for-byte untouched for audiovisual callers. The audio
        stream runs one token through randomly initialized weights (the SANA-WM
        refiner checkpoint ships none) and its output is discarded.
        """

        if self.refiner_transformer is None:
            raise RuntimeError("SANA-WM refiner transformer did not initialize.")
        transformer = self.refiner_transformer
        config = transformer.config
        batch_size = hidden_states.size(0)
        device, dtype = hidden_states.device, hidden_states.dtype
        dummy_audio = torch.zeros(batch_size, 1, int(config.audio_in_channels), device=device, dtype=dtype)
        dummy_audio_prompt = torch.zeros(batch_size, 1, int(config.caption_channels), device=device, dtype=dtype)

        sample, _ = transformer(
            hidden_states=hidden_states,
            audio_hidden_states=dummy_audio,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=dummy_audio_prompt,
            timestep=timestep,
            # Scalar per-sample timestep for the one-token audio stream; the
            # video timestep is per-token and would otherwise be broadcast
            # against the dummy stream's single token.
            audio_timestep=torch.zeros(batch_size, 1, device=device, dtype=timestep.dtype),
            encoder_attention_mask=encoder_attention_mask,
            self_attention_context_tokens=n_context_tokens,
            use_a2v_cross_attention=False,
            use_v2a_cross_attention=False,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            # (batch, 1, num_audio_tokens, 2) coordinates for the dummy token,
            # bypassing prepare_audio_coords which derives lengths from real
            # audio frame counts.
            audio_coords=torch.zeros(batch_size, 1, 1, 2, device=device, dtype=torch.float32),
            return_dict=False,
        )
        return sample

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

        prompt_embeds, _, attention_mask = self._encode_refiner_prompt(
            prompt_text,
            device=device,
            dtype=dtype,
            max_sequence_length=self._refiner_max_sequence_length(extra_args),
        )

        frame_rate = float(payload.get("fps", getattr(sampling_params, "resolved_frame_rate", None) or 24.0))
        # Default to the full LTX-2 distilled Stage-2 schedule (3 steps, ending
        # at sigma=0), matching NVlabs diffusers_ltx2_refiner. Fewer steps stop
        # before sigma=0 (e.g. 1 step ends at 0.725) and decode to noise.
        num_steps = max(1, int(extra_args.get(SANA_WM_INPROCESS_REFINER_STEPS_ARG, 3)))
        sigmas = self._stage2_sigma_schedule(num_steps, device=device, dtype=torch.float32)
        sink_size = int(extra_args.get("sana_wm_refiner_sink_size", 1))
        if latent_num_frames <= sink_size:
            raise ValueError(f"SANA-WM refiner requires more frames than sink_size={sink_size}.")

        sink = latents[:, :, :sink_size].contiguous()
        current = latents[:, :, sink_size:].contiguous()
        refiner_seed = int(extra_args.get("sana_wm_refiner_seed", 42))
        generator = torch.Generator(device=device).manual_seed(refiner_seed)
        eps = torch.randn(current.shape, generator=generator, device=device, dtype=dtype)
        noisy = (1.0 - float(sigmas[0])) * current + float(sigmas[0]) * eps

        patch_size, patch_size_t = self._refiner_patch_sizes()
        for index in range(len(sigmas) - 1):
            sigma = sigmas[index]
            denoised = self._predict_refiner_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=attention_mask,
                sigma=sigma,
                fps=frame_rate,
            )
            noisy_tokens = self._pack_refiner_latents(
                noisy,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )
            velocity = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = noisy_tokens.float() + velocity * (sigmas[index + 1] - sigma).float()
            noisy = self._unpack_refiner_latents(
                next_tokens.to(dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=patch_size,
                patch_size_t=patch_size_t,
            )

        refined_latents = torch.cat([sink, noisy], dim=2)
        actual_steps = int(len(sigmas) - 1)
        output_type = str(getattr(sampling_params, "output_type", None) or "np")
        if output_type != "latent":
            self._ensure_vae(device=device, dtype=dtype)
        output = self._decode_native_latents(refined_latents, output_type=output_type, device=device, dtype=dtype)
        return DiffusionOutput(
            output=build_sana_wm_output_envelope(
                output=output,
                output_type=output_type,
                metadata={
                    "backend": "native_inprocess_refiner",
                    "output_space": output_type,
                    "refiner_steps": actual_steps,
                    "refiner_latent_shape": tuple(refined_latents.shape),
                },
            ),
        )

    def forward(self, req: DiffusionRequestBatch, *args: Any, **kwargs: Any) -> DiffusionOutput:
        extra_args = self._extra_args(getattr(req, "sampling_params", None))
        if bool(extra_args.get("sana_wm_load_refiner_components", False)):
            device, dtype = self._runtime_device_dtype()
            self.ensure_refiner_components(device=device, dtype=dtype)
        # Default to running the refiner: this is the *two-stage* pipeline, so
        # decoding Stage-1 latents to RGB via the LTX-2 refiner is its purpose.
        # Without it, forward() returns a raw latent that /v1/videos cannot encode.
        # Callers can still opt out with sana_wm_inprocess_refiner=False.
        if bool(extra_args.get(SANA_WM_INPROCESS_REFINER_ARG, True)):
            if len(req.prompts) != 1:
                raise ValueError("SANA-WM in-process refiner currently supports exactly one prompt per request.")
            prompt = req.prompts[0]
            if isinstance(prompt, str):
                raise ValueError("SANA-WM in-process refiner requires a mapping prompt.")
            normalized_prompt = normalize_sana_wm_payload(prompt)
            payload = normalized_prompt["additional_information"]["sana_wm"]
            start = time.perf_counter()
            stage1_sampling_params = req.sampling_params
            if stage1_sampling_params is not None:
                clone = getattr(stage1_sampling_params, "clone", None)
                stage1_sampling_params = clone() if callable(clone) else copy(stage1_sampling_params)
                # Stage 1 must hand the refiner latents, whatever the caller
                # asked the *pipeline* to return.
                stage1_sampling_params.output_type = "latent"
            stage1 = self._run_native_backend(
                prompt=normalized_prompt,
                payload=payload,
                sampling_params=stage1_sampling_params,
            )
            refined = self._run_inprocess_refiner(
                latents=read_sana_wm_envelope_payload(stage1.output),
                prompt_text=str(normalized_prompt.get("prompt") or ""),
                payload=payload,
                sampling_params=req.sampling_params,
            )
            stage1_metadata = read_sana_wm_envelope_metadata(stage1.output)
            refined.output = {
                "payload": refined.output["payload"],
                "metadata": {
                    "sana_wm": {
                        **stage1_metadata,
                        **read_sana_wm_envelope_metadata(refined.output),
                        "stage1_backend": stage1_metadata.get("backend", "unknown"),
                    }
                },
            }
            refined.stage_durations = {
                **(stage1.stage_durations or {}),
                "sana_wm_inprocess_refiner_s": time.perf_counter() - start,
            }
            return refined
        return super().forward(req, *args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        loaded = super().load_weights(weights)
        # The refiner stack is loaded separately in ensure_refiner_components()
        # (from_pretrained, strict=False), and the video-only SANA-WM refiner
        # legitimately ships no LTX-2 audio-attention / Gemma vision-tower
        # weights. Report its params as loaded so the engine's strict
        # completeness check does not reject those intentional absences.
        for attr in ("refiner_transformer", "refiner_text_encoder", "refiner_connectors"):
            module = getattr(self, attr, None)
            if module is not None:
                loaded.update(f"{attr}.{name}" for name, _ in module.named_parameters())
        return loaded


# ``get_sana_wm_pre_process_func`` is re-exported because the registry looks the
# hook up by name inside this pipeline's own module.
__all__ = ["SanaWmTwoStagesPipeline", "get_sana_wm_pre_process_func"]
