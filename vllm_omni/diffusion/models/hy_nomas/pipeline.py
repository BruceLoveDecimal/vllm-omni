# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hy-Nomas native segment-autoregressive pipeline for realtime video.

The reference fork is an oracle only; no external Hy-Nomas runtime is imported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Qwen2_5_VLTextModel
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.modality_handlers.camera import CameraSession
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hy_nomas.cache import HyNomasKVCache
from vllm_omni.diffusion.models.hy_nomas.camera import normalize_camera
from vllm_omni.diffusion.models.hy_nomas.decoder import PixelUMMDecoder, PixelUMMStreamingDecodeSession
from vllm_omni.diffusion.models.hy_nomas.interaction import HyNomasPromptSession
from vllm_omni.diffusion.models.hy_nomas.prompt import EncodedPrompt, _action_windows, encode_segment_prompt
from vllm_omni.diffusion.models.hy_nomas.transformer import HyNomasTransformer3DModel
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery, SupportsStepExecution
from vllm_omni.diffusion.offloader.config import TEXT_ENCODER_COMPONENT, OffloadStrategy, resolve_offload
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionKVCacheSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState


@dataclass
class HyNomasSession:
    kv_cache: list[HyNomasKVCache]
    decoder: PixelUMMStreamingDecodeSession | None = None
    frame_offset: int = 0
    segment_token_lengths: list[int] = field(default_factory=list)
    prompt_key: tuple[str, int] | None = None
    encoded_prompt: EncodedPrompt | None = None
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None


def segment_lengths(num_frames: int, segment_frames: int, max_segments: int) -> list[int]:
    if min(num_frames, segment_frames, max_segments) <= 0:
        raise ValueError("Frame counts and session limits must be positive.")
    count = (num_frames + 3) // 4
    lengths = [min(segment_frames, count - start) for start in range(0, count, segment_frames)]
    if len(lengths) > max_segments:
        raise ValueError(f"Request exceeds Hy-Nomas's configured {max_segments} segment history budget.")
    return lengths


class HyNomasPipeline(nn.Module, SupportsComponentDiscovery, SupportsStepExecution, InteractionMixin):
    _decode_in_process: ClassVar[bool] = True
    supports_step_execution: ClassVar[bool] = True
    supports_chunk_step_grouping: ClassVar[bool] = True
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["decoder"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        parallel = od_config.parallel_config
        if any(
            getattr(parallel, name, 1) != 1
            for name in (
                "tensor_parallel_size",
                "sequence_parallel_size",
                "ulysses_degree",
                "ring_degree",
                "cfg_parallel_size",
            )
        ):
            raise ValueError("Hy-Nomas's precision baseline supports one worker; distributed layouts are unqualified.")
        if od_config.quantization_config is not None:
            raise ValueError("Hy-Nomas's precision baseline does not support quantized weights.")
        offload = resolve_offload(od_config)
        if offload.strategy is not OffloadStrategy.NONE and not (
            offload.strategy is OffloadStrategy.MODEL_LEVEL and offload.components == {TEXT_ENCODER_COMPONENT}
        ):
            raise ValueError("Hy-Nomas supports only module-mode text_encoder offload; keep DiT and decoder resident.")
        config = od_config.model_config or {}
        self.height = int(config.get("ar_diffusion_height", 704))
        self.width = int(config.get("ar_diffusion_width", 1280))
        self.segment_frames = int(config.get("hy_nomas_segment_latent_frames", 30))
        self.max_segments = int(config.get("hy_nomas_max_segments", 4))
        self.max_cache_segments = config.get("hy_nomas_max_cache_segments")
        if self.max_cache_segments is not None and (
            type(self.max_cache_segments) is not int or self.max_cache_segments <= 0
        ):
            raise ValueError("hy_nomas_max_cache_segments must be a positive integer or None.")
        if min(self.height, self.width) < 32 or self.height % 32 or self.width % 32:
            raise ValueError("Hy-Nomas height and width must be positive multiples of 32.")
        segment_lengths(1, self.segment_frames, self.max_segments)
        self.max_text_length = 1280
        self._sessions: dict[str, HyNomasSession] = {}
        self._bound_session: str | None = None
        revision = od_config.revision
        model = od_config.model
        tf_config = get_hf_file_to_dict("transformer/config.json", model, revision=revision)
        if tf_config is None:
            raise ValueError("Use the converted Hy-Nomas-Diffusers checkpoint with transformer and decoder configs.")
        self.transformer = HyNomasTransformer3DModel(
            od_config=od_config, **{k: v for k, v in tf_config.items() if not k.startswith("_")}
        )
        if self._decode_in_process:
            decoder_config = get_hf_file_to_dict("decoder/config.json", model, revision=revision)
            if decoder_config is None:
                raise ValueError("Hy-Nomas requires decoder/config.json.")
            self.decoder = PixelUMMDecoder(**{k: v for k, v in decoder_config.items() if not k.startswith("_")})
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", revision=revision)
        self.text_encoder = (
            Qwen2_5_VLTextModel.from_pretrained(
                model,
                subfolder="text_encoder",
                revision=revision,
                torch_dtype=od_config.dtype,
            )
            .to(self.device)
            .eval()
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", revision=revision
        )
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model, subfolder="transformer", prefix="transformer.", revision=revision
            ),
        ]
        if self._decode_in_process:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=model, subfolder="decoder", prefix="decoder.", revision=revision
                )
            )

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        cfg = self.transformer.config
        frame_tokens = (self.height // 32) * (self.width // 32)
        cache_segments = self.max_segments
        if self.max_cache_segments is not None:
            # Denoising sees retained history plus the uncommitted current segment.
            cache_segments = min(cache_segments, self.max_cache_segments + 1)
        tokens = cache_segments * (self.segment_frames * frame_tokens + self.max_text_length)
        element_size = next(self.transformer.parameters()).element_size()
        cache_bytes = (
            cfg.num_layers * tokens * (2 * cfg.num_attention_heads * cfg.attention_head_dim * element_size + 1)
        )
        # Growth and eviction copy one layer's one tensor at a time. They do
        # not overlap; the growth bound also covers eviction compaction.
        cache_bytes += (
            (cache_segments - 1)
            * (self.segment_frames * frame_tokens + self.max_text_length)
            * cfg.num_attention_heads
            * cfg.attention_head_dim
            * element_size
        )
        frames = self.max_segments * self.segment_frames
        # Initial noise, retained decoder latents, decoder pending pixels, and
        # Qwen embeddings. This is a persistent-state bound, not a measured peak;
        # decoder/attention workspace still requires separate GPU qualification.
        latent_bytes = 3 * frames * frame_tokens * cfg.in_channels * element_size
        pixel_bytes = (
            self.decoder.temporal_window_latents * 4 * self.height * self.width * 3 * 4
            if self._decode_in_process
            else 0
        )
        text_bytes = self.max_text_length * cfg.text_embed_dim * element_size
        # One segment's FP32 rotary tables, shared by all denoising/cache steps.
        rotary_bytes = 2 * self.segment_frames * frame_tokens * cfg.attention_head_dim * 4
        return ARDiffusionKVCacheSpec(
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_attention_heads,
            head_size=cfg.attention_head_dim,
            tokens_per_frame=frame_tokens,
            frames_per_block=1,
            window_frames=1,
            kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
            session_capacity=1,
            model_owned_state_bytes_per_session=cache_bytes + latent_bytes + pixel_bytes + text_bytes + rotary_bytes,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState):
        if self._bound_session is not None or state.session_id != session_id:
            raise ValueError("Invalid or nested Hy-Nomas AR session binding.")
        self._bound_session = session_id
        try:
            yield
        finally:
            self._bound_session = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self.reset_ar_diffusion_session(session_id)

    def _session(self, state: StepRequestState) -> HyNomasSession:
        if self._bound_session is not None and self._bound_session != state.request_id:
            raise ValueError("Hy-Nomas request does not match its bound AR session.")
        return self._sessions[state.request_id]

    @torch.no_grad()
    def prepare_encode(self, state: StepRequestState, **kwargs: object) -> StepRequestState:
        sampling = state.sampling
        height, width = (
            (self.height if sampling.height is None else sampling.height),
            (self.width if sampling.width is None else sampling.width),
        )
        if (height, width) != (self.height, self.width):
            raise ValueError("Hy-Nomas requests must match the fixed deployment resolution.")
        if sampling.guidance_scale not in (None, 1.0) or sampling.num_outputs_per_prompt != 1:
            raise ValueError("Hy-Nomas supports CFG=1 and one world per request.")
        steps = 8 if sampling.num_inference_steps is None else sampling.num_inference_steps
        if steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        fps = 24 if sampling.fps is None else sampling.fps
        if fps <= 0:
            raise ValueError("fps must be positive.")
        frames = 120 if sampling.num_frames is None else sampling.num_frames
        lengths = segment_lengths(frames, self.segment_frames, self.max_segments)
        if (
            self._decode_in_process
            and sampling.output_type != "latent"
            and len(lengths) > 1
            and self.segment_frames < self.decoder.temporal_window_latents
        ):
            raise ValueError("Realtime Hy-Nomas segments must cover at least one PixelUMM window.")
        extra_args = sampling.extra_args or {}
        if "ar_diffusion_tick" in extra_args:
            raise ValueError("Hy-Nomas uses one stepwise rollout per request; tick-mode continuations are unsupported.")
        prompts = extra_args.get("segment_prompts")
        if prompts is None:
            value = state.prompt if isinstance(state.prompt, str) else (state.prompt or {}).get("prompt")
            prompts = [value] * len(lengths)
        if (
            not isinstance(prompts, list)
            or len(prompts) != len(lengths)
            or not all(isinstance(p, str) and p for p in prompts)
        ):
            raise ValueError("segment_prompts must contain one nonempty string per segment.")
        max_length = self.max_text_length if sampling.max_sequence_length is None else sampling.max_sequence_length
        if not 1 <= max_length <= self.max_text_length:
            raise ValueError("Hy-Nomas text length must be between 1 and 1280.")
        generator = sampling.generator
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("Hy-Nomas requires one torch.Generator, not a generator list.")
        if generator is None:
            generator = torch.Generator(device=sampling.generator_device or "cpu").manual_seed(
                sampling.seed if sampling.seed is not None else 42
            )
            sampling.generator = generator
        noise = randn_tensor(
            (1, self.transformer.config.in_channels, sum(lengths), height // 32, width // 32),
            generator=generator,
            device=self.device,
            dtype=self.transformer.dtype,
        )
        cameras = extra_args.get("camera_extrinsics")
        if cameras is not None:
            cameras = torch.as_tensor(cameras)
            if cameras.shape != (1, frames, 4, 4) or not torch.isfinite(cameras).all():
                raise ValueError("camera_extrinsics must be normalized native poses with shape [1,num_frames,4,4].")
            pad = sum(lengths) * 4 - frames
            cameras = torch.cat([cameras[:, :1].expand(-1, pad, -1, -1), cameras], dim=1)
        pixels = [length * 4 for length in lengths]
        pixels[-1] -= sum(lengths) * 4 - frames
        capacity_segments = len(lengths)
        if self.max_cache_segments is not None:
            capacity_segments = min(capacity_segments, self.max_cache_segments + 1)
        capacity = sum(lengths[:capacity_segments]) * (height // 32) * (width // 32) + capacity_segments * max_length
        session = HyNomasSession(kv_cache=[HyNomasKVCache(capacity) for _ in self.transformer.double_blocks])
        if self._decode_in_process and sampling.output_type != "latent":
            session.decoder = self.decoder.create_streaming_decode_session(to_cpu=True, retain_history=False)
        self._sessions[state.request_id] = session
        state.total_chunks = len(lengths)
        state.chunk_index = state.step_index = state.step_in_chunk = 0
        state.chunk_num_steps = steps
        state.extra.update(
            initial_noise=noise,
            lengths=lengths,
            pixels=pixels,
            prompts=prompts,
            cameras=cameras,
            max_length=max_length,
            fps=fps,
        )
        # The runner applies initial interaction tracks then calls prepare_next_chunk.
        return state

    @torch.no_grad()
    def prepare_next_chunk(self, state: StepRequestState) -> None:
        if state.request_denoise_completed:
            return
        session = self._session(state)
        index = state.chunk_index
        length = state.extra["lengths"][index]
        start = sum(state.extra["lengths"][:index])
        if session.frame_offset != start:
            raise ValueError("Hy-Nomas chunks must advance contiguously.")
        prompt = state.extra["prompts"][index]
        prompt_session = state.interaction_sessions.get("prompt")
        if isinstance(prompt_session, HyNomasPromptSession) and prompt_session.prompt is not None:
            prompt = prompt_session.prompt
        prompt_key = (prompt, state.extra["max_length"])
        if session.prompt_key != prompt_key:
            encoded = encode_segment_prompt(
                prompt, self.tokenizer, self.text_encoder, self.device, state.extra["max_length"]
            )
            session.prompt_key, session.encoded_prompt = prompt_key, encoded
        encoded = session.encoded_prompt
        assert encoded is not None
        state.prompt_embeds, state.prompt_embeds_mask = encoded.embeds, encoded.mask
        state.extra["encoded"] = encoded
        state.extra["action_windows"] = _action_windows(prompt, length, state.extra["fps"])
        state.latents = state.extra["initial_noise"][:, :, start : start + length].clone()
        session.rotary_emb = self.transformer.rotary_embed(
            length, state.latents.shape[3], state.latents.shape[4], state.latents.device, start
        )
        state.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        sigmas = torch.linspace(1, 0, state.chunk_num_steps + 1)[:-1].tolist()
        state.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        state.timesteps = state.scheduler.timesteps
        state.step_index = state.step_in_chunk = 0
        camera = state.extra["cameras"]
        camera_session = state.interaction_sessions.get("camera")
        received = isinstance(camera_session, CameraSession) and camera_session.has_received_input
        if camera is not None:
            if received:
                raise ValueError("Cannot combine normalized camera_extrinsics with live camera events.")
            camera = camera[:, start * 4 : (start + length) * 4]
        elif received:
            # Native camera preprocessing uses 4k+1 poses; the transformer
            # repeats the first pose three times before packing four per latent.
            camera = normalize_camera(camera_session.last_absolute_poses, (length - 1) * 4 + 1)
        state.extra["segment_camera"] = (
            None if camera is None else camera.to(device=self.device, dtype=self.transformer.dtype)
        )

    def _predict(
        self, state: StepRequestState, latents: torch.Tensor, timestep: torch.Tensor, *, commit: bool = False
    ) -> torch.Tensor:
        session = self._session(state)
        encoded: EncodedPrompt = state.extra["encoded"]
        windows = state.extra["action_windows"]
        return self.transformer(
            hidden_states=latents,
            timestep=timestep.expand(1),
            encoder_hidden_states=encoded.embeds.to(self.transformer.dtype),
            encoder_attention_mask=encoded.mask,
            kv_cache=session.kv_cache,
            update_cache=commit,
            frame_offset=session.frame_offset,
            camera_extrinsics=state.extra["segment_camera"],
            action_text_offsets=[encoded.action_offsets] if windows is not None else None,
            action_frame_windows=windows,
            cache_action_start=encoded.cache_action_start,
            rotary_emb=session.rotary_emb,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def denoise_step(
        self, input_batch: object, *, states: Sequence[StepRequestState] | None = None, **kwargs: object
    ) -> torch.Tensor:
        if states is None or len(states) != 1:
            raise ValueError("Hy-Nomas supports one active world per worker.")
        state = states[0]
        if state.latents is None or state.current_timestep is None:
            raise RuntimeError("Hy-Nomas segment must be prepared before denoising.")
        return self._predict(state, state.latents, state.current_timestep)

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: object) -> None:
        dtype = state.latents.dtype
        state.latents = state.scheduler.step(
            noise_pred.float(), state.current_timestep, state.latents.float(), return_dict=False
        )[0].to(dtype)
        state.step_in_chunk += 1
        state.step_index += 1

    @torch.no_grad()
    def post_decode(self, state: StepRequestState, **kwargs: object) -> DiffusionOutput:
        session = self._session(state)
        latents = state.latents
        if latents is None:
            raise RuntimeError("Cannot commit an empty Hy-Nomas segment.")
        # No later segment reads the final segment's KV, so skip its cache-write forward.
        if state.chunk_index + 1 < state.total_chunks:
            sigma = 0.206897
            noise = randn_tensor(
                latents.shape, generator=state.sampling.generator, device=latents.device, dtype=latents.dtype
            )
            cache_latents = ((1 - sigma) * latents.float() + sigma * noise.float()).to(latents.dtype)
            # Match native causal_ar.py cache_write: denoising reads all retained
            # segments; evict whole segments only before committing this segment.
            # This optional bounded history is not equivalent to full-history output.
            if self.max_cache_segments is not None:
                while len(session.segment_token_lengths) >= self.max_cache_segments:
                    tokens = session.segment_token_lengths.pop(0)
                    for layer in session.kv_cache:
                        layer.evict(tokens)
            history = session.kv_cache[0].length
            self._predict(state, cache_latents, torch.tensor(sigma * 1000, device=latents.device), commit=True)
            # Commits keep only visible tokens, so record what was actually retained.
            session.segment_token_lengths.append(session.kv_cache[0].length - history)
        session.rotary_emb = None
        completed = state.chunk_index
        session.frame_offset += latents.shape[2]
        state.chunk_index += 1
        if not self._decode_in_process or state.sampling.output_type == "latent":
            output = latents
        else:
            if session.decoder is None:
                raise RuntimeError("PixelUMM session was not prepared.")
            parts = [session.decoder.push(latents, pixel_frames=state.extra["pixels"][completed])]
            if state.request_denoise_completed:
                parts.append(session.decoder.finish())
            parts = [part for part in parts if part.numel()]
            output = torch.cat(parts, dim=2) if parts else torch.empty(1, 3, 0, self.height, self.width)
        return DiffusionOutput(
            output=output,
            chunk_index=completed,
            total_chunks=state.total_chunks,
            finished=state.request_denoise_completed,
        )

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        return ChunkMediaSpec(
            num_media_frames=state.extra["pixels"][state.chunk_index],
            fps=state.extra["fps"],
            num_latent_frames=state.extra["lengths"][state.chunk_index],
        )

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.requests) != 1:
            raise ValueError("Hy-Nomas only supports one world per request batch.")
        request = req.requests[0]
        if self._bound_session is not None:
            raise ValueError(
                "Hy-Nomas AR serving requires step_execution; use the standard engine for offline forward."
            )
        state = StepRequestState(request_id=request.request_id, prompt=request.prompt, sampling=request.sampling_params)
        chunks = []
        try:
            self.prepare_encode(state)
            while not state.request_denoise_completed:
                self.prepare_next_chunk(state)
                while not state.chunk_denoise_completed:
                    prediction = self.denoise_step(None, states=[state])
                    self.step_scheduler(state, prediction)
                chunks.append(self.post_decode(state))
            return self._combine_chunk_outputs(chunks)
        finally:
            self.close_ar_diffusion_session(state.request_id)

    def _combine_chunk_outputs(self, chunks: list[DiffusionOutput]) -> DiffusionOutput:
        return DiffusionOutput(output=torch.cat([chunk.output for chunk in chunks], dim=2))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


def get_hy_nomas_post_process_func(od_config: OmniDiffusionConfig):
    processor = VideoProcessor(vae_scale_factor=32)

    def post_process(video: torch.Tensor, output_type: str = "np", sampling_params=None):
        if sampling_params is not None:
            output_type = sampling_params.output_type or output_type
        if output_type == "latent":
            return video
        if video.shape[2] == 0:
            return []
        frames = processor.postprocess_video(video, output_type=output_type)
        return frames[0] if isinstance(frames, list) else frames

    return post_process


def get_hy_nomas_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        sampling = request.sampling_params
        if sampling.generator is None and sampling.generator_device is None:
            sampling.generator_device = "cpu"
        return request

    return pre_process
