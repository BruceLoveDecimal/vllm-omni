# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM distilled Stage-1 realtime streaming pipeline.

Serves the chunk-causal ``SANA-WM_streaming`` release through the
step-execution contract (``SupportsStepExecution``) so the existing
``WS /v1/realtime/video`` transport streams one fragmented-MP4 chunk per
generated latent block instead of waiting for the whole clip. Everything
request-scoped — the GDN recurrence states, the softmax K/V of the cached
chunks, the temporal-conv tails, the latent history and the camera
trajectory — lives in ``StepRequestState.extra``; one request is one session.

Design: ``docs/design/feature/sana_wm_realtime_streaming.md``. The
bidirectional :class:`SanaWmPipeline` is inherited for loading, encoders,
VAE and camera handling and is otherwise untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.ltx2.ltx2_latents import resolve_video_latent_shape
from vllm_omni.diffusion.models.sana_wm.camera_control import (
    SanaWmCameraCondition,
    build_plucker_condition,
)
from vllm_omni.diffusion.models.sana_wm.config import (
    SANA_WM_VAE_SPATIAL_COMPRESSION,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import (
    SANA_WM_NATIVE_MAX_TOKENS,
    SanaWmPipeline,
    build_sana_wm_output_envelope,
    get_sana_wm_pre_process_func,
)
from vllm_omni.diffusion.models.sana_wm.request import normalize_sana_wm_payload
from vllm_omni.diffusion.models.sana_wm.self_forcing import (
    SanaWmSelfForcingSchedule,
    create_autoregressive_segments,
    self_forcing_euler_step,
)
from vllm_omni.diffusion.models.sana_wm.streaming_cache import SanaWmStreamingCache
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

logger = init_logger(__name__)

__all__ = [
    "SANA_WM_STREAMING_DEFAULT_NUM_FRAMES",
    "SANA_WM_STREAMING_MODEL_ID",
    "SanaWmStreamingPipeline",
    "get_sana_wm_pre_process_func",
]

# Diffusers-layout conversion of ``Efficient-Large-Model/SANA-WM_streaming``
# (``sana_dit/model.pt`` generator weights + the Stage-1 VAE), produced by
# ``tools/convert_sana_wm_streaming_to_diffusers.py``.
SANA_WM_STREAMING_MODEL_ID = "BBBBruce/SANA-WM_streaming-stage1-diffusers"
# 8 * chunk_size * k + 1 pixel frames: the nearest valid clip length at or
# above the bidirectional default of 161.
SANA_WM_STREAMING_DEFAULT_NUM_FRAMES = 169
SANA_WM_STREAMING_DEFAULT_GUIDANCE_SCALE = 1.0

_EXTRA_KEYS_TO_FREE = (
    "cache",
    "camera_full",
    "chunk_camera",
    "history_latents",
    "noise_full",
    "first_latent",
)


def _valid_num_frames_hint(num_frames: int, chunk_size: int) -> str:
    stride = SANA_WM_VAE_TEMPORAL_COMPRESSION * chunk_size
    below = ((num_frames - 1) // stride) * stride + 1
    candidates = sorted({max(below, stride + 1), below + stride, below + 2 * stride})
    return ", ".join(str(candidate) for candidate in candidates)


def _drop_first_frame(video: Any) -> Any:
    """Drop the leading pixel frame of a post-processed chunk.

    ``VideoProcessor.postprocess_video`` puts the frame axis at index 1 for
    ``np``/``pt`` batches and returns a list of per-video frame lists for
    ``pil``; all three keep the batch axis outermost.
    """
    if isinstance(video, (torch.Tensor,)) or hasattr(video, "shape"):
        return video[:, 1:]
    if isinstance(video, list):
        return [item[1:] if isinstance(item, (list, tuple)) or hasattr(item, "shape") else item for item in video]
    raise TypeError(f"Sana-WM streaming cannot drop the overlap frame from output type {type(video).__name__}.")


class SanaWmStreamingPipeline(SanaWmPipeline):
    """Chunk-causal SANA-WM Stage-1 served through step execution.

    Per chunk the model runs ``len(denoising_step_list) - 1`` denoising
    forwards with the cache read-only, then one clean forward at ``t = 0``
    with ``save_cache=True`` (NVlabs self-forcing recipe). Chunks are decoded
    with a one-latent-frame overlap so the pixel frame counts add up to
    ``num_frames`` without a streaming VAE decoder (design §8).
    """

    supports_step_execution: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig | None = None, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if od_config is not None and od_config.model is not None:
            if not self.sana_wm_config.streaming:
                raise ValueError(
                    "SanaWmStreamingPipeline needs a chunk-causal checkpoint whose transformer/config.json sets "
                    f"streaming=true (expected a conversion of Efficient-Large-Model/SANA-WM_streaming such as "
                    f"{SANA_WM_STREAMING_MODEL_ID}); {od_config.model!r} is a bidirectional release, serve it "
                    "with SanaWmPipeline instead."
                )
            parallel_config = getattr(od_config, "parallel_config", None)
            cfg_parallel_size = int(getattr(parallel_config, "cfg_parallel_size", 1) or 1)
            if cfg_parallel_size > 1:
                raise ValueError(
                    "SanaWmStreamingPipeline serves the distilled student at guidance_scale=1.0 and does not "
                    f"support cfg_parallel_size > 1 (got {cfg_parallel_size})."
                )
        # Validate the schedule at startup so a bad config fails before the
        # first request.
        self.self_forcing_schedule = SanaWmSelfForcingSchedule.from_config(self.sana_wm_config)

    # ------------------------------------------------------------------
    # Request mode is not supported: the causal checkpoint must not run
    # through the bidirectional loop.
    # ------------------------------------------------------------------

    def forward(self, req: DiffusionRequestBatch, *args: Any, **kwargs: Any) -> DiffusionOutput:
        del req, args, kwargs
        raise NotImplementedError(
            "SanaWmStreamingPipeline requires step execution; start the server with "
            "--diffusion-streaming-output and use WS /v1/realtime/video."
        )

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        prompt = state.prompt
        if prompt is None or isinstance(prompt, str):
            raise ValueError("Sana-WM requires a mapping prompt with first-frame image and camera/action metadata.")
        # The registered pre-process hook already normalised the payload; the
        # call is idempotent and covers offline callers that bypass it.
        prompt = normalize_sana_wm_payload(prompt)
        state.prompt = prompt
        payload = prompt["additional_information"]["sana_wm"]
        sampling = state.sampling
        extra_args = self._extra_args(sampling)
        config = self.sana_wm_config
        device, dtype = self._runtime_device_dtype()

        height = int(getattr(sampling, "height", None) or payload["height"])
        width = int(getattr(sampling, "width", None) or payload["width"])
        num_frames = int(getattr(sampling, "num_frames", None) or payload["num_frames"])
        if num_frames <= 1:
            num_frames = int(payload["num_frames"])

        if getattr(sampling, "guidance_scale_provided", False):
            guidance_scale = float(getattr(sampling, "guidance_scale", 1.0) or 1.0)
            if guidance_scale > 1.0:
                raise ValueError(
                    "Sana-WM streaming serves the distilled self-forcing student without classifier-free "
                    f"guidance; guidance_scale must be <= 1.0, got {guidance_scale}."
                )
        schedule = self.self_forcing_schedule
        schedule.check_num_inference_steps(getattr(sampling, "num_inference_steps", None))

        latent_frames, latent_height, latent_width = resolve_video_latent_shape(
            height,
            width,
            num_frames,
            vae_spatial_compression_ratio=SANA_WM_VAE_SPATIAL_COMPRESSION,
            vae_temporal_compression_ratio=SANA_WM_VAE_TEMPORAL_COMPRESSION,
        )
        chunk_size = int(config.chunk_size)
        if latent_frames <= chunk_size or (latent_frames - 1) % chunk_size != 0:
            raise ValueError(
                f"Sana-WM streaming generates {chunk_size} latent frames per chunk after the conditioning frame, so "
                f"num_frames must be {SANA_WM_VAE_TEMPORAL_COMPRESSION * chunk_size}k+1 with k >= 1 "
                f"(e.g. {_valid_num_frames_hint(num_frames, chunk_size)}); got {num_frames}."
            )
        boundaries = create_autoregressive_segments(latent_frames, chunk_size)
        total_chunks = len(boundaries) - 1
        token_count = latent_frames * latent_height * latent_width
        max_tokens = int(extra_args.get("sana_wm_native_max_tokens", SANA_WM_NATIVE_MAX_TOKENS))
        if token_count > max_tokens:
            raise ValueError(
                "Sana-WM latent token count exceeds the configured cap. "
                f"Requested latent tokens={token_count}, max={max_tokens}. Request a smaller "
                "`height`/`width`/`num_frames`, or raise the cap via `sana_wm_native_max_tokens`."
            )

        first_frame_image = (prompt.get("multi_modal_data") or {}).get("image")
        import numpy as _np

        if not (hasattr(first_frame_image, "convert") or isinstance(first_frame_image, (_np.ndarray, torch.Tensor))):
            raise TypeError(
                "Sana-WM first-frame image must be a PIL Image, numpy ndarray, or "
                f"torch.Tensor; got {type(first_frame_image).__name__}."
            )

        prompt_embeds = self._native_prompt_embeds(prompt, device=device, dtype=dtype)
        prompt_attention_mask = self._last_prompt_attention_mask
        first_latent = self._vae_encode_first_frame(
            first_frame_image,
            height=height,
            width=width,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
        )

        camera = payload.get("camera") or {}
        condition = SanaWmCameraCondition(
            poses=camera.get("poses") if isinstance(camera, dict) else None,
            intrinsics=payload.get("intrinsics"),
            action=payload.get("action"),
            num_frames=num_frames,
            height=height,
            width=width,
            translation_speed=float(payload.get("translation_speed", 0.05)),
            rotation_speed_deg=float(payload.get("rotation_speed_deg", 1.2)),
        )
        camera_tensors = build_plucker_condition(condition)
        camera_full = {
            "plucker": camera_tensors["chunk_plucker"].to(device=device, dtype=dtype),
            "raymap": camera_tensors["raymap"].to(device=device, dtype=dtype),
            "spatial_raymap": (
                camera_tensors["spatial_raymap"].to(device=device, dtype=dtype)
                if camera_tensors.get("spatial_raymap") is not None
                else None
            ),
        }
        if camera_full["raymap"].shape[0] != latent_frames:
            raise ValueError(
                f"Sana-WM camera trajectory covers {camera_full['raymap'].shape[0]} latent frames, "
                f"the request needs {latent_frames}."
            )

        generator = getattr(sampling, "generator", None)
        if generator is None:
            seed = int(getattr(sampling, "seed", None) or extra_args.get("seed", 0) or 0)
            generator = torch.Generator(device=device).manual_seed(seed)
            if sampling is not None:
                try:
                    sampling.generator = generator
                except AttributeError:
                    pass
        # NVlabs draws the noise for the whole clip once; each chunk then
        # takes its slice, so the RNG stream is independent of the chunking.
        # The runner hands out a generator on the worker device; a caller that
        # asked for another generator_device gets its noise drawn there.
        noise_device = generator.device if isinstance(generator, torch.Generator) else device
        noise_full = torch.randn(
            (1, first_latent.shape[1], latent_frames, latent_height, latent_width),
            device=noise_device,
            dtype=dtype,
            generator=generator,
        ).to(device=device)

        cache = SanaWmStreamingCache.new(
            num_blocks=len(self.transformer.blocks),
            chunk_size=chunk_size,
            num_cached_blocks=int(config.num_cached_blocks),
            sink_token=bool(config.sink_token),
        )

        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_attention_mask
        state.negative_prompt_embeds = None
        state.negative_prompt_embeds_mask = None
        state.do_true_cfg = False
        state.chunk_index = 0
        state.step_in_chunk = 0
        state.total_chunks = total_chunks
        state.extra.update(
            {
                "backend": "native_gdn_streaming",
                "boundaries": boundaries,
                "cache": cache,
                "camera_full": camera_full,
                "chunk_size": chunk_size,
                "device": device,
                "dtype": dtype,
                "first_latent": first_latent.to(dtype=dtype),
                "generator": generator,
                "height": height,
                "history_latents": first_latent.to(dtype=dtype).clone(),
                "latent_frames": latent_frames,
                "latent_height": latent_height,
                "latent_width": latent_width,
                "noise_full": noise_full,
                "num_frames": num_frames,
                "output_type": getattr(sampling, "output_type", None) or "np",
                "schedule": schedule,
                "width": width,
            }
        )
        self._prepare_next_chunk(state)
        return state

    def _prepare_next_chunk(self, state: StepRequestState) -> None:
        extra = state.extra
        boundaries: list[int] = extra["boundaries"]
        chunk_index = state.chunk_index
        chunk_start, chunk_end = boundaries[chunk_index], boundaries[chunk_index + 1]
        # Chunk 0 carries the clean conditioning frame inside the forward
        # (frame 0 at t = 0); only the frames after it are generated.
        gen_start = chunk_start + 1 if chunk_index == 0 else chunk_start
        extra["chunk_range"] = (chunk_start, chunk_end)
        extra["gen_range"] = (gen_start, chunk_end)
        extra["frame_index"] = torch.arange(chunk_start, chunk_end, dtype=torch.long, device=extra["device"])
        camera_full = extra["camera_full"]
        extra["chunk_camera"] = {
            "plucker": camera_full["plucker"][:, chunk_start:chunk_end],
            "raymap": camera_full["raymap"][chunk_start:chunk_end],
            "spatial_raymap": (
                camera_full["spatial_raymap"][:, chunk_start:chunk_end]
                if camera_full["spatial_raymap"] is not None
                else None
            ),
        }
        state.latents = extra["noise_full"][:, :, gen_start:chunk_end].clone()
        schedule: SanaWmSelfForcingSchedule = extra["schedule"]
        state.timesteps = schedule.timesteps_tensor(extra["device"])
        state.chunk_num_steps = schedule.num_steps
        state.step_in_chunk = 0
        state.step_index = 0

    def _chunk_model_inputs(
        self,
        state: StepRequestState,
        latents: torch.Tensor,
        timestep_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble ``(hidden_states, per-frame timestep)`` for one forward.

        For chunk 0 the clean conditioning latent is prepended and held at
        ``t = 0``; every generated frame carries ``timestep_value``.
        """
        extra = state.extra
        if state.chunk_index == 0:
            latents = torch.cat([extra["first_latent"].to(latents.dtype), latents], dim=2)
        frames = latents.shape[2]
        timestep = timestep_value.to(device=latents.device, dtype=torch.float32).reshape(1, 1, 1)
        timestep = timestep.expand(latents.shape[0], 1, frames).clone()
        if state.chunk_index == 0:
            timestep[:, :, 0] = 0.0
        return latents, timestep

    def _run_transformer(
        self,
        state: StepRequestState,
        latents: torch.Tensor,
        timestep_value: torch.Tensor,
        *,
        save_cache: bool,
    ) -> torch.Tensor:
        extra = state.extra
        hidden_states, timestep = self._chunk_model_inputs(state, latents, timestep_value)
        chunk_camera = extra["chunk_camera"]
        velocity = self.transformer.forward_streaming(
            hidden_states,
            timestep,
            frame_index=extra["frame_index"],
            cache=extra["cache"],
            save_cache=save_cache,
            encoder_hidden_states=state.prompt_embeds,
            encoder_attention_mask=state.prompt_embeds_mask,
            plucker=chunk_camera["plucker"],
            raymap=chunk_camera["raymap"],
            spatial_raymap=chunk_camera["spatial_raymap"],
        )
        if state.chunk_index == 0:
            velocity = velocity[:, :, 1:]
        return velocity

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del kwargs
        if states is None:
            states = input_batch.states
        if len(states) != 1:
            raise ValueError("Sana-WM streaming step execution supports a single request, not a batched request.")
        state = states[0]
        timestep = state.current_timestep
        if timestep is None:
            raise RuntimeError("Sana-WM streaming denoise_step called without a current timestep.")
        latents = state.latents.to(dtype=state.extra["dtype"])
        return self._run_transformer(state, latents, timestep, save_cache=False)

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        if noise_pred is None:
            raise RuntimeError("Sana-WM streaming step_scheduler called without a noise prediction.")
        schedule: SanaWmSelfForcingSchedule = state.extra["schedule"]
        sigma, sigma_next = schedule.sigma_pair(state.step_in_chunk)
        state.latents = self_forcing_euler_step(state.latents, noise_pred, sigma=sigma, sigma_next=sigma_next)
        state.step_in_chunk += 1
        state.step_index = state.step_in_chunk

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        extra = state.extra
        cache: SanaWmStreamingCache = extra["cache"]
        device, dtype = extra["device"], extra["dtype"]
        completed_chunk = state.chunk_index
        clean_latents = state.latents.to(dtype=dtype)

        with torch.no_grad():
            # Cache write: one clean forward at t = 0 (NVlabs ``save_kv_cache``).
            self._run_transformer(
                state,
                clean_latents,
                torch.zeros((), device=device, dtype=torch.float32),
                save_cache=True,
            )
            chunk_start, chunk_end = extra["chunk_range"]
            cache.commit(chunk_end - chunk_start)

            history = torch.cat([extra["history_latents"], clean_latents], dim=2)
            extra["history_latents"] = history

            output_type = extra["output_type"]
            chunk_size = extra["chunk_size"]
            if output_type == "latent":
                output = clean_latents
                num_pixel_frames = None
            else:
                # Overlap decode (design §8): the previous chunk's last latent
                # frame gives the decoder temporal context; its pixel frame is
                # dropped so the counts add up to num_frames.
                decode_latents = history[:, :, -(chunk_size + 1) :]
                output = self._decode_native_latents(
                    decode_latents, output_type=output_type, device=device, dtype=dtype
                )
                if completed_chunk > 0:
                    output = _drop_first_frame(output)
                num_pixel_frames = self._count_pixel_frames(output)

        envelope = build_sana_wm_output_envelope(
            output=output,
            output_type=output_type,
            metadata={
                "backend": extra["backend"],
                "output_space": output_type,
                "chunk_index": completed_chunk,
                "total_chunks": state.total_chunks,
                "frame_index": list(range(*extra["chunk_range"])),
                "generated_latent_frames": list(range(*extra["gen_range"])),
                "num_pixel_frames": num_pixel_frames,
                "cache_bytes": cache.nbytes(),
                "cached_latent_frames": cache.cached_frames(),
                "num_frames": extra["num_frames"],
                "height": extra["height"],
                "width": extra["width"],
                "sampling_steps": extra["schedule"].num_steps,
            },
        )

        state.chunk_index += 1
        finished = state.request_denoise_completed
        if not finished:
            self._prepare_next_chunk(state)
        else:
            self._release_request_state(state)
        return DiffusionOutput(
            output=envelope,
            chunk_index=completed_chunk,
            total_chunks=state.total_chunks,
            finished=finished,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else {},
        )

    @staticmethod
    def _count_pixel_frames(output: Any) -> int | None:
        if hasattr(output, "shape") and len(output.shape) >= 2:
            return int(output.shape[1])
        if isinstance(output, list) and output and isinstance(output[0], (list, tuple)):
            return len(output[0])
        return None

    @staticmethod
    def _release_request_state(state: StepRequestState) -> None:
        for key in _EXTRA_KEYS_TO_FREE:
            state.extra.pop(key, None)
        state.extra.pop("frame_index", None)
