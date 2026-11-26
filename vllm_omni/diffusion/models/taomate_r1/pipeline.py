# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-R1: reference-conditioned streaming MiniMax-H3 Ref2VA with two LoRA banks.

TaoMate-R1 (``TaoLiveAIGC/TaoMate-R1``) turns the MiniMax-H3 Ref2VA partition
into a causal streamer that generates audio and video jointly from a prompt
and ordered image references:

* the session's frame count is fixed at start; its whole Ref2VA packed
  sequence (text, references, target audio/video) is built once, and every
  chunk gathers its rows from it without resetting RoPE;
* chunk 0 (39 frames) runs the opening ("cold") adapter, every later chunk
  (34 frames, or a 17-frame tail) the continuation ("warm") adapter;
* each chunk takes three Euler steps at sigma indices (0, 16, 33, 49) of the
  50-point shifted schedule; reference rows keep the 0.999 condition timestep;
* every chunk except the last recomputes its clean (sigma = 0) audio/video K/V
  with the same adapter and commits it to the AR-Diffusion history (the first
  chunk's video as sink plus the two latest chunks).

The streaming attention, persistent KV policy and incremental decoders are
TaoMate-H3's; this pipeline replaces the conditioning, chunk plan, adapters
and the denoise update (no audio teacher, no prompt interaction).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import torch
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.taomate_h3.attention import StreamContext, StreamMode, stream_context
from vllm_omni.diffusion.models.taomate_h3.geometry import AUDIO_SAMPLE_RATE, CanvasGeometry
from vllm_omni.diffusion.models.taomate_h3.kv_cache import TaoMateKVHistory
from vllm_omni.diffusion.models.taomate_h3.lora import prepare_taomate_lora
from vllm_omni.diffusion.models.taomate_h3.pipeline import TaoMateH3Pipeline, use_cuda_fp8_activation_quant
from vllm_omni.diffusion.models.taomate_h3.schedule import student_sigmas
from vllm_omni.diffusion.models.taomate_h3.stream_decode import (
    StreamingAudioDecoder,
    StreamingVideoDecoder,
    samples_at_frame,
)
from vllm_omni.diffusion.models.taomate_h3.transformer import local_embed_plan
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState, StepWait
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionSegmentedKVSpec,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

from .chunks import (
    VIDEO_FPS,
    Bank,
    R1Chunk,
    R1SessionPlan,
    bank_for_chunk,
    compact_chunk_layout,
    plan_session,
)
from .conditioning import build_encoder_inputs, extract_reference_images, resize_references
from .lora import read_taomate_r1_banks

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

STUDENT_STEPS = 3
# Per-chunk noise seeds follow the release: torch.Generator(seed + chunk * 97),
# drawing the chunk's video rows before its audio rows.
_CHUNK_SEED_STRIDE = 97
DEFAULT_SEED = 42
DEFAULT_HEIGHT = 1344
DEFAULT_WIDTH = 768
# Ten seconds at 24 fps aligns to 243 frames (seven chunks), the release default.
DEFAULT_SESSION_FRAMES = 240
DEFAULT_MAX_SESSION_FRAMES = 60 * VIDEO_FPS
_VIDEO_ROW_WIDTH = 96
_AUDIO_ROW_WIDTH = 32
_AR_BRANCH = "main"
# Largest chunk: the opening chunk's 12 video latents and 65 audio latents per channel.
_MAX_CHUNK_VIDEO_LATENTS = 12
_MAX_CHUNK_AUDIO_LATENTS = 66
# Initial staging for the text + reference condition rows; staging grows on demand.
_INITIAL_CONDITION_TOKENS = 8192


@dataclass
class _ChunkState:
    chunk: R1Chunk
    branch: MiniMaxH3DenoiseBranch
    condition_rows: torch.Tensor
    media_rows: torch.Tensor
    seq_len: int
    condition_span: tuple[int, int]
    media_span: tuple[int, int]
    started_at: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)


class _R1Session:
    """Model-owned state of one TaoMate-R1 session."""

    def __init__(
        self,
        *,
        session_id: str,
        canvas: CanvasGeometry,
        seed: int,
        plan: R1SessionPlan,
        packed: dict[str, Any],
        reference_rows: int,
        anchors: torch.Tensor,
        text_embeddings: torch.Tensor,
        cache: Any,
        video_decoder: StreamingVideoDecoder,
        audio_decoder: StreamingAudioDecoder,
    ) -> None:
        self.session_id = session_id
        self.canvas = canvas
        self.seed = int(seed)
        self.plan = plan
        self.packed = packed
        self.reference_rows = int(reference_rows)
        self.anchors = anchors
        self.text_embeddings = text_embeddings
        self._cache = cache
        self.history = TaoMateKVHistory(cache)
        self.video_decoder = video_decoder
        self.audio_decoder = audio_decoder
        self.sigmas_video, self.sigmas_audio = student_sigmas()
        self.chunk: _ChunkState | None = None
        self.audio_velocity: torch.Tensor | None = None
        self.last_chunk_end: float | None = None

    @property
    def cache(self) -> ARDiffusionSegmentedKVState:
        return self._cache()

    def begin_chunk(self, index: int, *, device: torch.device) -> _ChunkState:
        chunk = self.plan.chunks[index]
        layout = compact_chunk_layout(
            self.packed,
            reference_rows=self.reference_rows,
            chunk=chunk,
            frame_rows=self.canvas.frame_rows,
            audio_latent_t=self.plan.audio_latent_t,
        )
        branch = MiniMaxH3DenoiseBranch(
            packed=layout,
            text_embeddings=self.text_embeddings,
            token_tags=layout["token_tags"],
            device=device,
        )
        text_rows = int(layout["text_pos"].numel())
        condition_count = text_rows + self.reference_rows
        media_count = int(layout["audio_pos"].numel()) + chunk.video_latent_count * self.canvas.frame_rows
        self.cache.prepare_layer_views(media_tokens=media_count, condition_tokens=condition_count)
        self.chunk = _ChunkState(
            chunk=chunk,
            branch=branch,
            condition_rows=torch.arange(condition_count, device=device),
            media_rows=torch.arange(condition_count, condition_count + media_count, device=device),
            seq_len=int(layout["seq_len"]),
            condition_span=(0, condition_count),
            media_span=(condition_count, media_count),
        )
        self.audio_velocity = None
        return self.chunk

    def initial_noise(self, chunk: R1Chunk, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(self.seed + chunk.index * _CHUNK_SEED_STRIDE)
        video = torch.randn(
            (chunk.video_latent_count * self.canvas.frame_rows, _VIDEO_ROW_WIDTH),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        audio = torch.randn(
            (2 * chunk.audio_latent_count, _AUDIO_ROW_WIDTH), device=device, dtype=torch.float32, generator=generator
        )
        return video, audio

    def stream_context(self, mode: StreamMode) -> StreamContext:
        chunk = self.chunk
        assert chunk is not None
        return StreamContext(
            cache=self.cache,
            mode=mode,
            condition_rows=chunk.condition_rows,
            media_rows=chunk.media_rows,
            seq_len=chunk.seq_len,
            condition_span=chunk.condition_span,
            media_span=chunk.media_span,
        )

    def close(self) -> None:
        self.history.clear()
        self.video_decoder.reset()
        self.audio_decoder.reset()
        self.chunk = None
        self.audio_velocity = None


def _single_lora_path(od_config: OmniDiffusionConfig) -> str | None:
    lora_path = getattr(od_config, "lora_path", None)
    if isinstance(lora_path, (list, tuple)):
        lora_path = lora_path[0] if len(lora_path) == 1 else None
    return str(lora_path) if lora_path else None


class TaoMateR1Pipeline(TaoMateH3Pipeline):
    """Streaming TaoMate-R1 (Ref2VA, opening/continuation banks) on the AR-Diffusion runtime."""

    # Video and audio velocities are both consumed in step_scheduler.
    denoise_step_updates_state: ClassVar[bool] = False
    diffusion_lora_fixed_adapter: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if self.partition != "ref2va":
            raise ValueError(
                f"TaoMate-R1 serves the MiniMax-H3 Ref2VA partition; point the model at MiniMax-H3/Ref2VA "
                f"(got partition {self.partition!r})"
            )
        if not self.load_vae_encoder:
            raise ValueError("TaoMate-R1 encodes its image references locally; keep the video VAE encoder loaded")
        if self._tm_lora_merge:
            raise ValueError("TaoMate-R1 does not support taomate_h3_lora_merge")
        model_config = dict(getattr(od_config, "model_config", None) or {})
        height = int(model_config.get("taomate_r1_height", DEFAULT_HEIGHT))
        width = int(model_config.get("taomate_r1_width", DEFAULT_WIDTH))
        self._tm_default_height, self._tm_default_width = height, width
        self._tm_default_canvas = CanvasGeometry(height=height, width=width)
        self._tm_default_seed = int(model_config.get("taomate_r1_seed", DEFAULT_SEED))
        self._r1_default_frames = int(model_config.get("taomate_r1_default_frames", DEFAULT_SESSION_FRAMES))
        self._r1_max_frames = int(model_config.get("taomate_r1_max_frames", DEFAULT_MAX_SESSION_FRAMES))
        if not 0 < self._r1_default_frames <= self._r1_max_frames:
            raise ValueError("taomate_r1_default_frames must be positive and at most taomate_r1_max_frames")
        # Three chunks visit the cold bank, the warm bank and the terminal (commit-free) chunk.
        self._r1_warmup_chunks = int(model_config.get("taomate_r1_warmup_chunks", 3))
        if self._r1_warmup_chunks < 1:
            raise ValueError("taomate_r1_warmup_chunks must be at least 1")
        # Parity debugging: save each chunk's final rows (output rank only) as
        # <dir>/<session>/chunk_<i>.pt. Adds a device-to-host copy per chunk.
        dump_dir = model_config.get("taomate_r1_dump_latents_dir")
        self._r1_dump_latents_dir = Path(str(dump_dir)) if dump_dir else None
        self._r1_active_bank: Bank = "cold"
        self._r1_bank_dirs: dict[Bank, Path] = {}
        self._r1_prepared: dict[Bank, tuple[LoRAModel, PEFTHelper]] = {}
        self._r1_sessions: dict[str, _R1Session] = {}
        # Re-log the decoder tile grid for the R1 canvas.
        self._apply_vae_decoder_tiling()

    # -- weights and LoRA banks ---------------------------------------------

    def load_weights(self, weights):  # type: ignore[override]
        # Skip TaoMate-H3's single-adapter and audio-teacher setup.
        loaded = MiniMaxH3Pipeline.load_weights(self, weights)
        lora_path = _single_lora_path(self.od_config)
        if lora_path is None:
            raise ValueError("TaoMate-R1 requires --lora-path pointing at the TaoLiveAIGC/TaoMate-R1 directory")
        self._r1_bank_dirs = read_taomate_r1_banks(lora_path)
        for role, bank_dir in self._r1_bank_dirs.items():
            model, peft, _ = prepare_taomate_lora(
                bank_dir, transformer=self.transformer, device=self.device, dtype=torch.bfloat16, merge=False
            )
            self._r1_prepared[role] = (model, peft)
        if self._tm_fp8_quant_cuda_op and self.device.type == "cuda":
            switched = use_cuda_fp8_activation_quant(self.transformer)
            if switched:
                logger.info("TaoMate-R1: %d FP8 linears quantize activations with the CUDA op", switched)
        if not self._tm_adaln_cache:
            cache = getattr(self.transformer, "adaln_cache", None)
            if cache is not None and hasattr(cache, "max_bytes"):
                cache.max_bytes = 0
        return loaded

    def diffusion_lora_fixed_banks(self, lora_path: str) -> dict[str, str]:
        """Bank adapter directories for the LoRA manager (see ``LoRAManager._fixed_bank_paths``)."""
        banks = self._r1_bank_dirs or read_taomate_r1_banks(lora_path)
        return {role: str(path) for role, path in banks.items()}

    def diffusion_lora_active_bank(self) -> str:
        return self._r1_active_bank

    def _select_bank(self, bank: Bank) -> None:
        self._r1_active_bank = bank
        if self._tm_lora_manager is not None:
            # Fixed-bank managers resolve the request adapter to the active bank.
            self._tm_lora_manager.set_request_adapter(None)

    def _load_diffusion_lora_adapter(  # type: ignore[override]
        self, *, lora_request: LoRARequest, lora_path: Any, dtype: torch.dtype
    ) -> tuple[LoRAModel, PEFTHelper]:
        del dtype
        requested = Path(str(lora_path or lora_request.lora_path)).expanduser().resolve()
        for role, bank_dir in self._r1_bank_dirs.items():
            if bank_dir == requested:
                prepared = self._r1_prepared.get(role)
                if prepared is None:
                    raise RuntimeError(f"TaoMate-R1 {role} bank was not prepared during weight loading")
                model, peft = prepared
                return model.clone(lora_request.lora_int_id), peft
        raise ValueError(
            f"TaoMate-R1 serves only its fixed cold/warm banks; {requested} is not one of "
            f"{sorted(str(path) for path in self._r1_bank_dirs.values())}"
        )

    def _create_diffusion_merged_lora(self, *, lora_model: Any, modules: Any, scale: float) -> None:  # type: ignore[override]
        del lora_model, modules, scale
        return None

    def _bind_diffusion_lora_manager(self, manager: DiffusionLoRAManager | None) -> None:  # type: ignore[override]
        self._tm_lora_manager = manager
        # The manager holds its own copies of both banks now.
        self._r1_prepared = {}

    # -- AR-Diffusion capability ----------------------------------------------

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:  # type: ignore[override]
        canvas = self._tm_default_canvas
        chunk_rows = _MAX_CHUNK_VIDEO_LATENTS * canvas.frame_rows + 2 * _MAX_CHUNK_AUDIO_LATENTS
        # The opening chunk's video is the sink; the two latest chunks carry audio and video.
        retained_rows = _MAX_CHUNK_VIDEO_LATENTS * canvas.frame_rows + 2 * chunk_rows
        decoder_bytes = (
            24 * 20 * canvas.latent_h * canvas.latent_w * 4 + 3 * 5 * canvas.height * canvas.width * 4 + 64 * 1024**2
        )
        audio_bytes = 2 * _AUDIO_ROW_WIDTH * 4 * (2 * _MAX_CHUNK_AUDIO_LATENTS + 48 + 24)
        anchor_bytes = _INITIAL_CONDITION_TOKENS * _VIDEO_ROW_WIDTH * 4
        return ARDiffusionKVCacheSpec(
            num_layers=int(self.transformer.arch.num_layers),
            num_kv_heads=int(self.transformer.streaming_local_heads),
            head_size=int(self.transformer.arch.attention_head_dim),
            tokens_per_frame=canvas.frame_rows,
            frames_per_block=1,
            window_frames=1,
            kv_branches=(ARDiffusionKVBranchSpec(_AR_BRANCH, 0),),
            session_capacity=1,
            segmented=ARDiffusionSegmentedKVSpec(
                block_size=256,
                max_retained_tokens=retained_rows,
                max_retained_segments=5,
                max_pending_tokens=chunk_rows,
                max_pending_segments=2,
                initial_staging_tokens=retained_rows + chunk_rows + _INITIAL_CONDITION_TOKENS,
            ),
            model_owned_state_bytes_per_session=decoder_bytes + audio_bytes + anchor_bytes,
        )

    def _release_session(self, session_id: str) -> None:  # type: ignore[override]
        session = self._r1_sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def _session(self, state: StepRequestState) -> _R1Session:  # type: ignore[override]
        key = state.extra.get("taomate_session_id") or self._bound_session_id(state)
        session = self._r1_sessions.get(key)
        if session is None:
            raise RuntimeError(f"TaoMate-R1 has no session {key!r} for request {state.request_id!r}")
        return session

    def ar_diffusion_step_wait(self, state: StepRequestState) -> StepWait | None:  # type: ignore[override]
        del state
        return None

    def ar_diffusion_resume_step(  # type: ignore[override]
        self, state: StepRequestState, reason: Literal["interaction", "timeout"]
    ) -> None:
        raise RuntimeError(f"TaoMate-R1 never waits between chunks (resume reason {reason!r})")

    def ar_diffusion_warmup_requests(self, session_id: str) -> Iterator[OmniDiffusionRequest]:  # type: ignore[override]
        """One throwaway session over both banks and the terminal chunk, with two synthetic references."""
        frames = 39 + 34 * (self._r1_warmup_chunks - 1)
        size = (512, 512)
        references = [
            Image.new("RGB", size, (200, 120, 60)),
            Image.fromarray(np.tile(np.linspace(0, 255, size[0], dtype=np.uint8)[None, :, None], (size[1], 1, 3))),
        ]
        self._tm_warmup_session_ids.add(session_id)
        yield OmniDiffusionRequest(
            prompt={
                "prompt": "The presenter from <Picture 1> holds the object from <Picture 2> and describes it.",
                "multi_modal_data": {"image": references},
            },
            sampling_params=OmniDiffusionSamplingParams(
                num_frames=frames,
                height=self._tm_default_height,
                width=self._tm_default_width,
                seed=self._tm_default_seed,
                extra_args={"session_id": session_id, "reset": True},
            ),
            request_id=f"taomate-r1-warmup-{session_id}",
        )

    # -- request mode -------------------------------------------------------

    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:  # type: ignore[override]
        """Generate the whole session in one runner call and return the joined media."""
        if request.num_reqs != 1:
            raise OmniClientError("TaoMate-R1 request mode serves one request at a time")
        req = request.requests[0]
        state = StepRequestState(request_id=req.request_id, sampling=req.sampling_params, prompt=req.prompt)
        self.prepare_encode(state)
        self.prepare_next_chunk(state)
        frames: list[np.ndarray] = []
        audio: list[np.ndarray] = []
        chunk_metadata: list[dict[str, Any]] = []
        while not state.request_denoise_completed:
            for _ in range(STUDENT_STEPS):
                check_request_cancellation()
                velocity = self.denoise_step(cast(InputBatch, None), states=[state])
                self.step_scheduler(state, velocity)
            last = self.post_decode(state)
            envelope = cast(dict[str, Any], last.output)
            frames.append(envelope["payload"]["video"])
            audio.append(envelope["payload"]["audio"])
            chunk_metadata.append(envelope["metadata"]["taomate_r1"])
            if not state.request_denoise_completed:
                self.prepare_next_chunk(state)
        session_id = self._bound_session_id(state)
        video = np.concatenate([item for item in frames if item.shape[0]], axis=0) if frames else np.zeros((0, 1, 1, 3))
        waveform = np.concatenate(audio, axis=0) if audio else np.zeros((0, 2), np.float32)
        return DiffusionOutput(
            output={
                "payload": {"video": video, "audio": waveform},
                "metadata": {
                    "video": {"fps": float(VIDEO_FPS)},
                    "audio": {"sample_rate": AUDIO_SAMPLE_RATE},
                    "ar_diffusion": ARDiffusionChunkMetadata(
                        session_id=session_id,
                        request_id=req.request_id,
                        chunk_index=state.chunk_index - 1,
                        applied_event_ids=(),
                    ).to_dict(),
                    "taomate_r1": {"chunks": chunk_metadata},
                },
            },
            chunk_index=state.chunk_index - 1,
            total_chunks=state.total_chunks,
            finished=True,
        )

    # -- step execution -----------------------------------------------------

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:  # type: ignore[override]
        del kwargs
        self._require_bound_ar_state()
        bound = self._ar_diffusion_kv_state
        assert bound is not None
        sampling = state.sampling
        if int(getattr(sampling, "num_outputs_per_prompt", 1) or 1) != 1:
            raise OmniClientError("TaoMate-R1 produces one stream per session; num_outputs_per_prompt must be 1")
        for name in ("fps", "frame_rate"):
            rate = getattr(sampling, name, None)
            if rate is not None and rate != VIDEO_FPS:
                raise OmniClientError(
                    f"TaoMate-R1 generates synchronized audio/video at {VIDEO_FPS} fps; {name} must be {VIDEO_FPS}"
                )
        prompt_text, multi_modal_data = self._extract_prompt(state.prompt)
        images = resize_references(extract_reference_images(multi_modal_data))
        height = int(getattr(sampling, "height", None) or self._tm_default_height)
        width = int(getattr(sampling, "width", None) or self._tm_default_width)
        try:
            canvas = CanvasGeometry(height=height, width=width)
        except ValueError as exc:
            raise OmniClientError(str(exc)) from exc
        if canvas != self._tm_default_canvas:
            raise OmniClientError(
                f"TaoMate-R1 was deployed for {self._tm_default_width}x{self._tm_default_height}; request asked "
                f"for {width}x{height}. Set model_config.taomate_r1_width/height."
            )
        num_frames = int(getattr(sampling, "num_frames", None) or 0) or self._r1_default_frames
        if num_frames > self._r1_max_frames:
            raise OmniClientError(
                f"TaoMate-R1 sessions are limited to {self._r1_max_frames} frames (taomate_r1_max_frames)"
            )
        plan = plan_session(num_frames)
        seed = getattr(sampling, "seed", None)
        seed = self._tm_default_seed if seed is None else int(seed)

        prepared = build_encoder_inputs(prompt_text, images, plan=plan, height=canvas.height, width=canvas.width)
        hidden, tags = MiniMaxH3Pipeline.encode_prompt(self, prepared)
        media = self._encode_local_media(prepared.media)
        if media.visual_condition is None or not media.visual_condition_shapes:
            raise RuntimeError("TaoMate-R1 reference encode produced no visual condition rows")
        clean = media.visual_condition.to("cpu", torch.float32)
        anchors = minimax_h3_imgvid_cond_noise_aug_rows(
            clean,
            condition_shapes=media.visual_condition_shapes,
            target_latent_t=plan.video_latent_t,
            imgvid_cond_num_frames=len(media.visual_condition_shapes),
            seed=seed,
            noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
        )
        text_len = int(hidden.shape[0])
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=text_len,
            latent_t=plan.video_latent_t,
            latent_h=canvas.latent_h,
            latent_w=canvas.latent_w,
            audio_t=plan.audio_latent_t,
            ref_blocks=media.ref_blocks,
        )
        packed["token_tags"][packed["text_pos"].view(-1)] = tags.detach().to("cpu", torch.long)
        reference_rows = int((~packed["update_mask"]).sum())
        if reference_rows != int(anchors.shape[0]):
            raise RuntimeError(f"TaoMate-R1 anchors have {anchors.shape[0]} rows, the layout {reference_rows}")

        session_id = self._bound_session_id(state)
        self._tm_vae_graph()
        if session_id in self._r1_sessions:
            self._release_session(session_id)
            bound.reset()
        session = _R1Session(
            session_id=session_id,
            canvas=canvas,
            seed=seed,
            plan=plan,
            packed=packed,
            reference_rows=reference_rows,
            anchors=anchors.to(self.device, torch.float32),
            text_embeddings=hidden.detach(),
            cache=lambda: self._bound_segmented_cache(session_id),
            video_decoder=StreamingVideoDecoder(
                self.video_vae,
                device=self.device,
                height=canvas.height,
                width=canvas.width,
                emit_frames=self._tm_output_rank,
            ),
            audio_decoder=StreamingAudioDecoder(self.audio_vae, device=self.device),
        )
        self._r1_sessions[session_id] = session
        state.prompt_embeds = hidden
        state.chunk_num_steps = STUDENT_STEPS
        state.total_chunks = plan.total_chunks
        state.chunk_index = 0
        state.step_index = 0
        state.step_in_chunk = 0
        state.do_true_cfg = False
        state.timesteps = torch.tensor(
            [1.0 - sigma for sigma in session.sigmas_video[:-1]], dtype=torch.float32, device=self.device
        )
        state.extra = {"taomate_session_id": session_id}
        logger.info(
            "TaoMate-R1 session %s: %dx%d, seed=%d, %d frames in %d chunk(s), %d reference(s) / %d rows, prompt=%.40r",
            session_id,
            canvas.width,
            canvas.height,
            seed,
            plan.frame_count,
            plan.total_chunks,
            len(images),
            reference_rows,
            prompt_text,
        )
        return state

    def prepare_next_chunk(self, state: StepRequestState) -> None:  # type: ignore[override]
        if state.request_denoise_completed:
            return
        session = self._session(state)
        chunk_state = session.begin_chunk(state.chunk_index, device=self.device)
        self._select_bank(chunk_state.chunk.bank)
        video, audio = session.initial_noise(chunk_state.chunk, device=self.device)
        state.latents = video
        state.extra["audio_rows"] = audio
        state.step_index = 0
        state.step_in_chunk = 0

    def _forward(
        self,
        session: _R1Session,
        chunk_state: _ChunkState,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        t_video: float,
        t_audio: float,
        mode: StreamMode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch = chunk_state.branch
        if "rope_table" not in branch.static_kwargs:
            branch.prepare_rope_table(self.transformer)
        if not hasattr(branch, "taomate_embed_plan"):
            branch.taomate_embed_plan = self.transformer.plan_local_embed(
                img_pos=branch.img_pos,
                audio_pos=branch.audio_pos,
                text_pos=branch.text_pos_dev,
                seq_len=branch.seq_len,
                device=self.device,
            )
        forward_kwargs = branch.forward_kwargs(
            video_rows=torch.cat((session.anchors, video_rows), dim=0),
            audio_rows=audio_rows,
            t_video=float(t_video),
            t_audio=float(t_audio),
            imgvid_cond_timestep=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            audio_ref_cond_timestep=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        )
        # The worker re-selects the fixed adapter before every step; keep the chunk's bank.
        self._select_bank(chunk_state.chunk.bank)
        with (
            local_embed_plan(branch.taomate_embed_plan),
            stream_context(session.stream_context(mode)),
            torch.inference_mode(),
        ):
            velocity_video, velocity_audio = self.transformer(**forward_kwargs)
        return velocity_video, velocity_audio

    def denoise_step(  # type: ignore[override]
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del input_batch, kwargs
        if states is None or len(states) != 1:
            raise ValueError("TaoMate-R1 step execution serves one session per forward")
        self._require_bound_ar_state()
        state = states[0]
        session = self._session(state)
        chunk_state = session.chunk
        if chunk_state is None or state.latents is None:
            raise RuntimeError("TaoMate-R1 denoise_step called before prepare_next_chunk")
        step = state.step_in_chunk
        if step >= STUDENT_STEPS:
            raise RuntimeError(f"TaoMate-R1 chunks have {STUDENT_STEPS} denoise steps, got step {step}")
        started = time.perf_counter()
        velocity_video, velocity_audio = self._forward(
            session,
            chunk_state,
            video_rows=state.latents,
            audio_rows=state.extra["audio_rows"],
            t_video=1.0 - session.sigmas_video[step],
            t_audio=1.0 - session.sigmas_audio[step],
            mode=StreamMode.NOISY,
        )
        if self._tm_log_timings and velocity_video.is_cuda:
            torch.accelerator.synchronize()
        chunk_state.timings["denoise"] = chunk_state.timings.get("denoise", 0.0) + (time.perf_counter() - started)
        # Output rows cover every visual row (references first, zeroed); keep the targets.
        session.audio_velocity = velocity_audio.float()
        return velocity_video[session.reference_rows :].float()

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor | None, **kwargs: Any) -> None:  # type: ignore[override]
        del kwargs
        session = self._session(state)
        if session.chunk is None or state.latents is None or noise_pred is None or session.audio_velocity is None:
            raise RuntimeError("TaoMate-R1 step_scheduler called out of order")
        step = state.step_in_chunk
        audio_rows = state.extra["audio_rows"]
        if tuple(noise_pred.shape) != tuple(state.latents.shape):
            raise RuntimeError(f"TaoMate-R1 video velocity {tuple(noise_pred.shape)} != {tuple(state.latents.shape)}")
        if tuple(session.audio_velocity.shape) != tuple(audio_rows.shape):
            raise RuntimeError(
                f"TaoMate-R1 audio velocity {tuple(session.audio_velocity.shape)} != {tuple(audio_rows.shape)}"
            )
        # Release update order: x + (sigma_left - sigma_right) * v, in float32.
        state.latents = state.latents + (session.sigmas_video[step] - session.sigmas_video[step + 1]) * noise_pred
        state.extra["audio_rows"] = (
            audio_rows + (session.sigmas_audio[step] - session.sigmas_audio[step + 1]) * session.audio_velocity
        )
        session.audio_velocity = None
        state.step_in_chunk += 1
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:  # type: ignore[override]
        del kwargs
        self._require_bound_ar_state()
        session = self._session(state)
        chunk_state = session.chunk
        if chunk_state is None or state.latents is None:
            raise RuntimeError("TaoMate-R1 post_decode called out of order")
        chunk = chunk_state.chunk
        video_rows = state.latents
        audio_rows = state.extra["audio_rows"]
        clean_commit = session.plan.commits_clean_history(chunk.index)
        cache = session.cache
        commit_started = time.perf_counter()
        if clean_commit:
            # Sigma-zero forward with this chunk's bank: its clean K/V become history.
            session.history.begin_commit(audio_tokens=int(audio_rows.shape[0]), video_tokens=int(video_rows.shape[0]))
            try:
                self._forward(
                    session,
                    chunk_state,
                    video_rows=video_rows,
                    audio_rows=audio_rows,
                    t_video=1.0,
                    t_audio=1.0,
                    mode=StreamMode.CLEAN_COMMIT,
                )
                session.history.commit()
            except BaseException:
                session.history.rollback()
                raise
        chunk_state.timings["commit"] = time.perf_counter() - commit_started

        decode_started = time.perf_counter()
        latent = minimax_h3_unpatchify_video_tokens(
            video_rows,
            latent_shape=(chunk.video_latent_count, session.canvas.latent_h // 2, session.canvas.latent_w // 2, 24),
            patch_size=(1, 2, 2),
        )
        frame_start, frames_device = session.video_decoder.push_device(latent)
        frames = session.video_decoder.to_host(frames_device)
        final_chunk = chunk.index + 1 >= state.total_chunks
        if final_chunk:
            flush_start, flushed = session.video_decoder.flush()
            if flushed is not None:
                if frames is None:
                    frames, frame_start = flushed, flush_start
                else:
                    frames = np.concatenate((frames, flushed), axis=0)
        num_frames = 0 if frames is None else int(frames.shape[0])
        session.audio_decoder.append(
            minimax_h3_unpack_audio_tokens(audio_rows, audio_t=int(audio_rows.shape[0]), audio_channel=2)
        )
        if self._tm_output_rank:
            audio = session.audio_decoder.decode_range(
                samples_at_frame(frame_start), samples_at_frame(frame_start + num_frames)
            )
        else:
            session.audio_decoder.skip_range(samples_at_frame(frame_start + num_frames))
            audio = np.zeros((0, 2), dtype=np.float32)
        chunk_state.timings["decode"] = time.perf_counter() - decode_started
        if self._r1_dump_latents_dir is not None and self._tm_output_rank:
            target = self._r1_dump_latents_dir / session.session_id
            target.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "chunk_index": chunk.index,
                    "lora_bank": chunk.bank,
                    "clean_commit": clean_commit,
                    "video_rows": video_rows.detach().float().cpu(),
                    "audio_rows": audio_rows.detach().float().cpu(),
                    "video_latent_range": (chunk.video_latent_start, chunk.video_latent_stop),
                    "audio_latent_range": (chunk.audio_latent_start, chunk.audio_latent_stop),
                    "anchors": session.anchors.detach().float().cpu() if chunk.index == 0 else None,
                    "text_embeddings": session.text_embeddings.detach().float().cpu() if chunk.index == 0 else None,
                },
                target / f"chunk_{chunk.index:03d}.pt",
            )

        completed_chunk_index = state.chunk_index
        state.chunk_index += 1
        chunk_end = time.perf_counter()
        chunk_seconds = chunk_end - chunk_state.started_at
        if self._tm_log_timings:
            logger.info(
                "TaoMate-R1 %s chunk %d/%d (%s): %.3f s (denoise x3 %.3f, commit %.3f, decode %.3f), "
                "%d frames, history %d rows",
                session.session_id,
                chunk.index + 1,
                session.plan.total_chunks,
                chunk.bank,
                chunk_seconds,
                chunk_state.timings.get("denoise", 0.0),
                chunk_state.timings.get("commit", 0.0),
                chunk_state.timings.get("decode", 0.0),
                num_frames,
                cache.history_tokens,
            )
        session.last_chunk_end = chunk_end
        chunk_facts = {
            "name": "taomate_r1",
            "chunk_index": chunk.index,
            "lora_bank": chunk.bank,
            "clean_commit": clean_commit,
        }
        payload: dict[str, Any] = {
            "video": frames
            if frames is not None
            else np.zeros((0, session.canvas.height, session.canvas.width, 3), np.uint8),
            "audio": audio,
        }
        metadata: dict[str, Any] = {
            "video": {"fps": float(VIDEO_FPS)},
            "audio": {"sample_rate": AUDIO_SAMPLE_RATE},
            "ar_diffusion": ARDiffusionChunkMetadata(
                session_id=session.session_id,
                request_id=state.request_id,
                chunk_index=completed_chunk_index,
                applied_event_ids=(),
            ).to_dict(),
            "stream": {"frame_start": frame_start, "model": chunk_facts},
            "taomate_r1": {
                **chunk_facts,
                "frame_start": frame_start,
                "num_frames": num_frames,
                "audio_samples": int(audio.shape[0]),
                "history_tokens": cache.history_tokens,
                "chunk_seconds": round(chunk_seconds, 4),
                **{f"{name}_seconds": round(value, 4) for name, value in chunk_state.timings.items()},
            },
        }
        session.chunk = None
        return DiffusionOutput(
            output={"payload": payload, "metadata": metadata},
            chunk_index=completed_chunk_index,
            total_chunks=state.total_chunks,
            finished=state.request_denoise_completed,
        )

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:  # type: ignore[override]
        session = self._r1_sessions.get(state.extra.get("taomate_session_id") or self._bound_session_id(state))
        plan = session.plan if session is not None else plan_session(int(state.sampling.num_frames or 1))
        chunk = plan.chunks[min(state.chunk_index, plan.total_chunks - 1)]
        return ChunkMediaSpec(
            num_media_frames=chunk.frame_count,
            fps=float(VIDEO_FPS),
            num_latent_frames=chunk.video_latent_count,
        )


def get_taomate_r1_post_process_func(od_config: OmniDiffusionConfig):
    """Pass the payload/metadata envelope through untouched."""
    del od_config

    def post_process_func(output: Any, output_type: str = "np", sampling_params: Any | None = None) -> Any:
        del output_type, sampling_params
        if isinstance(output, dict) and isinstance(output.get("payload"), dict):
            return output
        return {"payload": {"video": output}, "metadata": {}}

    return post_process_func


__all__ = ["TaoMateR1Pipeline", "bank_for_chunk", "get_taomate_r1_post_process_func"]
