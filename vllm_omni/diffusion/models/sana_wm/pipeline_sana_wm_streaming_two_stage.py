# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM realtime streaming with the Stage-2 LTX-2 refiner.

:class:`SanaWmStreamingTwoStagePipeline` runs the chunk-causal Stage-1 of
:class:`SanaWmStreamingPipeline` and, inside the same ``post_decode`` of every
chunk, refines the clean Stage-1 block with the streaming release's
chunk-causal LTX-2 refiner (``refiner.py``) before the overlap decode. The
Stage-1 loop, the WS transport and the runner are untouched: the refiner hooks
into ``_apply_stage2`` / ``_stage2_metadata`` of the Stage-1 pipeline.

Components (all resident, discovered through ``SupportsComponentDiscovery``):

* ``refiner_transformer`` -- vLLM-Omni's native ``LTX2VideoTransformer3DModel``
  built from ``refiner/transformer/config.json`` and weight-loaded by the
  diffusers loader like the Stage-1 DiT (TP-capable, strict coverage check);
* ``refiner_text_encoder`` / ``refiner_tokenizer`` -- Gemma-3-12B from
  ``refiner/text_encoder`` (the LTX-2 caption encoder; distinct from the
  Gemma-2-2B Stage-1 encoder);
* ``refiner_connectors`` -- diffusers ``LTX2TextConnectors`` from
  ``refiner/connectors`` with the Omni attention processor installed.

Design: ``docs/design/feature/sana_wm_realtime_streaming.md`` §17.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    _install_connector_attention,
    create_transformer_from_config,
)
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import get_sana_wm_pre_process_func
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm_streaming import (
    _EXTRA_KEYS_TO_FREE,
    SanaWmStreamingPipeline,
)
from vllm_omni.diffusion.models.sana_wm.refiner import (
    SANA_WM_REFINER_KV_MAX_FRAMES,
    SANA_WM_REFINER_SINK_FRAMES,
    SanaWmRefinerRunner,
    SanaWmRefinerSchedule,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

logger = init_logger(__name__)

__all__ = [
    "SANA_WM_REFINER_ROOT_ENV",
    "SANA_WM_REFINER_TEXT_ENCODER_ENV",
    "SANA_WM_STREAMING_TWO_STAGE_MODEL_ID",
    "SanaWmRefinerPaths",
    "SanaWmStreamingTwoStagePipeline",
    "get_sana_wm_pre_process_func",
    "resolve_sana_wm_refiner_paths",
]

# Diffusers-layout conversion of ``Efficient-Large-Model/SANA-WM_streaming``
# carrying Stage-1 (``transformer/`` + ``vae/``) and the refiner tree
# (``refiner/{transformer,connectors,text_encoder}``), produced by
# ``tools/convert_sana_wm_streaming_to_diffusers.py --refiner-dir ...``.
SANA_WM_STREAMING_TWO_STAGE_MODEL_ID = "BBBBruce/SANA-WM_streaming-two-stage-diffusers"
SANA_WM_STREAMING_DEFAULT_FPS = 16.0

SANA_WM_REFINER_SUBFOLDER = "refiner"
SANA_WM_REFINER_TRANSFORMER_SUBFOLDER = "transformer"
SANA_WM_REFINER_CONNECTORS_SUBFOLDER = "connectors"
SANA_WM_REFINER_TEXT_ENCODER_SUBFOLDER = "text_encoder"
SANA_WM_REFINER_PATTERNS = (
    f"{SANA_WM_REFINER_SUBFOLDER}/{SANA_WM_REFINER_TRANSFORMER_SUBFOLDER}/*",
    f"{SANA_WM_REFINER_SUBFOLDER}/{SANA_WM_REFINER_CONNECTORS_SUBFOLDER}/*",
    f"{SANA_WM_REFINER_SUBFOLDER}/{SANA_WM_REFINER_TEXT_ENCODER_SUBFOLDER}/*",
)
# Point the refiner at another local tree (e.g. the raw release's
# ``refiner_diffusers/``) without re-copying 40 GB into the converted repo; the
# text encoder can be redirected on its own (``gemma3_12b/`` of the release).
SANA_WM_REFINER_ROOT_ENV = "VLLM_OMNI_SANA_WM_REFINER_ROOT"
SANA_WM_REFINER_TEXT_ENCODER_ENV = "VLLM_OMNI_SANA_WM_REFINER_TEXT_ENCODER"
# NVlabs ``DiffusersLTX2Refiner(text_max_sequence_length=1024)``.
SANA_WM_REFINER_TEXT_MAX_LENGTH = 1024

# Modules of the native LTX-2 transformer that only the audio stream and the
# audio<->video cross-attention use. The refiner forward is video-only, so
# after the (strict) weight load they are parked on the host: about 11 GB of
# the 37.7 GB checkpoint.
_REFINER_AUDIO_TOP_LEVEL_MODULES = (
    "audio_proj_in",
    "audio_caption_projection",
    "audio_time_embed",
    "av_cross_attn_video_scale_shift",
    "av_cross_attn_audio_scale_shift",
    "av_cross_attn_video_a2v_gate",
    "av_cross_attn_audio_v2a_gate",
    "audio_norm_out",
    "audio_proj_out",
)
_REFINER_AUDIO_TOP_LEVEL_PARAMS = ("audio_scale_shift_table",)
_REFINER_AUDIO_BLOCK_MODULES = (
    "audio_norm1",
    "audio_attn1",
    "audio_norm2",
    "audio_attn2",
    "audio_to_video_norm",
    "audio_to_video_attn",
    "video_to_audio_norm",
    "video_to_audio_attn",
    "audio_norm3",
    "audio_ff",
)
_REFINER_AUDIO_BLOCK_PARAMS = (
    "audio_scale_shift_table",
    "video_a2v_cross_attn_scale_shift_table",
    "audio_a2v_cross_attn_scale_shift_table",
)


@dataclass(frozen=True)
class SanaWmRefinerPaths:
    """Resolved local directories of the refiner tree."""

    root: Path
    transformer_dir: Path
    connectors_dir: Path
    text_encoder_dir: Path


def resolve_sana_wm_refiner_paths(release_root: str | Path) -> SanaWmRefinerPaths:
    """Locate ``refiner/{transformer,connectors,text_encoder}`` under the release (or the env overrides)."""
    root = Path(os.environ.get(SANA_WM_REFINER_ROOT_ENV, "").strip() or Path(release_root) / SANA_WM_REFINER_SUBFOLDER)
    text_encoder_dir = Path(
        os.environ.get(SANA_WM_REFINER_TEXT_ENCODER_ENV, "").strip() or root / SANA_WM_REFINER_TEXT_ENCODER_SUBFOLDER
    )
    paths = SanaWmRefinerPaths(
        root=root,
        transformer_dir=root / SANA_WM_REFINER_TRANSFORMER_SUBFOLDER,
        connectors_dir=root / SANA_WM_REFINER_CONNECTORS_SUBFOLDER,
        text_encoder_dir=text_encoder_dir,
    )
    required = [
        paths.transformer_dir / "config.json",
        paths.connectors_dir / "config.json",
        paths.text_encoder_dir / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "SANA-WM streaming two-stage pipeline needs the LTX-2 refiner tree "
            f"({SANA_WM_REFINER_SUBFOLDER}/transformer, connectors, text_encoder); missing: {', '.join(missing)}. "
            f"Convert the release with tools/convert_sana_wm_streaming_to_diffusers.py --refiner-dir, or set "
            f"{SANA_WM_REFINER_ROOT_ENV} / {SANA_WM_REFINER_TEXT_ENCODER_ENV}."
        )
    return paths


class SanaWmStreamingTwoStagePipeline(SanaWmStreamingPipeline):
    """Chunk-causal Stage-1 + chunk-causal LTX-2 refiner, served through step execution.

    Per chunk ``post_decode`` runs the Stage-1 cache write, then
    ``refine_block`` on the clean block (3 refiner forwards + 1 K/V capture
    forward), then the overlap decode of the *refined* history. The refiner's
    frame count contract is unchanged: chunk 0 emits ``8 * chunk_size + 1``
    pixel frames (the conditioning frame is decoded from its raw latent, the
    sink of both stages), later chunks ``8 * chunk_size``.
    """

    _dit_modules: ClassVar[list[str]] = ["transformer", "refiner_transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "refiner_text_encoder", "refiner_connectors"]
    _extra_download_patterns: ClassVar[tuple[str, ...]] = SANA_WM_REFINER_PATTERNS
    _extra_keys_to_free: ClassVar[tuple[str, ...]] = (
        *_EXTRA_KEYS_TO_FREE,
        "refined_history",
        "refiner_cache",
        "refiner_prompt_embeds",
        "refiner_prompt_mask",
        "refiner_generator",
    )

    def __init__(self, *, od_config: OmniDiffusionConfig | None = None, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.refiner_transformer: nn.Module | None = None
        self.refiner_text_encoder: nn.Module | None = None
        self.refiner_connectors: nn.Module | None = None
        self.refiner_tokenizer: Any | None = None
        self.refiner_paths: SanaWmRefinerPaths | None = None
        self._refiner_runner: SanaWmRefinerRunner | None = None
        self.refiner_schedule = SanaWmRefinerSchedule()
        if od_config is None or od_config.model is None:
            return
        parallel_config = getattr(od_config, "parallel_config", None)
        sequence_parallel_size = int(getattr(parallel_config, "sequence_parallel_size", 1) or 1)
        if sequence_parallel_size > 1:
            raise ValueError(
                "SanaWmStreamingTwoStagePipeline does not support sequence parallelism "
                f"(got sequence_parallel_size={sequence_parallel_size}); the refiner's sliding K/V window is "
                "assembled per rank."
            )
        if self.release_paths is None:
            self.resolve_checkpoint()
        assert self.release_paths is not None
        self.refiner_paths = resolve_sana_wm_refiner_paths(self.release_paths.root)
        # The refiner DiT is built eagerly (like the Stage-1 transformer) so the
        # loader streams its checkpoint straight into the module and the
        # strict-coverage check covers it.
        self.weights_sources.append(
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(self.refiner_paths.root),
                subfolder=SANA_WM_REFINER_TRANSFORMER_SUBFOLDER,
                revision=None,
                prefix="refiner_transformer.",
                fall_back_to_pt=False,
            )
        )
        with (self.refiner_paths.transformer_dir / "config.json").open(encoding="utf-8") as handle:
            refiner_config = json.load(handle)
        self.refiner_transformer = create_transformer_from_config(refiner_config, quant_config=self.quant_config)
        self._load_refiner_text_components(dtype=self._native_dtype(self.device))
        self._place_aux_components()

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _load_refiner_text_components(self, *, dtype: torch.dtype) -> None:
        """Gemma-3-12B + LTX-2 text connectors of the refiner, built on CPU (placed by ``_place_aux_components``)."""
        from diffusers.pipelines.ltx2 import LTX2TextConnectors
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        assert self.refiner_paths is not None
        text_encoder_dir = str(self.refiner_paths.text_encoder_dir)
        self.refiner_tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir, local_files_only=True)
        if getattr(self.refiner_tokenizer, "pad_token", None) is None:
            self.refiner_tokenizer.pad_token = self.refiner_tokenizer.eos_token
        self.refiner_tokenizer.padding_side = "left"
        with torch.device("cpu"):
            self.refiner_text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                text_encoder_dir,
                torch_dtype=dtype,
                local_files_only=True,
            )
            self.refiner_connectors = LTX2TextConnectors.from_pretrained(
                str(self.refiner_paths.connectors_dir),
                torch_dtype=dtype,
                local_files_only=True,
            )
        _install_connector_attention(self.refiner_connectors)

    def _placement_is_managed(self) -> bool:
        parallel_config = getattr(self.od_config, "parallel_config", None)
        return bool(
            getattr(self.od_config, "enable_cpu_offload", False)
            or getattr(self.od_config, "enable_layerwise_offload", False)
            or getattr(parallel_config, "use_hsdp", False)
        )

    def _offload_refiner_audio_stream(self) -> None:
        """Park the audio-only LTX-2 modules on the host; the refiner forward never touches them."""
        transformer = self.refiner_transformer
        if transformer is None or self._placement_is_managed():
            return
        moved = 0

        def _park_param(owner: nn.Module, name: str) -> None:
            nonlocal moved
            param = getattr(owner, name, None)
            if isinstance(param, torch.Tensor) and param.device.type != "cpu":
                param.data = param.data.to("cpu")
                moved += param.numel()

        def _park_module(owner: nn.Module, name: str) -> None:
            nonlocal moved
            module = getattr(owner, name, None)
            if isinstance(module, nn.Module):
                moved += sum(p.numel() for p in module.parameters() if p.device.type != "cpu")
                module.to("cpu")

        for name in _REFINER_AUDIO_TOP_LEVEL_MODULES:
            _park_module(transformer, name)
        for name in _REFINER_AUDIO_TOP_LEVEL_PARAMS:
            _park_param(transformer, name)
        for block in transformer.transformer_blocks:
            for name in _REFINER_AUDIO_BLOCK_MODULES:
                _park_module(block, name)
            for name in _REFINER_AUDIO_BLOCK_PARAMS:
                _park_param(block, name)
        if moved:
            logger.info("Sana-WM refiner: parked %.2f B audio-stream parameters on the host.", moved / 1e9)

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        loaded = super().load_weights(weights)
        self._offload_refiner_audio_stream()
        return loaded

    def refiner_runner(self) -> SanaWmRefinerRunner:
        if self._refiner_runner is None:
            if self.refiner_transformer is None:
                raise RuntimeError("SANA-WM refiner transformer did not initialize.")
            self._refiner_runner = SanaWmRefinerRunner(
                self.refiner_transformer,
                schedule=self.refiner_schedule,
                sink_frames=SANA_WM_REFINER_SINK_FRAMES,
                block_size=int(self.sana_wm_config.chunk_size),
                kv_max_frames=SANA_WM_REFINER_KV_MAX_FRAMES,
            )
        return self._refiner_runner

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _encode_refiner_prompt(
        self,
        prompt_text: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gemma-3 per-layer hidden states -> LTX-2 connectors -> ``(video context, mask)``.

        The connectors normalise the stacked hidden states themselves
        (``per_layer_masked_mean_norm``), so the raw stack is handed over, as
        the LTX-2 pipelines do.
        """
        tokenizer, text_encoder, connectors = self.refiner_tokenizer, self.refiner_text_encoder, self.refiner_connectors
        if tokenizer is None or text_encoder is None or connectors is None:
            raise RuntimeError("SANA-WM refiner text components did not initialize.")
        encoded = tokenizer(
            [prompt_text.strip()],
            padding="max_length",
            max_length=SANA_WM_REFINER_TEXT_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        # The backbone alone: the LM head would materialise a [1024, 262k] logits
        # tensor per prompt that nothing reads.
        backbone = getattr(text_encoder, "model", text_encoder)
        hidden_states = backbone(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            output_hidden_states=True,
        ).hidden_states
        stacked = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=dtype)
        video_context, _, attention_mask = connectors(
            stacked, encoded.attention_mask, padding_side=tokenizer.padding_side
        )
        video_context = video_context.to(device=device, dtype=dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
            if bool(attention_mask.all()):
                attention_mask = None
        return video_context, attention_mask

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        state = super().prepare_encode(state, **kwargs)
        extra = state.extra
        sampling = state.sampling
        extra_args = self._extra_args(sampling)
        device, dtype = extra["device"], extra["dtype"]
        runner = self.refiner_runner()
        runner.schedule.check_num_steps(extra_args.get("sana_wm_refiner_steps"))

        prompt_text = str((state.prompt or {}).get("prompt") or "")
        prompt_embeds, prompt_mask = self._encode_refiner_prompt(prompt_text, device=device, dtype=dtype)
        # NVlabs seeds the refiner's noise separately (``--refiner_seed``);
        # default to the request seed so one seed reproduces the whole clip.
        refiner_seed = int(extra_args.get("sana_wm_refiner_seed", getattr(sampling, "seed", None) or 0))
        fps = float(getattr(sampling, "resolved_frame_rate", None) or SANA_WM_STREAMING_DEFAULT_FPS)

        extra.update(
            {
                "backend": "native_gdn_streaming+ltx2_refiner",
                "refined_history": extra["first_latent"].clone(),
                "refiner_cache": runner.new_cache(
                    latent_height=extra["latent_height"], latent_width=extra["latent_width"]
                ),
                "refiner_fps": fps,
                "refiner_generator": torch.Generator(device=device).manual_seed(refiner_seed),
                "refiner_prompt_embeds": prompt_embeds,
                "refiner_prompt_mask": prompt_mask,
                "refiner_seed": refiner_seed,
            }
        )
        return state

    def _apply_stage2(self, state: StepRequestState, clean_latents: torch.Tensor) -> torch.Tensor | None:
        extra = state.extra
        cache = extra["refiner_cache"]
        block_start, _ = extra["gen_range"]
        sink_latents = None
        if cache.sink_kv_pre is None:
            sink_latents = extra["first_latent"][:, :, : cache.sink_frames]
        return self.refiner_runner().refine_block(
            cache,
            clean_latents.to(dtype=extra["dtype"]),
            block_start=int(block_start),
            encoder_hidden_states=extra["refiner_prompt_embeds"],
            encoder_attention_mask=extra["refiner_prompt_mask"],
            fps=extra["refiner_fps"],
            generator=extra["refiner_generator"],
            sink_latents=sink_latents,
        )

    def _stage2_metadata(self, state: StepRequestState) -> dict[str, Any]:
        cache = state.extra.get("refiner_cache")
        if cache is None:
            return {}
        return {
            "refiner_backend": "native_ltx2_chunk_causal",
            "refiner_sigmas": list(self.refiner_schedule.sigmas),
            "refiner_blocks_refined": cache.blocks_refined,
            "refiner_cached_latent_frames": cache.cached_frames(),
            "refiner_cache_bytes": cache.nbytes(),
            "refiner_seed": state.extra.get("refiner_seed"),
        }
