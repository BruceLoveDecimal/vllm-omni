# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""dots.tts serving adapter."""

import math
import time

from pydantic import ValidationError
from transformers import AutoTokenizer
from vllm.logger import init_logger
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.dots_tts.dots_tts_prompt import MAX_AUDIO_PATCHES
from vllm_omni.model_executor.models.dots_tts.request_config import DotsTTSRequestConfig

logger = init_logger(__name__)


@register_tts_adapter
class DotsTTSAdapter(ARTTSAdapter):
    """Adapter for dots.tts (AR ``engine_client`` backend).

    Three conditioning modes:

    * zero-shot — ``input`` alone;
    * reference audio only — ``ref_audio``, which gives the DiT a CAM++
      x-vector for the speaker;
    * voice clone — ``ref_audio`` + ``ref_text``, which additionally prefills
      the reference's audio latents into the DiT history and the patch
      encoder's KV cache.

    The last mode makes the prompt length depend on the reference audio, so
    the adapter and the talker size it from one shared planner
    (``prompt_audio_plan``) rather than each doing its own arithmetic.
    """

    stage_keys = frozenset()
    model_archs = frozenset({"DotsTTSForConditionalGeneration"})
    name = "dots_tts"
    detect_priority = 5

    max_new_tokens_max = MAX_AUDIO_PATCHES

    def __init__(self, ctx):
        super().__init__(ctx)
        self.tokenizer = None
        self._build_prompt_async = None

    # ── prompt building ──

    def _build_prompt(
        self,
        text: str,
        *,
        ref_audio: list[float] | None = None,
        ref_sr: int | None = None,
        ref_text: str | None = None,
        prompt_patch_count: int = 0,
        prompt_audio_samples: int = 0,
        ref_audio_key: str | None = None,
        generation_config: dict | None = None,
        language: str | None = None,
    ) -> dict:
        from vllm_omni.model_executor.models.dots_tts.dots_tts_prompt import build_dots_tts_prompt

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.ctx.engine_client.model_config.model,
                trust_remote_code=True,
            )
        return build_dots_tts_prompt(
            self.tokenizer,
            text,
            ref_audio=ref_audio,
            ref_sr=ref_sr,
            ref_text=ref_text,
            prompt_patch_count=prompt_patch_count,
            prompt_audio_samples=prompt_audio_samples,
            ref_audio_key=ref_audio_key,
            generation_config=generation_config,
            language=language,
        )

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        server = self.ctx.server
        try:
            DotsTTSRequestConfig.model_validate(request.extra_params or {})
        except ValidationError as exc:
            return str(exc)
        if request.instructions:
            return (
                "dots.tts takes inline instructions in input; select "
                "extra_params.template_name=instruction_tts instead of the separate instructions field"
            )
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.speaker_embedding is not None:
            return "'speaker_embedding' is not supported for dots.tts"

        if request.x_vector_only_mode is not None:
            return "'x_vector_only_mode' is not supported for dots.tts"

        if request.voice is not None:
            request.voice = request.voice.lower()
            available_voices = server._get_available_speakers()
            if request.voice not in available_voices:
                supported = ", ".join(sorted(available_voices)) or "none"
                return f"Invalid voice '{request.voice}'. Supported: {supported}"

        # Resolves an uploaded voice into request.ref_audio/ref_text in place,
        # so the build path only ever sees inline reference audio.
        err = server._apply_uploaded_speaker(request)
        if err:
            return err

        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(self._single_ref_audio(request.ref_audio))
            if fmt_err:
                return fmt_err
        elif request.ref_text and request.ref_text.strip():
            return "ref_text requires ref_audio (the transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.ref_audio is not None and request.ref_text and request.ref_text.strip():
                if request.max_new_tokens < 2:
                    return "Voice cloning requires max_new_tokens >= 2 because the first patch is discarded"
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        from vllm_omni.model_executor.models.dots_tts.dots_tts_prompt import prompt_audio_plan

        server = self.ctx.server
        ref_audio: list[float] | None = None
        ref_sr: int | None = None
        ref_text: str | None = None
        ref_audio_key: str | None = None
        prompt_patch_count = 0
        prompt_audio_samples = 0

        if request.ref_audio is not None:
            samples: list[float] | None
            sample_rate: int | None
            samples, sample_rate, ref_audio_key = await server._resolve_ref_audio(
                self._single_ref_audio(request.ref_audio)
            )
            if samples is None or sample_rate is None:
                raise ValueError("Failed to resolve reference audio")
            ref_audio = samples
            ref_sr = sample_rate
            # Duration bounds are the shared speech API's: _resolve_ref_audio
            # already rejects clips outside [1 s, 30 s] before returning.
            samples_per_patch, target_sample_rate = self._audio_patch_geometry()
            prompt_patch_count, prompt_audio_samples = prompt_audio_plan(
                len(ref_audio),
                ref_sr,
                samples_per_patch=samples_per_patch,
                target_sample_rate=target_sample_rate,
            )
            if request.ref_text and request.ref_text.strip():
                if prompt_patch_count < 1:
                    raise ValueError(
                        "Reference audio is shorter than one audio patch "
                        f"({samples_per_patch / target_sample_rate * 1000:.0f} ms) and cannot seed "
                        "voice cloning. Supply a longer reference, or omit ref_text to condition "
                        "on the speaker's timbre alone."
                    )
                ref_text = request.ref_text

        if self._build_prompt_async is None:
            self._build_prompt_async = make_async(
                self._build_prompt,
                executor=self.ctx.server._tts_executor,
            )
        prompt = await self._build_prompt_async(
            request.input,
            ref_audio=ref_audio,
            ref_sr=ref_sr,
            ref_text=ref_text,
            prompt_patch_count=prompt_patch_count,
            prompt_audio_samples=prompt_audio_samples,
            ref_audio_key=ref_audio_key,
            generation_config=request.extra_params,
            language=request.language,
        )
        return PreparedRequest(
            prompt=prompt,
            tts_params={"prompt_patch_count": prompt_patch_count if ref_text else 0},
            model_type=self.name,
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list:
        """Cap ``max_tokens`` at the FM workspace the reference audio leaves free.

        Prompt patches and generated patches share the talker's per-request FM
        buffer. Without this the buffer would fill mid-utterance and the talker
        would cut the request short at an arbitrary point instead of the
        request being sized correctly up front.
        """
        import copy

        # ``prompt`` is whatever build() returned; only the engine-prompt dict
        # carries the prefill metadata this cap depends on.
        additional = prompt.get("additional_information") or {} if isinstance(prompt, dict) else {}
        prompt_patch_count = int(additional.get("prompt_patch_count", 0))
        budget = MAX_AUDIO_PATCHES - prompt_patch_count
        requested = request.max_new_tokens

        params = sampling_params_list[0]
        limit = min(requested, budget) if requested is not None else min(params.max_tokens or budget, budget)
        minimum = 2 if prompt_patch_count else 1
        if limit < minimum:
            raise ValueError(
                f"dots.tts requires at least {minimum} generation tokens with "
                f"{prompt_patch_count} prompt patches, but the effective budget is {limit}"
            )
        # Only max_tokens changes; nested sampling settings remain read-only.
        sampling_params_list = list(sampling_params_list)
        params = copy.copy(params)
        params.max_tokens = limit
        sampling_params_list[0] = params
        return sampling_params_list

    async def warmup(self) -> None:
        """Warm up dots.tts through a synthetic zero-shot serving request.

        The talker's side path (10-step DiT Euler, patch-encoder decode,
        streaming AudioVAE) allocates its per-request workspaces and pays every
        lazy CUDA/cuBLAS initialization on the first patch it produces. Doing
        that here moves it off the first real request's time-to-first-audio,
        and it needs a real request because the base LM only runs under a vLLM
        ``ForwardContext``.
        """
        server = self.ctx.server
        t0 = time.time()
        logger.info("Running warmup speech request for model_type=%s", self.name)
        # dots.tts ships no speaker presets: "default" is the zero-shot path,
        # which is also the cheapest warmup (no reference encode).
        warmup_req = OpenAICreateSpeechRequest(
            input="Warmup.",
            voice="default",
            response_format="wav",
            speed=1.0,
            stream=False,
            model=server.model_name,
        )
        try:
            await server._generate_audio_bytes(warmup_req, request_id="speech-warmup")
        except Exception as exc:
            logger.warning("Speech warmup failed (non-fatal): %s", exc)
            return
        logger.info("Speech warmup complete in %.1fs", time.time() - t0)

    def _load_supported_speakers(self) -> set[str]:
        # No speaker presets upstream; "default" names the zero-shot path.
        return {"default"}

    # ── helpers ──

    @staticmethod
    def _single_ref_audio(ref_audio: str | list[str]) -> str:
        """dots.tts conditions on one reference; take the first if a list came in."""
        return ref_audio[0] if isinstance(ref_audio, list) else ref_audio

    def _audio_patch_geometry(self) -> tuple[int, int]:
        """``(samples_per_patch, sample_rate)`` from the checkpoint's vocoder block.

        One audio patch is ``patch_size`` latent frames, each covering
        ``prod(downsample_rates)`` waveform samples.
        """
        hf_config = self.ctx.server.engine_client.model_config.hf_config
        vocoder = getattr(hf_config, "vocoder", None)
        if not isinstance(vocoder, dict) or not vocoder.get("downsample_rates"):
            raise ValueError(
                "dots.tts checkpoint config.json has no usable 'vocoder' block; cannot size reference-audio prefill."
            )
        hop_size = math.prod(int(rate) for rate in vocoder["downsample_rates"])
        return int(hf_config.patch_size) * hop_size, int(vocoder["sample_rate"])
