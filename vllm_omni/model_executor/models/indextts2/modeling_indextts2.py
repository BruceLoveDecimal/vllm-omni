# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS2 single-stage model wrapper for vLLM-Omni.

The wrapper follows the same AR-generator pattern as other single-stage TTS
models in vLLM-Omni:

* upstream weights are loaded lazily in ``load_weights()``;
* one upstream ``infer_generator()`` is stored per request id;
* each ``forward()`` returns only newly produced audio samples (delta mode);
* ``compute_logits()`` emits EOS when a request's final chunk has been yielded.

This MVP intentionally keeps the full upstream runtime in one stage. A later
split can move GPT mel-code generation and code-to-waveform decoding into
separate stages after full-sequence correctness is established.
"""

from __future__ import annotations

import random
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_DEFAULT_SAMPLE_RATE = 22050
_DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT = 120
_DEFAULT_INTERVAL_SILENCE_MS = 200


def _pick(info: dict[str, Any], key: str, default: Any = None) -> Any:
    """Extract a scalar value from vLLM additional_information."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


def _to_mono_1d(waveform: Any) -> torch.Tensor:
    """Normalize an upstream waveform event to 1-D float32 mono audio.

    Upstream IndexTTS2 streaming chunks are scaled to int16 range before they
    are yielded. vLLM-Omni's audio writer expects floating-point PCM in roughly
    [-1, 1], so large floating tensors are scaled back down.
    """
    chunk = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu()
    if chunk.ndim == 2:
        if chunk.shape[0] > 1:
            chunk = chunk.mean(dim=0)
        else:
            chunk = chunk.reshape(-1)
    else:
        chunk = chunk.reshape(-1)
    if chunk.numel() > 0 and torch.max(torch.abs(chunk)) > 2.0:
        chunk = chunk / 32767.0
    return chunk.contiguous()


def _resolve_cfg_path(model_path: str, explicit_cfg_path: str | None) -> str:
    """Resolve the upstream IndexTTS2 YAML config path."""
    if explicit_cfg_path:
        return explicit_cfg_path

    candidates = [
        Path(model_path) / "config.yaml",
        Path(model_path) / "checkpoints" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # Let upstream raise a clear file-not-found error that includes the path.
    return str(candidates[0])


class IndexTTS2ForGeneration(nn.Module):
    """Single-stage IndexTTS2 wrapper with delta audio output."""

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    inject_omni_request_id_into_runtime_info = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        self._runtime: Any | None = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()
        self._stream_gens: dict[str, Any] = {}
        self._ar_last_chunk_flags: list[bool] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the upstream IndexTTS2 runtime.

        The upstream package owns checkpoint loading, so vLLM's weight iterator
        is only exhausted to close the stream cleanly.
        """
        with self._lock:
            if self._runtime is not None:
                return set()

            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device

            cfg_path = _resolve_cfg_path(
                self.model_path,
                getattr(self.config, "indextts2_config_path", None),
            )
            use_fp16 = device.type == "cuda"
            use_cuda_kernel = getattr(self.config, "use_cuda_kernel", None)
            use_deepspeed = bool(getattr(self.config, "use_deepspeed", False))
            use_accel = bool(getattr(self.config, "use_accel", False))
            use_torch_compile = bool(getattr(self.config, "use_torch_compile", False))

            try:
                from indextts.infer_v2 import IndexTTS2
            except Exception as exc:
                raise ImportError(
                    "IndexTTS2 support requires the upstream 'indextts' package. "
                    "Install it from https://github.com/index-tts/index-tts and "
                    "make sure its optional audio dependencies are available."
                ) from exc

            logger.info(
                "Loading IndexTTS2 runtime from model_dir=%s cfg_path=%s device=%s",
                self.model_path,
                cfg_path,
                device,
            )
            self._runtime = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=self.model_path,
                use_fp16=use_fp16,
                device=str(device),
                use_cuda_kernel=use_cuda_kernel,
                use_deepspeed=use_deepspeed,
                use_accel=use_accel,
                use_torch_compile=use_torch_compile,
            )
            logger.info("IndexTTS2 runtime loaded on %s", device)

        for _ in weights:
            pass
        return set()

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        return [{"text": "hello", "_is_dummy": True}] * num_reqs

    def _materialize_audio_array(
        self,
        audio_array: Any,
        suffix: str = ".wav",
    ) -> str | None:
        """Write a resolved ``(samples, sr)`` tuple to a temp WAV path."""
        if audio_array is None:
            return None
        try:
            import numpy as np
            import soundfile as sf

            wav_list, sr_in = audio_array
            wav_np = np.asarray(wav_list, dtype=np.float32)
            if wav_np.ndim > 1:
                wav_np = np.mean(wav_np, axis=-1)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                path = tmp.name
            sf.write(path, wav_np, int(sr_in))
            return path
        except Exception:
            logger.exception("IndexTTS2 failed to materialize reference audio")
            return None

    def _create_stream_gen(self, info: dict[str, Any]):
        """Create an upstream ``infer_generator`` for a request.

        Yields ``(audio_delta, is_last)`` tuples. Audio deltas are 1-D float32
        tensors in [-1, 1] at 22.05 kHz.
        """
        text = str(_pick(info, "text", "") or "")
        if not text.strip():
            logger.warning("IndexTTS2 received empty text; yielding silence.")
            yield torch.zeros((_DEFAULT_SAMPLE_RATE,), dtype=torch.float32), True
            return

        spk_audio_prompt = _pick(info, "spk_audio_prompt", None)
        spk_audio_tmp = None
        if spk_audio_prompt is None:
            spk_audio_tmp = self._materialize_audio_array(_pick(info, "spk_audio_array", None))
            spk_audio_prompt = spk_audio_tmp

        emo_audio_prompt = _pick(info, "emo_audio_prompt", None)
        emo_audio_tmp = None
        if emo_audio_prompt is None:
            emo_audio_tmp = self._materialize_audio_array(_pick(info, "emo_audio_array", None))
            emo_audio_prompt = emo_audio_tmp

        if not spk_audio_prompt:
            raise ValueError("IndexTTS2 requires spk_audio_prompt or spk_audio_array for voice cloning")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        seed = _pick(info, "seed", None)
        if seed is not None:
            seed = int(seed)
            py_rng_state = random.getstate()
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        else:
            py_rng_state = None
            cpu_rng_state = None
            cuda_rng_state = None

        generation_kwargs: dict[str, Any] = {}
        for key, typ in (
            ("top_p", float),
            ("top_k", int),
            ("temperature", float),
            ("num_beams", int),
            ("length_penalty", float),
            ("repetition_penalty", float),
            ("max_mel_tokens", int),
        ):
            val = _pick(info, key, None)
            if val is not None:
                generation_kwargs[key] = typ(val)

        try:
            iterator = self._runtime.infer_generator(
                spk_audio_prompt=str(spk_audio_prompt),
                text=text,
                output_path=output_path,
                emo_audio_prompt=str(emo_audio_prompt) if emo_audio_prompt else None,
                emo_alpha=float(_pick(info, "emo_alpha", 1.0)),
                emo_vector=_pick(info, "emo_vector", None),
                use_emo_text=bool(_pick(info, "use_emo_text", False)),
                emo_text=_pick(info, "emo_text", None),
                use_random=bool(_pick(info, "use_random", False)),
                interval_silence=int(_pick(info, "interval_silence", _DEFAULT_INTERVAL_SILENCE_MS)),
                verbose=bool(_pick(info, "verbose", False)),
                max_text_tokens_per_segment=int(
                    _pick(info, "max_text_tokens_per_segment", _DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT)
                ),
                stream_return=True,
                **generation_kwargs,
            )
            yielded = False
            for event in iterator:
                chunk = _to_mono_1d(event)
                yielded = yielded or chunk.numel() > 0
                yield chunk, False
            if not yielded:
                logger.warning("IndexTTS2 produced no audio chunks for text=%r", text[:80])
        except Exception:
            logger.exception("IndexTTS2 inference failed for text=%r", text[:80])
            raise
        finally:
            for path in (output_path, spk_audio_tmp, emo_audio_tmp):
                if path is None:
                    continue
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove temp audio file: %s", path)
            if py_rng_state is not None:
                random.setstate(py_rng_state)
            if cpu_rng_state is not None:
                torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        yield torch.zeros((0,), dtype=torch.float32), True

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        device = self._device or torch.device("cpu")
        hidden = int(getattr(self.config, "hidden_size", 1280))
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        return torch.zeros((n, hidden), device=device, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        sr = int(getattr(self.config, "audio_sample_rate", _DEFAULT_SAMPLE_RATE))
        sr_tensor = torch.tensor(sr, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._make_dummy_hidden(input_ids)

        infos = runtime_additional_information or [{}]
        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            self._ar_last_chunk_flags = [True] * len(infos)
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "model_outputs": [empty] * len(infos),
                    "sr": [sr_tensor] * len(infos),
                },
            )

        if self._runtime is None:
            raise RuntimeError("IndexTTS2 model not loaded. Was load_weights() called?")

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        last_chunk_flags: list[bool] = []

        for info in infos:
            if info.get("_is_dummy"):
                outputs.append(empty)
                srs.append(sr_tensor)
                last_chunk_flags.append(True)
                continue

            request_key = str(info.get("global_request_id") or info.get("_omni_req_id") or id(info))
            if request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(info)

            generator = self._stream_gens[request_key]
            try:
                chunk, is_last = next(generator)
            except StopIteration:
                self._stream_gens.pop(request_key, None)
                outputs.append(empty)
                last_chunk_flags.append(True)
            else:
                if is_last:
                    self._stream_gens.pop(request_key, None)
                outputs.append(chunk)
                last_chunk_flags.append(bool(is_last))

            srs.append(sr_tensor)

        self._ar_last_chunk_flags = last_chunk_flags
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for req_id in finished_req_ids:
            gen = self._stream_gens.pop(str(req_id), None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    logger.exception("IndexTTS2 failed to close stream generator for request %s", req_id)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Emit EOS for rows whose final audio chunk has been yielded."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            device = self._device or torch.device("cpu")
            hidden_states = torch.zeros((0, 1), device=device, dtype=torch.float32)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = int(getattr(self.config, "vocab_size", 12000))
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros((num_rows, vocab_size), dtype=torch.float32, device=hidden_states.device)
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0

        flags = self._ar_last_chunk_flags
        for row in range(num_rows):
            is_last = flags[row] if row < len(flags) else True
            if is_last:
                logits[row, eos_id] = 1.0e6
            else:
                logits[row, eos_id] = -1.0e9
                logits[row, safe_id] = 1.0e6
        return logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden = int(getattr(self.config, "hidden_size", 1280))
        return torch.zeros((input_ids.shape[0], hidden), device=input_ids.device, dtype=torch.float32)
