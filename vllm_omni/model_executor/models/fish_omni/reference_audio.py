# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Content-addressed reference codes using the bounded speaker artifact cache."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from vllm_omni.model_executor.models.fish_speech.dac_encoder import encode_reference_audio_codes
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache


def encode_reference_audio_codes_cached(
    model_path: str,
    wav: np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device,
    cache: SpeakerEmbeddingCache,
    revision: str | None = None,
    voice_name: str | None = None,
    voice_created_at: int = 0,
) -> torch.Tensor:
    """Return CPU int64 ``[frames, codebooks]`` codes, encoding only on a miss.

    Hash the actual input, not its URL or a caller-provided digest. Shape and
    sample rate are part of the identity since both affect DAC preprocessing.
    Use float32, just as the encoder does, without changing channel reduction
    or resampling. The model path/revision and device isolate encoder outputs.

    Uploaded voices retain their name/generation in the cache key so existing
    voice invalidation still works. Inline references use a separate namespace.
    Entries own only CPU codes and share the existing byte-bounded LRU budget.
    Concurrent misses for the same key share one encoding within this cache
    instance; different voices/inputs have independent flights.
    """
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().to(device="cpu", dtype=torch.float32).numpy()
    samples = np.ascontiguousarray(wav, dtype=np.float32)
    speaker_name = voice_name.strip() if isinstance(voice_name, str) else ""
    identity = (
        "fish_omni_reference_v1",
        model_path,
        revision,
        str(device),
        int(sample_rate),
        samples.shape,
        "voice" if speaker_name else "inline",
    )
    digest = hashlib.blake2b(digest_size=32)
    digest.update(json.dumps(identity, separators=(",", ":")).encode())
    digest.update(memoryview(samples).cast("B"))
    key = cache.make_cache_key(
        speaker_name or "reference_audio",
        model_type=f"fish_omni:{digest.hexdigest()}",
        created_at=voice_created_at if speaker_name else 0,
    )

    def encode() -> dict[str, torch.Tensor]:
        codes = encode_reference_audio_codes(model_path, samples, sample_rate, device=device).cpu()
        return {"ref_codes_fq": codes}

    return cache.get_or_compute(key, encode)["ref_codes_fq"]
