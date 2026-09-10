# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio preprocessing for AuK's VAE and semantic conditioner."""

import numpy as np
import torch
import torchaudio


def prepare_audio(value, sample_rate: int) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("AuK reference audio must be a (waveform, sample_rate) tuple")
    waveform, sr = value
    if isinstance(sr, bool) or not isinstance(sr, int) or sr <= 0:
        raise ValueError("Reference sample rate must be a positive integer")
    waveform = torch.as_tensor(np.asarray(waveform) if not isinstance(waveform, torch.Tensor) else waveform)
    waveform = waveform.detach().float().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2) or waveform.shape[-1] == 0:
        raise ValueError("Reference audio must have shape [samples] or [1 or 2, samples]")
    if not torch.isfinite(waveform).all():
        raise ValueError("Reference audio contains NaN or Inf")
    waveform = waveform.mean(dim=0, keepdim=True)
    return torchaudio.transforms.Resample(sr, sample_rate)(waveform) if sr != sample_rate else waveform


def resample_semantic_audio(waveform: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Match qwen-omni-utils' soxr_hq resampling and ceil-length padding."""
    if original_sr == target_sr:
        return waveform
    import soxr

    length = int(np.ceil(waveform.shape[-1] * (float(target_sr) / original_sr)))
    resampled = soxr.resample(waveform, original_sr, target_sr, quality="HQ")
    if len(resampled) < length:
        resampled = np.pad(resampled, (0, length - len(resampled)))
    return resampled[:length].astype(waveform.dtype, copy=False)
