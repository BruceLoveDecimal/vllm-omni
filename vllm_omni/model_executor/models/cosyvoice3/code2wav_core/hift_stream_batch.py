# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched streaming HiFT over request-owned bounded mel windows.

Rows are left aligned and right padded. The F0 predictor reads at most
``trim`` frames ahead and the decoder at most ``conv_pre_look_right`` frames;
every other layer, the oscillator phase and the position-indexed noise are
causal. A non-final row emits only samples that end before both look-ahead
spans and the ISTFT frame held back for overlap-add, so its padding never
reaches them. A final row emits its tail, so it always runs at exact width.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .packed_dit import to_device_async

# Non-final batches pad to a row slot and a frame step so convolution shapes
# recur: cuDNN reuses its plans and loaded kernels instead of building them for
# each new rows-by-width pair, and decode graphs, when on, stay few.
_ROW_SLOTS = (1, 2, 4, 8, 16)
_FRAME_STEP = 64


@dataclass
class HiftWindow:
    """One request's streaming HiFT input and the samples it releases."""

    mel: torch.Tensor  # (1, mels, frames): F0 input, its context margin first
    margin: int  # F0 context frames before the decoded window
    phase_acc: torch.Tensor | None  # (1, 1, harmonics) phase carried into this window
    noise_offset: int  # absolute sample index of the decoded window's start
    next_overlap: int  # frames the next window re-reads; the phase carry point
    emit_start: int  # emitted sample range of the decoded window
    emit_end: int


@contextmanager
def _tf32_disabled():
    """F0 runs in FP32 without TF32, as in CausalHiFTGenerator.inference."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class StreamingHiFTBatcher:
    """Run the streaming HiFT step of several requests as one call.

    Mirrors ``CausalHiFTGenerator.inference`` with ``SineGen2``: F0, the
    harmonic source and the decoder take every row at once; per-row margins,
    noise offsets and phase carries become index tensors instead of host
    loops. Device copies of the fixed noise tables are made on first use.
    """

    def __init__(self, hift: torch.nn.Module) -> None:
        self.hift = hift
        self.scale = int(hift.m_source.l_sin_gen.upsample_scale)
        self.trim = int(hift.f0_predictor.condnet[0].causal_padding)
        self.look_right = int(hift.conv_pre_look_right)
        self._noise: torch.Tensor | None = None
        self._rand_ini: torch.Tensor | None = None

    def _tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._noise is None or self._noise.device != device:
            sine_gen = self.hift.m_source.l_sin_gen
            self._noise = sine_gen.sine_waves[0].to(device)
            self._rand_ini = sine_gen.rand_ini.to(device)
        return self._noise, self._rand_ini

    @torch.inference_mode()
    def run(self, windows: list[HiftWindow], *, finalize: bool) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Return each window's emitted speech ``(1, 1, samples)`` and, for
        non-final windows, the phase carried into its next window."""
        if finalize and len(windows) != 1:
            raise ValueError("A final window emits its tail and runs alone at exact width")
        widths = [window.mel.shape[-1] for window in windows]
        rows, width = len(windows), max(widths)
        if not finalize:
            rows = next(slots for slots in _ROW_SLOTS if slots >= rows)
            width = -(-width // _FRAME_STEP) * _FRAME_STEP
        first = windows[0].mel
        device = first.device
        mel = first.new_zeros((rows, first.shape[1], width))
        for row, window in enumerate(windows):
            mel[row, :, : widths[row]] = window.mel[0]
        with _tf32_disabled():
            f0 = self.hift.f0_predictor(mel.float(), finalize=True).to(mel.dtype)
        margins = [window.margin for window in windows] + [0] * (rows - len(windows))
        if finalize:
            # Exact width: a gathered row would repeat its last frame past its end.
            f0, mel = f0[:, margins[0] :], mel[:, :, margins[0] :]
        elif any(margins):
            index = torch.arange(width).unsqueeze(0) + torch.tensor(margins).unsqueeze(1)
            index = to_device_async(index.clamp_(max=width - 1), device)
            f0 = f0.gather(1, index)
            mel = mel.gather(2, index.unsqueeze(1).expand_as(mel))
        source, phase = self._source(f0, windows)
        speech = self.hift.decode(x=mel, s=source, finalize=True)
        speeches = [
            speech[row : row + 1, window.emit_start : window.emit_end].reshape(1, 1, -1)
            for row, window in enumerate(windows)
        ]
        if finalize:
            return speeches, [None]
        return speeches, self._carry(phase, windows, widths)

    def _source(self, f0: torch.Tensor, windows: list[HiftWindow]) -> tuple[torch.Tensor, torch.Tensor]:
        """SineGen2 + SourceModuleHnNSF for every row; returns the source and
        the frame-rate phase (mod-1 cycles, carry already added)."""
        m_source = self.hift.m_source
        sine_gen = m_source.l_sin_gen
        rows = f0.shape[0]
        noise_table, rand_ini = self._tables(f0.device)
        upsampled = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        rad = (upsampled * sine_gen.harmonic_ids / sine_gen.sampling_rate) % 1
        fresh = [window.phase_acc is None for window in windows] + [True] * (rows - len(windows))
        if any(fresh):
            # A stream's first window starts at a random harmonic phase.
            mask = to_device_async(torch.tensor(fresh, dtype=rad.dtype), f0.device)
            rad[:, 0, :] = rad[:, 0, :] + mask[:, None] * rand_ini
        rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.scale, mode="linear").transpose(1, 2)
        carried = [
            rad.new_zeros((1, 1, rad.shape[-1])) if window.phase_acc is None else window.phase_acc for window in windows
        ]
        carried += [rad.new_zeros((1, 1, rad.shape[-1]))] * (rows - len(windows))
        phase = torch.cumsum(rad, dim=1) + torch.cat(carried)
        angle = phase * 2 * np.pi
        angle = F.interpolate(angle.transpose(1, 2) * self.scale, scale_factor=self.scale, mode="nearest")
        sine_waves = torch.sin(angle.transpose(1, 2)) * sine_gen.sine_amp
        uv = (upsampled > sine_gen.voiced_threshold).type(torch.float32)
        noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
        offsets = [window.noise_offset for window in windows] + [0] * (rows - len(windows))
        offsets = to_device_async(torch.tensor(offsets, dtype=torch.long), f0.device)
        positions = (offsets[:, None] + torch.arange(sine_waves.shape[1], device=f0.device)) % noise_table.shape[0]
        sine_waves = sine_waves * uv + noise_amp * noise_table[positions]
        source = m_source.l_tanh(m_source.l_linear(sine_waves)).transpose(1, 2)
        return source, phase

    def _carry(self, phase: torch.Tensor, windows: list[HiftWindow], widths: list[int]) -> list[torch.Tensor | None]:
        """Phase at the frame before the next window's start, as
        ``_carry_phase_at_boundary`` takes it from a non-final window."""
        carries: list[torch.Tensor | None] = []
        for row, (window, frames) in enumerate(zip(windows, widths)):
            decoded = frames - window.margin
            if window.next_overlap >= decoded:
                carries.append(window.phase_acc)
            else:
                index = decoded - window.next_overlap - 1
                carries.append(phase[row : row + 1, index : index + 1] % 1)
        return carries

    def emit_range(self, decoded_frames: int, new_frames: int, *, finalize: bool) -> tuple[int, int]:
        """Samples a window of ``decoded_frames`` releases for ``new_frames``
        new mel frames, matching ``_stream_hift_from_feat``."""
        samples_per_frame = self.scale
        if finalize:
            end = decoded_frames * samples_per_frame
            released = (new_frames + self.trim + self.look_right + 1) * samples_per_frame
        else:
            end = (decoded_frames - self.trim - self.look_right - 1) * samples_per_frame
            released = new_frames * samples_per_frame
        return max(0, end - released), end
