# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Self-forcing rollout of the SolarWM-H3 Stage-2 student.

Each five-latent chunk is denoised with four evaluations on the shift-12
rectified-flow grid. After every evaluation the clean estimate is re-noised to
the next grid level with fresh noise; the last estimate is committed and a
fifth forward at ``t = 1`` writes its raw keys and values into the window
cache. MiniMax-H3 predicts the data-ward velocity ``x0 - noise`` with ``t = 1``
clean, so ``x0 = x_t + (1 - t) * v``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
)

from .camera import CameraProjection, camera_projection
from .layout import (
    SOLARWM_H3_CHUNK_LATENTS,
    SOLARWM_H3_LATENT_CHANNELS,
    SOLARWM_H3_LATENT_HEIGHT,
    SOLARWM_H3_LATENT_WIDTH,
    SOLARWM_H3_PATCH_SIZE,
    SOLARWM_H3_ROWS_PER_LATENT,
    SOLARWM_H3_WINDOW_CHUNKS,
    VIDEO_TAG,
    ChunkWindow,
    PrefixLayout,
    RolloutGeometry,
    video_position_grid,
)
from .solarwm_h3_transformer import (
    SolarWMAttentionPlan,
    SolarWMChunkInputs,
    SolarWMH3DiTModel,
    SolarWMKVCache,
)

SOLARWM_H3_VIDEO_SHIFT = 12.0
SOLARWM_H3_AUDIO_SHIFT = 3.0
SOLARWM_H3_DENOISE_STEPS = 4
SOLARWM_H3_ANCHOR_TIMESTEP = 0.999
SOLARWM_H3_TEXT_TIMESTEP = 1.0
SOLARWM_H3_CLEAN_TIMESTEP = 1.0


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """MiniMax-H3's exponential time shift ``s * sigma / (1 + (s - 1) * sigma)``."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def denoise_timesteps(
    *, num_steps: int = SOLARWM_H3_DENOISE_STEPS, shift: float = SOLARWM_H3_VIDEO_SHIFT
) -> list[float]:
    """Ascending model timesteps ``1 - sigma`` for the ``num_steps + 1`` point shifted grid."""
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    sigmas = shift_sigma(base, shift)
    return [float(1.0 - sigma) for sigma in sigmas[:-1]]


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    return minimax_h3_patchify_video_latent(latents, patch_size=SOLARWM_H3_PATCH_SIZE)


def unpatchify_chunk(rows: torch.Tensor) -> torch.Tensor:
    return minimax_h3_unpatchify_video_tokens(
        rows,
        latent_shape=(
            SOLARWM_H3_CHUNK_LATENTS,
            SOLARWM_H3_LATENT_HEIGHT // 2,
            SOLARWM_H3_LATENT_WIDTH // 2,
            SOLARWM_H3_LATENT_CHANNELS,
        ),
        patch_size=SOLARWM_H3_PATCH_SIZE,
    )


@dataclass
class RolloutConditions:
    """Request-level inputs shared by every chunk forward."""

    prompt_embeds: torch.Tensor
    """``[text_len, 5120]`` bfloat16 Qwen hidden states."""
    layout: PrefixLayout
    anchor_rows: torch.Tensor
    """``[1008, 96]`` noised first-frame latent rows."""
    audio_rows: torch.Tensor
    """``[526, 32]`` noised silence rows."""
    audio_timestep: float
    viewmats: torch.Tensor
    """``[1 + rollout_latents, 4, 4]`` relative world-to-camera poses; slot 0 is the anchor."""
    geometry: RolloutGeometry


class SolarWMRollout:
    """Run the causal rollout for one request."""

    def __init__(
        self,
        transformer: SolarWMH3DiTModel,
        conditions: RolloutConditions,
        *,
        cache_device: torch.device,
        generator: torch.Generator,
    ) -> None:
        self.transformer = transformer
        self.conditions = conditions
        self.generator = generator
        layout = conditions.layout
        device = conditions.prompt_embeds.device
        self.device = device
        self.cache = SolarWMKVCache(num_layers=len(transformer.blocks), device=cache_device)

        self.prefix_freqs = transformer.rope_frequencies(layout.position_ids).to(device)
        # Window-local positions restart at the text length for every window
        # size the six-chunk schedule can produce.
        self.window_freqs = {
            latents: transformer.rope_frequencies(video_position_grid(latents, float(layout.text_len))).to(device)
            for latents in range(SOLARWM_H3_CHUNK_LATENTS, SOLARWM_H3_WINDOW_CHUNKS * SOLARWM_H3_CHUNK_LATENTS + 1, 5)
        }
        # Prefix rows carry identity poses (the anchor is the reference frame),
        # and PRoPE still applies the fixed focal lift to them.
        prefix_views = torch.eye(4, dtype=torch.float32, device=device).repeat(layout.prefix_len, 1, 1)
        prefix_views[layout.text_len : layout.condition_len] = conditions.viewmats[0].to(device)
        attention_dtype = transformer.blocks[0].attn.qkv_proj.weight.dtype
        self.prefix_projection = camera_projection(prefix_views, attention_dtype)
        self.frame_projection = camera_projection(conditions.viewmats.to(device), attention_dtype)

        self.text_pos = torch.arange(layout.text_len, device=device)
        anchor_pos = torch.arange(layout.text_len, layout.condition_len, device=device)
        chunk_pos = torch.arange(layout.prefix_len, layout.sequence_len, device=device)
        self.img_pos = torch.cat((anchor_pos, chunk_pos))
        self.audio_pos = torch.arange(layout.condition_len, layout.prefix_len, device=device)
        chunk_tags = torch.full((chunk_pos.shape[0],), VIDEO_TAG, dtype=torch.long)
        self.token_tags = torch.cat((layout.token_tags, chunk_tags)).to(device)

    def _frame_rows(self, latent_start: int, latent_stop: int) -> torch.Tensor:
        """Camera slots of every row in latents ``[start, stop)``; slot 0 is the anchor."""
        latents = torch.arange(1 + latent_start, 1 + latent_stop, device=self.device)
        return latents.repeat_interleave(SOLARWM_H3_ROWS_PER_LATENT)

    def _plan(self, window: ChunkWindow, *, commit: bool) -> SolarWMAttentionPlan:
        layout = self.conditions.layout
        window_freqs = self.window_freqs[window.local_latents]
        current_rows = self._frame_rows(window.current_latent, window.stop_latent)
        window_rows = self._frame_rows(window.first_latent, window.stop_latent)
        current = self.frame_projection
        row_projection = CameraProjection(
            query=torch.cat((self.prefix_projection.query, current.query[current_rows])),
            key_value=torch.cat((self.prefix_projection.key_value, current.key_value[current_rows])),
            output=torch.cat((self.prefix_projection.output, current.output[current_rows])),
        )
        current_start = window.local_latents - SOLARWM_H3_CHUNK_LATENTS
        return SolarWMAttentionPlan(
            window=window,
            condition_len=layout.condition_len,
            prefix_len=layout.prefix_len,
            row_freqs=torch.cat((self.prefix_freqs, window_freqs[current_start * SOLARWM_H3_ROWS_PER_LATENT :])),
            row_projection=row_projection,
            window_freqs=window_freqs,
            window_key_value=current.key_value[window_rows],
            cache=self.cache,
            commit=commit,
        )

    def _inputs(self, chunk_latents: torch.Tensor, timestep: float) -> SolarWMChunkInputs:
        layout = self.conditions.layout
        row_timesteps = torch.empty(layout.sequence_len, dtype=torch.float32, device=self.device)
        row_timesteps[: layout.text_len] = SOLARWM_H3_TEXT_TIMESTEP
        row_timesteps[layout.text_len : layout.condition_len] = SOLARWM_H3_ANCHOR_TIMESTEP
        row_timesteps[layout.condition_len : layout.prefix_len] = self.conditions.audio_timestep
        row_timesteps[layout.prefix_len :] = timestep
        unique_timesteps, inverse_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        return SolarWMChunkInputs(
            prompt_embeds=self.conditions.prompt_embeds,
            video_rows=torch.cat((self.conditions.anchor_rows, patchify_latents(chunk_latents))),
            audio_rows=self.conditions.audio_rows,
            text_pos=self.text_pos,
            img_pos=self.img_pos,
            audio_pos=self.audio_pos,
            token_tags=self.token_tags,
            unique_timesteps=unique_timesteps,
            inverse_indices=inverse_indices,
        )

    def _predict_clean(self, noisy: torch.Tensor, timestep: float, window: ChunkWindow) -> torch.Tensor:
        velocity_rows = self.transformer.forward_chunk(self._inputs(noisy, timestep), self._plan(window, commit=False))
        velocity = unpatchify_chunk(velocity_rows.float())
        return noisy + (1.0 - timestep) * velocity

    def _commit(self, clean: torch.Tensor, window: ChunkWindow) -> None:
        self.transformer.forward_chunk(self._inputs(clean, SOLARWM_H3_CLEAN_TIMESTEP), self._plan(window, commit=True))

    def _noise_like(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.randn(latents.shape, generator=self.generator, dtype=torch.float32, device=self.generator.device)

    @torch.inference_mode()
    def run(self, noise: torch.Tensor, on_chunk: Callable[[int], None] | None = None) -> torch.Tensor:
        """``noise``: ``[1, 24, rollout_latents, 48, 84]`` -> generated latents of the same shape."""
        timesteps = denoise_timesteps()
        generated = torch.empty_like(noise)
        try:
            for chunk_index in range(self.conditions.geometry.num_chunks):
                window = ChunkWindow(chunk_index)
                start = window.current_latent
                stop = window.stop_latent
                current = noise[:, :, start:stop]
                for step, timestep in enumerate(timesteps):
                    clean = self._predict_clean(current, timestep, window)
                    if step + 1 < len(timesteps):
                        next_timestep = timesteps[step + 1]
                        current = next_timestep * clean + (1.0 - next_timestep) * self._noise_like(clean)
                generated[:, :, start:stop] = clean
                self._commit(clean, window)
                if on_chunk is not None:
                    on_chunk(chunk_index)
        finally:
            self.cache.clear()
        return generated


__all__ = [
    "SOLARWM_H3_ANCHOR_TIMESTEP",
    "SOLARWM_H3_AUDIO_SHIFT",
    "SOLARWM_H3_DENOISE_STEPS",
    "SOLARWM_H3_VIDEO_SHIFT",
    "RolloutConditions",
    "SolarWMRollout",
    "denoise_timesteps",
    "patchify_latents",
    "shift_sigma",
    "unpatchify_chunk",
]
