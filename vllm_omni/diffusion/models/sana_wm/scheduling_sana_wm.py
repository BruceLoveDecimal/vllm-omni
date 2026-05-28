# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 flow scheduler helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT = 9.8


def shift_flow_timestep(
    timestep: torch.Tensor,
    shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT,
) -> torch.Tensor:
    """Apply the inference flow-shift transform used by SANA-WM."""

    return shift * timestep / (1.0 + (shift - 1.0) * timestep)


class SanaWmFlowMatchScheduler:
    """Production flow-DPM-Solver++ scheduler wrapping diffusers.

    Replaces the shifted-Euler smoke with the production ``vis_sampler:
    flow_dpm-solver`` path (``inference_flow_shift=9.8``, 20-30 steps).
    Provides ``add_noise`` for first-frame latent conditioning (A.1).
    """

    def __init__(
        self,
        num_inference_steps: int,
        shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("Sana-WM scheduler num_inference_steps must be positive.")
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        from diffusers import DPMSolverMultistepScheduler

        self._sched = DPMSolverMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )
        self._timesteps_device: torch.device | None = None

    def _ensure_timesteps(self, device: torch.device) -> None:
        if self._timesteps_device != device:
            self._sched.set_timesteps(self.num_inference_steps, device=device)
            self._timesteps_device = device

    def timesteps(self, *, device: torch.device) -> torch.Tensor:
        self._ensure_timesteps(device)
        return self._sched.timesteps

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        return self._sched.step(noise_pred, timestep, latents).prev_sample

    def _sigma_index_for(self, timestep: torch.Tensor) -> int:
        """Return the index ``i`` such that ``self._sched.timesteps[i] == timestep``.

        The DPMSolverMultistepScheduler keeps a `(num_steps,)` ``timesteps``
        tensor and a `(num_steps + 1,)` ``sigmas`` tensor; the terminal
        sigma at index ``num_steps`` is the ``0`` we use as ``sigma_next``
        for the last step. We use ``argmin(abs(diff))`` so the lookup
        survives the fp16/bf16/fp32 round-trips of ``timestep``.
        """
        ts = self._sched.timesteps
        t_val = timestep.to(device=ts.device, dtype=ts.dtype)
        return int((ts - t_val).abs().argmin().item())

    def step_flow_euler_per_token(
        self,
        noise_pred: torch.Tensor,           # (B, C, F, H, W)
        timestep: torch.Tensor,             # scalar current step
        latents: torch.Tensor,              # (B, C, F, H, W)
        per_token_timesteps: torch.Tensor,  # (B, F*H*W) per-token integer timestep
    ) -> torch.Tensor:
        """Per-token flow-matching Euler step matching NVlabs ``LTXFlowEuler.sample``.

        NVlabs invokes the diffusers ``FlowMatchEulerDiscreteScheduler.step``
        with the noise prediction sign-flipped:

            prev = sample - (sigma_next - sigma) * (-noise_pred)
                 = sample + (sigma_next - sigma) * noise_pred

        For conditioning tokens (``per_token_timesteps[i] == 0``), the
        per-token sigma is already 0, so ``sigma_next - sigma = 0`` and
        the step is a no-op for them by construction. That makes the
        ``tokens_to_denoise_mask`` ``torch.where`` an exact no-op safety
        belt rather than a hard discontinuity.

        Reuses the existing ``DPMSolverMultistepScheduler`` sigma table
        (computed under ``use_flow_sigmas=True`` with the configured
        ``flow_shift``) so this method does not re-implement scheduling.

        Args:
            noise_pred: ``(B, C, F, H, W)`` model output.
            timestep: scalar integer timestep at this sampling step.
            latents: ``(B, C, F, H, W)`` current latent.
            per_token_timesteps: ``(B, F*H*W)`` per-token current
                timestep, with conditioning tokens set to 0.

        Returns:
            ``(B, C, F, H, W)`` updated latent (``prev_sample``).
        """
        if noise_pred.shape != latents.shape:
            raise ValueError(
                "Sana-WM per-token step: noise_pred / latents shape mismatch "
                f"({tuple(noise_pred.shape)} vs {tuple(latents.shape)})."
            )
        if latents.ndim != 5:
            raise ValueError(
                f"Sana-WM per-token step expects (B, C, F, H, W); got {tuple(latents.shape)}."
            )
        self._ensure_timesteps(latents.device)
        sigmas = self._sched.sigmas.to(device=latents.device, dtype=torch.float32)
        cur_idx = self._sigma_index_for(timestep)
        sigma_cur_scalar = sigmas[cur_idx]
        sigma_next_scalar = sigmas[cur_idx + 1] if cur_idx + 1 < sigmas.numel() else sigmas[-1]
        # Use the SCHEDULER's actual sigma value for the current step, not
        # a hand-computed `t/num_train`. Under `use_flow_sigmas=True` the
        # sigmas have a flow-shift baked in (shift=9.8 for SANA-WM) so the
        # mapping t → sigma is not the identity. Tokens whose per-token
        # timestep is 0 are conditioning tokens at sigma=0; everything
        # else carries this step's `sigma_cur_scalar`.
        is_cond = per_token_timesteps.float().to(latents.device) < 1e-6
        sigma_cur_token = torch.where(
            is_cond,
            torch.zeros_like(is_cond, dtype=torch.float32),
            sigma_cur_scalar.expand_as(is_cond).to(torch.float32),
        )
        sigma_next_token = torch.where(
            is_cond,
            sigma_cur_token,
            sigma_next_scalar.expand_as(is_cond).to(torch.float32),
        )
        # dt is negative for denoising (sigma decreases from 1 toward 0).
        dt = (sigma_next_token - sigma_cur_token)  # (B, FHW)

        # Sign convention follows NVlabs `LTXFlowEuler.sample`, which calls
        # the diffusers FlowMatchEulerDiscrete step with `-noise_pred`. The
        # diffusers step is `prev = sample - (sigma_next - sigma) * model_output`,
        # so substituting `-noise_pred` for model_output gives
        # `prev = sample + (sigma_next - sigma) * noise_pred`. The direct
        # form below applies the negation explicitly so the math line is
        # `latents + dt * (-noise_pred)`.
        #
        # 2026-05-28 GPU A/B (audit §6.13c verification): the
        # `-noise_pred` direction gives MAE=69.06 / PSNR=9.68 on the
        # 9-frame harness. `+noise_pred` gives MAE=112.5 / PSNR=6.0 —
        # clearly the wrong direction. `VLLM_OMNI_SANA_WM_FLIP_FLOW_SIGN=1`
        # swaps to the opposite convention for future verification.
        flip = os.environ.get("VLLM_OMNI_SANA_WM_FLIP_FLOW_SIGN", "").lower() in {
            "1", "true", "yes", "on"
        }
        sign = 1.0 if flip else -1.0  # default: NVlabs -noise_pred
        b, c, f, h, w = latents.shape
        latents_flat = latents.permute(0, 2, 3, 4, 1).reshape(b, -1, c).float()
        noise_flat = noise_pred.permute(0, 2, 3, 4, 1).reshape(b, -1, c).float()
        prev_flat = latents_flat + dt.unsqueeze(-1) * (sign * noise_flat)
        prev = prev_flat.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return prev.to(latents.dtype)

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Linear flow interpolation: sigma*noise + (1-sigma)*sample.

        ``sigma = timestep / num_train_timesteps`` maps the scheduler's integer
        timestep (0–1000 range) back to the [0, 1] noise level.
        """
        num_train = float(self._sched.config.num_train_timesteps)
        sigma = (timestep.float() / num_train).clamp(0.0, 1.0)
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return (sigma * noise + (1.0 - sigma) * sample).to(sample.dtype)


@dataclass(frozen=True)
class SanaWmFlowDpmScheduler:
    """Shifted-Euler smoke scheduler — kept for backward-compat with existing tests.

    New code should use :class:`SanaWmFlowMatchScheduler`.
    """

    num_inference_steps: int
    shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 0:
            raise ValueError("Sana-WM scheduler num_inference_steps must be positive.")

    def timesteps(self, *, device: torch.device) -> torch.Tensor:
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device, dtype=torch.float32)[:-1]
        return shift_flow_timestep(base, self.shift)

    def deltas(self, *, device: torch.device) -> torch.Tensor:
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device, dtype=torch.float32)
        shifted = shift_flow_timestep(base, self.shift)
        return shifted[:-1] - shifted[1:]

    def step(self, latents: torch.Tensor, noise_pred: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        while delta.ndim < latents.ndim:
            delta = delta.unsqueeze(-1)
        return latents - delta.to(device=latents.device, dtype=latents.dtype) * noise_pred
