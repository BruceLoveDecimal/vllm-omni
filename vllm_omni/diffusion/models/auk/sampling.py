# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-only Euler integration with the AuK Base and Flash recipes."""

import math

import torch

FLASH_TIME_GRID = (0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0)


def time_grid(*, flash: bool, steps: int, sway: float, device) -> torch.Tensor:
    if flash:
        return torch.tensor(FLASH_TIME_GRID, device=device, dtype=torch.float32)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise ValueError("num_inference_steps must be an integer between 1 and 1000")
    if not isinstance(sway, (int, float)) or not math.isfinite(sway) or not -1.0 <= sway <= 0.0:
        raise ValueError("sway_sampling_coef must be between -1 and 0")
    times = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float32)
    return times + sway * (torch.cos(torch.pi / 2 * times) - 1 + times)


@torch.inference_mode()
def sample_latents(transformer, noise, reference, semantic, semantic_mask, times, cfg_strength):
    """Integrate one request; no model-global cache or random state is changed."""
    if not isinstance(cfg_strength, (int, float)) or not math.isfinite(cfg_strength) or cfg_strength < 0:
        raise ValueError("cfg_strength must be finite and non-negative")
    ref_mask = torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
    x = noise
    for t0, t1 in zip(times[:-1], times[1:]):
        velocity = transformer(
            x=x,
            text=semantic,
            time=t0,
            c_mask=semantic_mask,
            ref=reference,
            ref_mask=ref_mask,
            cfg_infer=cfg_strength >= 1e-5,
        )
        if cfg_strength >= 1e-5:
            cond, uncond = velocity.chunk(2)
            velocity = cond + (cond - uncond) * cfg_strength
        # torchdiffeq's Euler solver casts the interval to the state dtype.
        x = x + (t1 - t0).to(x.dtype) * velocity
    return x
