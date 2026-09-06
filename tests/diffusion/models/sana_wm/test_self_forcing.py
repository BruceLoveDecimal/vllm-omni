# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-forcing schedule validation, chunking and the Euler update."""

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.self_forcing import (
    SanaWmSelfForcingSchedule,
    create_autoregressive_segments,
    self_forcing_euler_step,
    validate_chunk_split_strategy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_schedule_from_config_defaults() -> None:
    schedule = SanaWmSelfForcingSchedule.from_config(SanaWmConfig())
    assert schedule.timesteps == (1000, 960, 889, 727)
    assert schedule.num_steps == 4
    assert schedule.sigmas == pytest.approx((1.0, 0.96, 0.889, 0.727))
    assert schedule.sigma_pair(0) == pytest.approx((1.0, 0.96))
    assert schedule.sigma_pair(3) == pytest.approx((0.727, 0.0))
    assert schedule.timesteps_tensor().tolist() == [1000.0, 960.0, 889.0, 727.0]


def test_schedule_override_and_config_coercion() -> None:
    schedule = SanaWmSelfForcingSchedule.from_config(SanaWmConfig(), override=[1000, 500, 0])
    assert schedule.timesteps == (1000, 500)
    config = SanaWmConfig.from_dict({"denoising_step_list": [1000, 700, 0], "streaming": True})
    assert config.denoising_step_list == (1000, 700, 0)
    assert config.streaming is True
    assert SanaWmSelfForcingSchedule.from_config(config).timesteps == (1000, 700)


@pytest.mark.parametrize(
    "bad",
    [
        [1000, 960],  # must end with 0
        [0],  # too short
        [960, 900, 0],  # must start at 1000
        [1000, 1000, 0],  # strictly decreasing
        [1000, 960, 970, 0],
        [1000, -5, 0],
    ],
)
def test_schedule_rejects_invalid_lists(bad) -> None:
    with pytest.raises(ValueError):
        SanaWmSelfForcingSchedule.from_step_list(bad)


def test_num_inference_steps_check() -> None:
    schedule = SanaWmSelfForcingSchedule.from_config(SanaWmConfig())
    schedule.check_num_inference_steps(None)
    schedule.check_num_inference_steps(4)
    with pytest.raises(ValueError, match="num_inference_steps must be 4"):
        schedule.check_num_inference_steps(50)


def test_create_autoregressive_segments_first_chunk_absorbs_remainder() -> None:
    # 22 latent frames (169 pixel frames): chunk 0 = [0, 4), then 3-frame chunks.
    assert create_autoregressive_segments(22, 3) == [0, 4, 7, 10, 13, 16, 19, 22]
    assert create_autoregressive_segments(4, 3) == [0, 4]
    assert create_autoregressive_segments(9, 3) == [0, 3, 6, 9]
    with pytest.raises(ValueError):
        create_autoregressive_segments(3, 3)
    with pytest.raises(ValueError):
        create_autoregressive_segments(10, 0)


def test_chunk_split_strategy_must_be_the_implemented_rule() -> None:
    validate_chunk_split_strategy("first_chunk_plus_one")
    with pytest.raises(ValueError, match="chunk_split_strategy"):
        validate_chunk_split_strategy("uniform")
    with pytest.raises(ValueError, match="chunk_split_strategy"):
        create_autoregressive_segments(22, 3, strategy="uniform")


def test_euler_step_matches_per_token_flow_scheduler_convention() -> None:
    """``x_next = x - (sigma - sigma_next) * v`` (the ``scheduler.step(-v)`` sign)."""
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(1, 4, 3, 2, 2, generator=generator, dtype=torch.bfloat16)
    velocity = torch.randn(1, 4, 3, 2, 2, generator=generator, dtype=torch.bfloat16)
    sigma, sigma_next = 0.96, 0.889
    stepped = self_forcing_euler_step(latents, velocity, sigma=sigma, sigma_next=sigma_next)
    expected = (latents.float() - (sigma - sigma_next) * velocity.float()).to(torch.bfloat16)
    assert stepped.dtype == torch.bfloat16
    torch.testing.assert_close(stepped, expected, rtol=0, atol=0)
    # Final step lands on x0 = x - sigma * v.
    x0 = self_forcing_euler_step(latents, velocity, sigma=0.727, sigma_next=0.0)
    torch.testing.assert_close(x0, (latents.float() - 0.727 * velocity.float()).to(torch.bfloat16), rtol=0, atol=0)
    with pytest.raises(ValueError):
        self_forcing_euler_step(latents, velocity[:, :, :2], sigma=1.0, sigma_next=0.5)
