# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched AuK DiT integration reproduces each request's single-request result."""

import pytest
import torch

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, integrate_latents, sample_latents

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_dit(device: str) -> AuKTransformer:
    torch.manual_seed(12)
    return (
        AuKTransformer(
            dim=32,
            heads=2,
            dim_head=16,
            ff_mult=2,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=2,
            num_single_layers=2,
        )
        .eval()
        .to(device)
    )


def _request_inputs(device: str, *, tokens: int, ref_frames: int, seed: int) -> dict[str, torch.Tensor]:
    """One request's conditioning with its own lengths and content."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "text": torch.randn(1, tokens, 8, generator=generator, device=device),
        "c_mask": torch.ones(1, tokens, dtype=torch.bool, device=device),
        "ref": torch.randn(1, ref_frames, 4, generator=generator, device=device),
        "ref_mask": torch.ones(1, ref_frames, dtype=torch.bool, device=device),
    }


def _pad_rows(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``[1, n_i, D]`` rows into ``[B, max n_i, D]`` with a validity mask."""
    length = max(row.shape[1] for row in rows)
    batch = torch.zeros(len(rows), length, rows[0].shape[2], device=rows[0].device, dtype=rows[0].dtype)
    mask = torch.zeros(len(rows), length, dtype=torch.bool, device=rows[0].device)
    for i, row in enumerate(rows):
        batch[i, : row.shape[1]] = row[0]
        mask[i, : row.shape[1]] = True
    return batch, mask


# Three requests with different text, reference and target lengths; the middle
# one carries no reference at all, which is the mixed case a real batch produces.
_BATCH_SPECS = (
    {"tokens": 7, "ref_frames": 4, "gen_frames": 9, "seed": 1},
    {"tokens": 4, "ref_frames": 0, "gen_frames": 5, "seed": 2},
    {"tokens": 6, "ref_frames": 2, "gen_frames": 12, "seed": 3},
)


@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_batched_integration_matches_per_request(cfg_strength: float) -> None:
    """Integrate the requests one at a time and as one padded batch; both must agree per row."""
    dit = _make_dit("cpu")
    t_grid = [0.0, 0.4, 1.0]
    singles, noises, inputs = [], [], []
    for spec in _BATCH_SPECS:
        request = _request_inputs("cpu", tokens=spec["tokens"], ref_frames=spec["ref_frames"], seed=spec["seed"])
        noise = torch.randn(1, spec["gen_frames"], 4, generator=torch.Generator().manual_seed(spec["seed"]))
        singles.append(integrate_latents(dit, x=noise, mask=None, **request, t_grid=t_grid, cfg_strength=cfg_strength))
        noises.append(noise)
        inputs.append(request)

    x, mask = _pad_rows(noises)
    text, c_mask = _pad_rows([request["text"] for request in inputs])
    ref, ref_mask = _pad_rows([request["ref"] for request in inputs])
    batched = integrate_latents(
        dit,
        x=x,
        mask=mask,
        text=text,
        c_mask=c_mask,
        ref=ref,
        ref_mask=ref_mask,
        t_grid=t_grid,
        cfg_strength=cfg_strength,
    )

    assert batched.shape == x.shape
    for i, (spec, single) in enumerate(zip(_BATCH_SPECS, singles)):
        torch.testing.assert_close(batched[i : i + 1, : spec["gen_frames"]], single, atol=1e-5, rtol=1e-4)


@torch.inference_mode()
def test_sample_latents_is_the_single_request_case() -> None:
    """The legacy entry point draws its noise and delegates to the batched integrator."""
    dit = _make_dit("cpu")
    request = _request_inputs("cpu", tokens=7, ref_frames=4, seed=5)
    t_grid = [0.0, 0.5, 1.0]

    legacy = sample_latents(
        dit, **request, gen_frames=9, t_grid=t_grid, cfg_strength=2.0, generator=torch.Generator().manual_seed(9)
    )
    noise = torch.randn(9, 4, generator=torch.Generator().manual_seed(9)).unsqueeze(0)
    direct = integrate_latents(dit, x=noise, mask=None, **request, t_grid=t_grid, cfg_strength=2.0)

    torch.testing.assert_close(legacy, direct)


def test_rotary_positions_skip_padding() -> None:
    """Masked positions do not advance the rotary phase, so a padded row matches an unpadded one."""
    dit = _make_dit("cpu")
    mask = torch.tensor([[True, True, False, False, True], [True, True, True, True, True]])

    freqs = dit.rotary_embed(5, mask)
    dense = dit.rotary_embed(3)

    assert freqs.shape == (2, 1, 5, 16)
    # Row 0's valid positions are 0, 1, 2 regardless of the padding in between.
    torch.testing.assert_close(freqs[0, 0, [0, 1, 4]], dense[0, 0])
    torch.testing.assert_close(freqs[1, 0, :3], dense[0, 0])
