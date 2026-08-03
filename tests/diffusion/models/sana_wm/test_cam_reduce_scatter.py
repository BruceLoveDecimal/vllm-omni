# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Equivalence tests for SANA-WM's camera-branch reduce-scatter.

``out_proj_cam`` is row-parallel, so every rank produces a partial sum over the
full hidden width, while the GDN stream it is added to is TP-local. Instead of
all-reducing and then discarding all but one slice, the branch reduce-scatters
straight to the local shard. These tests pin the shard each rank ends up with,
including the bias slice.

CPU-only: the collective and the rank/world-size lookups are mocked, so a
single process can simulate an arbitrary TP degree.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.diffusion.models.sana_wm.sana_wm_transformer"

HIDDEN = 24


def _host(bias: torch.Tensor | None) -> SimpleNamespace:
    """A stand-in ``self`` exposing only what the method reads."""
    return SimpleNamespace(out_proj_cam=SimpleNamespace(bias=bias))


def _fake_reduce_scatter(rank: int, tp_size: int):
    """Sum the per-rank partials, then hand back chunk ``rank``.

    The test drives a single process, so the "partials" of the other ranks are
    already folded into the input; scattering is all this has to model.
    """

    def _impl(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1
        return tensor.chunk(tp_size, dim=-1)[rank]

    return _impl


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("rank", [0, 1])
def test_matches_all_reduce_then_slice(tp_size: int, rank: int) -> None:
    """The reduce-scatter shard equals the slice the old all-reduce path took."""
    local_width = HIDDEN // tp_size
    torch.manual_seed(0)
    contrib = torch.randn(2, 3, HIDDEN)
    bias = torch.randn(HIDDEN)
    reference = torch.zeros(2, 3, local_width)

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=tp_size),
        patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=rank),
        patch(f"{_MODULE}.tensor_model_parallel_reduce_scatter", side_effect=_fake_reduce_scatter(rank, tp_size)),
    ):
        out = SanaWmSelfAttention._reduce_scatter_cam_contrib(_host(bias), contrib, reference)

    start = rank * local_width
    expected = (contrib + bias)[..., start : start + local_width]
    assert out.shape == (2, 3, local_width)
    torch.testing.assert_close(out, expected)


def test_tp1_adds_bias_without_a_collective() -> None:
    contrib = torch.randn(2, 3, HIDDEN)
    bias = torch.randn(HIDDEN)

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{_MODULE}.tensor_model_parallel_reduce_scatter") as collective,
    ):
        out = SanaWmSelfAttention._reduce_scatter_cam_contrib(_host(bias), contrib, contrib)

    assert collective.call_count == 0
    torch.testing.assert_close(out, contrib + bias)


def test_rejects_a_width_that_does_not_shard() -> None:
    """A local width that is not 1/tp of the contribution is a wiring bug."""
    contrib = torch.randn(1, 2, HIDDEN)
    reference = torch.zeros(1, 2, HIDDEN // 3)  # tp=2 would need HIDDEN // 2

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=2),
        patch(f"{_MODULE}.tensor_model_parallel_reduce_scatter"),
        pytest.raises(ValueError, match="camera contribution width mismatch"),
    ):
        SanaWmSelfAttention._reduce_scatter_cam_contrib(_host(None), contrib, reference)
