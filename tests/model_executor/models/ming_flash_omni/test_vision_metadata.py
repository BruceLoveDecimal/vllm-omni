# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.core_model


@pytest.mark.parametrize("grid", [[[1, 4, 8]], [[1, 8, 4]], [[1, 4, 4], [1, 4, 8]], [[3, 4, 4]]])
@pytest.mark.parametrize("deepstack", [True, False])
@torch.inference_mode()
def test_precomputed_metadata_matches_upstream(ming_vision_encoder, grid, deepstack):
    vision = ming_vision_encoder
    if not deepstack:
        vision.encoder.deepstack_visual_indexes = None
    grid_thw = torch.tensor(grid, dtype=torch.int64)
    num_patches = sum(t * h * w for t, h, w in grid)
    pixels = torch.randn(num_patches, 24, device="cuda", dtype=torch.bfloat16)

    expected = vision(pixels, grid_thw)
    metadata = vision.prepare_encoder_metadata(grid_thw, max_sequences=8, max_seqlen_override=num_patches)
    actual = vision(pixels, encoder_metadata=metadata)
    assert metadata["max_seqlen"].device.type == "cpu"
    # The final repeated offsets describe empty sequences, never negative ones.
    offsets = metadata["cu_seqlens"].cpu()
    assert offsets.shape == (9,)
    assert torch.all(offsets[1:] >= offsets[:-1])
    assert offsets[-1] == num_patches
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


def test_metadata_rejects_device_grid(ming_vision_encoder):
    with pytest.raises(ValueError, match="must stay on CPU"):
        ming_vision_encoder.prepare_encoder_metadata(torch.tensor([[1, 4, 4]], device="cuda"))


@pytest.mark.parametrize("grid", [[[0, 4, 4]], [[1, 3, 4]], [[1, 4, 3]]])
def test_metadata_rejects_invalid_grid(ming_vision_encoder, grid):
    with pytest.raises(ValueError, match="spatial dimensions"):
        ming_vision_encoder.prepare_encoder_metadata(torch.tensor(grid))


def test_metadata_rejects_sequence_overflow(ming_vision_encoder):
    with pytest.raises(ValueError, match="sequence capacity"):
        ming_vision_encoder.prepare_encoder_metadata(torch.tensor([[3, 4, 4]]), max_sequences=2)


def test_metadata_rejects_small_capture_bound(ming_vision_encoder):
    with pytest.raises(ValueError, match="max_seqlen"):
        ming_vision_encoder.prepare_encoder_metadata(torch.tensor([[1, 4, 4]]), max_seqlen_override=8)


@torch.inference_mode()
def test_precomputed_vision_can_be_captured(ming_vision_encoder):
    pixels = torch.randn(32, 24, device="cuda", dtype=torch.bfloat16)
    metadata = ming_vision_encoder.prepare_encoder_metadata(torch.tensor([[1, 4, 8]]))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ming_vision_encoder(pixels, encoder_metadata=metadata)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ming_vision_encoder(pixels, encoder_metadata=metadata)
    pixels.normal_()
    graph.replay()
    expected = ming_vision_encoder(pixels, torch.tensor([[1, 4, 8]]))
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
