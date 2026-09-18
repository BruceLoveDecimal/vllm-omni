# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.ming_flash_omni.encoder_cudagraph import pad_cumulative_lengths

pytestmark = pytest.mark.core_model


@pytest.mark.parametrize("grid", [[[3, 4, 4]], [[8, 2, 2]], [[2, 4, 4], [4, 2, 2]], [[5, 4, 4]]])
@torch.inference_mode()
def test_video_graph_parity(ming_encoder_graph_manager, grid):
    manager = ming_encoder_graph_manager
    assert manager.supports_modality("video")
    num_patches = sum(t * h * w for t, h, w in grid)
    inputs = {
        "pixel_values_videos": torch.randn(num_patches, 24, device="cuda", dtype=torch.bfloat16),
        "video_grid_thw": torch.tensor(grid),
    }
    lengths = [t * h * w // 4 for t, h, w in grid]
    expected = manager.model.encoder_eager_forward(inputs).split(lengths)
    actual = manager.execute(inputs)
    assert len(actual) == len(grid)
    assert manager.graph_hits == sum(length <= 16 for length in lengths)
    assert manager.graph_misses == sum(length > 16 for length in lengths)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-2)


@torch.inference_mode()
def test_images_and_videos_share_graphs(ming_encoder_graph_manager):
    manager = ming_encoder_graph_manager
    saved_outputs = []
    for grid, modality in [([[1, 4, 8]], "image"), ([[8, 2, 2]], "video"), ([[1, 8, 4]], "image")]:
        pixel_key = "pixel_values"
        if modality == "video":
            pixel_key = "pixel_values_videos"
        inputs = {
            pixel_key: torch.randn(32, 24, device="cuda", dtype=torch.bfloat16),
            f"{modality}_grid_thw": torch.tensor(grid),
        }
        result = manager.execute(inputs)[0]
        torch.testing.assert_close(result, manager.model.encoder_eager_forward(inputs), atol=2e-3, rtol=2e-2)
        saved_outputs.append((result, result.clone()))
    assert manager.graph_hits == 3
    assert len(manager.budget_graphs["default"]) == 3
    for output, snapshot in saved_outputs:
        torch.testing.assert_close(output, snapshot, atol=0, rtol=0)


def test_cumulative_length_padding():
    destination = torch.empty(7, dtype=torch.int32)
    pad_cumulative_lengths(destination, torch.tensor([0, 4, 12], dtype=torch.int32), spatial_merge_unit=4)
    torch.testing.assert_close(destination, torch.tensor([0, 4, 12, 20, 20, 20, 20], dtype=torch.int32))
    # A shorter subsequent request must also replace the previously used tail.
    pad_cumulative_lengths(destination, torch.tensor([0, 4], dtype=torch.int32), spatial_merge_unit=4)
    torch.testing.assert_close(destination, torch.tensor([0, 4, 20, 20, 20, 20, 20], dtype=torch.int32))
    with pytest.raises(ValueError, match="token budget"):
        pad_cumulative_lengths(destination, torch.arange(8, dtype=torch.int32), spatial_merge_unit=4)


def test_processor_keeps_grid_metadata_on_cpu():
    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_thinker import (
        MingFlashOmniThinkerMultiModalProcessor,
    )

    processor = object.__new__(MingFlashOmniThinkerMultiModalProcessor)
    fields = processor._get_mm_fields_config(
        {"image_grid_thw": torch.tensor([[1, 4, 4]]), "video_grid_thw": torch.tensor([[3, 4, 4]])}, {}
    )
    assert fields["image_grid_thw"].field.keep_on_cpu
    assert fields["video_grid_thw"].field.keep_on_cpu
