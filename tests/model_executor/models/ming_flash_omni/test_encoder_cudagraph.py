# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

pytestmark = pytest.mark.core_model


def test_default_budget_range_is_bounded():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.ming_flash_omni.encoder_cudagraph import MingVisionCudaGraphMixin

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
        model_config=SimpleNamespace(max_model_len=32768),
    )
    assert MingVisionCudaGraphMixin.get_encoder_cudagraph_budget_range(None, config) == (512, 2048)

    config.scheduler_config.max_num_batched_tokens = 256
    assert MingVisionCudaGraphMixin.get_encoder_cudagraph_budget_range(None, config) == (256, 256)


@torch.inference_mode()
def test_image_graph_packing_and_output_lifetime(ming_encoder_graph_manager):
    manager = ming_encoder_graph_manager
    model = manager.model
    assert supports_encoder_cudagraph(model)
    assert manager.config.out_hidden_size == 96
    assert manager.supports_modality("image")
    assert not manager.supports_modality("audio")

    grid = torch.tensor([[1, 4, 8], [1, 4, 4], [1, 8, 4], [1, 4, 16]])
    pixels = torch.randn(144, 24, device="cuda", dtype=torch.bfloat16)
    inputs = {"pixel_values": pixels, "image_grid_thw": grid}
    expected = model.encoder_eager_forward(inputs).split([8, 4, 8, 16])
    actual = manager.execute(inputs)
    assert manager.graph_hits == 4
    assert manager.graph_misses == 0
    assert len(actual) == 4
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-2)
    saved = [output.clone() for output in actual]

    # Reuse the same budgets with different pixels, aspect ratio and padding.
    smaller_inputs = {
        "pixel_values": torch.randn(32, 24, device="cuda", dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 8, 4]]),
    }
    smaller = manager.execute(smaller_inputs)
    torch.testing.assert_close(smaller[0], model.encoder_eager_forward(smaller_inputs), atol=2e-3, rtol=2e-2)
    for output, snapshot in zip(actual, saved):
        torch.testing.assert_close(output, snapshot, atol=0, rtol=0)

    # A single item beyond the largest budget must use the same projected,
    # normalized eager result, without invalidating earlier cache entries.
    large_inputs = {
        "pixel_values": torch.randn(96, 24, device="cuda", dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 8, 12]]),
    }
    fallback = manager.execute(large_inputs)
    assert manager.graph_misses == 1
    torch.testing.assert_close(fallback[0], model.encoder_eager_forward(large_inputs), atol=2e-3, rtol=2e-2)
    for output, snapshot in zip(actual, saved):
        torch.testing.assert_close(output, snapshot, atol=0, rtol=0)


def test_item_selection_preserves_patch_boundaries(ming_vision_thinker):
    inputs = {
        "pixel_values": torch.arange(80 * 24).reshape(80, 24),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 8], [1, 8, 4]]),
    }
    specs = ming_vision_thinker.get_encoder_cudagraph_item_specs(inputs)
    assert [spec.input_size for spec in specs] == [16, 32, 32]
    assert [spec.output_tokens for spec in specs] == [4, 8, 8]
    selected = ming_vision_thinker.select_encoder_cudagraph_items(inputs, [2, 0])
    torch.testing.assert_close(selected["image_grid_thw"], inputs["image_grid_thw"][[2, 0]])
    torch.testing.assert_close(
        selected["pixel_values"], torch.cat([inputs["pixel_values"][48:], inputs["pixel_values"][:16]])
    )
    empty = ming_vision_thinker.select_encoder_cudagraph_items(inputs, [])
    assert empty["pixel_values"].shape == (0, 24)
    assert empty["image_grid_thw"].shape == (0, 3)


def test_replay_retains_capture_max_seqlen(ming_encoder_graph_manager):
    manager = ming_encoder_graph_manager
    inputs = {
        "pixel_values": torch.randn(16, 24, device="cuda", dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    replay = manager.model.prepare_encoder_cudagraph_replay_buffers(inputs, 2, 0)
    assert "max_seqlen" not in replay.values
    manager.execute(inputs)
    assert manager.budget_graphs["default"][4].input_buffers["max_seqlen"].item() == 16


def test_graph_rejects_encoder_data_parallel(ming_vision_thinker):
    ming_vision_thinker.model_config.multimodal_config.mm_encoder_tp_mode = "data"
    with pytest.raises(ValueError, match="mm_encoder_tp_mode=weights"):
        ming_vision_thinker.get_encoder_cudagraph_config()
