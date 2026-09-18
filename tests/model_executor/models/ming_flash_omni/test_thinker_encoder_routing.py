# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.core_model


@torch.inference_mode()
def test_thinker_mixed_embedding_contract(ming_encoder_graph_manager, monkeypatch):
    manager = ming_encoder_graph_manager
    thinker = manager.model.encoder_cudagraph_model
    audio_output = torch.full((3, 96), 0.125, device="cuda", dtype=torch.bfloat16)
    # Audio is outside this feature. Keep its output identifiable to check
    # that visual graph results preserve the thinker's mixed-modality order.
    monkeypatch.setattr(thinker, "extract_audio_feature", lambda *args: (audio_output,))
    image_inputs = {
        "pixel_values": torch.randn(32, 24, device="cuda", dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 4, 8]]),
    }
    video_inputs = {
        "pixel_values_videos": torch.randn(32, 24, device="cuda", dtype=torch.bfloat16),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }
    inputs = {
        **image_inputs,
        "audio_feats": torch.zeros(1, 8, 80, device="cuda", dtype=torch.bfloat16),
        "audio_feats_lengths": torch.tensor([[8]]),
        **video_inputs,
    }
    eager = thinker.embed_multimodal(**inputs)
    graph_outputs = [*manager.execute(image_inputs), audio_output, *manager.execute(video_inputs)]
    assert len(eager) == len(graph_outputs) == 3
    assert eager[1] is audio_output
    for expected, actual in zip(eager, graph_outputs):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    assert manager.graph_hits == 2
    assert not manager.supports_modality("audio")
    assert thinker.embed_multimodal() == []


def test_video_subset_preserves_temporal_patch_boundaries(ming_vision_thinker):
    pixels = torch.arange(112 * 24).reshape(112, 24)
    grid = torch.tensor([[2, 4, 4], [1, 4, 8], [3, 4, 4]])
    inputs = {"pixel_values_videos": pixels, "video_grid_thw": grid}
    selected = ming_vision_thinker.select_encoder_cudagraph_items(inputs, [2, 0])
    assert set(selected) == {"pixel_values_videos", "video_grid_thw"}
    torch.testing.assert_close(selected["video_grid_thw"], grid[[2, 0]])
    torch.testing.assert_close(selected["pixel_values_videos"], torch.cat([pixels[64:], pixels[:32]]))
    specs = ming_vision_thinker.get_encoder_cudagraph_item_specs(selected)
    assert [spec.output_tokens for spec in specs] == [12, 8]
