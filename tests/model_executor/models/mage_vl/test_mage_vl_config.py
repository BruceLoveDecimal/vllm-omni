# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for the Mage-VL config, registration, and patch positions."""

import pytest
import torch

from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig, MageVLVisionConfig
from vllm_omni.transformers_utils.processors.mage_vl import build_patch_positions


@pytest.mark.core_model
@pytest.mark.cpu
def test_vision_config_defaults_match_checkpoint():
    cfg = MageVLVisionConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        patch_size=16,
        spatial_merge_size=2,
        out_hidden_size=2560,
        frame_windows_size=4,
    )
    # head_dim//2 must split 4:6:6 across (t, h, w) for the 3D rotary table.
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    assert head_dim == 64
    assert (head_dim // 2) % 16 == 0


@pytest.mark.core_model
@pytest.mark.cpu
def test_config_resolves_text_and_vision_subconfigs():
    cfg = MageVLConfig(
        text_config={"model_type": "qwen3", "hidden_size": 2560, "eos_token_id": 151645},
        vision_config={"hidden_size": 1024, "num_attention_heads": 16},
    )
    assert cfg.vision_config.hidden_size == 1024
    assert cfg.text_config.model_type == "qwen3"
    # Generation ids are mirrored to the top level for callers that read them there.
    assert cfg.eos_token_id == 151645
    # No mrope section, which is what decides whether vLLM switches to M-RoPE
    # positions; Mage-VL feeds the decoder plain 1-D positions.
    rope_scaling = getattr(cfg.text_config, "rope_scaling", None) or {}
    assert "mrope_section" not in rope_scaling


@pytest.mark.core_model
@pytest.mark.cpu
def test_registered_in_omni_registry():
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert _OMNI_MODELS["MageVLForConditionalGeneration"] == (
        "mage_vl",
        "mage_vl",
        "MageVLForConditionalGeneration",
    )


@pytest.mark.core_model
@pytest.mark.cpu
def test_pipeline_registered_single_stage():
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    pipeline = OMNI_PIPELINES["mage_vl"]
    assert len(pipeline.stages) == 1
    stage = pipeline.stages[0]
    assert stage.final_output and stage.final_output_type == "text"
    assert stage.owns_tokenizer and stage.requires_multimodal_data


@pytest.mark.core_model
@pytest.mark.cpu
def test_patch_positions_block_layout():
    """Positions must land in the 2x2 block order the image processor emits."""
    grid = torch.tensor([[1, 4, 4]])
    pos = build_patch_positions(grid, spatial_merge_size=2)

    assert pos.shape == (16, 3)
    assert (pos[:, 0] == 0).all()  # single frame -> t is 0 everywhere
    # First block covers rows 0-1 x cols 0-1 before moving on.
    assert pos[:4, 1:].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert pos[4:8, 1:].tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]
    # Every (h, w) coordinate appears exactly once.
    assert {tuple(p) for p in pos[:, 1:].tolist()} == {(h, w) for h in range(4) for w in range(4)}


@pytest.mark.core_model
@pytest.mark.cpu
def test_patch_positions_use_real_frame_indices():
    """Under sparse sampling ``t`` is the source frame number, not a dense index."""
    grid = torch.tensor([[3, 2, 2]])
    pos = build_patch_positions(grid, spatial_merge_size=2, frame_indices=[torch.tensor([0, 7, 19])])
    assert sorted(set(pos[:, 0].tolist())) == [0, 7, 19]


@pytest.mark.core_model
@pytest.mark.cpu
def test_window_cu_seqlens_splits_on_frame_windows():
    from vllm_omni.model_executor.models.mage_vl.vision import build_window_cu_seqlens

    # 10 temporal steps, 2x2 patches each, windows of 4 -> 4 + 4 + 2.
    grid = torch.tensor([[10, 2, 2]])
    cu = build_window_cu_seqlens(grid, total_patches=40, frame_windows_size=4)
    assert cu.tolist() == [0, 16, 32, 40]

    # A single image (t=1) is one window.
    cu = build_window_cu_seqlens(torch.tensor([[1, 4, 4]]), total_patches=16, frame_windows_size=4)
    assert cu.tolist() == [0, 16]


@pytest.mark.core_model
@pytest.mark.cpu
def test_window_cu_seqlens_rejects_mismatched_totals():
    from vllm_omni.model_executor.models.mage_vl.vision import build_window_cu_seqlens

    with pytest.raises(ValueError, match="cu_seqlens mismatch"):
        build_window_cu_seqlens(torch.tensor([[1, 4, 4]]), total_patches=99, frame_windows_size=4)
