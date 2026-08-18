# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for the Mage-VL config, registration, and patch positions."""

import inspect

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


@pytest.mark.core_model
@pytest.mark.cpu
def test_vision_config_declares_temporal_patch_size():
    """Qwen2-VL-derived processing helpers read this straight off the vision config."""
    assert MageVLVisionConfig().temporal_patch_size == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_reused_encoder_mlp_keeps_the_config_activation():
    """The tower reuses ``Qwen2VisionMLP``, whose ``act_layer`` defaults to QuickGELU.

    Mage-VL is a GELU model, so the config activation must be bound explicitly at the
    call site. Getting this wrong swaps the activation silently -- weights still load and
    nothing raises, only the numbers drift.
    """
    from functools import partial

    from vllm.model_executor.layers.activation import get_act_fn
    from vllm.model_executor.models.qwen2_vl import Qwen2VisionMLP

    default_act = inspect.signature(Qwen2VisionMLP.__init__).parameters["act_layer"].default
    assert default_act is not None, "upstream signature changed; re-check the binding"

    cfg = MageVLVisionConfig(hidden_act="gelu")
    bound = partial(get_act_fn, cfg.hidden_act)
    assert type(bound()) is not default_act, (
        "the config activation must differ from the upstream default, otherwise this "
        "test cannot detect a missing act_layer binding"
    )


@pytest.mark.core_model
@pytest.mark.cpu
def test_merger_reuse_keeps_checkpoint_key_layout():
    """``Qwen2VisionPatchMerger`` must expose ln_q / mlp.0 / mlp.2, as in the checkpoint."""
    from vllm.model_executor.models.qwen2_vl import Qwen2VisionPatchMerger

    params = inspect.signature(Qwen2VisionPatchMerger.__init__).parameters
    for name in ("d_model", "context_dim", "norm_layer", "spatial_merge_size", "prefix"):
        assert name in params, f"upstream merger signature lost {name!r}"


@pytest.mark.core_model
@pytest.mark.cpu
def test_processor_drops_the_video_attribute():
    """The vendored processor must not inherit Qwen2-VL's ``video_processor`` attribute.

    transformers derives a processor's attribute list from the *subclass* ``__init__``
    signature, and resolves every attribute in it against the checkpoint directory. The
    Mage-VL checkpoint ships no ``video_preprocessor_config.json``, so keeping
    ``video_processor`` in the signature turns ``from_pretrained`` into an ``OSError``
    (and passing ``None`` into a ``TypeError``). Frames arrive as images here, so there is
    nothing for a video processor to do.
    """
    from transformers.models.qwen2_vl import Qwen2VLProcessor

    from vllm_omni.transformers_utils.processors.mage_vl import MageVLProcessor

    assert "video_processor" in Qwen2VLProcessor.get_attributes(), (
        "upstream no longer declares video_processor; re-check whether this subclass "
        "still needs its own __init__"
    )
    assert MageVLProcessor.get_attributes() == ["image_processor", "tokenizer"]


def _stub_processing_info(max_pixels: int = 4_000_000):
    """A ``MageVLProcessingInfo`` over stub context, for the CPU-only size arithmetic."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    from vllm_omni.model_executor.models.mage_vl.mage_vl import MageVLProcessingInfo

    config = MageVLConfig(
        vision_config={
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        },
        text_config={"model_type": "qwen3", "hidden_size": 2560},
    )
    image_processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, temporal_patch_size=1,
        min_pixels=32 * 32, max_pixels=max_pixels,
    )

    class _StubProcessor:
        pass

    processor = _StubProcessor()
    processor.image_processor = image_processor

    class _StubCtx:
        def get_hf_config(self, cls):
            return config

        def get_hf_processor(self, cls, **kwargs):
            return processor

        def get_merged_mm_kwargs(self, kwargs):
            return dict(kwargs)

    return MageVLProcessingInfo(_StubCtx())


@pytest.mark.core_model
@pytest.mark.cpu
def test_profiling_size_comes_from_the_checkpoint_max_pixels():
    """Dummy data must be the largest image the processor accepts, not a fixed guess.

    ``get_image_size_with_most_features`` is reused from Qwen2-VL: it factorises
    ``max_pixels / (patch_size * merge_size)**2`` into a non-extreme aspect ratio. With the
    checkpoint's 4,000,000 max_pixels that is 2016x1984 -> 3906 vision tokens, versus the
    196 a hardcoded 448x448 would have profiled.
    """
    info = _stub_processing_info()

    width, height = info.get_image_size_with_most_features()
    assert (width, height) == (2016, 1984)
    assert info.get_max_image_tokens() == 3906
    assert info.get_supported_mm_limits() == {"image": None}
    assert info.get_mm_max_tokens_per_item(seq_len=8192, mm_counts={"image": 1}) == {
        "image": 3906
    }


@pytest.mark.core_model
@pytest.mark.cpu
def test_dummy_mm_data_uses_the_max_feature_size():
    from vllm_omni.model_executor.models.mage_vl.mage_vl import MageVLDummyInputsBuilder

    info = _stub_processing_info()
    dummy = MageVLDummyInputsBuilder(info).get_dummy_mm_data(
        seq_len=8192, mm_counts={"image": 2}, mm_options={}
    )

    images = dummy["image"]
    assert len(images) == 2
    assert all(image.size == (2016, 1984) for image in images)


@pytest.mark.core_model
@pytest.mark.cpu
def test_processor_override_keeps_vllm_off_the_mm_only_path():
    """``_call_hf_processor`` must stay overridden, or ``patch_positions`` disappears.

    vLLM decides its mm-only path by whether this method is overridden. Left to the base
    implementation it calls ``call_hf_processor_mm_only``, which bypasses the processor's
    ``__call__`` and runs ``processor.image_processor`` directly -- the positions table is
    then never built and the tower fails with ``KeyError: 'patch_positions'`` at decode
    time, long after startup looked healthy.
    """
    from vllm.multimodal.processing import BaseMultiModalProcessor

    from vllm_omni.model_executor.models.mage_vl.mage_vl import MageVLMultiModalProcessor

    assert (
        MageVLMultiModalProcessor._call_hf_processor
        is not BaseMultiModalProcessor._call_hf_processor
    )
