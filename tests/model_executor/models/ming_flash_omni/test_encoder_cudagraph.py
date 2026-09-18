# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationConfig, MultiModalConfig
from vllm.model_executor.models import vision
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_omni.model_executor.models.ming_flash_omni.encoder_cudagraph import pad_cumulative_lengths
from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni import MingFlashOmniForConditionalGeneration
from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_thinker import (
    MingFlashOmniThinkerForConditionalGeneration,
    MingFlashOmniThinkerMultiModalProcessor,
)
from vllm_omni.model_executor.models.ming_flash_omni.projectors import VisionProjector
from vllm_omni.model_executor.models.ming_flash_omni.vision_encoder import MingVisionEncoder

pytestmark = pytest.mark.core_model


@pytest.fixture
def graph_model(init_fake_tp_group, monkeypatch):
    """Real small ViT/projector and production protocol, without the large LLM."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    mm_config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")
    monkeypatch.setattr(vision, "get_multimodal_config", lambda: mm_config)
    vision_config = SimpleNamespace(
        hidden_size=128,
        num_heads=2,
        image_size=32,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=64,
        intermediate_size=256,
        hidden_act="gelu",
        depth=2,
        deepstack_visual_indexes=[0],
        apply_vit_abs_pos_embed=True,
    )
    thinker = MingFlashOmniThinkerForConditionalGeneration.__new__(MingFlashOmniThinkerForConditionalGeneration)
    torch.nn.Module.__init__(thinker)
    with set_default_torch_dtype(torch.bfloat16):
        thinker.vision = MingVisionEncoder(vision_config).cuda().eval()
        thinker.linear_proj = VisionProjector(64, 96, mlp_depth=2).cuda().eval()
    thinker.config = SimpleNamespace(hidden_size=96)
    thinker.model_config = SimpleNamespace(max_model_len=32768, multimodal_config=mm_config)
    # vLLM allocates uninitialized checkpoint storage; seed all test weights.
    generator = torch.Generator(device="cuda").manual_seed(17)
    with torch.no_grad():
        for name, parameter in thinker.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=0.02, generator=generator)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    wrapper = MingFlashOmniForConditionalGeneration.__new__(MingFlashOmniForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.thinker = thinker.eval()
    wrapper.model = thinker
    return wrapper.eval()


@pytest.fixture
def manager(graph_model):
    config = SimpleNamespace(
        compilation_config=CompilationConfig(),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
        model_config=graph_model.thinker.model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    manager = EncoderCudaGraphManager(config, torch.device("cuda"), torch.bfloat16, graph_model)
    assert manager.token_budgets == [512, 1024, 2048]
    assert manager.max_batch_size == 4
    # Use small capture buffers for numerical regression; same production code.
    config.compilation_config.encoder_cudagraph_token_budgets = [4, 8, 16]
    config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 2
    manager = EncoderCudaGraphManager(config, torch.device("cuda"), torch.bfloat16, graph_model)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        manager.capture(graph_pool=torch.cuda.graph_pool_handle())
    torch.cuda.current_stream().wait_stream(stream)
    yield manager
    manager.clear()


@pytest.mark.parametrize(
    ("modality", "pixel_key", "grid"),
    [
        ("image", "pixel_values", [[1, 4, 8], [1, 4, 20], [1, 4, 4]]),
        ("video", "pixel_values_videos", [[8, 2, 2], [5, 4, 4], [1, 4, 4]]),
    ],
)
@torch.inference_mode()
def test_encoder_graph_parity(manager, modality, pixel_key, grid):
    # Pack small items around one eager fallback, preserving original order.
    # Video has more temporal sequences than the maximum item count.
    inputs = {
        pixel_key: torch.randn(128, 24, device="cuda", dtype=torch.bfloat16),
        f"{modality}_grid_thw": torch.tensor(grid),
    }
    reference = manager.model.encoder_eager_forward(inputs).split([8, 20, 4])
    outputs = manager.execute(inputs)
    assert len(outputs) == 3
    assert manager.graph_hits == 2
    assert manager.graph_misses == 1
    for actual, expected in zip(outputs, reference):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    # Replay must not replace the CPU launch bound baked into the capture.
    replay = manager.model.prepare_encoder_cudagraph_replay_buffers(inputs, 2, 0)
    assert "max_seqlen" not in replay.values
    inputs[f"{modality}_grid_thw"] = inputs[f"{modality}_grid_thw"].cuda()
    with pytest.raises(ValueError, match="CPU grid"):
        manager.model.get_encoder_cudagraph_item_specs(inputs)


def test_padding_covers_all_patches():
    # Uncovered padding rows caused NaNs with real checkpoint weights.
    destination = torch.empty(7, dtype=torch.int32)
    for offsets in ([0, 4, 12], [0, 4]):
        pad_cumulative_lengths(destination, torch.tensor(offsets, dtype=torch.int32), spatial_merge_unit=4)
        expected = offsets + [20] * (7 - len(offsets))
        torch.testing.assert_close(destination, torch.tensor(expected, dtype=torch.int32))


def test_processor_keeps_grids_on_cpu():
    processor = object.__new__(MingFlashOmniThinkerMultiModalProcessor)
    fields = processor._get_mm_fields_config(
        {"image_grid_thw": torch.tensor([[1, 4, 4]]), "video_grid_thw": torch.tensor([[3, 4, 4]])}, {}
    )
    assert fields["image_grid_thw"].field.keep_on_cpu
    assert fields["video_grid_thw"].field.keep_on_cpu


def test_graph_rejects_encoder_data_parallel(graph_model):
    graph_model.thinker.model_config.multimodal_config.mm_encoder_tp_mode = "data"
    with pytest.raises(ValueError, match="mm_encoder_tp_mode=weights"):
        graph_model.get_encoder_cudagraph_config()
