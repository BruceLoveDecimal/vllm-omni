# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in parity against Ming's real vision/projector checkpoint on one GPU.

Set MING_CHECKPOINT to an existing local snapshot. Only vision and projector
weights are read; this does not claim full language-model generation parity.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_omni.model_executor.models.ming_flash_omni.projectors import VisionProjector
from vllm_omni.model_executor.models.ming_flash_omni.vision_encoder import MingVisionEncoder
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMM2Config

pytestmark = [pytest.mark.core_model, pytest.mark.slow]


@pytest.fixture
def checkpoint_graph_manager(ming_encoder_graph_manager):
    checkpoint = os.environ.get("MING_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set MING_CHECKPOINT to a local Ming-flash-omni-2.0 checkpoint")

    snapshot = Path(checkpoint)
    config = BailingMM2Config(**json.loads((snapshot / "config.json").read_text()))
    thinker = ming_encoder_graph_manager.model.encoder_cudagraph_model
    ming_encoder_graph_manager.clear()
    with set_default_torch_dtype(torch.bfloat16):
        thinker.vision = MingVisionEncoder(config.vision_config).cuda().eval()
        thinker.linear_proj = (
            VisionProjector(
                config.vision_config.out_hidden_size,
                config.llm_config.hidden_size,
                mlp_depth=getattr(config, "mlp_depth", 2),
            )
            .cuda()
            .eval()
        )
    thinker.config = config.llm_config
    thinker.thinker_config = config

    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    tower_weights = []
    projector_weights = []
    shards = sorted({shard for name, shard in weight_map.items() if name.startswith(("vision.", "linear_proj."))})
    for shard in shards:
        with safe_open(snapshot / shard, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name.startswith("vision."):
                    tower_weights.append((name.removeprefix("vision."), tensors.get_tensor(name)))
                elif name.startswith("linear_proj."):
                    projector_weights.append((name.removeprefix("linear_proj."), tensors.get_tensor(name)))
    assert tower_weights and projector_weights, "checkpoint must contain both vision and projector weights"
    loaded_vision = thinker.vision.load_weights(tower_weights)
    loaded_projector = thinker.linear_proj.load_weights(projector_weights)
    assert set(dict(thinker.vision.named_parameters())) <= loaded_vision
    assert set(dict(thinker.linear_proj.named_parameters())) <= loaded_projector
    del tower_weights, projector_weights

    runtime_config = ming_encoder_graph_manager.vllm_config
    runtime_config.compilation_config.encoder_cudagraph_token_budgets = [64, 256, 512]
    runtime_config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 4
    manager = EncoderCudaGraphManager(
        runtime_config, torch.device("cuda"), torch.bfloat16, ming_encoder_graph_manager.model
    )
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        manager.capture(graph_pool=torch.cuda.graph_pool_handle())
    torch.cuda.current_stream().wait_stream(capture_stream)
    yield manager
    manager.clear()


@pytest.mark.parametrize(
    ("modality", "grid"),
    [
        ("image", [[1, 32, 32]]),
        ("image", [[1, 16, 64]]),
        ("image", [[1, 16, 16], [1, 16, 32]]),
        ("video", [[4, 16, 16]]),
        ("image", [[1, 48, 48]]),
    ],
)
@torch.inference_mode()
def test_checkpoint_embedding_parity(checkpoint_graph_manager, modality, grid, record_property):
    manager = checkpoint_graph_manager
    vision = manager.model.encoder_cudagraph_model.vision.encoder
    patch_width = vision.patch_embed.proj.in_channels * vision.temporal_patch_size * vision.patch_size**2
    num_patches = sum(t * h * w for t, h, w in grid)
    pixel_key = "pixel_values"
    if modality == "video":
        pixel_key = "pixel_values_videos"
    generator = torch.Generator(device="cuda").manual_seed(2026)
    inputs = {
        pixel_key: torch.randn(num_patches, patch_width, device="cuda", dtype=torch.bfloat16, generator=generator),
        f"{modality}_grid_thw": torch.tensor(grid),
    }
    lengths = [t * h * w // vision.spatial_merge_size**2 for t, h, w in grid]
    expected = manager.model.encoder_eager_forward(inputs).split(lengths)
    actual = manager.execute(inputs)
    errors = []
    cosines = []
    for output, reference in zip(actual, expected):
        assert torch.isfinite(reference).all(), "original eager encoder produced non-finite embeddings"
        assert torch.isfinite(output).all(), "encoder graph produced non-finite embeddings"
        torch.testing.assert_close(output, reference, atol=5e-4, rtol=2e-2)
        errors.append((output.float() - reference.float()).abs().max())
        cosines.append(torch.nn.functional.cosine_similarity(output.float(), reference.float()).min())
    assert len(actual) == len(expected)
    assert manager.graph_hits == sum(length <= 512 for length in lengths)
    assert manager.graph_misses == sum(length > 512 for length in lengths)
    record_property("max_abs_error", torch.stack(errors).max().item())
    record_property("min_token_cosine", torch.stack(cosines).min().item())
