"""Codec-path parity: our vision tower vs the reference, on real codec canvases.

The image path is checked by ``run_parity_vision.py`` / ``run_fp32_equivalence.py`` on
frame-sampled inputs, where ``patch_positions`` is a dense ``arange``. Codec inputs are the
case those drivers cannot reach: patches are sparse in time, ``t`` is the source frame
index, and one canvas mixes several frames. This drives the tower with exactly those
tensors and compares against the checkpoint's own model.

Needs the codec engine to have produced assets. Either let the processor run it
(``pip install codec-video-prep`` in a separate venv, ``CV_PREINFER_BIN`` pointing at it),
or pass ``MAGE_CODEC_ASSETS=<dir>`` for a directory it already produced.

    python run_codec_parity.py
"""

import json
import os

import torch

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
VIDEO = os.environ.get("MAGE_VIDEO", f"{MODEL}/examples/soccer-broadcast.mp4")
ASSETS = os.environ.get("MAGE_CODEC_ASSETS")

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29609")

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402

ctx = set_current_vllm_config(VllmConfig())
ctx.__enter__()
init_distributed_environment(
    world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl"
)
initialize_model_parallel(tensor_model_parallel_size=1)

from safetensors import safe_open  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

from vllm_omni.model_executor.models.mage_vl.vision import MageVLVisionTransformer  # noqa: E402
from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig  # noqa: E402
from vllm_omni.transformers_utils.processors.mage_vl import MageVLProcessor  # noqa: E402


def build_codec_inputs():
    """Canvases -> (pixel_values, image_grid_thw, patch_positions, prompt)."""
    processor = MageVLProcessor.from_pretrained(MODEL)
    prompt = (
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n"
        "What happens in this video?<|im_end|>\n<|im_start|>assistant\n"
    )
    codec_config = {"asset_dir": ASSETS} if ASSETS else None
    out = processor(text=prompt, videos=VIDEO, video_backend="codec", codec_config=codec_config)
    return out


def load_tower(dtype):
    cfg = MageVLConfig.from_pretrained(MODEL)
    torch.set_default_dtype(dtype)
    with torch.device("cuda"):
        tower = MageVLVisionTransformer(cfg.vision_config, prefix="visual")
    tower.eval()
    for layer in tower.encoder.layers:
        layer.self_attn.attn.attn_backend = AttentionBackendEnum.TORCH_SDPA
        layer.self_attn.attn.is_flash_attn_backend = False

    with open(f"{MODEL}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key in [k for k in weight_map if k.startswith("model.visual.")]:
        by_file.setdefault(weight_map[key], []).append(key)

    def weights():
        for filename, keys in by_file.items():
            with safe_open(f"{MODEL}/{filename}", framework="pt", device="cpu") as f:
                for key in keys:
                    tensor = f.get_tensor(key).cuda()
                    yield key[len("model.visual.") :], tensor.to(dtype)

    loaded = tower.load_weights(weights())
    print(f"loaded {len(loaded)} params (dtype={dtype})", flush=True)
    return tower


def compare(ours, reference, label):
    diff = ours.float() - reference.float()
    rel_l2 = (diff.norm() / reference.float().norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        ours.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    print(
        f"{label:<10} max_abs={diff.abs().max().item():.4e} rel_l2={rel_l2:.4e} cos={cos:.10f}",
        flush=True,
    )
    return rel_l2


def main():
    inputs = build_codec_inputs()
    grid = inputs["image_grid_thw"].cuda()
    positions = inputs["patch_positions"].cuda()
    pixels = inputs["pixel_values"].cuda()
    print(
        f"codec inputs: {grid.shape[0]} canvases, {pixels.shape[0]} patches, "
        f"{int(grid.prod(-1).sum()) // 4} visual tokens, "
        f"t range [{int(positions[:, 0].min())}, {int(positions[:, 0].max())}]",
        flush=True,
    )

    from transformers import AutoModelForCausalLM

    # fp32 on both sides: the decisive check, free of accumulation noise.
    hf = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    tower_fp32 = load_tower(torch.float32)
    with torch.inference_mode():
        ours = tower_fp32(pixels.float(), grid, positions)
        reference = hf.model.visual(
            pixels.float(), grid_thw=grid, patch_positions=positions
        ).last_hidden_state
    reference = reference.squeeze(0) if reference.dim() == 3 else reference
    rel_fp32 = compare(ours, reference, "fp32")

    del hf, tower_fp32
    torch.cuda.empty_cache()

    hf_bf16 = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    tower_bf16 = load_tower(torch.bfloat16)
    with torch.inference_mode():
        ours_bf16 = tower_bf16(pixels.to(torch.bfloat16), grid, positions)
        ref_bf16 = hf_bf16.model.visual(
            pixels.to(torch.bfloat16), grid_thw=grid, patch_positions=positions
        ).last_hidden_state
    ref_bf16 = ref_bf16.squeeze(0) if ref_bf16.dim() == 3 else ref_bf16
    compare(ours_bf16, ref_bf16, "bf16")

    # fp32 equivalence is the acceptance criterion; bf16 drift is accumulation.
    print(f"CODEC_PARITY: {'PASS' if rel_fp32 < 1e-4 else 'FAIL'} (fp32 rel_l2={rel_fp32:.4e})")


if __name__ == "__main__":
    main()
