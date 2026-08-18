"""Parity: vLLM-Omni Mage-ViT tower vs the HF reference vision output.

Loads the checkpoint's vision weights into the ported tower, replays the exact inputs the
HF baseline saw, and compares the merged visual embeddings.
"""

import os
import sys
from pathlib import Path

import torch

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
REF = Path(os.environ.get("REF_OUT", "/root/autodl-tmp/mage_ref"))


def main():
    # vLLM parallel layers need the distributed env even for TP=1.
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")

    from vllm.config import VllmConfig, set_current_vllm_config

    vllm_config = VllmConfig()
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()

    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from safetensors import safe_open

    from vllm_omni.model_executor.models.mage_vl.vision import MageVLVisionTransformer
    from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig

    cfg = MageVLConfig.from_pretrained(MODEL)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        tower = MageVLVisionTransformer(cfg.vision_config, prefix="visual")
    tower.eval()

    if os.environ.get("MAGE_FORCE_SDPA") == "1":
        # Match the HF baseline's attention kernel so any remaining delta is logic,
        # not kernel numerics.
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        for layer in tower.encoder:
            layer.self_attn.attn.attn_backend = AttentionBackendEnum.TORCH_SDPA
            layer.self_attn.attn.is_flash_attn_backend = False
        print("forced TORCH_SDPA for the ported tower")
    print("tower attn backend:", tower.encoder[0].self_attn.attn.attn_backend)

    # Stream just the vision weights out of the shards.
    index = torch.load if False else None  # noqa: F841 - placeholder to keep imports tidy
    import json

    with open(f"{MODEL}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    vis_keys = [k for k in weight_map if k.startswith("model.visual.")]
    by_file: dict[str, list[str]] = {}
    for k in vis_keys:
        by_file.setdefault(weight_map[k], []).append(k)

    def iter_weights():
        for fname, keys in by_file.items():
            with safe_open(f"{MODEL}/{fname}", framework="pt", device="cpu") as f:
                for k in keys:
                    yield k[len("model.visual.") :], f.get_tensor(k).cuda()

    loaded = tower.load_weights(iter_weights())
    expected = {n for n, _ in tower.named_parameters(remove_duplicate=False)}
    missing = expected - loaded
    print(f"loaded {len(loaded)} params, missing {len(missing)}")
    if missing:
        print("  MISSING:", sorted(missing)[:10])
        return 1

    failures = 0
    for case in sorted(REF.glob("*.pt")):
        ref = torch.load(case, map_location="cuda")
        with torch.inference_mode():
            got = tower(
                ref["pixel_values"].cuda().to(tower.dtype),
                ref["image_grid_thw"].cuda(),
                ref["patch_positions"].cuda(),
            )
        want = ref["vision_embeds"].cuda().float()
        got_f = got.float()
        if got_f.shape != want.shape:
            print(f"[{case.stem}] SHAPE MISMATCH got={tuple(got_f.shape)} want={tuple(want.shape)}")
            failures += 1
            continue
        max_abs = (got_f - want).abs().max().item()
        rel = ((got_f - want).norm() / want.norm()).item()
        cos = torch.nn.functional.cosine_similarity(got_f.flatten(), want.flatten(), dim=0).item()
        ok = cos > 0.9995 and rel < 0.02
        failures += 0 if ok else 1
        print(
            f"[{case.stem}] shape={tuple(got_f.shape)} max_abs={max_abs:.4e} "
            f"rel_l2={rel:.4e} cos={cos:.8f} {'OK' if ok else 'FAIL'}"
        )

    print("PARITY_RESULT:", "PASS" if failures == 0 else f"FAIL({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
