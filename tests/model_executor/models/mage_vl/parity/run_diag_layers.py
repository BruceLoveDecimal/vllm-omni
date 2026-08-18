"""Layer-by-layer diff between the HF Mage-ViT tower and the ported one.

Gradual growth across layers => bf16/kernel drift. A jump at one layer => a real bug.
"""

import json
import os

import torch

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
REF = "/root/autodl-tmp/mage_ref/img_dog_short.pt"


def stats(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (
        (a - b).abs().max().item(),
        ((a - b).norm() / b.norm()).item(),
        torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    )


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29593")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from safetensors import safe_open

    from vllm_omni.model_executor.models.mage_vl.vision import MageVLVisionTransformer
    from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig

    ref = torch.load(REF, map_location="cuda")
    pv, gthw, pp = ref["pixel_values"].cuda(), ref["image_grid_thw"].cuda(), ref["patch_positions"].cuda()

    # ---- ported tower ----
    cfg = MageVLConfig.from_pretrained(MODEL)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        mine = MageVLVisionTransformer(cfg.vision_config, prefix="visual")
    mine.eval()

    with open(f"{MODEL}/model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]
    by_file = {}
    for k in [k for k in wmap if k.startswith("model.visual.")]:
        by_file.setdefault(wmap[k], []).append(k)

    def it():
        for fn, keys in by_file.items():
            with safe_open(f"{MODEL}/{fn}", framework="pt", device="cpu") as f:
                for k in keys:
                    yield k[len("model.visual.") :], f.get_tensor(k).cuda()

    mine.load_weights(it())

    mine_acts = {}
    for i, layer in enumerate(mine.encoder):
        layer.register_forward_hook(
            lambda m, inp, out, i=i: mine_acts.__setitem__(i, out.detach())
        )
    with torch.inference_mode():
        mine_out = mine(pv.to(mine.dtype), gthw, pp)

    # ---- HF tower ----
    from transformers import AutoModelForCausalLM

    hf = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    visual = hf.model.visual
    hf_acts = {}
    for i, layer in enumerate(visual.encoder.layers):
        layer.register_forward_hook(
            lambda m, inp, out, i=i: hf_acts.__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach()
            )
        )
    with torch.inference_mode():
        hf_out = visual(pv.to(hf.dtype), grid_thw=gthw, patch_positions=pp).last_hidden_state

    print(f"{'layer':>6} {'max_abs':>12} {'rel_l2':>12} {'cos':>12}")
    for i in sorted(mine_acts):
        a = mine_acts[i]
        b = hf_acts[i]
        b = b.squeeze(0) if b.dim() == 3 else b
        a = a.squeeze(0) if a.dim() == 3 else a
        m, r, c = stats(a, b)
        print(f"{i:>6} {m:>12.4e} {r:>12.4e} {c:>12.8f}")

    m, r, c = stats(mine_out, hf_out.squeeze(0) if hf_out.dim() == 3 else hf_out)
    print(f"{'merged':>6} {m:>12.4e} {r:>12.4e} {c:>12.8f}")
    print("value scale: mine.std=%.4f hf.std=%.4f" % (mine_out.float().std(), hf_out.float().std()))


if __name__ == "__main__":
    main()
