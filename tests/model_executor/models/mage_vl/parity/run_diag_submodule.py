"""Bisect the Mage-ViT delta: compare layer-0 submodules on identical inputs."""

import json
import os

import torch

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")


def stats(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (
        (a - b).abs().max().item(),
        ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
        torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    )


def show(tag, a, b):
    m, r, c = stats(a, b)
    print(f"{tag:34s} max_abs={m:.4e} rel_l2={r:.4e} cos={c:.10f}")


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from safetensors import safe_open
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_omni.model_executor.models.mage_vl.vision import (
        MageVLVisionTransformer,
        build_window_cu_seqlens,
    )
    from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig

    cfg = MageVLConfig.from_pretrained(MODEL)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        mine = MageVLVisionTransformer(cfg.vision_config, prefix="visual")
    mine.eval()
    for layer in mine.encoder.layers:
        layer.self_attn.attn.attn_backend = AttentionBackendEnum.TORCH_SDPA
        layer.self_attn.attn.is_flash_attn_backend = False

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

    from transformers import AutoModelForCausalLM

    hf = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    hvis = hf.model.visual

    ref = torch.load("/root/autodl-tmp/mage_ref/img_dog_short.pt", map_location="cuda")
    pv, gthw, pp = ref["pixel_values"].cuda(), ref["image_grid_thw"].cuda(), ref["patch_positions"].cuda()
    if pp.dim() == 3:
        pp = pp.squeeze(0)

    with torch.inference_mode():
        # --- patch embedding ---
        e_mine = mine.embeddings(pv.to(mine.dtype))
        e_hf = hvis.embeddings(pv.to(hf.dtype))
        show("patch_embedding", e_mine, e_hf)

        # --- pre-norm ---
        n_mine = mine.layernorm_pre(e_mine)
        n_hf = hvis.layernorm_pre(e_hf.unsqueeze(0)).squeeze(0)
        show("layernorm_pre", n_mine, n_hf)

        # --- rotary tables ---
        cos, sin = mine.video_rope(pp)
        freqs_hf = hvis.video_rope.forward_from_positions(pp)
        freqs_hf = torch.cat([freqs_hf, freqs_hf], dim=-1)
        show("rope cos", cos, freqs_hf.cos())

        # --- layer 0 pieces, fed the SAME input ---
        x = n_mine
        L_mine, L_hf = mine.encoder.layers[0], hvis.encoder.layers[0]

        show("layer0.layer_norm1", L_mine.layer_norm1(x), L_hf.layer_norm1(x.unsqueeze(0)).squeeze(0))

        h = L_mine.layer_norm1(x)
        cu_np = build_window_cu_seqlens(gthw, x.shape[0], mine.frame_windows_size)
        from vllm.model_executor.layers.attention import MMEncoderAttention

        backend = L_mine.self_attn.attn.attn_backend
        seq_lens = MMEncoderAttention.maybe_compute_seq_lens(backend, cu_np, x.device)
        max_s = torch.tensor(MMEncoderAttention.compute_max_seqlen(backend, cu_np), dtype=torch.int32)
        cu = MMEncoderAttention.maybe_recompute_cu_seqlens(
            backend, cu_np, hidden_size=cfg.vision_config.hidden_size, tp_size=1, device=x.device
        )
        a_mine = L_mine.self_attn(h, cu, cos, sin, max_s, seq_lens)
        cu_hf = torch.tensor(cu_np, dtype=torch.int32, device=x.device)
        a_hf, _ = L_hf.self_attn(
            hidden_states=h.unsqueeze(0),
            rotary_pos_emb=freqs_hf.unsqueeze(0),
            cu_seqlens=cu_hf,
        )
        show("layer0.self_attn", a_mine, a_hf.squeeze(0))

        show("layer0.mlp", L_mine.mlp(h), L_hf.mlp(h.unsqueeze(0)).squeeze(0))

        # activation only
        act_mine = L_mine.mlp.act
        probe = torch.randn(512, dtype=torch.float32, device="cuda").to(mine.dtype)
        import transformers.activations as A

        show("gelu impl", act_mine(probe), A.ACT2FN[cfg.vision_config.hidden_act](probe))

    print("done")


if __name__ == "__main__":
    main()
