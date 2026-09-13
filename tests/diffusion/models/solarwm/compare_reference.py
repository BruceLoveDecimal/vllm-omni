# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU attention parity against a separately checked-out, pinned SolarWM.

Run from the vLLM-Omni root with a compatible torch/vllm environment:
python -m tests.diffusion.models.solarwm.compare_reference /path/to/SolarWM
The reference is used only in this validation command, never in the runtime.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

REFERENCE_REVISION = "a3a3fac16466102a2b97df867f7703df7172cb2a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--transformer", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_REVISION:
        raise ValueError(f"Expected SolarWM {REFERENCE_REVISION}, got {revision}")
    if subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip():
        raise ValueError("Reference checkout must be clean")
    sys.path.insert(0, str(args.reference / "src"))
    from solarwm.backends.wan22.runtime.modeling.camera_prope import prope_qkv
    from solarwm.backends.wan22.runtime.modeling.causal_model import build_inference_window_block_mask

    from tests.diffusion.models.solarwm.window_attention import SolarWMWindowAttention
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

    with (
        patch("vllm_omni.diffusion.attention.layer.get_attn_backend_for_role", return_value=(SDPABackend, None)),
        patch.object(SDPAImpl, "forward", SDPAImpl.forward_cuda),
    ):
        native = SolarWMWindowAttention(2, 8)
        generator = torch.Generator().manual_seed(91)
        q, k, v = [torch.randn(2, 12, 2, 8, generator=generator) for _ in range(3)]
        maximum_error = 0.0
        cases = 0
        for cameras in (3, 12):
            views = torch.eye(4).repeat(2, cameras, 1, 1)
            views[..., :3, 3] = torch.randn(2, cameras, 3, generator=generator)
            views[:, 1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
            intrinsics = torch.eye(3).repeat(2, cameras, 1, 1)
            intrinsics[..., 0, 0], intrinsics[..., 1, 1] = 1.7, 0.8
            intrinsics[..., :2, 2] = 0.5
            for transform in ("linear", "logd4"):
                rq, rk, rv, restore = prope_qkv(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    viewmats=views,
                    Ks=intrinsics,
                    camera_translation_transform=transform,
                )
                for chunk in (None, 4, 5):
                    mask = None
                    if chunk is not None:
                        block, _ = build_inference_window_block_mask(
                            num_clean_frames=0,
                            num_noisy_frames=12,
                            frame_seqlen=1,
                            num_frame_per_block=chunk,
                            device="cpu",
                        )
                        positions = torch.arange(12)
                        mask = block.mask_mod(0, 0, positions[:, None], positions[None, :])
                    expected = restore(F.scaled_dot_product_attention(rq, rk, rv, attn_mask=mask)).transpose(1, 2)
                    actual = native(
                        q,
                        k,
                        v,
                        viewmats=views,
                        intrinsics=intrinsics,
                        tokens_per_chunk=chunk,
                        translation_transform=transform,
                    )
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    maximum_error = max(maximum_error, (actual - expected).abs().max().item())
                    cases += 1
        print(f"SolarWM {revision}: {cases} attention cases passed; max_abs_error={maximum_error:.9g}")
        if args.transformer:
            compare_transformer()


@torch.no_grad()
def compare_transformer():
    from solarwm.backends.wan22.runtime.modeling import causal_model, model

    from vllm_omni.diffusion.models.solarwm.transformer import SolarWMCache, SolarWMTransformer

    def sdpa(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    config = dict(
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=2,
        in_dim=4,
        out_dim=4,
        freq_dim=16,
        text_dim=16,
        text_len=8,
        eps=1e-6,
    )
    torch.manual_seed(13)
    reference = causal_model.CausalWanModel(
        **config,
        model_type="ti2v",
        local_attn_size=18,
        sink_size=0,
        add_control_adapter=True,
        camera_attention_mode="fused_prope",
        use_echorope=False,
        frame_seq_length=4,
    ).eval()
    # Upstream initializes the head to zero. Randomize it so comparisons can
    # detect errors in every preceding layer, rather than passing trivially.
    reference.head.head.weight.normal_(std=0.02)
    native = SolarWMTransformer(**config).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    cache = SolarWMCache()
    ref_cache = [
        {
            "k": torch.zeros(1, 72, 4, 8),
            "v": torch.zeros(1, 72, 4, 8),
            "global_end_index": torch.tensor(0),
            "local_end_index": torch.tensor(0),
        }
        for _ in range(2)
    ]
    cross_cache = [{"is_init": False} for _ in range(2)]
    text = torch.randn(1, 5, 16)
    max_error = 0.0
    with patch.object(causal_model, "attention", sdpa), patch.object(model, "flash_attention", sdpa):
        for start in range(0, 24, 3):
            latents = torch.randn(1, 4, 3, 4, 4)
            views = torch.eye(4).repeat(1, 3, 1, 1)
            views[..., :3, 3] = torch.randn(1, 3, 3) * 0.1
            ks = torch.eye(3).repeat(1, 3, 1, 1)
            for step in (750, 250, 0):
                time = torch.full((1, 3), float(step))
                camera = {"viewmats": views.repeat_interleave(4, dim=1), "K": ks.repeat_interleave(4, dim=1)}
                expected = reference(
                    list(latents),
                    time.repeat_interleave(4, dim=1),
                    list(text),
                    seq_len=12,
                    y_camera=camera,
                    kv_cache=ref_cache,
                    crossattn_cache=cross_cache,
                    current_start=start * 4,
                    cache_update_policy="commit_detached" if step == 0 else "none",
                )
                actual = native(latents, time, text, views, ks, cache=cache, start_frame=start, commit=step == 0)
                torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
                max_error = max(max_error, (actual - expected).abs().max().item())
            assert cache.keys[0].shape[1] <= 72
    print(
        f"24 full-DiT forwards including read-only denoising, commit and eviction passed; max_abs_error={max_error:.9g}"
    )


if __name__ == "__main__":
    main()
