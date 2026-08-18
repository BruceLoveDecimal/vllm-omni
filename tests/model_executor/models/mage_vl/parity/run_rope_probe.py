"""Decide which upstream rotary form (if any) reproduces Mage-VL's HF rotation."""

import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/models/Mage-VL")

torch.manual_seed(0)
L, H, D = 37, 4, 64
half = D // 2

# Build freqs the way the HF vision tower does: [L, half], then cat -> [L, D]
freqs_half = torch.randn(L, half, dtype=torch.float32).abs()
freqs_full = torch.cat([freqs_half, freqs_half], dim=-1)  # [L, D]

q = torch.randn(1, H, L, D, dtype=torch.float32)
k = torch.randn(1, H, L, D, dtype=torch.float32)


def hf_rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def hf_apply(q, k, freqs):
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()
    return (q * cos) + (hf_rotate_half(q) * sin), (k * cos) + (hf_rotate_half(k) * sin)


ref_q, _ = hf_apply(q, k, freqs_full.unsqueeze(0))  # tower passes [B, L, D]

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

# vLLM wants [seq, num_heads, head_size]; HF q is [b, h, L, D] -> [L, H, D]
q_vllm = q[0].transpose(0, 1).contiguous()


def report(tag, got):
    got_hf = got.transpose(0, 1).unsqueeze(0)
    diff = (got_hf - ref_q).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        got_hf.flatten(), ref_q.flatten(), dim=0
    ).item()
    print(f"{tag:52s} max_abs_diff={diff:.3e}  cos={cos_sim:.8f}")


# Candidate 1: non-neox (GPT-J interleaved) with cos/sin from the half-width freqs
report(
    "non-neox, cos=freqs_half",
    ApplyRotaryEmb.forward_static(
        q_vllm, freqs_half.cos(), freqs_half.sin(), is_neox_style=False
    ),
)

# Candidate 2: neox (half-split) with cos/sin from the half-width freqs
report(
    "neox, cos=freqs_half",
    ApplyRotaryEmb.forward_static(
        q_vllm, freqs_half.cos(), freqs_half.sin(), is_neox_style=True
    ),
)

# Candidate 3: non-neox but feeding the *even* slice of the full freqs
report(
    "non-neox, cos=freqs_full[..., ::2]",
    ApplyRotaryEmb.forward_static(
        q_vllm,
        freqs_full[..., ::2].cos(),
        freqs_full[..., ::2].sin(),
        is_neox_style=False,
    ),
)


# Candidate 4: exact HF replication written against the [L,H,D] layout
def mage_apply(x, cos_full, sin_full):
    cos = cos_full.unsqueeze(-2)
    sin = sin_full.unsqueeze(-2)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rot = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos + rot * sin


report("exact-replication helper", mage_apply(q_vllm, freqs_full.cos(), freqs_full.sin()))

# Sanity: is the HF op even a norm-preserving rotation?
print(
    "norm(q) =", q.norm().item(),
    " norm(rotated) =", ref_q.norm().item(),
    " -> rotation-like" if abs(q.norm() - ref_q.norm()) < 1e-3 else " -> NOT norm preserving",
)
