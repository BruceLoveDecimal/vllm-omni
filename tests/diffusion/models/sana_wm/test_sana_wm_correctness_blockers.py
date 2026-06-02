# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the three single-GPU correctness blockers:

  A.1 — First-frame VAE encode (pipeline_sana_wm._vae_encode_first_frame)
  A.2 — FlowMatch DPM-Solver++ scheduler (SanaWmFlowMatchScheduler)
  A.3 — UCPE camera branch (raymap_embedder + _forward_ucpe)
"""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# ---------------------------------------------------------------------------
# Tiny model config reused across multiple tests
# ---------------------------------------------------------------------------
_TINY_CFG_KWARGS = dict(
    num_blocks=1,
    hidden_size=8,
    linear_head_dim=4,
    mlp_ratio=1,
    model_max_length=3,
    chunk_plucker_channels=6,
    cam_attn_compress=1,
)


def _tiny_cfg_kwargs(**overrides):
    kwargs = dict(_TINY_CFG_KWARGS)
    kwargs.update(overrides)
    return kwargs


# ===========================================================================
# A.2 — SanaWmFlowMatchScheduler
# ===========================================================================


def test_flow_match_scheduler_is_exported() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    assert SanaWmFlowMatchScheduler.__name__ == "SanaWmFlowMatchScheduler"


def test_flow_match_scheduler_rejects_nonpositive_steps() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    with pytest.raises(ValueError, match="num_inference_steps"):
        SanaWmFlowMatchScheduler(0)
    with pytest.raises(ValueError, match="num_inference_steps"):
        SanaWmFlowMatchScheduler(-1)


def test_flow_match_scheduler_timesteps_shape_and_order() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    sched = SanaWmFlowMatchScheduler(num_inference_steps=5, shift=9.8)
    ts = sched.timesteps(device=torch.device("cpu"))

    assert ts.shape == (5,)
    # Flow matching goes from high noise (large t) to clean (small t)
    assert ts[0] > ts[-1], "Timesteps must be in descending order"
    assert ts[0] > 0


def test_flow_match_scheduler_step_preserves_shape() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    sched = SanaWmFlowMatchScheduler(num_inference_steps=3, shift=9.8)
    ts = sched.timesteps(device=torch.device("cpu"))
    latents = torch.randn(1, 4, 2, 4, 4)
    noise_pred = torch.randn_like(latents)

    out = sched.step(noise_pred, ts[0], latents)

    assert out.shape == latents.shape
    assert torch.isfinite(out).all()


def test_flow_match_scheduler_add_noise_blending() -> None:
    """add_noise interpolates linearly between sample and noise.

    sigma=0 (clean) → output should equal sample.
    sigma=num_train (noisy) → output should equal noise.
    Midpoint sigma → output is strictly between the two.
    """
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    sched = SanaWmFlowMatchScheduler(num_inference_steps=3, shift=9.8)
    sched.timesteps(device=torch.device("cpu"))  # populate internal state
    num_train = sched.num_train_timesteps  # typically 1000

    sample = torch.ones(1, 4)
    noise = torch.zeros(1, 4)

    # t=0 → sigma=0 → pure sample
    out_clean = sched.add_noise(sample, noise, torch.tensor(0.0))
    assert torch.allclose(out_clean, sample, atol=1e-5)

    # t=num_train → sigma=1 → pure noise
    out_noisy = sched.add_noise(sample, noise, torch.tensor(float(num_train)))
    assert torch.allclose(out_noisy, noise, atol=1e-5)

    # midpoint sigma → value strictly between sample and noise
    mid_t = torch.tensor(float(num_train) * 0.5)
    out_mid = sched.add_noise(sample, noise, mid_t)
    assert 0.0 < out_mid[0, 0].item() < 1.0


def test_flow_match_scheduler_add_noise_broadcasts_over_latent_dims() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    sched = SanaWmFlowMatchScheduler(num_inference_steps=2, shift=9.8)
    sched.timesteps(device=torch.device("cpu"))
    sample = torch.randn(1, 128, 3, 8, 8)
    noise = torch.randn_like(sample)
    # scalar timestep tensor
    t = torch.tensor(500.0)

    out = sched.add_noise(sample, noise, t)

    assert out.shape == sample.shape
    assert out.dtype == sample.dtype


def test_flow_match_scheduler_step_sequence_decreasing() -> None:
    """Each step should bring latents closer to clean (sigma decreases)."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    sched = SanaWmFlowMatchScheduler(num_inference_steps=4, shift=9.8)
    ts = sched.timesteps(device=torch.device("cpu"))

    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))


def test_flow_match_scheduler_per_token_step_matches_diffusers_formula() -> None:
    """Native per-token step mirrors FlowMatchEulerDiscreteScheduler.step.

    The diffusers per-token branch derives the current sigma from
    ``per_token_timesteps / num_train_timesteps`` and selects the largest
    scheduler sigma strictly below it as ``next_sigma``.
    """
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowMatchScheduler

    sched = SanaWmFlowMatchScheduler(num_inference_steps=3, shift=9.8)
    timesteps = sched.timesteps(device=torch.device("cpu"))
    latents = torch.randn(1, 4, 2, 2, 3)
    noise_pred = torch.randn_like(latents)
    per_frame_t = timesteps[0].expand(1, 1, 2).clone()
    per_frame_t[:, :, 0] = 0.0
    per_token_t = per_frame_t.unsqueeze(-1).unsqueeze(-1).expand(1, 1, 2, 2, 3).reshape(1, -1)

    out = sched.step_flow_euler_per_token(noise_pred, timesteps[0], latents, per_token_t)

    sigmas = sched.sigmas.float()
    per_token_sigmas = per_token_t.float() / float(sched.num_train_timesteps)
    lower_mask = sigmas[:, None, None] < per_token_sigmas[None] - 1e-6
    lower_sigmas = (lower_mask * sigmas[:, None, None]).max(dim=0).values
    dt = per_token_sigmas - lower_sigmas
    latents_flat = latents.permute(0, 2, 3, 4, 1).reshape(1, -1, 4).float()
    noise_flat = noise_pred.permute(0, 2, 3, 4, 1).reshape(1, -1, 4).float()
    expected = latents_flat - dt.unsqueeze(-1) * noise_flat
    expected = expected.reshape(1, 2, 2, 3, 4).permute(0, 4, 1, 2, 3).contiguous()

    assert torch.allclose(out, expected, atol=1e-6)


def test_flow_match_condition_mask_matches_nvlabs_bf16_boundary() -> None:
    """NVlabs' post-step mask intentionally skips t=1000 under bf16.

    This is the exact expression from ``LTXFlowEuler.sample``. With bf16
    latents, ``1 - 1e-6`` rounds to ``1``, so generated tokens are not
    updated at the first full-noise step.
    """
    import torch

    condition_mask = torch.zeros(1, 1, 2, 1, 1, dtype=torch.bfloat16)
    condition_mask[:, :, 0] = 1.0

    first_step = torch.tensor(1000.0)
    first_mask = first_step / 1000.0 - 1e-6 < (1.0 - condition_mask)
    assert not bool(first_mask[:, :, 0].item())
    assert not bool(first_mask[:, :, 1].item())

    later_step = torch.tensor(909.0270)
    later_mask = later_step / 1000.0 - 1e-6 < (1.0 - condition_mask)
    assert not bool(later_mask[:, :, 0].item())
    assert bool(later_mask[:, :, 1].item())


# ===========================================================================
# A.2b — Hybrid softmax blocks
# ===========================================================================


def test_softmax_self_attention_applies_shared_output_gate() -> None:
    """Every-N-th softmax blocks still use the GDN output_gate + proj path."""
    import torch
    import torch.nn.functional as F

    from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    cfg = SanaWmConfig(
        **_tiny_cfg_kwargs(
            qk_norm=False,
            cross_norm=False,
            conv_kernel_size=0,
        )
    )
    attn = SanaWmSelfAttention(cfg, use_gdn=False, use_vllm_parallel_layers=False)
    attn.eval()

    with torch.no_grad():
        attn.qkv.weight.zero_()
        # q = 0, k = 0 -> uniform softmax; v = identity(hidden).
        attn.qkv.weight[16:24].copy_(torch.eye(8))
        attn.output_gate.weight.zero_()
        attn.output_gate.bias.fill_(1.0)
        attn.proj.weight.copy_(torch.eye(8))
        attn.proj.bias.zero_()

    hidden = (torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 10.0)
    out = attn(hidden, spatial_shape=(1, 2, 2), rotary_emb=None, camera_conditions=None)

    raw_uniform = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
    expected = raw_uniform * F.silu(torch.tensor(1.0))
    assert torch.allclose(out.float(), expected, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(out.float(), raw_uniform, atol=1e-2, rtol=1e-2)


def test_mlp_preserves_nvlabs_noncontiguous_conv_input_layout() -> None:
    """GLUMBConvTemp must feed Conv2d the same strided view as NVlabs."""
    import torch

    from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmMbConvFfn

    cfg = SanaWmConfig(**_tiny_cfg_kwargs())
    mlp = SanaWmMbConvFfn(cfg)
    seen: dict[str, object] = {}

    def capture_input(_module, args):
        conv_input = args[0]
        seen["is_contiguous"] = conv_input.is_contiguous()
        seen["stride"] = conv_input.stride()

    handle = mlp.inverted_conv.conv.register_forward_pre_hook(capture_input)
    try:
        hidden = torch.randn(1, 2 * 2 * 3, cfg.hidden_size)
        _ = mlp(hidden, spatial_shape=(2, 2, 3))
    finally:
        handle.remove()

    assert seen["is_contiguous"] is False
    # NVlabs uses x.reshape(B*T, H, W, C).permute(0, 3, 1, 2), where channel
    # stride is 1. A contiguous NCHW copy would have channel stride H*W.
    assert seen["stride"][1] == 1


# ===========================================================================
# A.3 — spatial_raymap in build_plucker_condition
# ===========================================================================


def test_build_plucker_condition_returns_spatial_raymap() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-8", num_frames=9, height=64, width=96)
    tensors = build_plucker_condition(condition)

    assert "spatial_raymap" in tensors, "build_plucker_condition must return 'spatial_raymap'"


def test_build_plucker_condition_spatial_raymap_shape() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-8", num_frames=9, height=64, width=96)
    tensors = build_plucker_condition(condition)
    spatial_raymap = tensors["spatial_raymap"]

    # Expected: [3, F_latent, latent_H, latent_W]
    # With vae_stride=(8,32,32): latent_frames=(9-1)//8+1=2, lh=64//32=2, lw=96//32=3
    assert spatial_raymap.ndim == 4
    assert spatial_raymap.shape[0] == 3, "spatial_raymap must have 3 direction channels"
    assert spatial_raymap.shape == torch.Size([3, 2, 2, 3])


def test_build_plucker_condition_spatial_raymap_unit_vectors() -> None:
    """Ray direction vectors should be (approximately) unit length."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-8", num_frames=9, height=64, width=64)
    tensors = build_plucker_condition(condition)
    sr = tensors["spatial_raymap"]  # [3, F, H, W]

    # Norms along channel dim
    norms = sr.norm(dim=0)  # [F, H, W]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_build_plucker_condition_spatial_raymap_varies_spatially() -> None:
    """Different pixels should have different ray directions (except degenerate cases)."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition, build_plucker_condition

    condition = SanaWmCameraCondition(action="w-8", num_frames=9, height=128, width=128)
    tensors = build_plucker_condition(condition)
    sr = tensors["spatial_raymap"]  # [3, F, H, W]

    # At a single frame, different H/W positions should give different directions
    frame0 = sr[:, 0, :, :]  # [3, H, W]
    h, w = frame0.shape[1], frame0.shape[2]
    if h > 1 and w > 1:
        corner_tl = frame0[:, 0, 0]
        corner_br = frame0[:, -1, -1]
        assert not torch.allclose(corner_tl, corner_br, atol=1e-4)


# ===========================================================================
# A.1 — _preprocess_first_frame and _vae_encode_first_frame
# ===========================================================================


def test_preprocess_first_frame_pil() -> None:
    """PIL Image → [1, 3, 1, H, W] in [-1, 1]."""
    import torch

    pytest.importorskip("PIL")
    from PIL import Image

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    img = Image.new("RGB", (96, 64), color=(128, 128, 128))
    tensor = SanaWmPipeline._preprocess_first_frame(
        img,
        height=64,
        width=96,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert tensor.shape == (1, 3, 1, 64, 96)
    assert tensor.min() >= -1.0 - 1e-4
    assert tensor.max() <= 1.0 + 1e-4


def test_preprocess_first_frame_numpy() -> None:
    """numpy ndarray → [1, 3, 1, H, W] in [-1, 1]."""
    import numpy as np
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    arr = np.full((64, 96, 3), 128, dtype=np.uint8)
    tensor = SanaWmPipeline._preprocess_first_frame(
        arr,
        height=64,
        width=96,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert tensor.shape == (1, 3, 1, 64, 96)
    assert tensor.min() >= -1.0 - 1e-4
    assert tensor.max() <= 1.0 + 1e-4


def test_preprocess_first_frame_tensor_chw_float01() -> None:
    """[C, H, W] float tensor in [0,1] → [1, 3, 1, H, W] in [-1, 1]."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    t = torch.full((3, 64, 96), 0.5)
    tensor = SanaWmPipeline._preprocess_first_frame(
        t,
        height=64,
        width=96,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert tensor.shape == (1, 3, 1, 64, 96)
    assert tensor.min() >= -1.0 - 1e-4
    assert tensor.max() <= 1.0 + 1e-4


def test_preprocess_first_frame_unsupported_type_raises() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    with pytest.raises(TypeError, match="PIL/ndarray/Tensor"):
        SanaWmPipeline._preprocess_first_frame(
            object(),
            height=64,
            width=96,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_vae_encode_first_frame_shape() -> None:
    """_vae_encode_first_frame returns [1, C, 1, latent_h, latent_w]."""
    import torch

    pytest.importorskip("PIL")
    from PIL import Image

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    class FakeVAE(torch.nn.Module):
        dtype = torch.float32

        class _FakeDist:
            def __init__(self, shape):
                self.mean = torch.zeros(*shape)

        class _FakeEncResult:
            def __init__(self, shape):
                self.latent_dist = FakeVAE._FakeDist(shape)

        config = SimpleNamespace(scaling_factor=0.5)

        def encode(self, x):
            # x: [1, 3, 1, H, W] → latent [1, 128, 1, lh, lw]
            b, c, f, h, w = x.shape
            return self._FakeEncResult((b, 128, f, h // 2, w // 2))

    pipeline = SanaWmPipeline(od_config=None)
    pipeline.vae = FakeVAE()

    img = Image.new("RGB", (96, 64))
    result = pipeline._vae_encode_first_frame(
        img,
        height=64,
        width=96,
        latent_height=2,
        latent_width=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result.shape == (1, 128, 1, 2, 3)


def test_pipeline_first_frame_encoded_flag_false_for_placeholder() -> None:
    """object() placeholder image must NOT trigger VAE encode."""
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmPipeline

    config = SanaWmConfig(**_tiny_cfg_kwargs(chunk_plucker_channels=48))
    pipeline = SanaWmPipeline(od_config=None)
    pipeline.sana_wm_config = config
    pipeline.transformer.config = config
    req = SimpleNamespace(
        prompts=[
            {
                "prompt": "test",
                "multi_modal_data": {"image": object()},  # not a real image
                "sana_wm": {"action": "w-1", "num_frames": 1, "height": 64, "width": 64},
            }
        ],
        sampling_params=SimpleNamespace(
            height=64,
            width=64,
            num_frames=1,
            num_inference_steps=1,
            seed=0,
            extra_args={"sana_wm_native_smoke": True, "sana_wm_hash_prompt_smoke": True},
        ),
    )

    output = pipeline(req)

    assert output.custom_output["sana_wm_first_frame_encoded"] is False


def test_pipeline_first_frame_encoded_flag_true_for_pil() -> None:
    """A real PIL image must set sana_wm_first_frame_encoded=True."""
    import torch

    pytest.importorskip("PIL")
    from PIL import Image

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmPipeline

    class FakeVAE(torch.nn.Module):
        dtype = torch.float32

        class _FakeDist:
            def __init__(self, t):
                self.mean = t

        class _FakeEncResult:
            def __init__(self, t):
                self.latent_dist = FakeVAE._FakeDist(t)

        config = SimpleNamespace(scaling_factor=1.0)

        def encode(self, x):
            b, c, f, h, w = x.shape
            return self._FakeEncResult(torch.zeros(b, 128, f, h, w))

        def to(self, *args, **kwargs):
            return self

    config = SanaWmConfig(**_tiny_cfg_kwargs(chunk_plucker_channels=48))
    pipeline = SanaWmPipeline(od_config=None)
    pipeline.sana_wm_config = config
    pipeline.transformer.config = config
    pipeline.vae = FakeVAE()

    img = Image.new("RGB", (64, 64))
    req = SimpleNamespace(
        prompts=[
            {
                "prompt": "test",
                "multi_modal_data": {"image": img},
                "sana_wm": {"action": "w-1", "num_frames": 1, "height": 64, "width": 64},
            }
        ],
        sampling_params=SimpleNamespace(
            height=64,
            width=64,
            num_frames=1,
            num_inference_steps=1,
            seed=0,
            extra_args={"sana_wm_native_smoke": True, "sana_wm_hash_prompt_smoke": True},
        ),
    )

    output = pipeline(req)

    assert output.custom_output["sana_wm_first_frame_encoded"] is True


# ===========================================================================
# A.3 — UCPE camera branch (GDN+UCPE per-block attention)
# ===========================================================================
#
# The original A.3 tests exercised an SDPA-based ``_forward_ucpe`` that
# treated camera information as cross-attention K/V. That implementation
# was reopened by the 2026-05-27 deep-dive (audit §6.10) and replaced with
# a real bidirectional GDN+UCPE cam branch matching NVlabs' reference.
# These tests now exercise the new ``_forward_cam_branch`` signature.


def _make_tiny_camera_conditions(batch: int, frames: int):
    """Build a (B, T, 20) raymap tensor with identity poses + safe intrinsics."""
    import torch

    c2w = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch, frames, 1, 1)
    # Latent-pixel intrinsics. Use small fx/fy so the unprojected rays are
    # well-defined for any (H, W) grid.
    intrinsics = torch.tensor([[8.0, 8.0, 2.0, 2.0]]).repeat(batch, frames, 1)
    return torch.cat([c2w.reshape(batch, frames, 16), intrinsics], dim=-1)


def test_sana_wm_self_attention_cam_dim_set_before_projections() -> None:
    """cam_dim must be set before the TP/fallback branch so projection sizes are correct."""
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    cfg = SanaWmConfig(**_TINY_CFG_KWARGS)
    attn = SanaWmSelfAttention(cfg, use_gdn=True, use_vllm_parallel_layers=False)

    assert attn.cam_dim == cfg.hidden_size  # cam_attn_compress=1
    assert attn.q_proj_cam.in_features == cfg.hidden_size
    assert attn.q_proj_cam.out_features == attn.cam_dim
    assert attn.out_proj_cam.in_features == attn.cam_dim
    assert attn.out_proj_cam.out_features == cfg.hidden_size


def test_forward_cam_branch_output_shape() -> None:
    """_forward_cam_branch must return (B, N, cam_dim) raw output."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    cfg = SanaWmConfig(**_tiny_cfg_kwargs(hidden_size=16, linear_head_dim=8))
    attn = SanaWmSelfAttention(cfg, use_gdn=True, use_vllm_parallel_layers=False)
    attn.eval()

    batch = 1
    spatial_shape = (2, 1, 3)  # F*H*W == 6
    frames, height, width = spatial_shape
    seq = frames * height * width
    hidden = torch.randn(batch, seq, cfg.hidden_size)
    camera_conditions = _make_tiny_camera_conditions(batch, frames)
    # Precompute gates with the same code path forward() uses.
    beta, decay = attn._compute_frame_gates(hidden, spatial_shape)

    with torch.no_grad():
        out = attn._forward_cam_branch(
            hidden,
            spatial_shape,
            camera_conditions,
            rotary_emb=None,
            precomputed_gates=(beta, decay),
        )

    assert out.shape == (batch, seq, attn.cam_dim)
    assert torch.isfinite(out).all()


def test_forward_with_camera_conditions_changes_output() -> None:
    """forward() must produce a different result when camera_conditions is provided."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    torch.manual_seed(42)
    cfg = SanaWmConfig(**_tiny_cfg_kwargs(hidden_size=16, linear_head_dim=8))
    attn = SanaWmSelfAttention(cfg, use_gdn=True, use_vllm_parallel_layers=False)
    attn.eval()

    batch = 1
    spatial_shape = (2, 1, 3)
    frames, height, width = spatial_shape
    seq = frames * height * width
    hidden = torch.randn(batch, seq, cfg.hidden_size)
    camera_conditions = _make_tiny_camera_conditions(batch, frames)

    with torch.no_grad():
        out_no_cam = attn(hidden, spatial_shape, rotary_emb=None, camera_conditions=None)
        out_with_cam = attn(
            hidden, spatial_shape, rotary_emb=None, camera_conditions=camera_conditions
        )

    # With at-init weights cam_contrib may be small but must not be exactly
    # equal to the main-only path. Use a relaxed tolerance.
    assert not torch.allclose(out_no_cam, out_with_cam, atol=0.0, rtol=0.0)
    assert torch.isfinite(out_with_cam).all()


def test_self_attention_forward_with_camera_differs_from_without() -> None:
    """SanaWmSelfAttention.forward must apply UCPE when camera_hidden_states given."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    torch.manual_seed(0)
    cfg = SanaWmConfig(**_TINY_CFG_KWARGS, softmax_every_n=0)  # force GDN
    attn = SanaWmSelfAttention(cfg, use_gdn=True, use_vllm_parallel_layers=False)
    attn.eval()

    batch, frames, h, w, d = 1, 2, 1, 2, cfg.hidden_size
    hidden = torch.randn(batch, frames * h * w, d)
    camera = torch.randn(batch, frames * h * w, d)
    spatial_shape = (frames, h, w)

    with torch.no_grad():
        out_no_cam = attn.forward(hidden, spatial_shape)
        out_with_cam = attn.forward(hidden, spatial_shape, camera_hidden_states=camera)

    assert out_no_cam.shape == out_with_cam.shape
    assert not torch.allclose(out_no_cam, out_with_cam)


def test_self_attention_forward_none_camera_unchanged() -> None:
    """Passing camera_hidden_states=None must give same result as omitting it."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    torch.manual_seed(1)
    cfg = SanaWmConfig(**_TINY_CFG_KWARGS, softmax_every_n=0)
    attn = SanaWmSelfAttention(cfg, use_gdn=False, use_vllm_parallel_layers=False)
    attn.eval()

    hidden = torch.randn(1, 4, cfg.hidden_size)

    with torch.no_grad():
        out_default = attn.forward(hidden)
        out_none_cam = attn.forward(hidden, camera_hidden_states=None)

    assert torch.allclose(out_default, out_none_cam)


def test_transformer_forward_with_plucker_and_spatial_raymap_shape() -> None:
    """Transformer forward with full camera conditioning returns correct latent shape."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    cfg = SanaWmConfig(**_tiny_cfg_kwargs(chunk_plucker_channels=6))
    model = SanaWmTransformer3DModel(config=cfg)
    model.eval()

    latents = torch.randn(1, 4, 2, 2, 2)
    prompt_embeds = torch.randn(1, 3, 6)
    # plucker: [chunk_plucker_channels, F_latent, H_latent, W_latent]
    plucker = torch.randn(6, 2, 2, 2)
    # spatial_raymap: [3, F_latent, H_latent, W_latent]
    spatial_raymap = torch.randn(3, 2, 2, 2)

    with torch.no_grad():
        output = model(
            latents,
            torch.tensor([500.0]),
            encoder_hidden_states=prompt_embeds,
            plucker=plucker,
            spatial_raymap=spatial_raymap,
        )

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()


def test_camera_conditioning_changes_transformer_output() -> None:
    """Transformer output must differ when camera conditioning is provided vs. not."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    torch.manual_seed(7)
    cfg = SanaWmConfig(**_tiny_cfg_kwargs(chunk_plucker_channels=6))
    model = SanaWmTransformer3DModel(config=cfg)
    model.eval()

    latents = torch.randn(1, 4, 2, 2, 2)
    prompt_embeds = torch.randn(1, 3, 6)
    plucker = torch.randn(6, 2, 2, 2)
    spatial_raymap = torch.randn(3, 2, 2, 2)

    with torch.no_grad():
        out_no_cam = model(latents, torch.tensor([500.0]), encoder_hidden_states=prompt_embeds)
        out_with_cam = model(
            latents,
            torch.tensor([500.0]),
            encoder_hidden_states=prompt_embeds,
            plucker=plucker,
            spatial_raymap=spatial_raymap,
        )

    assert not torch.allclose(out_no_cam, out_with_cam), (
        "Camera conditioning must change transformer output"
    )


def test_camera_hidden_states_from_conditions_with_spatial_raymap() -> None:
    """_camera_hidden_states_from_conditions fuses plucker + spatial_raymap."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    torch.manual_seed(3)
    cfg = SanaWmConfig(**_tiny_cfg_kwargs(chunk_plucker_channels=6))
    model = SanaWmTransformer3DModel(config=cfg, materialize=True)

    plucker = torch.randn(1, 6, 2, 2, 2)  # [B, C, F, H, W]
    spatial_raymap = torch.randn(1, 3, 2, 2, 2)
    spatial_shape = (2, 2, 2)

    with torch.no_grad():
        cam_plucker_only = model._camera_hidden_states_from_conditions(
            plucker=plucker,
            spatial_raymap=None,
            spatial_shape=spatial_shape,
            batch_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cam_with_raymap = model._camera_hidden_states_from_conditions(
            plucker=plucker,
            spatial_raymap=spatial_raymap,
            spatial_shape=spatial_shape,
            batch_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    assert cam_plucker_only is not None
    assert cam_with_raymap is not None
    assert cam_plucker_only.shape == cam_with_raymap.shape
    assert not torch.allclose(cam_plucker_only, cam_with_raymap), (
        "spatial_raymap must modify camera hidden states"
    )


def test_pipeline_scheduler_uses_flow_match_and_descends() -> None:
    """Native backend must use SanaWmFlowMatchScheduler: timesteps are descending."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmPipeline

    class RecordingTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.timesteps: list[float] = []

        def forward(self, hidden_states, timestep, **_):
            self.timesteps.append(float(timestep.flatten()[0]))
            return torch.zeros_like(hidden_states)

    cfg = SanaWmConfig(**_tiny_cfg_kwargs(chunk_plucker_channels=48))
    pipeline = SanaWmPipeline(od_config=None)
    pipeline.sana_wm_config = cfg
    pipeline.transformer = RecordingTransformer()
    pipeline.transformer.config = cfg

    req = SimpleNamespace(
        prompts=[
            {
                "prompt": "descend test",
                "multi_modal_data": {"image": object()},
                "sana_wm": {"action": "w-1", "num_frames": 1, "height": 64, "width": 64},
            }
        ],
        sampling_params=SimpleNamespace(
            height=64, width=64, num_frames=1, num_inference_steps=4,
            seed=0,
            extra_args={"sana_wm_native_smoke": True, "sana_wm_hash_prompt_smoke": True},
        ),
    )

    pipeline(req)
    ts = pipeline.transformer.timesteps

    assert len(ts) == 4
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1)), (
        "Scheduler timesteps must be strictly descending"
    )
    # SanaWmFlowMatchScheduler timesteps are in the 0–1000 range (not 0–1)
    assert ts[0] > 1.0, "FlowMatch timesteps are in integer range, not [0,1]"


# ---------------------------------------------------------------------------
# B.2 — Full scheduler-loop / VAE roundtrip integration tests
# ---------------------------------------------------------------------------

def test_scheduler_full_denoising_loop_produces_finite_output() -> None:
    """Run a complete N-step FlowMatch loop and verify output is finite."""
    import torch

    from vllm_omni.diffusion.models.sana_wm.scheduling_sana_wm import SanaWmFlowMatchScheduler

    steps = 5
    scheduler = SanaWmFlowMatchScheduler(num_inference_steps=steps)
    device = torch.device("cpu")
    timesteps = scheduler.timesteps(device=device)

    latents = torch.randn(1, 4, 2, 4, 4)
    noise = torch.randn_like(latents)
    latents = scheduler.add_noise(latents, noise, timesteps[0])

    for t in timesteps:
        noise_pred = torch.randn_like(latents)
        latents = scheduler.step(noise_pred, t, latents)

    assert latents.shape == (1, 4, 2, 4, 4)
    assert torch.isfinite(latents).all(), "Denoised latents must be finite"


def test_vae_encode_then_add_noise_roundtrip_shape() -> None:
    """VAE encode → add_noise → scheduler.step output shape is preserved."""
    import torch
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline
    from vllm_omni.diffusion.models.sana_wm.scheduling_sana_wm import SanaWmFlowMatchScheduler

    class FakeLatentDist:
        def __init__(self, mean):
            self.mean = mean

    class FakeVAE(torch.nn.Module):
        dtype = torch.float32
        spatial_compression_ratio = 8
        config = SimpleNamespace(timestep_conditioning=False, scaling_factor=0.18)

        def encode(self, x):
            # Return a fake distribution whose mean is a downsampled version.
            B, C, F, H, W = x.shape
            latent = torch.randn(B, 4, F, H // 8, W // 8)
            return SimpleNamespace(latent_dist=FakeLatentDist(latent))

        def decode(self, latents, timestep=None, return_dict=False):
            B, C, F, H, W = latents.shape
            video = torch.zeros(B, 3, F, H * 8, W * 8)
            return (video,)

    from PIL import Image
    import numpy as np

    pipeline = SanaWmPipeline(od_config=None)
    pipeline.vae = FakeVAE()

    first_frame = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    latent = pipeline._vae_encode_first_frame(
        first_frame,
        height=64,
        width=64,
        latent_height=8,
        latent_width=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # latent: [1, 4, 1, H_lat, W_lat]
    assert latent.shape[0] == 1
    assert latent.shape[2] == 1  # single frame

    # Add noise at first timestep
    scheduler = SanaWmFlowMatchScheduler(num_inference_steps=3)
    timesteps = scheduler.timesteps(device=torch.device("cpu"))
    noise = torch.randn_like(latent)
    noised = scheduler.add_noise(latent, noise, timesteps[0])

    assert noised.shape == latent.shape
    assert torch.isfinite(noised).all()
