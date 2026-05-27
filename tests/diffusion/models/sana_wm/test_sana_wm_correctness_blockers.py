# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the three single-GPU correctness blockers:

  A.1 — First-frame VAE encode (pipeline_sana_wm._vae_encode_first_frame)
  A.2 — FlowMatchDPMSolverMultistepScheduler (SanaWmFlowMatchScheduler)
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
    num_train = sched._sched.config.num_train_timesteps  # typically 1000

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

    config = SanaWmConfig(**_TINY_CFG_KWARGS, chunk_plucker_channels=48)
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
            return self._FakeEncResult(torch.zeros(b, 4, f, h, w))

        def to(self, *args, **kwargs):
            return self

    config = SanaWmConfig(**_TINY_CFG_KWARGS, chunk_plucker_channels=48)
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
# A.3 — _forward_ucpe and transformer camera conditioning
# ===========================================================================


def test_sana_wm_self_attention_cam_dim_set_before_projections() -> None:
    """cam_dim must be set before the TP/fallback branch so projection sizes are correct."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    cfg = SanaWmConfig(**_TINY_CFG_KWARGS)
    attn = SanaWmSelfAttention(cfg, use_gdn=True, use_vllm_parallel_layers=False)

    assert attn.cam_dim == cfg.hidden_size  # cam_attn_compress=1
    assert attn.q_proj_cam.in_features == cfg.hidden_size
    assert attn.q_proj_cam.out_features == attn.cam_dim
    assert attn.out_proj_cam.in_features == attn.cam_dim
    assert attn.out_proj_cam.out_features == cfg.hidden_size


def test_forward_ucpe_output_shape() -> None:
    """_forward_ucpe must return tensor with same shape as hidden_states."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    cfg = SanaWmConfig(**_TINY_CFG_KWARGS)
    attn = SanaWmSelfAttention(cfg, use_gdn=True, use_vllm_parallel_layers=False)
    attn.eval()

    batch, seq, d = 1, 6, cfg.hidden_size
    hidden = torch.randn(batch, seq, d)
    camera = torch.randn(batch, seq, d)
    spatial_shape = (2, 1, 3)  # F*H*W == 6

    with torch.no_grad():
        out = attn._forward_ucpe(hidden, camera, spatial_shape)

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()


def test_forward_ucpe_changes_output() -> None:
    """With fixed weights, UCPE should produce a non-zero contribution."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmSelfAttention

    torch.manual_seed(42)
    cfg = SanaWmConfig(**_TINY_CFG_KWARGS)
    attn = SanaWmSelfAttention(cfg, use_gdn=True, use_vllm_parallel_layers=False)
    attn.eval()

    batch, seq, d = 1, 4, cfg.hidden_size
    hidden = torch.randn(batch, seq, d)
    camera = torch.randn(batch, seq, d)

    with torch.no_grad():
        ucpe_out = attn._forward_ucpe(hidden, camera, None)

    assert not torch.allclose(ucpe_out, torch.zeros_like(ucpe_out))


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

    cfg = SanaWmConfig(**_TINY_CFG_KWARGS, chunk_plucker_channels=6)
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
    cfg = SanaWmConfig(**_TINY_CFG_KWARGS, chunk_plucker_channels=6)
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
    cfg = SanaWmConfig(**_TINY_CFG_KWARGS, chunk_plucker_channels=6)
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

    cfg = SanaWmConfig(**_TINY_CFG_KWARGS, chunk_plucker_channels=48)
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
