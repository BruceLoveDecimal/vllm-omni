# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubTransformer, StubVAE
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_animate import (
    Wan22AnimatePipeline,
    get_wan22_animate_pre_process_func,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_animate_transformer import (
    WanAnimateFaceBlockCrossAttention,
    create_animate_transformer_from_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_animate_pipeline() -> Wan22AnimatePipeline:
    pipeline = object.__new__(Wan22AnimatePipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(in_channels=36, out_channels=16)
    pipeline.vae = StubVAE(z_dim=16)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    return pipeline


def _frames(count: int, size: tuple[int, int] = (64, 64), mode: str = "RGB") -> list[Image.Image]:
    return [Image.new(mode, size, "black") for _ in range(count)]


# ---------------------------------------------------------------------------
# Pre-processing / mode dispatch
# ---------------------------------------------------------------------------


def test_animate_preprocess_collects_pose_and_face_videos() -> None:
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())
    ref = Image.new("RGB", (320, 160), "green")
    pose = _frames(3, (200, 100))
    face = _frames(3, (512, 512))

    request = SimpleNamespace(
        prompt={
            "prompt": "a dancer",
            "multi_modal_data": {"image": ref, "pose_video": pose, "face_video": face},
        },
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)
    additional_info = result.prompt["additional_information"]

    # Target size falls back to the pose video's, floored to the 16px model grid.
    assert result.sampling_params.width == 192
    assert result.sampling_params.height == 96
    assert len(additional_info["pose_video"]) == 3
    assert len(additional_info["face_video"]) == 3
    # Animation mode: no background / mask conditioning.
    assert additional_info["background_video"] is None
    assert additional_info["mask_video"] is None


def test_animate_preprocess_collects_replacement_inputs() -> None:
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())
    request = SimpleNamespace(
        prompt={
            "prompt": "swap the character",
            "multi_modal_data": {
                "image": Image.new("RGB", (64, 64), "green"),
                "pose_video": _frames(2),
                "face_video": _frames(2, (512, 512)),
                "video": _frames(2),
                "mask": _frames(2, mode="L"),
            },
        },
        sampling_params=SimpleNamespace(height=64, width=64),
    )

    additional_info = preprocess(request).prompt["additional_information"]

    assert len(additional_info["background_video"]) == 2
    assert additional_info["mask_video"][0].mode == "L"


def test_animate_preprocess_rejects_half_specified_replacement() -> None:
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())
    request = SimpleNamespace(
        prompt={
            "prompt": "p",
            "multi_modal_data": {
                "image": Image.new("RGB", (64, 64), "green"),
                "pose_video": _frames(2),
                "face_video": _frames(2, (512, 512)),
                "video": _frames(2),
            },
        },
        sampling_params=SimpleNamespace(height=64, width=64),
    )

    with pytest.raises(ValueError, match="Replacement mode needs both"):
        preprocess(request)


def test_animate_preprocess_requires_pose_and_face_videos() -> None:
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())
    request = SimpleNamespace(
        prompt={
            "prompt": "p",
            "multi_modal_data": {"image": Image.new("RGB", (64, 64), "green"), "pose_video": _frames(2)},
        },
        sampling_params=SimpleNamespace(height=64, width=64),
    )

    with pytest.raises(ValueError, match="pose video and a face video"):
        preprocess(request)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_check_inputs_rejects_mismatched_driving_lengths() -> None:
    pipeline = _make_animate_pipeline()

    with pytest.raises(ValueError, match="must have the same length"):
        pipeline.check_inputs(
            prompt="p",
            image=Image.new("RGB", (64, 64)),
            pose_video=_frames(4),
            face_video=_frames(3),
            background_video=None,
            mask_video=None,
            height=64,
            width=64,
            segment_frame_length=5,
            prev_segment_cond_frames=1,
        )


def test_check_inputs_rejects_unaligned_resolution() -> None:
    pipeline = _make_animate_pipeline()

    with pytest.raises(ValueError, match="divisible by 16"):
        pipeline.check_inputs(
            prompt="p",
            image=Image.new("RGB", (64, 64)),
            pose_video=_frames(4),
            face_video=_frames(4),
            background_video=None,
            mask_video=None,
            height=70,
            width=64,
            segment_frame_length=5,
            prev_segment_cond_frames=1,
        )


def test_check_inputs_rejects_oversized_conditioning_window() -> None:
    pipeline = _make_animate_pipeline()

    with pytest.raises(ValueError, match="must be smaller than `segment_frame_length`"):
        pipeline.check_inputs(
            prompt="p",
            image=Image.new("RGB", (64, 64)),
            pose_video=_frames(10),
            face_video=_frames(10),
            background_video=None,
            mask_video=None,
            height=64,
            width=64,
            segment_frame_length=5,
            prev_segment_cond_frames=5,
        )


# ---------------------------------------------------------------------------
# Latent helpers
# ---------------------------------------------------------------------------


def test_get_i2v_mask_marks_conditioning_frames_across_channel_axis() -> None:
    pipeline = _make_animate_pipeline()

    mask = pipeline.get_i2v_mask(batch_size=1, latent_t=2, latent_h=1, latent_w=1, mask_len=1)

    # The per-pixel-frame mask is folded into 4 channels by the VAE's temporal compression.
    assert mask.shape == (1, 4, 2, 1, 1)
    torch.testing.assert_close(mask[:, :, 0], torch.ones(1, 4, 1, 1))
    torch.testing.assert_close(mask[:, :, 1], torch.zeros(1, 4, 1, 1))


def test_get_i2v_mask_is_all_zero_for_the_first_segment() -> None:
    pipeline = _make_animate_pipeline()

    mask = pipeline.get_i2v_mask(batch_size=1, latent_t=2, latent_h=1, latent_w=1, mask_len=0)

    torch.testing.assert_close(mask, torch.zeros(1, 4, 2, 1, 1))


def test_prepare_reference_image_latents_prepends_mask_channels() -> None:
    pipeline = _make_animate_pipeline()
    image = torch.zeros(1, 3, 32, 32)

    latents = pipeline.prepare_reference_image_latents(image, batch_size=1, device=torch.device("cpu"))

    # 4 mask channels + 16 latent channels, one latent frame.
    assert latents.shape == (1, 20, 1, 4, 4)
    torch.testing.assert_close(latents[:, :4], torch.ones(1, 4, 1, 4, 4))


def test_prepare_latents_adds_the_leading_reference_frame() -> None:
    pipeline = _make_animate_pipeline()

    latents = pipeline.prepare_latents(
        1, num_channels_latents=16, height=64, width=64, num_frames=77, device=torch.device("cpu")
    )

    # (77 - 1) / 4 + 1 = 20 latent frames, plus 1 for the reference image.
    assert latents.shape == (1, 16, 21, 8, 8)


def test_pad_video_frames_bounces_back_and_forth() -> None:
    assert Wan22AnimatePipeline.pad_video_frames([1, 2, 3, 4, 5], 10) == [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]


def test_pad_video_frames_handles_single_frame_input() -> None:
    assert Wan22AnimatePipeline.pad_video_frames(["a"], 3) == ["a", "a", "a"]


# ---------------------------------------------------------------------------
# Transformer pieces
# ---------------------------------------------------------------------------


def test_create_animate_transformer_from_config_maps_animate_specific_keys(monkeypatch) -> None:
    captured = {}

    class FakeAnimateTransformer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.wan2_2_animate_transformer.WanAnimateTransformer3DModel",
        FakeAnimateTransformer,
    )

    transformer = create_animate_transformer_from_config(
        {
            "patch_size": [1, 2, 2],
            "in_channels": 36,
            "latent_channels": 16,
            "num_layers": 40,
            "inject_face_latents_blocks": 5,
            "motion_encoder_size": 512,
            "face_encoder_num_heads": 4,
            "unknown": "ignored",
        }
    )

    assert isinstance(transformer, FakeAnimateTransformer)
    assert captured == {
        "patch_size": (1, 2, 2),
        "in_channels": 36,
        "latent_channels": 16,
        "num_layers": 40,
        "inject_face_latents_blocks": 5,
        "motion_encoder_size": 512,
        "face_encoder_num_heads": 4,
    }


def test_face_adapter_preserves_hidden_state_shape() -> None:
    torch.manual_seed(0)
    attn = WanAnimateFaceBlockCrossAttention(dim=8, heads=2, dim_head=4, cross_attention_dim_head=4)
    hidden_states = torch.randn(1, 6, 8)
    # 3 motion frames x 2 tokens each; 6 video tokens => 2 tokens per frame.
    motion_vec = torch.randn(1, 3, 2, 8)

    out = attn(hidden_states, motion_vec)

    assert out.shape == hidden_states.shape


def test_face_adapter_rejects_sequence_not_divisible_by_motion_frames() -> None:
    attn = WanAnimateFaceBlockCrossAttention(dim=8, heads=2, dim_head=4, cross_attention_dim_head=4)
    hidden_states = torch.randn(1, 7, 8)
    motion_vec = torch.randn(1, 3, 2, 8)

    with pytest.raises(ValueError, match="divisible by the motion frame count"):
        attn(hidden_states, motion_vec)
