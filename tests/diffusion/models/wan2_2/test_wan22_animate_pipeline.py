# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_animate import (
    get_wan22_animate_pre_process_func,
    pad_video_frames,
    preprocess_video,
    resize_and_fill,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_animate_transformer import (
    WanAnimateFaceBlockCrossAttention,
    WanAnimateFaceEncoder,
    WanAnimateMotionEncoder,
    WanAnimateTransformer3DModel,
    check_unsupported_parallelism,
    create_animate_transformer_from_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# The backbone is built from vLLM layers, whose norm kernels dispatch to Triton
# whenever a CUDA driver is present -- and Triton rejects CPU tensors. Run the
# real-module tests on the accelerator when there is one, and on CPU otherwise
# (where Triton disables itself and the native kernels take over).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tiny stand-in for the 14B config: 2 blocks, 16-dim, 8x8 face frames.
TINY_CONFIG = {
    "patch_size": [1, 2, 2],
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "latent_channels": 4,
    "out_channels": 4,
    "text_dim": 8,
    "freq_dim": 16,
    "ffn_dim": 16,
    "num_layers": 2,
    "eps": 1e-6,
    "motion_encoder_channel_sizes": {"8": 4, "4": 8},
    "motion_encoder_size": 8,
    "motion_style_dim": 8,
    "motion_dim": 4,
    "motion_encoder_dim": 8,
    "face_encoder_hidden_dim": 8,
    "face_encoder_num_heads": 2,
    "inject_face_latents_blocks": 1,
    "motion_encoder_batch_size": 2,
}

# A 5-frame segment yields 2 latent frames, so hidden_states carries 3 frames
# (2 generated + 1 reference slot) and the face encoder emits 2 + 1 tokens.
SEGMENT_FRAMES = 5
LATENT_FRAMES = 2


@pytest.fixture(scope="module")
def single_rank_env():
    """Single-rank distributed env so vLLM parallel layers can be built."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            data_parallel_size=1,
            cfg_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    yield
    destroy_distributed_env()


@pytest.fixture(scope="module")
def tiny_transformer(single_rank_env) -> WanAnimateTransformer3DModel:
    """Full tiny model. Needs the platform to supply an attention backend.

    The Animate-specific submodules are plain torch and are exercised
    separately, so they still run where the backbone cannot be built.
    """
    del single_rank_env
    try:
        model = create_animate_transformer_from_config(TINY_CONFIG).to(_DEVICE).eval()
    except Exception as exc:  # noqa: BLE001 - platform capability probe
        pytest.skip(f"cannot build the backbone on this platform: {exc}")

    # vLLM layers allocate parameters with torch.empty and rely on load_weights
    # to fill them. These tests never load a checkpoint, so without this the
    # forward runs on uninitialised memory -- which happens to be finite on CPU
    # but is routinely NaN on CUDA. Seed every parameter with small finite
    # values so the numerics are meaningful and identical on both devices.
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.empty(param.shape, dtype=param.dtype).normal_(0.0, 0.02, generator=generator))
    return model


@pytest.fixture(scope="module")
def tiny_motion_encoder() -> WanAnimateMotionEncoder:
    return WanAnimateMotionEncoder(
        size=TINY_CONFIG["motion_encoder_size"],
        style_dim=TINY_CONFIG["motion_style_dim"],
        motion_dim=TINY_CONFIG["motion_dim"],
        out_dim=TINY_CONFIG["motion_encoder_dim"],
        channels=TINY_CONFIG["motion_encoder_channel_sizes"],
    ).eval()


def _make_request(multi_modal_data: dict, height=None, width=None, extra_args=None) -> SimpleNamespace:
    return SimpleNamespace(
        prompt={"prompt": "a dancing character", "multi_modal_data": multi_modal_data},
        sampling_params=SimpleNamespace(height=height, width=width, extra_args=extra_args or {}),
    )


# ---------------------------------------------------------------------------
# Pre-process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["image", "pose_video", "face_video"])
def test_animate_preprocess_requires_all_three_inputs(missing: str) -> None:
    multi_modal_data = {
        "image": Image.new("RGB", (64, 64), "green"),
        "pose_video": "pose.mp4",
        "face_video": "face.mp4",
    }
    del multi_modal_data[missing]
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())

    with pytest.raises(ValueError, match=missing.split("_")[0]):
        preprocess(_make_request(multi_modal_data))


def test_animate_preprocess_derives_size_and_fits_reference_image() -> None:
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())
    request = _make_request(
        {
            "image": Image.new("RGB", (640, 480), "green"),
            "pose_video": "pose.mp4",
            "face_video": "face.mp4",
        }
    )

    result = preprocess(request)

    height, width = result.sampling_params.height, result.sampling_params.width
    assert height % 16 == 0 and width % 16 == 0
    assert result.prompt["multi_modal_data"]["image"].size == (width, height)
    assert result.prompt["additional_information"]["pose_video"] == "pose.mp4"
    assert result.prompt["additional_information"]["face_video"] == "face.mp4"


def test_animate_preprocess_reads_serving_keys() -> None:
    """Online serving delivers pose as `video` and face via extra_args."""
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())
    frame = Image.new("RGB", (32, 32), "black")
    request = _make_request(
        {"image": Image.new("RGB", (64, 64), "green"), "video": [frame]},
        height=64,
        width=64,
        extra_args={"face_video": "face.mp4"},
    )

    result = preprocess(request)

    assert result.prompt["additional_information"]["pose_video"] == [frame]
    assert result.prompt["additional_information"]["face_video"] == "face.mp4"


def test_animate_preprocess_rejects_misaligned_resolution() -> None:
    preprocess = get_wan22_animate_pre_process_func(SimpleNamespace())
    request = _make_request(
        {"image": Image.new("RGB", (64, 64), "green"), "pose_video": "p.mp4", "face_video": "f.mp4"},
        height=100,
        width=64,
    )

    with pytest.raises(ValueError, match="divisible by 16"):
        preprocess(request)


def test_resize_and_fill_centers_on_black_canvas() -> None:
    # A 2:1 source in a 1:1 canvas leaves black bars top and bottom.
    filled = resize_and_fill(Image.new("RGB", (64, 32), "white"), height=64, width=64)

    assert filled.size == (64, 64)
    assert filled.getpixel((32, 32)) == (255, 255, 255)
    assert filled.getpixel((32, 0)) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Segment arithmetic
# ---------------------------------------------------------------------------


def test_pad_video_frames_bounces_at_the_ends() -> None:
    assert pad_video_frames([1, 2, 3, 4, 5], 10) == [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
    assert pad_video_frames([1, 2, 3], 3) == [1, 2, 3]
    assert pad_video_frames([7], 4) == [7, 7, 7, 7]


def test_preprocess_video_normalizes_to_unit_range() -> None:
    frames = [Image.new("RGB", (32, 16), "white"), Image.new("RGB", (32, 16), "black")]

    video = preprocess_video(frames, height=16, width=32)

    assert video.shape == (1, 3, 2, 16, 32)
    torch.testing.assert_close(video[:, :, 0], torch.ones_like(video[:, :, 0]))
    torch.testing.assert_close(video[:, :, 1], -torch.ones_like(video[:, :, 1]))


# ---------------------------------------------------------------------------
# Config plumbing and guards
# ---------------------------------------------------------------------------


def test_create_animate_transformer_maps_animate_specific_keys(monkeypatch) -> None:
    captured = {}

    class FakeAnimateTransformer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.wan2_2_animate_transformer.WanAnimateTransformer3DModel",
        FakeAnimateTransformer,
    )

    create_animate_transformer_from_config(
        {
            "patch_size": [1, 2, 2],
            "in_channels": 36,
            "num_layers": 40,
            "inject_face_latents_blocks": 5,
            "motion_encoder_size": 512,
            "motion_encoder_channel_sizes": None,
            "unknown_key": "ignored",
        }
    )

    assert captured["patch_size"] == (1, 2, 2)
    assert captured["in_channels"] == 36
    assert captured["inject_face_latents_blocks"] == 5
    assert "unknown_key" not in captured
    # Null entries in config.json must not override the constructor defaults.
    assert "motion_encoder_channel_sizes" not in captured


def test_animate_transformer_infers_in_channels_from_latent_channels(tiny_transformer) -> None:
    """TINY_CONFIG carries only latent_channels, so in_channels is derived."""
    assert tiny_transformer.config.in_channels == 2 * TINY_CONFIG["latent_channels"] + 4


def test_check_unsupported_parallelism_rejects_sequence_and_pipeline_parallel() -> None:
    check_unsupported_parallelism(1, 1)

    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        check_unsupported_parallelism(2, 1)
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        check_unsupported_parallelism(1, 2)


def test_animate_transformer_declares_no_sequence_parallel_plan() -> None:
    assert WanAnimateTransformer3DModel._sp_plan == {}


def test_animate_transformer_rejects_indivisible_injection_interval() -> None:
    """Rejected before the backbone is allocated, so no vLLM layers are built."""
    config = {**TINY_CONFIG, "num_layers": 4, "inject_face_latents_blocks": 3}

    with pytest.raises(ValueError, match="divisible by inject_face_latents_blocks"):
        create_animate_transformer_from_config(config)


def test_animate_transformer_rejects_mismatched_channel_packing() -> None:
    config = {**TINY_CONFIG, "in_channels": 20, "latent_channels": 4}

    with pytest.raises(ValueError, match=r"2 \* latent_channels \+ 4"):
        create_animate_transformer_from_config(config)


# ---------------------------------------------------------------------------
# Submodule shapes
# ---------------------------------------------------------------------------


def test_motion_encoder_maps_face_frames_to_motion_vectors(tiny_motion_encoder) -> None:
    faces = torch.randn(3, 3, 8, 8)

    motion_vec = tiny_motion_encoder(faces)

    assert motion_vec.shape == (3, TINY_CONFIG["motion_encoder_dim"])


def test_motion_encoder_rejects_wrong_face_resolution(tiny_motion_encoder) -> None:
    with pytest.raises(ValueError, match="resolution"):
        tiny_motion_encoder(torch.randn(1, 3, 16, 16))


def test_face_encoder_halves_time_twice_and_appends_padding_token() -> None:
    encoder = WanAnimateFaceEncoder(in_dim=8, out_dim=16, hidden_dim=8, num_heads=2).eval()

    tokens = encoder(torch.randn(1, SEGMENT_FRAMES, 8))

    # Two stride-2 causal convs: 5 -> 3 -> 2 frames, plus one padding token.
    assert tokens.shape == (1, LATENT_FRAMES, TINY_CONFIG["face_encoder_num_heads"] + 1, 16)


def test_encode_face_aligns_tokens_with_latent_frames(tiny_transformer) -> None:
    faces = torch.randn(1, 3, SEGMENT_FRAMES, 8, 8, device=_DEVICE)

    motion_vec = tiny_transformer.encode_face(faces)

    inner_dim = TINY_CONFIG["num_attention_heads"] * TINY_CONFIG["attention_head_dim"]
    face_tokens = TINY_CONFIG["face_encoder_num_heads"] + 1
    assert motion_vec.shape == (1, LATENT_FRAMES + 1, face_tokens, inner_dim)
    # The prepended frame is zero so the reference slot gets no motion.
    torch.testing.assert_close(motion_vec[:, 0], torch.zeros_like(motion_vec[:, 0]))


def test_face_adapter_requires_sequence_divisible_by_frames() -> None:
    adapter = WanAnimateFaceBlockCrossAttention(dim=16, heads=2, dim_head=8)
    hidden_states = torch.randn(1, 7, 16)
    motion_vec = torch.randn(1, 3, 3, 16)

    with pytest.raises(ValueError, match="divide evenly"):
        adapter(hidden_states, motion_vec)


def test_face_adapter_preserves_hidden_state_shape() -> None:
    adapter = WanAnimateFaceBlockCrossAttention(dim=16, heads=2, dim_head=8)
    hidden_states = torch.randn(1, 12, 16)
    motion_vec = torch.randn(1, 3, 3, 16)

    assert adapter(hidden_states, motion_vec).shape == hidden_states.shape


# ---------------------------------------------------------------------------
# Transformer forward and weight naming
# ---------------------------------------------------------------------------


def _tiny_forward_inputs() -> dict:
    latent_channels = TINY_CONFIG["latent_channels"]
    in_channels = 2 * latent_channels + 4
    return {
        "hidden_states": torch.randn(1, in_channels, LATENT_FRAMES + 1, 4, 4, device=_DEVICE),
        "timestep": torch.tensor([500], device=_DEVICE),
        "encoder_hidden_states": torch.randn(1, 6, TINY_CONFIG["text_dim"], device=_DEVICE),
        "pose_hidden_states": torch.randn(1, latent_channels, LATENT_FRAMES, 4, 4, device=_DEVICE),
        "face_pixel_values": torch.randn(1, 3, SEGMENT_FRAMES, 8, 8, device=_DEVICE),
    }


@torch.no_grad()
def test_animate_transformer_forward_returns_latent_shaped_output(tiny_transformer) -> None:
    inputs = _tiny_forward_inputs()

    output = tiny_transformer(**inputs, return_dict=False)[0]

    assert output.shape == (1, TINY_CONFIG["out_channels"], LATENT_FRAMES + 1, 4, 4)
    assert torch.isfinite(output).all()


@torch.no_grad()
def test_animate_transformer_accepts_precomputed_motion_vectors(tiny_transformer) -> None:
    inputs = _tiny_forward_inputs()
    motion_vec = tiny_transformer.encode_face(inputs.pop("face_pixel_values"))

    output = tiny_transformer(**inputs, motion_vec=motion_vec, return_dict=False)[0]

    assert output.shape == (1, TINY_CONFIG["out_channels"], LATENT_FRAMES + 1, 4, 4)


@torch.no_grad()
def test_animate_transformer_pose_skips_the_reference_frame(tiny_transformer) -> None:
    """Pose latents condition every frame except the reference slot.

    Checked at the injection point rather than on the output, because
    self-attention spreads the perturbation across all frames afterwards.
    """
    captured = []
    handle = tiny_transformer.blocks[0].register_forward_pre_hook(lambda _m, args: captured.append(args[0].clone()))
    try:
        inputs = _tiny_forward_inputs()
        tiny_transformer(**inputs, return_dict=False)
        tiny_transformer(**{**inputs, "pose_hidden_states": inputs["pose_hidden_states"] + 10.0}, return_dict=False)
    finally:
        handle.remove()

    baseline, perturbed = captured
    # Post-patchify tokens are frame-major, so the reference frame owns the
    # leading height/patch * width/patch tokens.
    tokens_per_frame = baseline.shape[1] // (LATENT_FRAMES + 1)
    torch.testing.assert_close(baseline[:, :tokens_per_frame], perturbed[:, :tokens_per_frame])
    assert not torch.allclose(baseline[:, tokens_per_frame:], perturbed[:, tokens_per_frame:])


@torch.no_grad()
def test_animate_transformer_rejects_pose_frame_mismatch(tiny_transformer) -> None:
    inputs = _tiny_forward_inputs()
    inputs["pose_hidden_states"] = torch.randn(
        1, TINY_CONFIG["latent_channels"], LATENT_FRAMES + 1, 4, 4, device=_DEVICE
    )

    with pytest.raises(ValueError, match="one fewer"):
        tiny_transformer(**inputs, return_dict=False)


def test_animate_parameter_names_match_diffusers_checkpoint(tiny_transformer) -> None:
    """Names the inherited load_weights relies on to need no Animate-specific remapping."""
    params = dict(tiny_transformer.named_parameters())

    # to_out is a bare Linear here, unlike the backbone's `attn1.to_out.0`.
    assert "face_adapter.0.to_out.weight" in params
    assert "face_adapter.0.to_out.0.weight" not in params
    # QK norms are per-head, so they must not be sharded like the backbone norms.
    assert params["face_adapter.0.norm_q.weight"].shape == (TINY_CONFIG["attention_head_dim"],)
    # Plain Conv1d, not a causal-conv wrapper that would add a `.conv` level.
    assert "face_encoder.conv1_local.weight" in params
    assert "face_encoder.conv1_local.conv.weight" not in params
    assert "face_encoder.padding_tokens" in params
    assert "motion_encoder.motion_synthesis_weight" in params
    assert "motion_encoder.conv_in.act_fn.bias" in params
    assert "motion_encoder.res_blocks.0.conv_skip.weight" in params
    assert "pose_patch_embedding.weight" in params
    # The checkpoint's top-level scale_shift_table is remapped by the base class.
    assert "output_scale_shift_prepare.scale_shift_table" in params


def test_animate_load_weights_accepts_diffusers_key_names(tiny_transformer) -> None:
    params = dict(tiny_transformer.named_parameters())
    weights = [
        ("face_adapter.0.to_out.weight", torch.zeros_like(params["face_adapter.0.to_out.weight"])),
        ("motion_encoder.motion_synthesis_weight", torch.zeros_like(params["motion_encoder.motion_synthesis_weight"])),
        ("face_encoder.conv1_local.weight", torch.zeros_like(params["face_encoder.conv1_local.weight"])),
        ("pose_patch_embedding.weight", torch.zeros_like(params["pose_patch_embedding.weight"])),
        ("scale_shift_table", torch.zeros_like(params["output_scale_shift_prepare.scale_shift_table"])),
    ]

    loaded = tiny_transformer.load_weights(weights)

    # Assert on where the values landed rather than on the names echoed back:
    # the inherited loader renames before it records, so a remapped key such as
    # `scale_shift_table` is only ever reported under its destination name.
    assert "output_scale_shift_prepare.scale_shift_table" in loaded
    assert {name for name, _ in weights if name != "scale_shift_table"} <= loaded

    reloaded = dict(tiny_transformer.named_parameters())
    for name, _ in weights:
        target = "output_scale_shift_prepare.scale_shift_table" if name == "scale_shift_table" else name
        assert torch.count_nonzero(reloaded[target]) == 0, f"{name} was not written to {target}"
