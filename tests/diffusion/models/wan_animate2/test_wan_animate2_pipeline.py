# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan2.2-Animate-2 pipeline-level contracts (M1).

Covers the segment arithmetic, conditioning geometry, checkpoint-variant
detection and registration that the end-to-end run depends on, without needing
real weights.
"""

import json
from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import torch

from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.wan_animate2.pipeline_wan_animate2 import (
    ANIMATE2_DEFAULT_MAX_DRIVING_FRAMES,
    Animate2Request,
    LetterboxInfo,
    Wan22Animate2Pipeline,
    _is_distilled_checkpoint,
    _load_scheduler,
    get_frame_indices,
    get_i2v_mask,
    get_padding_len,
    resample_frames,
    resize_by_area,
    zigzag_padding,
)
from vllm_omni.diffusion.models.wan_animate2.wan_animate2_transformer import WanAnimate2TransformerConfig
from vllm_omni.diffusion.registry import (
    _DIFFUSION_MODELS,
    _DIFFUSION_POST_PROCESS_FUNCS,
    _DIFFUSION_PRE_PROCESS_FUNCS,
    _NO_CACHE_ACCELERATION,
)
from vllm_omni.model_extras import get_video_generation_defaults, should_preserve_reference_image_size

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _segment_starts(total_frames: int, segment_frames: int, overlap: int) -> list[tuple[int, int]]:
    """Replay the pipeline's segment walk, returning ``(start, clip_len)`` pairs."""
    starts = []
    start = 0
    while start + overlap < total_frames:
        clip_len = min(segment_frames, total_frames - start)
        starts.append((start, clip_len))
        start += clip_len - overlap
    return starts


# ---------------------------------------------------------------------------
# Segment arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("real_len", [1, 30, 81, 82, 120, 161, 200, 401])
def test_padding_length_leaves_a_usable_trailing_segment(real_len):
    """A1.6 precursor: after padding, no segment is degenerately short and the
    padded length never drops below the real one."""
    segment_frames, overlap = 81, 1
    padded = get_padding_len(real_len, segment_frames, overlap)
    assert padded >= real_len

    segments = _segment_starts(padded, segment_frames, overlap)
    assert segments, "at least one segment must be produced"
    for _, clip_len in segments:
        assert clip_len > overlap
        assert clip_len >= 28

    produced = sum(clip_len - (overlap if idx else 0) for idx, (_, clip_len) in enumerate(segments))
    assert produced >= real_len, "the segment walk must cover the whole driving video"


def test_segment_walk_covers_frames_without_gaps():
    padded = get_padding_len(200, 81, 1)
    segments = _segment_starts(padded, 81, 1)
    for (prev_start, prev_len), (next_start, _) in zip(segments, segments[1:]):
        assert next_start == prev_start + prev_len - 1, "segments must overlap by exactly one frame"


def test_zigzag_padding_bounces_instead_of_clamping():
    frames = [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(5)]
    padded = zigzag_padding(frames, 9)
    assert [int(frame[0, 0, 0]) for frame in padded] == [0, 1, 2, 3, 4, 3, 2, 1, 0]
    assert len(zigzag_padding(frames[:3], 3)) == 3
    assert len(zigzag_padding(frames[:1], 4)) == 4
    with pytest.raises(ValueError, match="empty frame sequence"):
        zigzag_padding([], 4)


def test_frame_indices_resample_to_target_fps():
    # 30 fps source, 24 fps target: 5 source frames per 4 output frames.
    indices = get_frame_indices(frame_count=60, video_fps=30.0, target_count=48, target_fps=24.0)
    assert len(indices) == 48
    assert indices[0] == 0
    assert indices == sorted(indices)
    assert max(indices) <= 59


def test_resample_frames_keeps_duration():
    frames = [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(60)]
    resampled = resample_frames(frames, video_fps=30.0, target_fps=24.0)
    assert len(resampled) == 48
    with pytest.raises(ValueError, match="too short"):
        resample_frames(frames[:1], video_fps=60.0, target_fps=24.0)


def test_online_single_path_list_is_decoded_as_driving_video(monkeypatch, tmp_path):
    """The multipart video API persists uploads and passes ``list[str]``."""
    video_path = tmp_path / "driving.mp4"
    source_frames = [np.full((16, 16, 3), value, dtype=np.uint8) for value in range(60)]
    decoded_paths = []

    def _decode(path):
        decoded_paths.append(path)
        return source_frames, 30.0

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan_animate2.pipeline_wan_animate2.decode_video_file",
        _decode,
    )
    request = Animate2Request(
        prompt="static background.",
        negative_prompt="",
        prompt_ref="人物动作的参考视频",
        image=PIL.Image.new("RGB", (16, 16)),
        video=[str(video_path)],
        width=16,
        height=16,
        fps=24,
        max_frames=3,
        segment_frames=81,
        num_inference_steps=10,
        guidance_scale=1.0,
    )

    pipeline = SimpleNamespace(resolution_divisor=16)
    frames = Wan22Animate2Pipeline._load_driving_frames(pipeline, request)

    assert decoded_paths == [str(video_path)]
    assert len(frames) == 3
    assert all(frame.shape == (16, 16, 3) for frame in frames)


def test_multiple_driving_video_paths_are_rejected(tmp_path):
    request = SimpleNamespace(
        video=[str(tmp_path / "first.mp4"), str(tmp_path / "second.mp4")],
        fps=24,
        max_frames=81,
        width=16,
        height=16,
    )

    with pytest.raises(ValueError, match="exactly one driving video"):
        Wan22Animate2Pipeline._load_driving_frames(SimpleNamespace(resolution_divisor=16), request)


# ---------------------------------------------------------------------------
# Conditioning geometry
# ---------------------------------------------------------------------------


def test_letterbox_preserves_aspect_and_records_crop():
    # 3:5 does not divide into a 16-aligned 720x1280-area box, so bars are needed.
    source = PIL.Image.new("RGB", (600, 1000), (255, 255, 255))  # fully white: any black is padding
    canvas, info = resize_by_area(source, 720 * 1280, divisor=16)

    height, width = canvas.shape[:2]
    assert height % 16 == 0 and width % 16 == 0
    assert height * width <= 720 * 1280
    assert info.pad_axis in {"width", "height"}

    video = torch.from_numpy(canvas).permute(2, 0, 1)[None, :, None]  # [1, 3, 1, H, W]
    cropped = info.crop(video)
    assert (cropped > 0).all()
    assert cropped.shape[-1] * cropped.shape[-2] < height * width


def test_letterbox_crop_axes():
    video = torch.arange(2 * 6).view(1, 1, 1, 2, 6)
    assert torch.equal(LetterboxInfo("width", 1, 3).crop(video), video[..., 1:4])
    assert torch.equal(LetterboxInfo("height", 1, 1).crop(video), video[..., 1:2, :])


def test_i2v_mask_shape_and_conditioned_extent():
    lat_t, lat_h, lat_w = 21, 10, 6
    mask = get_i2v_mask(lat_t, lat_h, lat_w, mask_len=1, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (4, lat_t, lat_h, lat_w)
    # Only the first pixel frame is conditioned, which lands in latent frame 0.
    assert mask[:, 0].sum() > 0
    assert torch.equal(mask[:, 1:], torch.zeros_like(mask[:, 1:]))

    full = get_i2v_mask(lat_t, lat_h, lat_w, mask_len=81, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(full, torch.ones_like(full))


def test_condition_channel_count_matches_transformer_input():
    """The transformer takes 36 channels: 16 noise + 4 mask + 16 conditioning."""
    lat_t, lat_h, lat_w = 22, 10, 6
    mask = get_i2v_mask(lat_t, lat_h, lat_w, 1, torch.device("cpu"), torch.float32).unsqueeze(0)
    condition = torch.cat([mask, torch.zeros(1, 16, lat_t, lat_h, lat_w)], dim=1)
    assert condition.shape[1] + 16 == WanAnimate2TransformerConfig().in_dim


def test_latent_frame_count_leaves_room_for_the_reference_slot():
    """``lat_t = (clip_len + 1) // 4 + 2``: one slot more than the driving
    video's latents, which is where the reference image sits."""
    for clip_len in (81, 61, 45, 29):
        assert (clip_len + 1) // 4 + 2 == clip_len // 4 + 1 + 1


# ---------------------------------------------------------------------------
# Checkpoint variants
# ---------------------------------------------------------------------------


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_distilled_release_is_detected_from_the_modular_index(tmp_path):
    _write_json(tmp_path / "modular_model_index.json", {"_blocks_class_name": "WanAnimate2DistilledBlocks"})
    assert _is_distilled_checkpoint(str(tmp_path), local_files_only=True)

    _write_json(tmp_path / "modular_model_index.json", {"_blocks_class_name": "WanAnimate2Blocks"})
    assert not _is_distilled_checkpoint(str(tmp_path), local_files_only=True)

    (tmp_path / "modular_model_index.json").unlink()
    assert not _is_distilled_checkpoint(str(tmp_path), local_files_only=True)


def test_scheduler_follows_the_checkpoint_config(tmp_path):
    """A1.2 precursor: base ships DPM-Solver++ over flow sigmas, distilled a
    shifted flow-match Euler; the request's flow_shift overrides both."""
    from diffusers import DPMSolverMultistepScheduler

    from vllm_omni.diffusion.models.schedulers import FlowMatchEulerDiscreteScheduler

    base = tmp_path / "base"
    _write_json(
        base / "scheduler" / "scheduler_config.json",
        {
            "_class_name": "DPMSolverMultistepScheduler",
            "algorithm_type": "dpmsolver++",
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
            "flow_shift": 5.0,
            "final_sigmas_type": "zero",
            "solver_order": 2,
        },
    )
    scheduler = _load_scheduler(str(base), local_files_only=True, flow_shift=7.0)
    assert isinstance(scheduler, DPMSolverMultistepScheduler)
    assert scheduler.config.flow_shift == 7.0
    assert scheduler.config.use_flow_sigmas

    distilled = tmp_path / "distilled"
    _write_json(
        distilled / "scheduler" / "scheduler_config.json",
        {"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 5.0, "use_dynamic_shifting": False},
    )
    scheduler = _load_scheduler(str(distilled), local_files_only=True, flow_shift=5.0)
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert scheduler.config.shift == 5.0

    other = tmp_path / "other"
    _write_json(other / "scheduler" / "scheduler_config.json", {"_class_name": "UniPCMultistepScheduler"})
    with pytest.raises(ValueError, match="unsupported"):
        _load_scheduler(str(other), local_files_only=True, flow_shift=5.0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registry_entries_resolve():
    arch = "WanAnimate2Pipeline"
    assert _DIFFUSION_MODELS[arch] == ("wan_animate2", "pipeline_wan_animate2", "Wan22Animate2Pipeline")
    assert _DIFFUSION_PRE_PROCESS_FUNCS[arch] == "get_wan_animate2_pre_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS[arch] == "get_wan_animate2_post_process_func"
    assert arch in _NO_CACHE_ACCELERATION


def test_model_metadata_allows_image_plus_video_references():
    metadata = get_diffusion_model_metadata("WanAnimate2Pipeline")
    assert metadata.supports_multimodal_inputs
    assert metadata.supports_mixed_reference_inputs
    assert metadata.max_multimodal_image_inputs == 1
    assert metadata.final_output_type == "video"


def test_pipeline_declares_offload_components():
    assert Wan22Animate2Pipeline._dit_modules == ["transformer"]
    assert Wan22Animate2Pipeline._encoder_modules == ["text_encoder", "image_encoder"]
    assert Wan22Animate2Pipeline._vae_modules == ["vae"]
    # Upstream is effectively batch-1; batching would need one K/V cache per request.
    assert Wan22Animate2Pipeline.supports_request_batch is False


def test_reference_video_decode_spec_caps_request_length():
    """A5.3 precursor: a driving video must not be able to expand a request
    without bound, since segment count scales with its duration."""
    default = Wan22Animate2Pipeline.reference_video_decode_spec()
    assert default.max_frames == ANIMATE2_DEFAULT_MAX_DRIVING_FRAMES
    assert default.keep == "first"

    raised = Wan22Animate2Pipeline.reference_video_decode_spec(extra_args={"max_driving_frames": 96})
    assert raised.max_frames == 96

    # An explicit num_frames only ever tightens the cap.
    assert Wan22Animate2Pipeline.reference_video_decode_spec(num_frames=48).max_frames == 48
    assert (
        Wan22Animate2Pipeline.reference_video_decode_spec(num_frames=10**6).max_frames
        == ANIMATE2_DEFAULT_MAX_DRIVING_FRAMES
    )


def test_shared_example_hooks():
    """The shared I2V example must not pre-resize the reference image (the
    pipeline letterboxes it) and takes the official demo defaults."""
    assert should_preserve_reference_image_size("WanAnimate2Pipeline", model=None)
    defaults = get_video_generation_defaults("WanAnimate2Pipeline")
    assert (defaults.width, defaults.height) == (720, 1280)
    assert defaults.num_inference_steps == 40
    assert defaults.guidance_scale == 3.0
    assert defaults.fps == 24
    assert defaults.default_negative_prompt is None
