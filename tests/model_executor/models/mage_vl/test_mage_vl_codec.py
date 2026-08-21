# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for Mage-VL codec-native video preprocessing.

Every expected value here was cross-checked elementwise against the checkpoint's own
``codec_video_processing_mage_vl.py`` over randomized inputs before being pinned.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.transformers_utils.processors.mage_vl_codec import (
    CodecConfig,
    codec_image_processor_outputs,
    codec_positions_for_processor,
    drop_padding_canvases,
    load_codec_result,
    process_codec_video,
    rewrite_text_with_codec_positions,
)


@pytest.mark.core_model
@pytest.mark.cpu
def test_codec_positions_are_reordered_per_canvas():
    """Each canvas is reordered on its own, so row i lines up with pixel patch i."""
    # Two canvases of 2x2 patches; distinct t per canvas makes reordering visible.
    src = np.array(
        [[7, 0, 0], [7, 0, 1], [7, 1, 0], [7, 1, 1], [9, 0, 0], [9, 0, 1], [9, 1, 0], [9, 1, 1]],
        dtype=np.int64,
    )
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])

    out = codec_positions_for_processor(src, grid)

    assert out.shape == (8, 3)
    assert out.dtype == torch.int64
    # A single 2x2 block is already in block order, so values survive intact...
    assert out[:4, 0].tolist() == [7, 7, 7, 7]
    assert out[4:, 0].tolist() == [9, 9, 9, 9]
    # ...and every (h, w) coordinate is still present exactly once per canvas.
    assert {tuple(p) for p in out[:4, 1:].tolist()} == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.core_model
@pytest.mark.cpu
def test_codec_positions_reject_a_table_that_does_not_match_the_grid():
    src = np.zeros((7, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="position length mismatch"):
        codec_positions_for_processor(src, torch.tensor([[1, 2, 2]]))


@pytest.mark.core_model
@pytest.mark.cpu
def test_padding_canvases_are_dropped_with_their_patches():
    """The codec pads up to ``target_canvas``; padded canvases carry t = -1 throughout."""
    images = [Image.new("RGB", (8, 8)) for _ in range(3)]
    src = np.array(
        [[3, 0, 0], [3, 0, 1], [5, 0, 0], [5, 0, 1], [-1, 0, 0], [-1, 0, 1]],
        dtype=np.int64,
    )

    kept_images, kept_positions, dropped = drop_padding_canvases(images, src)

    assert dropped == 1
    assert len(kept_images) == 2
    assert kept_positions.shape == (4, 3)
    assert (kept_positions[:, 0] >= 0).all()


@pytest.mark.core_model
@pytest.mark.cpu
def test_half_padding_canvas_is_an_error_not_a_silent_trim():
    """Padding is canvas-granular by construction; anything else means the asset is wrong."""
    images = [Image.new("RGB", (8, 8))]
    src = np.array([[4, 0, 0], [-1, 0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="half-padding canvas"):
        drop_padding_canvases(images, src)


@pytest.mark.core_model
@pytest.mark.cpu
def test_prompt_rewrite_emits_one_timestamped_run_per_source_frame():
    """Codec canvases mix frames, so the placeholder becomes per-frame image-pad runs."""
    text = (
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n"
        "What happens?<|im_end|>\n"
    )
    # 8 patches at frame 25, then 4 at frame 50 -> 2 and 1 tokens after the 2x2 merge.
    positions = torch.tensor([[25, 0, 0]] * 8 + [[50, 0, 0]] * 4, dtype=torch.int64)

    out = rewrite_text_with_codec_positions(text, positions, fps=25.0, decimals=1)

    assert "<1.0 seconds><|vision_start|><|image_pad|><|image_pad|><|vision_end|>" in out
    assert "<2.0 seconds><|vision_start|><|image_pad|><|vision_end|>" in out
    assert "<|video_pad|>" not in out
    assert out.count("<|image_pad|>") == 3
    # The surrounding prompt is untouched.
    assert out.startswith("<|im_start|>user\n") and "What happens?<|im_end|>" in out


@pytest.mark.core_model
@pytest.mark.cpu
def test_prompt_rewrite_skips_padding_patches_and_passes_plain_text_through():
    positions = torch.tensor([[-1, 0, 0]] * 4 + [[10, 0, 0]] * 4, dtype=torch.int64)
    text = "<|vision_start|><|video_pad|><|vision_end|>"

    out = rewrite_text_with_codec_positions(text, positions, fps=10.0, decimals=2)

    assert out.count("<|image_pad|>") == 1
    assert "<1.00 seconds>" in out
    # A prompt with no video placeholder is not this function's business.
    assert rewrite_text_with_codec_positions("no video here", positions, 10.0) == "no video here"


@pytest.mark.core_model
@pytest.mark.cpu
def test_canvas_pixel_budget_is_clamped_so_the_processor_cannot_resize():
    """A canvas below the processor's default ``min_pixels`` must not be upscaled.

    224x224 and 160x160 at patch 16 are 14x14 and 10x10 patch grids. Without the clamp
    the processor's default ``min_pixels=200704`` would grow the 160x160 canvas, and the
    extra patches would no longer line up with the codec's position table.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    image_processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, temporal_patch_size=1, min_pixels=3136, max_pixels=4_000_000
    )
    canvases = [Image.new("RGB", (224, 224)), Image.new("RGB", (160, 160))]

    out = codec_image_processor_outputs(image_processor, canvases, max_pixels=150_000)

    assert out["image_grid_thw"].tolist() == [[1, 14, 14], [1, 10, 10]]


@pytest.mark.core_model
@pytest.mark.cpu
def test_codec_config_matches_the_checkpoint_budget():
    """Defaults mirror ``preprocessor_config.json``'s ``codec`` block."""
    cfg = CodecConfig(dcvc={"qp": 21, "per_frame_cap_ratio": 1.2})

    # 32 canvases / 4 per group x 32 frames per group.
    assert cfg.num_sampled_frames() == 256
    assert cfg.dcvc.qp == 21  # nested dict is coerced...
    assert not hasattr(cfg.dcvc, "per_frame_cap_ratio")  # ...and engine-only knobs dropped
    cfg.validate()

    with pytest.raises(ValueError, match="divisible by images_per_group"):
        CodecConfig(target_canvas=30, images_per_group=4).validate()


@pytest.mark.core_model
@pytest.mark.cpu
def test_missing_codec_tooling_fails_with_an_actionable_message(monkeypatch, tmp_path):
    """Say which tool is missing and what to do instead, rather than failing downstream."""
    # hevc drives an external binary; without it the message has to name the package.
    monkeypatch.setenv("CV_PREINFER_BIN", str(tmp_path / "definitely-not-installed"))
    with pytest.raises(RuntimeError, match="codec-video-prep") as hevc:
        process_codec_video("video.mp4", CodecConfig(engine="hevc", cache_root=tmp_path))
    assert "asset_dir" in str(hevc.value)

    # dcvc-rt has no driver here at all.
    with pytest.raises(NotImplementedError, match="neural_codec") as dcvc:
        process_codec_video("video.mp4", CodecConfig(engine="dcvc-rt", cache_root=tmp_path))
    assert "asset_dir" in str(dcvc.value)

    with pytest.raises(ValueError, match="unknown codec engine"):
        process_codec_video("video.mp4", CodecConfig(engine="jpeg2000", cache_root=tmp_path))


@pytest.mark.core_model
@pytest.mark.cpu
def test_codec_asset_directory_round_trips(tmp_path):
    """A directory the reference pipeline produced loads back into canvases + positions."""
    positions = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=np.int64)
    np.save(tmp_path / "src_patch_position.npy", positions)
    (tmp_path / "meta.json").write_text('{"fps": 25.0}')
    Image.new("RGB", (32, 32)).save(tmp_path / "canvas_00000.jpg")

    result = load_codec_result(tmp_path)

    assert len(result["images"]) == 1
    assert result["fps"] == 25.0
    assert np.array_equal(result["src_positions"], positions)

    with pytest.raises(FileNotFoundError, match="not a codec asset directory"):
        load_codec_result(tmp_path / "nope")


def _codec_asset_dir(tmp_path, *, canvas_px: int = 32, frames=(4, 9), pad_canvases: int = 1):
    """A synthetic codec asset dir: one canvas per entry in ``frames``, then padding."""
    patch, merge = 16, 2
    side = canvas_px // patch  # patches per canvas side
    per_canvas = side * side

    rows = []
    for frame in frames:
        for h in range(side):
            for w in range(side):
                rows.append([frame, h, w])
    for _ in range(pad_canvases):
        rows.extend([[-1, 0, 0]] * per_canvas)

    np.save(tmp_path / "src_patch_position.npy", np.array(rows, dtype=np.int64))
    (tmp_path / "meta.json").write_text('{"fps": 25.0}')
    for index in range(len(frames) + pad_canvases):
        Image.new("RGB", (canvas_px, canvas_px)).save(tmp_path / f"canvas_{index:05d}.jpg")
    return per_canvas // (merge**2)


class _StubTokenizer:
    """Enough tokenizer to assemble a BatchFeature; the real one lives on the GPU box."""

    model_input_names = ["input_ids"]
    chat_template = None

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, text, return_tensors=None, **kwargs):
        self.seen.extend(text)
        return {"input_ids": torch.zeros((len(text), 8), dtype=torch.int64)}


def _processor_over_codec_assets():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    from vllm_omni.transformers_utils.processors.mage_vl import MageVLProcessor

    # Built field by field: ProcessorMixin's constructor validates sub-processor classes,
    # and a real tokenizer would drag a checkpoint download into a CPU test.
    processor = MageVLProcessor.__new__(MageVLProcessor)
    processor.image_processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, temporal_patch_size=1, min_pixels=3136, max_pixels=4_000_000
    )
    processor.tokenizer = _StubTokenizer()
    return processor


@pytest.mark.core_model
@pytest.mark.cpu
def test_processor_codec_backend_builds_inputs_from_an_asset_dir(tmp_path):
    """The codec branch must emit the same keys as the image path, plus a rewritten prompt."""
    tokens_per_canvas = _codec_asset_dir(tmp_path, frames=(4, 9), pad_canvases=1)
    processor = _processor_over_codec_assets()

    out = processor(
        text="<|vision_start|><|video_pad|><|vision_end|>\nWhat happens?",
        video_backend="codec",
        codec_config={"asset_dir": str(tmp_path)},
    )

    assert set(out.keys()) >= {"input_ids", "pixel_values", "image_grid_thw", "patch_positions"}
    # The padding canvas is gone: two canvases in, two grids out.
    assert out["image_grid_thw"].shape[0] == 2
    assert out["patch_positions"].shape[0] == out["pixel_values"].shape[0]

    prompt = processor.tokenizer.seen[0]
    assert "<|video_pad|>" not in prompt
    # Frames 4 and 9 at 25 fps, one timestamped run each.
    assert "<0.2 seconds>" in prompt and "<0.4 seconds>" in prompt
    assert prompt.count("<|image_pad|>") == 2 * tokens_per_canvas
    assert prompt.endswith("What happens?")


@pytest.mark.core_model
@pytest.mark.cpu
def test_processor_rejects_unknown_backends_and_video_tensors():
    processor = _processor_over_codec_assets()

    with pytest.raises(ValueError, match="unknown video_backend"):
        processor(text="hi", video_backend="mpeg")

    with pytest.raises(ValueError, match="samples frames upstream"):
        processor(text="hi", videos=["clip.mp4"])


@pytest.mark.core_model
@pytest.mark.cpu
def test_codec_patch_size_is_taken_from_the_image_processor(tmp_path):
    """The codec grid and the image processor grid must be the same grid.

    The checkpoint ships ``codec.patch = 14`` while its image processor is ``patch_size =
    16``. Measured on a real video, that combination tiles 504x280 canvases as 36x20 = 720
    patches while the processor reads 527 -- ``codec_positions_for_processor`` then raises
    a length mismatch. Deriving the value keeps the two grids identical by construction.
    """
    _codec_asset_dir(tmp_path, frames=(4,), pad_canvases=0)
    processor = _processor_over_codec_assets()

    with pytest.raises(ValueError, match="patch=14 but the image processor"):
        processor(
            text="<|vision_start|><|video_pad|><|vision_end|>",
            video_backend="codec",
            codec_config={"asset_dir": str(tmp_path), "patch": 14},
        )

    # Left unset it follows the processor, and the two grids agree.
    out = processor(
        text="<|vision_start|><|video_pad|><|vision_end|>",
        video_backend="codec",
        codec_config={"asset_dir": str(tmp_path)},
    )
    assert out["patch_positions"].shape[0] == out["pixel_values"].shape[0]


def _codec_engine_available() -> bool:
    import shutil

    binary = os.environ.get("CV_PREINFER_BIN", "cv-preinfer")
    return (shutil.which(binary) is not None or os.path.isfile(binary)) and shutil.which(
        "ffprobe"
    ) is not None


@pytest.mark.advanced_model
@pytest.mark.skipif(
    not _codec_engine_available(),
    reason="needs cv-preinfer (pip install codec-video-prep) and ffprobe on PATH",
)
@pytest.mark.skipif(
    not os.environ.get("MAGE_VIDEO"), reason="set MAGE_VIDEO to a video file to run the engine"
)
def test_codec_engine_produces_a_consistent_asset_directory(tmp_path):
    """Run the real engine and check the asset it writes is self-consistent.

    Guarded on the tooling rather than skipped silently in code: `engine="hevc"` shells out
    to `cv-preinfer`, which ships on PyPI as `codec-video-prep` and pins `numpy<2.0`, so it
    normally lives in its own virtualenv reached through `CV_PREINFER_BIN`.
    """
    cfg = CodecConfig(patch=16, cache_root=tmp_path)

    result = process_codec_video(os.environ["MAGE_VIDEO"], cfg)

    canvases = result["images"]
    positions = result["src_positions"]
    assert canvases, "engine produced no canvases"
    assert len(canvases) <= cfg.target_canvas
    assert positions.shape[1] == 3
    assert positions.shape[0] % len(canvases) == 0
    assert result["fps"] > 0

    # The canvases tile exactly on the configured patch grid -- the property that lets the
    # image processor read them back without resizing.
    per_canvas = positions.shape[0] // len(canvases)
    width, height = canvases[0].size
    assert (width % cfg.patch, height % cfg.patch) == (0, 0)
    assert (width // cfg.patch) * (height // cfg.patch) == per_canvas

    # A second call must hit the cache rather than re-running the engine.
    cached = process_codec_video(os.environ["MAGE_VIDEO"], cfg)
    assert np.array_equal(cached["src_positions"], positions)
