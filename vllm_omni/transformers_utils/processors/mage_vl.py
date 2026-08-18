# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch-position construction for Mage-VL.

Mage-ViT's 3D RoPE is driven by an explicit per-patch ``(t, h, w)`` table rather than
being derived from ``grid_thw``: under codec-native sampling the retained patches are
sparse in both time and space, and ``t`` is the *source frame index*. This module builds
that table for the dense (frame-sampled / image) case, matching the reference
implementation's ``video_processing_mage_vl.build_patch_positions``.

The table is emitted in the same 2x2 block order the Qwen2-VL image processor uses for
``pixel_values``, so row *i* of the positions lines up with patch *i* of the pixels.
"""

import torch
from transformers.models.qwen2_vl import Qwen2VLProcessor


def convert_positions_to_block_layout(
    positions: torch.Tensor,
    t: int,
    h: int,
    w: int,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Reorder a row-major ``[t*h*w, 3]`` position table into 2x2 block order."""
    sms = int(spatial_merge_size)
    if sms == 1:
        return positions
    total = int(t) * int(h) * int(w)
    indices = torch.arange(total, device=positions.device).view(t, h, w)
    h_m, w_m = int(h) // sms, int(w) // sms
    indices = (
        indices.view(t, h_m, sms, w_m, sms).permute(0, 1, 3, 2, 4).contiguous().view(total)
    )
    return positions[indices]


def build_patch_positions(
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    frame_indices: list[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Build the block-layout ``[sum(t*h*w), 3]`` (t, h, w) table.

    Args:
        grid_thw: ``[num_samples, 3]`` of per-sample (T, H_patches, W_patches).
        spatial_merge_size: vision tower spatial merge factor.
        frame_indices: optional per-sample real frame indices to use as the temporal
            coordinate. The reference pipeline passes the source video's frame numbers so
            the 3D RoPE encodes actual temporal position; ``None`` falls back to a dense
            ``arange(T)``, which is what a still image or uniformly sampled clip needs.
    """
    out = []
    for sample_idx, row in enumerate(grid_thw):
        t_v, h_v, w_v = int(row[0]), int(row[1]), int(row[2])
        h_coords = torch.arange(h_v, dtype=torch.int64).repeat_interleave(w_v).repeat(t_v)
        w_coords = torch.arange(w_v, dtype=torch.int64).repeat(h_v).repeat(t_v)

        sample_frame_idx = None
        if frame_indices is not None and sample_idx < len(frame_indices):
            sample_frame_idx = frame_indices[sample_idx]
        if sample_frame_idx is not None:
            fi = torch.as_tensor(sample_frame_idx, dtype=torch.int64)
            if fi.numel() != t_v:
                raise ValueError(
                    f"frame_indices[{sample_idx}] has length {fi.numel()} but "
                    f"grid_thw[{sample_idx}, 0] = {t_v}"
                )
            t_coords = fi.repeat_interleave(h_v * w_v)
        else:
            t_coords = torch.arange(t_v, dtype=torch.int64).repeat_interleave(h_v * w_v)

        positions = torch.stack([t_coords, h_coords, w_coords], dim=1)
        out.append(convert_positions_to_block_layout(positions, t_v, h_v, w_v, spatial_merge_size))
    return torch.cat(out, dim=0)


class MageVLProcessor(Qwen2VLProcessor):
    """Image+text processor for Mage-VL.

    The checkpoint ships its own ``MageVLProcessor`` that deliberately does **not**
    inherit :class:`~transformers.ProcessorMixin`, which vLLM's processor cache requires.
    Mage-VL's image path is Qwen2-VL's -- the checkpoint's own preprocessor config *is* a
    ``Qwen2VLImageProcessor`` (patch 16 / merge 2 / temporal patch 1) -- so this subclasses
    the upstream processor and adds only what differs:

    1. ``__init__`` drops ``video_processor``. This is load-bearing, not cosmetic:
       transformers derives the attribute list from the *subclass* ``__init__`` signature
       (``ProcessorMixin.get_attributes``), so dropping the parameter is what keeps
       ``from_pretrained`` from demanding the ``video_preprocessor_config.json`` this
       checkpoint does not ship. Inheriting the signature verbatim fails both ways:
       ``from_pretrained`` raises ``OSError`` on the missing config, and passing ``None``
       raises ``TypeError`` ("a BaseVideoProcessor was expected").
    2. ``patch_positions`` is appended to the outputs. Mage-ViT's 3D RoPE is driven by an
       explicit per-patch table rather than by ``grid_thw`` (see the module docstring).

    Video is intentionally out of scope: the reference pipeline routes frames through the
    image tensors after its own sampling/codec stage, so callers hand this processor a
    list of frames as images.
    """

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @property
    def spatial_merge_size(self) -> int:
        return int(self.image_processor.merge_size)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        video_backend: str = "frames",
        codec_config=None,
        **kwargs,
    ):
        """Image / frame-sampled path by default; ``video_backend="codec"`` for canvases.

        Frame sampling keeps video in the image tensors (one block per frame), which is
        what the reference serves online, so ``videos=`` is not a separate tensor path
        here. Codec sampling is different in kind -- it tiles patches from many frames
        onto shared canvases and rewrites the prompt with per-frame timestamps -- so it
        gets its own branch.
        """
        if video_backend == "codec":
            return self._call_codec(text=text, videos=videos, codec_config=codec_config)
        if video_backend != "frames":
            raise ValueError(
                f"unknown video_backend {video_backend!r} (use 'frames' or 'codec')"
            )
        if videos is not None:
            raise ValueError(
                "Mage-VL has no video processor: with video_backend='frames' the caller "
                "samples frames upstream and passes them as `images`."
            )

        outputs = super().__call__(images=images, text=text, **kwargs)
        if "image_grid_thw" in outputs:
            outputs["patch_positions"] = build_patch_positions(
                torch.as_tensor(outputs["image_grid_thw"]),
                spatial_merge_size=self.spatial_merge_size,
            )
        return outputs

    def _call_codec(self, text, videos, codec_config):
        """Build model inputs from codec canvases.

        The canvases carry patches from several source frames each, so unlike the frame
        path the prompt cannot keep a single placeholder: it is rewritten into one
        timestamped ``<|image_pad|>`` run per source frame.
        """
        from transformers import BatchFeature

        from vllm_omni.transformers_utils.processors.mage_vl_codec import (
            CodecConfig,
            codec_image_processor_outputs,
            codec_positions_for_processor,
            drop_padding_canvases,
            load_codec_result,
            process_codec_video,
            rewrite_text_with_codec_positions,
        )

        if text is None:
            raise ValueError("`text` is required")
        prompts = [text] if isinstance(text, str) else list(text)
        if len(prompts) != 1:
            raise ValueError("the codec path handles one prompt at a time")

        settings = dict(codec_config or {})
        asset_dir = settings.pop("asset_dir", None)
        cfg = CodecConfig(**settings)

        # An asset directory is a codec run someone already did; without one we would
        # have to run an engine, and process_codec_video says which tool is missing.
        result = load_codec_result(asset_dir) if asset_dir else process_codec_video(videos, cfg)

        images, src_positions, _ = drop_padding_canvases(result["images"], result["src_positions"])
        if not images:
            raise ValueError("every codec canvas was padding; nothing to encode")

        image_outputs = codec_image_processor_outputs(
            self.image_processor, images, max_pixels=cfg.max_pixels
        )
        patch_positions = codec_positions_for_processor(
            src_positions, image_outputs["image_grid_thw"]
        )
        prompt = rewrite_text_with_codec_positions(
            prompts[0], patch_positions, fps=result["fps"], decimals=result["decimals"]
        )

        text_inputs = self.tokenizer([prompt], return_tensors="pt")
        return BatchFeature(
            data={
                **text_inputs,
                "pixel_values": image_outputs["pixel_values"],
                "image_grid_thw": image_outputs["image_grid_thw"],
                "patch_positions": patch_positions,
            },
            tensor_type="pt",
        )

    @property
    def model_input_names(self):
        return list(dict.fromkeys([*super().model_input_names, "patch_positions"]))


__all__ = [
    "MageVLProcessor",
    "build_patch_positions",
    "convert_positions_to_block_layout",
]
