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
from transformers.processing_utils import ProcessorMixin


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


class MageVLProcessor(ProcessorMixin):
    """Image+text processor for Mage-VL.

    The checkpoint ships its own ``MageVLProcessor`` that deliberately does **not**
    inherit :class:`ProcessorMixin`, which vLLM's processor cache requires. This is a
    compliant re-implementation of the image path: it wraps the checkpoint's
    ``Qwen2VLImageProcessor`` (patch 16 / merge 2 / temporal patch 1), expands each
    ``<|image_pad|>`` placeholder to the merged token count, and emits the per-patch
    position table the vision tower needs.

    Video is intentionally out of scope here: the reference pipeline routes frames
    through the image tensors after its own sampling/codec stage, so callers hand this
    processor a list of frames as images.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.spatial_merge_size = int(getattr(image_processor, "merge_size", 2))
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        from transformers import BatchFeature

        if text is None:
            raise ValueError("`text` is required")
        if isinstance(text, str):
            text = [text]
        text = list(text)

        out: dict = {}
        if images is not None:
            image_outputs = self.image_processor(images=images, return_tensors="pt")
            grid_thw = image_outputs["image_grid_thw"]
            merge_factor = self.spatial_merge_size**2
            token_counts = (grid_thw.prod(-1) // merge_factor).tolist()

            # Expand one placeholder per image, in prompt order. The two-step swap via a
            # sentinel keeps freshly written placeholders from being re-matched.
            idx = 0

            def expand(s: str) -> str:
                nonlocal idx
                while self.image_token in s and idx < len(token_counts):
                    s = s.replace(
                        self.image_token, "<|placeholder|>" * int(token_counts[idx]), 1
                    )
                    idx += 1
                return s.replace("<|placeholder|>", self.image_token)

            text = [expand(s) for s in text]

            out["pixel_values"] = image_outputs["pixel_values"]
            out["image_grid_thw"] = grid_thw
            out["patch_positions"] = build_patch_positions(
                grid_thw, spatial_merge_size=self.spatial_merge_size
            )

        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        return BatchFeature(data={**text_inputs, **out}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                [*self.tokenizer.model_input_names, "pixel_values", "image_grid_thw",
                 "patch_positions"]
            )
        )


__all__ = [
    "MageVLProcessor",
    "build_patch_positions",
    "convert_positions_to_block_layout",
]
