# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from vllm_omni.diffusion.models.minimax_h3.world import (
    action_attention,
    action_script,
    action_sentence,
    append_action_embeddings,
    bind_action_spans,
    build_action_block_mask,
    directed_visibility,
)
from vllm_omni.diffusion.models.minimax_h3.world_lora import (
    load_h3_world_lora,
    validate_world_checkpoint,
)
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "keys, expected",
    [
        ([], "the man stands still, camera holds steady"),
        (["w"], "the man walks forward, camera follows him"),
        (["S", "A"], "the man walks backward and strafes left, camera follows him"),
        (["W", "D", "J", "F", "K"], "the man walks forward and strafes right, camera pans left sharply and tilts up"),
        (["I"], "the man stands still, camera tilts down"),
        (["L"], "the man stands still, camera pans right slowly"),
        (["W", "S", "A", "D", "I", "K", "J", "L", "F"], "the man stands still, camera holds steady"),
        (["F"], "the man stands still, camera holds steady"),
    ],
)
def test_official_keyboard_language(keys, expected):
    assert action_sentence(keys) == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    "value",
    [None, [], {}, {"keys": "W"}, {"keys": ["Q"]}, {"frames": [["W"]]}, {"keys": [], "frames": []}, {"keys": [1]}],
)
def test_invalid_controls(value):
    with pytest.raises(OmniClientError):
        action_script(value, 2)


@pytest.mark.cpu
def test_per_latent_script_and_independent_encoding():
    script = action_script({"frames": [["W"], [], ["W"]]}, 3)
    encode = Mock(side_effect=lambda sentence: torch.full((4, 8), float(sentence == script[0])))
    head, tags = torch.zeros(7, 8), torch.tensor([0, 0, 1, 1, 1, 1, 1])
    hidden, actual_tags, spans = append_action_embeddings(head, tags, script, encode)
    assert encode.call_count == 2
    assert spans == ((7, 11), (11, 15), (15, 19))
    torch.testing.assert_close(hidden[:7], head)
    torch.testing.assert_close(hidden[7:11], hidden[15:19])
    assert actual_tags.tolist() == tags.tolist() + [1] * 12
    assert action_script({"keys": ["W"]}, 3) == (script[0],) * 3
    # Dedup state cannot leak between requests.
    append_action_embeddings(head, tags, script, encode)
    assert encode.call_count == 4


def _layout():
    # Real H3 temporal grid for 22 frames: seven video latents.
    spans = tuple((5 + index * 12, 5 + (index + 1) * 12) for index in range(7))
    packed = minimax_h3_packed_sequence(
        text_len=89,
        latent_t=7,
        latent_h=4,
        latent_w=6,
        audio_t=37,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0],
        frame_count=22,
    )
    return packed, spans


@pytest.mark.cpu
def test_mirrored_rope_and_refiner_isolation():
    packed, spans = _layout()
    before = packed["img_position_ids"].clone()
    bounds = bind_action_spans(packed, spans)
    assert bounds == (0, 5, 17, 29, 41, 53, 65, 77, 89)
    start = int(packed["video_row_start"])
    video_times = packed["img_position_ids"][start : start + 42 : 6, 0]
    offsets = []
    for (lo, hi), time in zip(spans, video_times, strict=True):
        assert packed["img_position_ids"][lo:hi, 0].unique().numel() == 1
        offsets.append(float(time - packed["img_position_ids"][lo, 0]))
    assert len(set(round(value, 8) for value in offsets)) == 1
    torch.testing.assert_close(packed["img_position_ids"][:5], before[:5])
    torch.testing.assert_close(packed["img_position_ids"][89:], before[89:])


@pytest.mark.cpu
@pytest.mark.parametrize(
    "spans", [(), ((0, 89),), ((5, 17),) * 7, tuple((5 + i * 12, 6 + (i + 1) * 12) for i in range(7))]
)
def test_reject_non_tiling_spans(spans):
    packed, _ = _layout()
    with pytest.raises(ValueError):
        bind_action_spans(packed, spans)


def _labels(packed, spans, device="cpu"):
    length = int(packed["seq_len"])
    ann = torch.full((length,), -1, dtype=torch.int32, device=device)
    frame = torch.full_like(ann, -1)
    for index, (lo, hi) in enumerate(spans):
        ann[lo:hi] = index
    start = int(packed["video_row_start"])
    frame[start : start + 42] = torch.arange(42, device=device) // 6
    return ann, frame


@pytest.mark.cpu
def test_directed_edges_and_no_multilayer_bypass():
    packed, spans = _layout()
    ann, frame = _labels(packed, spans)
    index = torch.arange(len(ann))
    mask = directed_visibility(ann, frame, int(packed["cu_seqlens"][1]), index[:, None], index[None, :])
    assert mask.diag().all()  # including padding, never an empty softmax row
    used = int(packed["cu_seqlens"][1])
    assert not mask[:used, used:].any()
    assert not mask[used:, :used].any()
    for k, (lo, hi) in enumerate(spans):
        assert not mask[(ann < 0) & (frame != k), lo:hi].any()
        assert mask[frame == k, lo:hi].all()
        assert not mask[lo:hi, (frame >= 0) & (frame != k)].any()
        assert mask[lo:hi, :5].all()  # scene may ground an action, but cannot read it
        # Remove V_k. No number of layers may route A_k into another token.
        reachable = torch.zeros(len(ann), dtype=torch.bool)
        reachable[lo:hi] = True
        for _ in range(4):
            reachable |= mask[:, reachable].any(dim=1) & (frame != k)
        assert not reachable[(ann != k)].any()


class _Checkpoint:
    def __init__(self):
        # Captured from the published safetensors header, without weight data.
        self.shapes = json.loads((Path(__file__).parent / "fixtures/h3_world_weight_shapes.json.fixture").read_text())

    def keys(self):
        return self.shapes.keys()

    def get_slice(self, name):
        return SimpleNamespace(get_shape=lambda: self.shapes[name])

    def get_tensor(self, name):
        return torch.empty(self.shapes[name], device="meta")


@pytest.mark.cpu
def test_released_lora_contract():
    checkpoint = _Checkpoint()
    tensors = validate_world_checkpoint(checkpoint)
    assert len(tensors) == 208
    assert all(name.startswith("transformer.") for name in tensors)
    checkpoint.shapes.pop(next(iter(checkpoint.shapes)))
    with pytest.raises(ValueError, match="104"):
        validate_world_checkpoint(checkpoint)
    checkpoint = _Checkpoint()
    checkpoint.shapes[next(iter(checkpoint.shapes))] = (64, 5376)
    with pytest.raises(ValueError, match="shape"):
        validate_world_checkpoint(checkpoint)
    assert (
        load_h3_world_lora(partition="fl2va", lora_request=None, lora_path="other.safetensors", dtype=torch.float32)
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FlexAttention numerical check")
@pytest.mark.cuda
@pytest.mark.parametrize("seed", [1, 2])
def test_sparse_kernel_matches_dense_and_changed_spans(seed):
    packed, original_spans = _layout()
    torch.manual_seed(seed)
    length = int(packed["seq_len"])
    q, k, v = [torch.randn(length, 2, 32, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    # Same total shape, different action boundaries, then return to the first
    # layout: detect a compiled mask incorrectly retaining an earlier request.
    changed = list(original_spans)
    changed[0] = (5, 18)
    changed[1] = (18, 29)
    shorter = dict(packed)
    shorter["cu_seqlens"] = packed["cu_seqlens"].clone()
    shorter["cu_seqlens"][1] -= 2
    cases = ((packed, original_spans), (packed, tuple(changed)), (shorter, original_spans), (packed, original_spans))
    for layout, spans in cases:
        ann, frame = _labels(layout, spans, "cuda")
        idx = torch.arange(length, device="cuda")
        mask = directed_visibility(ann, frame, int(layout["cu_seqlens"][1]), idx[:, None], idx[None, :])
        actual = action_attention(q, k, v, build_action_block_mask(layout, spans, torch.device("cuda")), 32**-0.5)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                attn_mask=mask,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
