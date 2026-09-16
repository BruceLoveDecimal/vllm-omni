# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed varlen layout for the AuK DiT: the plan, and packed-versus-padded parity."""

import pytest
import torch

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper
from vllm_omni.diffusion.models.auk.packing import PackPlan, build_pack_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_dit(device: str) -> AuKTransformer:
    torch.manual_seed(12)
    return (
        AuKTransformer(
            dim=32,
            heads=2,
            dim_head=16,
            ff_mult=2,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=2,
            num_single_layers=2,
        )
        .eval()
        .to(device)
    )


def _mask(lengths: list[int], width: int, device: str = "cpu") -> torch.Tensor:
    return torch.arange(width, device=device)[None] < torch.tensor(lengths, device=device)[:, None]


# Three rows: reference lengths 4/0/2, target lengths 9/5/12, text 7/4/6.
_REF = [4, 0, 2]
_TARGET = [9, 5, 12]
_TEXT = [7, 4, 6]


def _inputs(device: str) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(3)
    rows = len(_REF)
    return {
        "x": torch.randn(rows, max(_TARGET), 4, generator=g, device=device)
        * _mask(_TARGET, max(_TARGET), device)[..., None],
        "mask": _mask(_TARGET, max(_TARGET), device),
        "text": torch.randn(rows, max(_TEXT), 8, generator=g, device=device)
        * _mask(_TEXT, max(_TEXT), device)[..., None],
        "c_mask": _mask(_TEXT, max(_TEXT), device),
        "ref": torch.randn(rows, max(_REF), 4, generator=g, device=device) * _mask(_REF, max(_REF), device)[..., None],
        "ref_mask": _mask(_REF, max(_REF), device),
    }


class TestPackPlan:
    def test_plan_packs_rows_back_to_back_with_row_positions(self):
        audio = _mask([3, 1, 2], 4)
        text = _mask([2, 2, 1], 3)

        plan = build_pack_plan(audio, text)

        assert plan.audio_real == 6 and plan.text_real == 5
        assert plan.audio_idx.tolist() == [0, 1, 2, 4, 8, 9]
        assert plan.audio_seg.tolist() == [0, 0, 0, 1, 2, 2]
        assert plan.audio_pos.tolist() == [0, 1, 2, 0, 0, 1]
        assert plan.text_idx.tolist() == [0, 1, 3, 4, 6]
        # Joint order: row 0 audio, row 0 text, row 1 audio, ... ; no filler segment.
        joint = torch.cat([plan.audio_seg * 2, plan.text_seg * 2 + 1])[plan.joint_perm]
        assert joint.tolist() == [0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5]
        assert plan.joint_cu.tolist() == [0, 5, 8, 11]
        assert plan.joint_max == 5
        assert torch.equal(plan.joint_perm[plan.joint_inv], torch.arange(11))
        # Single order puts text first, and audio positions continue after the text.
        single_seg = torch.cat([plan.text_seg, plan.audio_seg])[plan.single_perm]
        assert single_seg.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert plan.single_pos.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2]
        assert plan.single_cu.tolist() == [0, 5, 8, 11]

    def test_capacities_add_a_trailing_filler_segment(self):
        audio = _mask([3, 1, 2], 4)
        text = _mask([2, 2, 1], 3)

        plan = build_pack_plan(audio, text, audio_capacity=8, text_capacity=8)

        assert plan.audio_capacity == 8 and plan.text_capacity == 8
        # Fillers read the scratch slot past the flattened stream and belong to a row past the end.
        assert plan.audio_idx[6:].tolist() == [12, 12] and plan.audio_seg[6:].tolist() == [3, 3]
        assert plan.joint_cu.tolist() == [0, 5, 8, 11, 16]
        assert plan.joint_max == 5
        # Fillers sort last in both orders.
        assert plan.joint_perm[11:].tolist() == [6, 7, 13, 14, 15]
        assert torch.cat([plan.text_seg, plan.audio_seg])[plan.single_perm][11:].tolist() == [3] * 5

    def test_static_plan_round_trip(self):
        plan = build_pack_plan(_mask([3, 1, 2], 4), _mask([2, 2, 1], 3), audio_capacity=8, text_capacity=8)
        static = plan.clone()
        other = build_pack_plan(_mask([2, 2, 2], 4), _mask([1, 3, 1], 3), audio_capacity=8, text_capacity=8)

        other.copy_into(static)

        for name, value in other.tensors().items():
            assert torch.equal(getattr(static, name), value), name
        assert isinstance(static, PackPlan) and static.rows == 3


@torch.inference_mode()
@pytest.mark.parametrize("cfg_infer", [False, True])
def test_packed_blocks_match_the_padded_blocks(cfg_infer: bool) -> None:
    """Packing changes the layout, not the numbers: every real target frame agrees."""
    dit = _make_dit("cpu")
    inputs = _inputs("cpu")
    t = torch.tensor(0.3)

    padded = dit(
        inputs["x"],
        inputs["text"],
        t,
        **{k: v for k, v in inputs.items() if k not in ("x", "text")},
        cfg_infer=cfg_infer,
    )
    dit.packed_attention = True
    packed = dit(
        inputs["x"],
        inputs["text"],
        t,
        **{k: v for k, v in inputs.items() if k not in ("x", "text")},
        cfg_infer=cfg_infer,
    )

    rows = padded.shape[0]
    for row in range(rows):
        frames = _TARGET[row % len(_TARGET)]
        torch.testing.assert_close(packed[row, :frames], padded[row, :frames], atol=1e-5, rtol=1e-4)


@torch.inference_mode()
def test_a_plan_with_filler_capacity_gives_the_same_result() -> None:
    dit = _make_dit("cpu")
    dit.packed_attention = True
    inputs = _inputs("cpu")
    t = torch.tensor(0.3)
    kwargs = {k: v for k, v in inputs.items() if k not in ("x", "text")}
    exact = dit(inputs["x"], inputs["text"], t, **kwargs)

    audio_mask = torch.cat([inputs["ref_mask"], inputs["mask"]], dim=1)
    plan = build_pack_plan(audio_mask, inputs["c_mask"], audio_capacity=64, text_capacity=64)
    plan.joint_max = plan.single_max = 128
    with_filler = dit(inputs["x"], inputs["text"], t, **kwargs, plan=plan)

    for row, frames in enumerate(_TARGET):
        torch.testing.assert_close(with_filler[row, :frames], exact[row, :frames], atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_packed_graph_replay_matches_packed_eager(cfg_strength: float) -> None:
    dit = _make_dit("cuda")
    dit.packed_attention = True
    inputs = _inputs("cuda")
    t = torch.tensor(0.3, device="cuda")
    kwargs = {k: v for k, v in inputs.items() if k not in ("x", "text")}
    wrapper = AuKCUDAGraphWrapper(dit)

    if cfg_strength >= 1e-5:
        pred = dit(inputs["x"], inputs["text"], t, **kwargs, cfg_infer=True)
        cond, uncond = pred.chunk(2, dim=0)
        eager = cond + (cond - uncond) * cfg_strength
    else:
        eager = dit(inputs["x"], inputs["text"], t, **kwargs)
    replayed = wrapper(
        x=inputs["x"],
        mask=inputs["mask"],
        text=inputs["text"],
        c_mask=inputs["c_mask"],
        ref=inputs["ref"],
        ref_mask=inputs["ref_mask"],
        timestep=t,
        cfg_strength=cfg_strength,
    )
    # Replay twice so the static plan is refreshed on a cache hit as well.
    replayed_again = wrapper(
        x=inputs["x"],
        mask=inputs["mask"],
        text=inputs["text"],
        c_mask=inputs["c_mask"],
        ref=inputs["ref"],
        ref_mask=inputs["ref_mask"],
        timestep=t,
        cfg_strength=cfg_strength,
    )

    assert len(wrapper._cache) == 1
    key = next(iter(wrapper._cache))
    assert len(key) == 8 and key[6] % 64 == 0 and key[7] % 64 == 0
    for row, frames in enumerate(_TARGET):
        torch.testing.assert_close(replayed[row, :frames], eager[row, :frames], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(replayed_again[row, :frames], replayed[row, :frames])
