# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Folding a VDN-H3 release into H3's native checkpoint names.

The failure this file exists for is silent: a delta placed in the wrong half of
a fused matrix, or aimed at the wrong QKV slot, still loads and still generates.
Every check below therefore compares against the reconstruction the release
states - ``W_native = W_base + sum_adapters scale * B @ A`` - in the layout H3
actually stores.
"""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.minimax_h3.vdn_release import (
    CHANNELS,
    FFN,
    HEAD_DIM,
    HIDDEN,
    TIME_EMBED_DIM,
    adapter_tensors,
    branch_tensors,
    write_release,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv
from vllm_omni.diffusion.models.minimax_h3.vdn.checkpoint import (
    VdnCheckpointError,
    VdnSpec,
    resolve_vdn_checkpoint,
)
from vllm_omni.diffusion.models.minimax_h3.vdn.weight_fusion import (
    BRANCH_BLOCK_SUFFIXES,
    VdnWeightFusion,
    branch_parameter_names,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCKS = 2
REFINERS = 1


def _fusion(tmp_path, **release):
    release.setdefault("num_blocks", BLOCKS)
    release.setdefault("num_refiner_blocks", REFINERS)
    root = write_release(tmp_path / "release", **release)
    checkpoint = resolve_vdn_checkpoint(VdnSpec.from_mapping({"checkpoint": str(root)}))
    return VdnWeightFusion.from_checkpoint(
        checkpoint,
        head_dim=HEAD_DIM,
        num_blocks=release["num_blocks"],
        num_refiner_blocks=release["num_refiner_blocks"],
    )


def _delta(tensors: dict, module: str, adapter: str) -> torch.Tensor:
    a = tensors[f"{module}.lora_A.{adapter}.weight"]
    b = tensors[f"{module}.lora_B.{adapter}.weight"]
    return b.float() @ a.float()


def _all_adapter_tensors() -> dict[str, dict[str, torch.Tensor]]:
    return {
        adapter: adapter_tensors(adapter, num_blocks=BLOCKS, num_refiner_blocks=REFINERS)[0]
        for adapter in ("default", "turbo")
    }


def test_the_branch_is_assigned_under_native_names(tmp_path):
    fusion = _fusion(tmp_path)

    expected = set(branch_parameter_names(BLOCKS))
    assert fusion.injection_names == expected
    assert len(expected) == len(BRANCH_BLOCK_SUFFIXES) * BLOCKS
    # The published release carries 16 tensors for each of its 50 blocks.
    assert len(BRANCH_BLOCK_SUFFIXES) * 50 == 800
    assert "blocks.0.attn.hybrid.linear.alpha.A_log" in expected
    assert "blocks.1.attn.hybrid.softmax_gate.up.bias" in expected
    assert "blocks.0.attn.hybrid.to_out_linear.weight" in expected


def test_the_assigned_branch_tensors_are_the_checkpoint_s_own(tmp_path):
    fusion = _fusion(tmp_path)
    source = branch_tensors(BLOCKS)

    assigned = dict(fusion.iter_injections())
    torch.testing.assert_close(
        assigned["blocks.1.attn.hybrid.linear.short_conv.k_tm.weight"],
        source["transformer_blocks.1.attn.linear_attention.short_conv.k_tm.weight"],
    )
    torch.testing.assert_close(
        assigned["blocks.0.attn.hybrid.linear.output_gate.up.bias"],
        source["transformer_blocks.0.attn.linear_attention.output_gate.up.bias"],
    )


def test_both_adapters_sum_into_the_grouped_qkv_projection(tmp_path):
    """The one placement that cannot be checked by shape alone.

    H3 stores attention as one grouped ``[q, k, v]`` matrix per head group; the
    adapters are written against the three separate diffusers projections. The
    reconstruction is only correct if unpacking the fused parameter the way the
    loader does yields the base plus each projection's own delta.
    """
    fusion = _fusion(tmp_path)
    tensors = _all_adapter_tensors()
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(3 * CHANNELS, HIDDEN, generator=generator, dtype=torch.float32)

    fused = fusion.fuse("blocks.0.attn.qkv_proj.weight", base).cpu()

    def unpack(weight: torch.Tensor) -> torch.Tensor:
        return _reorder_grouped_qkv_to_qkv(
            weight, num_query_groups=CHANNELS // HEAD_DIM, heads_per_group=1, head_dim=HEAD_DIM
        )

    expected = unpack(base)
    for projection, offset in (("to_q", 0), ("to_k", 1), ("to_v", 2)):
        module = f"transformer_blocks.0.attn.orig.{projection}"
        delta = sum(_delta(tensors[adapter], module, adapter) for adapter in ("default", "turbo"))
        rows = slice(offset * CHANNELS, (offset + 1) * CHANNELS)
        expected[rows] += delta
    torch.testing.assert_close(unpack(fused), expected, rtol=1e-5, atol=1e-5)


def test_the_fused_gate_up_delta_is_swapped_into_native_order(tmp_path):
    """diffusers stores the feed-forward projection value-first, H3 gate-first."""
    fusion = _fusion(tmp_path)
    tensors = _all_adapter_tensors()["turbo"]
    generator = torch.Generator().manual_seed(12)
    base = torch.randn(2 * FFN, HIDDEN, generator=generator, dtype=torch.float32)

    fused = fusion.fuse("blocks.0.mlp.fc1.weight", base).cpu()

    delta = _delta(tensors, "transformer_blocks.0.ff.net.0.proj", "turbo")
    value, gate = delta.chunk(2, dim=0)
    torch.testing.assert_close(fused, base + torch.cat([gate, value]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("native", "module", "shape"),
    [
        ("blocks.0.attn.out_proj.weight", "transformer_blocks.0.attn.orig.to_out.0", (HIDDEN, CHANNELS)),
        ("blocks.1.mlp.fc2.weight", "transformer_blocks.1.ff.net.2", (HIDDEN, FFN)),
        ("blocks.0.adaln_proj.linear.weight", "transformer_blocks.0.adaln_proj.linear", (18 * HIDDEN, TIME_EMBED_DIM)),
        ("final_layer.adaln_proj.linear.weight", "norm_out.linear", (2 * HIDDEN, TIME_EMBED_DIM)),
        (
            "token_refiner.blocks.0.attn.out_proj.weight",
            "token_refiner.refiner_blocks.0.attn.to_out.0",
            (HIDDEN, CHANNELS),
        ),
    ],
)
def test_every_other_target_is_added_as_it_comes(native, module, shape, tmp_path):
    fusion = _fusion(tmp_path)
    tensors = _all_adapter_tensors()
    generator = torch.Generator().manual_seed(13)
    base = torch.randn(*shape, generator=generator, dtype=torch.float32)

    fused = fusion.fuse(native, base).cpu()

    expected = base.clone()
    for adapter, payload in tensors.items():
        if f"{module}.lora_A.{adapter}.weight" in payload:
            expected += _delta(payload, module, adapter)
    torch.testing.assert_close(fused, expected, rtol=1e-5, atol=1e-5)


def test_a_parameter_no_adapter_edits_passes_through(tmp_path):
    fusion = _fusion(tmp_path)
    weight = torch.ones(HIDDEN)

    assert fusion.fuse("blocks.0.norm1.weight", weight) is weight


def test_the_stream_assigns_the_branch_after_the_checkpoint(tmp_path):
    fusion = _fusion(tmp_path)
    stream = [("blocks.0.norm1.weight", torch.ones(HIDDEN))]

    names = [name for name, _ in fusion.apply(stream)]

    assert names[0] == "blocks.0.norm1.weight"
    assert set(names[1:]) == set(branch_parameter_names(BLOCKS))
    with pytest.raises(VdnCheckpointError, match="already been fused"):
        next(fusion.apply(stream))


def test_a_checkpoint_that_already_carries_the_branch_is_refused(tmp_path):
    """Assigning over it would discard the checkpoint's own weight."""
    fusion = _fusion(tmp_path)
    stream = [("blocks.0.attn.hybrid.to_out_linear.weight", torch.ones(HIDDEN, CHANNELS))]

    with pytest.raises(VdnCheckpointError, match="already provides"):
        list(fusion.apply(stream))


def test_a_branch_parameter_that_never_reached_the_model_is_reported(tmp_path):
    """An assigned parameter lands on a module the base checkpoint lacks.

    If the model never built it, ``load_weights`` only logs a skip - so the
    fusion is closed against the names the DiT actually consumed.
    """
    fusion = _fusion(tmp_path)
    # Stand in for a checkpoint that carried every parameter the adapters edit.
    fusion._applied.update(fusion._patches)
    consumed = set(branch_parameter_names(BLOCKS)) - {"blocks.0.attn.hybrid.linear.norm.weight"}

    with pytest.raises(VdnCheckpointError, match="never reached the model"):
        fusion.validate_fully_applied(consumed)


def test_an_adapter_target_the_checkpoint_never_provided_is_reported(tmp_path):
    fusion = _fusion(tmp_path)

    with pytest.raises(VdnCheckpointError, match="never provided"):
        fusion.validate_fully_applied(set(branch_parameter_names(BLOCKS)))


def test_a_branch_missing_one_tensor_is_refused(tmp_path):
    with pytest.raises(VdnCheckpointError, match="block 0 carries"):
        _fusion(tmp_path, drop_branch_key="transformer_blocks.0.attn.linear_attention.norm.weight")


def test_a_branch_that_does_not_cover_every_block_is_refused(tmp_path):
    root = write_release(tmp_path / "release", num_blocks=BLOCKS, num_refiner_blocks=REFINERS)
    checkpoint = resolve_vdn_checkpoint(VdnSpec.from_mapping({"checkpoint": str(root)}))

    with pytest.raises(VdnCheckpointError, match="block 2 carries 0"):
        VdnWeightFusion.from_checkpoint(
            checkpoint, head_dim=HEAD_DIM, num_blocks=BLOCKS + 1, num_refiner_blocks=REFINERS
        )


def test_an_unpaired_adapter_factor_is_refused(tmp_path):
    with pytest.raises(VdnCheckpointError, match="unpaired factor"):
        _fusion(tmp_path, drop_adapter_key="transformer_blocks.0.attn.orig.to_q.lora_B.default.weight")


def test_an_exact_target_adapter_is_held_to_its_declaration(tmp_path):
    """A truncated turbo file would otherwise leave most blocks undistilled."""
    with pytest.raises(VdnCheckpointError, match="declares .* targets but carries"):
        _fusion(tmp_path, drop_adapter_module="transformer_blocks.0.adaln_proj.linear")


def test_a_stage_b_adapter_must_reach_every_block(tmp_path):
    root = write_release(tmp_path / "release", num_blocks=BLOCKS, num_refiner_blocks=REFINERS)
    checkpoint = resolve_vdn_checkpoint(VdnSpec.from_mapping({"checkpoint": str(root)}))

    with pytest.raises(VdnCheckpointError, match="token_refiner.blocks."):
        VdnWeightFusion.from_checkpoint(
            checkpoint, head_dim=HEAD_DIM, num_blocks=BLOCKS, num_refiner_blocks=REFINERS + 1
        )
