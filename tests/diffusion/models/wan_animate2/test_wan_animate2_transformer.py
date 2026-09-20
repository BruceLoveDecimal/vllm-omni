# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan2.2-Animate-2 transformer structure tests (M0).

Covers the two-phase forward contract, the reference K/V cache lifecycle, the
vectorised RoPE against upstream's per-sample complex implementation, and the
Diffusers checkpoint weight-name remapping.
"""

import gc
import math
import os
import weakref
from dataclasses import dataclass

import pytest
import torch

from tests.diffusion.models.wan_animate2.test_wan_animate2_attention import (
    _upstream_rope_apply,
    _upstream_rope_params,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig
from vllm_omni.diffusion.models.wan_animate2 import (
    ReferenceGridInfo,
    WanAnimate2RotaryPosEmbed,
    WanAnimate2Transformer3DModel,
    WanAnimate2TransformerConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# vLLM's CustomOp layers dispatch on the *platform*, not on tensor device, so
# these run on the accelerator whenever there is one and on CPU otherwise.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# head_dim 64: the smallest width every attention backend (cuDNN included) accepts.
_TINY_CONFIG = WanAnimate2TransformerConfig(
    dim=128,
    num_heads=2,
    in_dim=8,
    out_dim=4,
    text_dim=8,
    freq_dim=8,
    ffn_dim=64,
    num_layers=12,
)


@pytest.fixture(autouse=True)
def _init_distributed():
    """Minimal tensor-parallel group, required by the vLLM linear layers."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend=backend)
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@dataclass(frozen=True)
class _ParallelStub:
    ring_degree: int = 1


@dataclass(frozen=True)
class _AttentionOnlyConfig:
    """The slice of ``OmniDiffusionConfig`` the attention layer reads at construction."""

    diffusion_attention_config: AttentionConfig
    parallel_config: _ParallelStub


@pytest.fixture(autouse=True)
def _pin_torch_sdpa():
    """These tests run in fp32 on tiny shapes; the platform default (cuDNN on
    Blackwell, FA3 on Hopper) rejects that, so pin the portable backend."""
    config = _AttentionOnlyConfig(AttentionConfig(default="TORCH_SDPA"), _ParallelStub())
    with set_current_diffusion_config(config):
        yield


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    """vLLM dispatches GEMM by platform, not by tensor device; pin the CPU path."""
    if _DEVICE.type != "cpu":
        return
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *args, **kwargs: default_unquantized_gemm,
    )


def _tiny_model(log_scale: float = 0.0) -> WanAnimate2Transformer3DModel:
    config = WanAnimate2TransformerConfig(**{**_TINY_CONFIG.__dict__, "log_scale": log_scale})
    model = WanAnimate2Transformer3DModel(config).to(_DEVICE).eval()

    # vLLM's parallel linear layers allocate their parameters with
    # torch.empty(), so an unloaded model runs on uninitialised memory.  Fill
    # them deterministically so these structural tests exercise real arithmetic.
    generator = torch.Generator(device="cpu").manual_seed(1234)
    with torch.no_grad():
        for param in model.parameters():
            values = torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.02
            param.copy_(values.to(param.dtype))
    return model


def _segment_inputs(*, segment_frames=9, reference_frames=8, height=8, width=8):
    """Latents for one segment, sized so the mask window matches exactly."""
    latent_channels = _TINY_CONFIG.out_dim
    condition_channels = _TINY_CONFIG.in_dim - latent_channels
    generator = torch.Generator().manual_seed(5)

    def _rand(channels, frames):
        return torch.randn(1, channels, frames, height, width, generator=generator).to(_DEVICE)

    # origin_len is chosen so that origin_len // 4 + 1 == reference_frames.
    origin_len = (reference_frames - 1) * 4
    return {
        "hidden_states": _rand(latent_channels, segment_frames),
        "condition_latents": _rand(condition_channels, segment_frames),
        "reference_latents": _rand(latent_channels, reference_frames),
        "reference_condition": _rand(condition_channels, reference_frames),
        "encoder_hidden_states": torch.randn(1, 512, _TINY_CONFIG.text_dim, generator=generator).to(_DEVICE),
        "encoder_hidden_states_image": torch.randn(1, 257, 1280, generator=generator).to(_DEVICE),
        "origin_len": origin_len,
        "origin_area": (width * 8, height * 8),
        "reference_grid": (reference_frames, height // 2, width // 2),
    }


def _run_segment(model, inputs, *, is_uncondition=False):
    generation_grid = (
        inputs["hidden_states"].shape[2],
        inputs["hidden_states"].shape[3] // 2,
        inputs["hidden_states"].shape[4] // 2,
    )
    kv_cache = model.extract_reference(
        inputs["reference_latents"],
        inputs["reference_condition"],
        inputs["encoder_hidden_states"],
        inputs["encoder_hidden_states_image"],
        generation_grid,
    )
    with torch.no_grad():
        output = model(
            inputs["hidden_states"],
            torch.tensor([500.0], device=_DEVICE),
            inputs["encoder_hidden_states"],
            inputs["encoder_hidden_states_image"],
            inputs["condition_latents"],
            kv_cache,
            inputs["reference_grid"],
            inputs["origin_len"],
            inputs["origin_area"],
            is_uncondition=is_uncondition,
            return_dict=False,
        )[0]
    return output, kv_cache


def test_two_phase_forward_produces_latent_shaped_output():
    model = _tiny_model()
    inputs = _segment_inputs()
    output, kv_cache = _run_segment(model, inputs)

    assert output.shape == inputs["hidden_states"].shape
    assert torch.isfinite(output).all()
    assert kv_cache.is_populated()


def test_reference_cache_is_not_part_of_module_state():
    """A0.4: segment state must not leak into state_dict / parameters."""
    model = _tiny_model()
    state_keys_before = set(model.state_dict())
    _run_segment(model, _segment_inputs())

    assert set(model.state_dict()) == state_keys_before
    assert not any("cache" in name for name in dict(model.named_parameters()))
    assert not any("cache" in name for name in dict(model.named_buffers()))


def test_reference_cache_release_drops_all_tensors():
    """A0.4: releasing the cache at the segment boundary frees its tensors."""
    model = _tiny_model()
    _, kv_cache = _run_segment(model, _segment_inputs())

    tracked = weakref.ref(kv_cache.get(0)[0])
    kv_cache.release()
    gc.collect()

    assert tracked() is None
    assert not kv_cache.is_populated()
    with pytest.raises(RuntimeError, match="never populated"):
        kv_cache.get(0)


def test_unconditional_branch_skips_block_nine():
    """A0.6: the negative CFG branch runs one block fewer, and it is block 9."""
    model = _tiny_model()
    inputs = _segment_inputs()

    denoise_visits: list[int] = []
    for idx, block in enumerate(model.blocks):
        original = block.forward

        def _wrapped(*args, _idx=idx, _original=original, **kwargs):
            # Both phases route through forward(); only count the denoising one.
            if not kwargs.get("extract", False):
                denoise_visits.append(_idx)
            return _original(*args, **kwargs)

        block.forward = _wrapped

    _run_segment(model, inputs, is_uncondition=False)
    conditional = list(denoise_visits)
    denoise_visits.clear()
    _run_segment(model, inputs, is_uncondition=True)
    unconditional = list(denoise_visits)

    assert conditional == list(range(_TINY_CONFIG.num_layers))
    assert 9 not in unconditional
    assert len(unconditional) == len(conditional) - 1


def test_both_phases_go_through_module_call():
    """FSDP2 unshard and layerwise-offload are pre-forward hooks, so the
    extraction pass must not bypass ``Module.__call__``."""
    model = _tiny_model()

    calls: list[bool] = []
    handles = [
        block.register_forward_pre_hook(
            lambda _module, _args, _kwargs, _calls=calls: _calls.append(_kwargs.get("extract", False)),
            with_kwargs=True,
        )
        for block in model.blocks
    ]
    try:
        _run_segment(model, _segment_inputs())
    finally:
        for handle in handles:
            handle.remove()

    assert calls.count(True) == _TINY_CONFIG.num_layers, "every block must run the extraction pass through __call__"
    assert calls.count(False) == _TINY_CONFIG.num_layers, "every block must run the denoising pass through __call__"


def test_conditional_and_unconditional_outputs_differ():
    model = _tiny_model()
    inputs = _segment_inputs()
    cond, _ = _run_segment(model, inputs, is_uncondition=False)
    uncond, _ = _run_segment(model, inputs, is_uncondition=True)
    assert (cond - uncond).abs().max().item() > 1e-6


@pytest.mark.parametrize(
    "offsets,time_stride",
    [((0, 0, 0), 1), ((1, 0, 5), 1), ((1, 0, 5), 2)],
)
def test_rope_matches_upstream_complex_implementation(offsets, time_stride):
    """A0.5: vectorised RoPE reproduces upstream's per-sample complex path."""
    from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWanS2V

    head_dim, num_heads = 128, 2
    grid = (4, 3, 5)
    seq_len = grid[0] * grid[1] * grid[2]
    x = torch.randn(1, seq_len, num_heads, head_dim, generator=torch.Generator().manual_seed(2)).to(_DEVICE)

    rope = WanAnimate2RotaryPosEmbed(head_dim)
    freqs = rope(grid, _DEVICE, offsets=offsets, time_stride=time_stride)
    actual = RotaryEmbeddingWanS2V()(x.clone(), freqs).cpu()

    upstream_freqs = torch.cat(
        [
            _upstream_rope_params(512, head_dim - 4 * (head_dim // 6), offset=offsets[0]),
            _upstream_rope_params(512, 2 * (head_dim // 6), offset=offsets[1]),
            _upstream_rope_params(512, 2 * (head_dim // 6), offset=offsets[2]),
        ],
        dim=1,
    )
    assert upstream_freqs.shape[1] == head_dim // 2
    expected = _upstream_rope_apply(x.cpu().clone(), [grid], upstream_freqs, time_stride=time_stride)

    assert (actual - expected).abs().max().item() <= 1e-5


def test_rope_frequency_table_is_cached():
    rope = WanAnimate2RotaryPosEmbed(32)
    first = rope((2, 3, 4), _DEVICE)
    second = rope((2, 3, 4), _DEVICE)
    third = rope((2, 3, 4), _DEVICE, offsets=(1, 0, 4))

    assert first is second
    assert third is not first
    rope.clear_cache()
    assert rope((2, 3, 4), _DEVICE) is not first


def test_negative_reference_offset_resolves_against_generation_grid():
    """`refer_offset_w = -1` means "one generation grid width to the right", so
    the reference positions never collide with the generated ones."""
    model = _tiny_model()
    generation_grid, reference_grid = (5, 3, 4), (4, 3, 4)
    freqs = model._reference_freqs(reference_grid, generation_grid, _DEVICE)
    expected = model.rope(reference_grid, _DEVICE, offsets=(1, 0, 4), time_stride=1)
    assert torch.equal(freqs, expected)


def test_grid_bookkeeping_accepts_a_wider_mask_window():
    """`origin_area` is the requested area; letterboxing rounds the real grid
    down, so the mask window is allowed to be wider -- the remainder is padding."""
    grid = ReferenceGridInfo.from_grids((5, 3, 4), (4, 3, 4), origin_len=12, origin_area=(64, 64))
    assert grid.frame_tokens == 12
    assert grid.padded_frame_tokens == 16
    assert grid.num_frames == 5
    assert grid.num_reference_frames == 4
    assert grid.padded_reference_frames == 4
    assert grid.padded_gen_frames == 5


def test_mask_window_smaller_than_the_latent_grid_is_rejected():
    with pytest.raises(ValueError, match="cannot be smaller than the latent grid"):
        ReferenceGridInfo.from_grids((5, 3, 4), (4, 3, 4), origin_len=12, origin_area=(32, 32))


def test_reference_grid_must_match_the_generation_grid_spatially():
    with pytest.raises(ValueError, match="share spatial extents"):
        ReferenceGridInfo.from_grids((5, 3, 4), (4, 2, 4), origin_len=12, origin_area=(64, 64))


@pytest.mark.parametrize(
    "original,expected",
    [
        ("blocks.0.self_attn.to_q.weight", "blocks.0.attn1.to_q.weight"),
        ("blocks.7.self_attn.to_out.0.bias", "blocks.7.attn1.to_out.bias"),
        ("blocks.3.self_attn.norm_k.weight", "blocks.3.attn1.norm_k.weight"),
        ("blocks.1.cross_attn.add_k_proj.weight", "blocks.1.attn2.add_k_proj.weight"),
        ("blocks.1.cross_attn.norm_added_k.weight", "blocks.1.attn2.norm_added_k.weight"),
        ("blocks.2.cross_attn.to_out.0.weight", "blocks.2.attn2.to_out.weight"),
        ("blocks.2.ffn.0.weight", "blocks.2.ffn.net_0.proj.weight"),
        ("blocks.2.ffn.2.bias", "blocks.2.ffn.net_2.bias"),
        ("blocks.4.norm3.weight", "blocks.4.norm2.weight"),
        ("blocks.5.modulation", "blocks.5.scale_shift_table"),
        ("head.head.weight", "proj_out.weight"),
        ("head.modulation", "output_scale_shift_prepare.scale_shift_table"),
        ("text_embedding.0.weight", "condition_embedder.text_embedder.linear_1.weight"),
        ("time_embedding.2.bias", "condition_embedder.time_embedder.linear_2.bias"),
        ("time_projection.1.weight", "condition_embedder.time_proj.weight"),
        ("img_emb.proj.1.weight", "condition_embedder.image_embedder.ff.net.0.proj.weight"),
        ("img_emb.proj.4.bias", "condition_embedder.image_embedder.norm2.bias"),
        ("patch_embedding.weight", "patch_embedding.weight"),
    ],
)
def test_weight_name_remapping(original, expected):
    assert WanAnimate2Transformer3DModel.remap_weight_name(original) == expected


def _diffusers_checkpoint_names(num_layers: int) -> list[str]:
    """The key set of `transformer/diffusion_pytorch_model.safetensors.index.json`."""
    names = ["patch_embedding.weight", "patch_embedding.bias"]
    for stem in ("text_embedding.0", "text_embedding.2", "time_embedding.0", "time_embedding.2", "time_projection.1"):
        names += [f"{stem}.weight", f"{stem}.bias"]
    for idx in (0, 1, 3, 4):
        names += [f"img_emb.proj.{idx}.weight", f"img_emb.proj.{idx}.bias"]
    names += ["head.head.weight", "head.head.bias", "head.modulation"]
    for layer in range(num_layers):
        prefix = f"blocks.{layer}"
        names.append(f"{prefix}.modulation")
        names += [f"{prefix}.norm3.weight", f"{prefix}.norm3.bias"]
        for attn in ("self_attn", "cross_attn"):
            for proj in ("to_q", "to_k", "to_v", "to_out.0"):
                names += [f"{prefix}.{attn}.{proj}.weight", f"{prefix}.{attn}.{proj}.bias"]
            names += [f"{prefix}.{attn}.norm_q.weight", f"{prefix}.{attn}.norm_k.weight"]
        for proj in ("add_k_proj", "add_v_proj"):
            names += [f"{prefix}.cross_attn.{proj}.weight", f"{prefix}.cross_attn.{proj}.bias"]
        names.append(f"{prefix}.cross_attn.norm_added_k.weight")
        names += [f"{prefix}.ffn.0.weight", f"{prefix}.ffn.0.bias", f"{prefix}.ffn.2.weight", f"{prefix}.ffn.2.bias"]
    return names


def test_remapped_names_cover_every_parameter():
    """A0.7: every checkpoint key lands on a parameter, and every parameter is
    covered; the fused QKV projection is the only many-to-one mapping."""
    model = _tiny_model()
    params = set(dict(model.named_parameters()))

    remapped = {
        WanAnimate2Transformer3DModel.remap_weight_name(name)
        for name in _diffusers_checkpoint_names(_TINY_CONFIG.num_layers)
    }
    fused = {name for name in remapped if ".attn1.to_q" in name or ".attn1.to_k" in name or ".attn1.to_v" in name}
    resolved = {
        name.replace(".attn1.to_q", ".attn1.to_qkv")
        .replace(".attn1.to_k", ".attn1.to_qkv")
        .replace(".attn1.to_v", ".attn1.to_qkv")
        for name in fused
    }

    assert (remapped - fused) <= params, f"unmapped destinations: {(remapped - fused) - params}"
    assert resolved <= params
    assert params - ((remapped - fused) | resolved) == set()


def test_load_weights_populates_every_parameter():
    """A0.7 with real loading: a full synthetic checkpoint loads without
    unexpected keys and leaves no parameter untouched."""
    model = _tiny_model()
    shapes = {}
    for name, param in model.named_parameters():
        shapes[name] = param.shape
    weights = []
    for name in _diffusers_checkpoint_names(_TINY_CONFIG.num_layers):
        target = WanAnimate2Transformer3DModel.remap_weight_name(name)
        if ".attn1.to_q" in target or ".attn1.to_k" in target or ".attn1.to_v" in target:
            fused_shape = shapes[target.replace("to_q", "to_qkv").replace("to_k", "to_qkv").replace("to_v", "to_qkv")]
            shape = (fused_shape[0] // 3,) + tuple(fused_shape[1:])
        else:
            shape = shapes[target]
        weights.append((name, torch.ones(shape)))

    loaded = model.load_weights(weights)
    assert loaded == set(shapes)


def test_load_weights_rejects_unknown_keys():
    model = _tiny_model()
    with pytest.raises(KeyError, match="unexpected weight"):
        model.load_weights([("blocks.0.mystery.weight", torch.zeros(1))])


def test_distilled_log_scale_changes_the_output():
    """The distilled bias must actually reach the attention branch."""
    inputs = _segment_inputs()
    base = _tiny_model(log_scale=0.0)
    distilled = _tiny_model(log_scale=-1.3)
    distilled.load_state_dict(base.state_dict())

    base_out, _ = _run_segment(base, inputs)
    distilled_out, _ = _run_segment(distilled, inputs)
    assert (base_out - distilled_out).abs().max().item() > 1e-5


def test_image_embedder_uses_upstream_layernorm_eps():
    model = _tiny_model()
    assert model.condition_embedder.image_embedder.norm1.eps == 1e-5
    assert model.condition_embedder.image_embedder.norm2.eps == 1e-5


def test_hsdp_and_offload_declarations():
    assert WanAnimate2Transformer3DModel._repeated_blocks == ["WanAnimate2TransformerBlock"]
    assert WanAnimate2Transformer3DModel._layerwise_offload_blocks_attrs == ["blocks"]
    assert WanAnimate2Transformer3DModel.packed_modules_mapping == {"to_qkv": ["to_q", "to_k", "to_v"]}
    condition = WanAnimate2Transformer3DModel._hsdp_shard_conditions[0]
    assert condition("blocks.7", None)
    assert not condition("blocks.7.attn1", None)


def test_config_bridges_diffusers_config_names():
    """`transformer/config.json` uses upstream's spelling and carries no log_scale."""
    config = WanAnimate2TransformerConfig.from_dict(
        {
            "patch_size": [1, 2, 2],
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "num_heads": 40,
            "num_layers": 40,
            "text_dim": 4096,
            "out_dim": 16,
            "eps": 1e-6,
            "use_img_emb": True,
            "refer_offset_t": 1,
            "refer_offset_h": 0,
            "refer_offset_w": -1,
            "refer_stride": 1,
            "text_len": 512,
        },
        log_scale=-1.3,
    )
    assert config.attention_head_dim == 128
    assert config.patch_size == (1, 2, 2)
    assert math.isclose(config.log_scale, -1.3)
