# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for EasyCache: config validation, hook forward numerics,
sequence-parallel statistic reduction, backend lifecycle and selector wiring.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.cache.base import resolve_denoise_steps
from vllm_omni.diffusion.cache.easycache import (
    EasyCacheBackend,
    EasyCacheConfig,
    apply_easy_cache_hook,
    remove_easy_cache_hook,
    synchronized_mean_abs,
)
from vllm_omni.diffusion.cache.easycache import hook as easycache_hook
from vllm_omni.diffusion.cache.easycache.hook import (
    _EASY_CACHE_BLOCK_HOOK,
    _EASY_CACHE_HEAD_HOOK,
    iter_easy_cache_hooks,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.hooks.base import HookRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------


class _AffineBlock(nn.Module):
    """Single-stream block: hidden_states * scale + delta, counting calls."""

    def __init__(self, scale: float = 1.0, delta: float = 0.0):
        super().__init__()
        self.scale = scale
        self.delta = delta
        self.forward_calls = 0

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None):
        self.forward_calls += 1
        return hidden_states * self.scale + self.delta


class _DualStreamBlock(nn.Module):
    """Dual-stream block returning (hidden_states, encoder_hidden_states)."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        self.forward_calls = 0

    def forward(self, hidden_states, encoder_hidden_states):
        self.forward_calls += 1
        return hidden_states + self.delta, encoder_hidden_states * 2


class _PackedThdBlock(nn.Module):
    """MiniMax H3-shaped block: positional ``x`` plus keyword-only conditioning.

    H3 packs ``[total_rows, hidden]`` and names its first parameter ``x``, so it
    exercises the positional fallback of the hook's argument resolution rather
    than the ``hidden_states`` keyword every diffusers block uses.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        self.forward_calls = 0

    def forward(self, x, *, t_emb, cu_seqlens, num_requests=1):
        del t_emb, cu_seqlens, num_requests
        self.forward_calls += 1
        return x + self.delta


class _ToyPackedTransformer(nn.Module):
    """Block stack called the way ``MiniMaxH3DiTModel.forward`` calls its own."""

    def __init__(self, n_blocks: int = 3, delta: float = 1.0):
        super().__init__()
        self.blocks = nn.ModuleList([_PackedThdBlock(delta) for _ in range(n_blocks)])

    def forward(self, x, t_emb, cu_seqlens):
        for block in self.blocks:
            x = block(x, t_emb=t_emb, cu_seqlens=cu_seqlens, num_requests=1)
        return x

    @property
    def total_forward_calls(self) -> int:
        return sum(block.forward_calls for block in self.blocks)


class _FakeMultiDiTPipeline:
    """Pipeline exposing two DiT partitions, as MiniMax H3 does for ref2va."""

    _dit_modules = ["transformer", "transformers_ref"]

    def __init__(self, transformer, transformers_ref):
        self.transformer = transformer
        self.transformers_ref = transformers_ref


class _ToyTransformer(nn.Module):
    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, encoder_hidden_states=None):
        for block in self.blocks:
            out = block(hidden_states, encoder_hidden_states)
            if isinstance(out, tuple):
                hidden_states, encoder_hidden_states = out
            else:
                hidden_states = out
        return hidden_states

    @property
    def total_forward_calls(self) -> int:
        return sum(block.forward_calls for block in self.blocks)


def _affine_model(n_blocks: int = 3, scale: float = 1.0, delta: float = 1.0) -> _ToyTransformer:
    return _ToyTransformer([_AffineBlock(scale, delta) for _ in range(n_blocks)])


def _config(**overrides) -> EasyCacheConfig:
    base = dict(threshold=0.1, warmup_steps=1, cooldown_steps=1, max_skip_steps=0, num_inference_steps=6)
    base.update(overrides)
    return EasyCacheConfig(**base)


def _head_state(model: _ToyTransformer, branch: str = "positive"):
    registry = HookRegistry.check_if_exists_or_initialize(model.blocks[0])
    hook = registry.get_hook(_EASY_CACHE_HEAD_HOOK)
    hook.state_manager.set_context(f"easycache_{branch}")
    return hook.state_manager.get_state()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestEasyCacheConfig:
    def test_defaults(self):
        config = EasyCacheConfig()
        assert config.threshold == 0.1
        assert config.warmup_steps == 5
        assert config.cooldown_steps == 1
        assert config.max_skip_steps == 0
        assert config.first_cooldown_step == 49

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": 0.0},
            {"threshold": -1.0},
            {"threshold": float("nan")},
            {"warmup_steps": -1},
            {"cooldown_steps": -1},
            {"max_skip_steps": -1},
            {"max_skip_steps": True},
            {"num_inference_steps": 0},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            EasyCacheConfig(**kwargs)


# ---------------------------------------------------------------------------
# Hook numerics
# ---------------------------------------------------------------------------


class TestEasyCacheHook:
    def test_unchanged_input_skips_after_estimator_warmup(self):
        """Zero input drift predicts zero output change, so every eligible step skips."""
        n_blocks, delta = 3, 1.0
        model = _affine_model(n_blocks, delta=delta)
        apply_easy_cache_hook(model, _config(num_inference_steps=6))
        x = torch.zeros(1, 4, 8)
        state = _head_state(model)

        # step 0: warmup; step 1: estimator initialization (k unknown).
        assert torch.allclose(model(x), x + n_blocks * delta)
        assert torch.allclose(model(x), x + n_blocks * delta)
        assert model.total_forward_calls == 2 * n_blocks
        assert state.transform_rate == 0.0

        # steps 2-4: cache hit, residual of the whole stack applied exactly once.
        for i in range(3):
            out = model(x)
            assert model.total_forward_calls == 2 * n_blocks, f"step {i + 2} was not skipped"
            assert torch.allclose(out, x + n_blocks * delta), "cached residual mis-applied on skip"
        assert state.num_skipped == 3

        # step 5: cooldown always computes and finishes the run.
        model(x)
        assert model.total_forward_calls == 3 * n_blocks

        # Run finished -> state reset for the next request.
        assert state.step_index == 0
        assert state.cached_residual is None
        assert state.num_computed == 0 and state.num_skipped == 0

    def test_input_insensitive_blocks_skip_on_drifting_input(self):
        """Blocks whose output ignores the input have k == 0 and skip despite input drift."""
        n_blocks = 3
        model = _affine_model(n_blocks, scale=0.0, delta=1.0)
        apply_easy_cache_hook(model, _config(num_inference_steps=6))
        state = _head_state(model)

        model(torch.zeros(1, 4, 8))
        x_last = torch.full((1, 4, 8), 5.0)
        model(x_last)
        assert state.transform_rate == 0.0

        x = torch.full((1, 4, 8), 9.0)
        out = model(x)
        assert model.total_forward_calls == 2 * n_blocks
        # y_t = x_t + (y_last - x_last) with y_last == 1.
        assert torch.allclose(out, x + (1.0 - x_last))

    def test_constant_offset_blocks_have_unit_transform_rate(self):
        """y = x + c gives k == 1: the estimate equals the relative input drift."""
        model = _affine_model(3, delta=1.0)
        apply_easy_cache_hook(model, _config(threshold=0.1, num_inference_steps=6))
        state = _head_state(model)
        x = torch.zeros(1, 4, 8)

        model(x)
        model(x + 0.5)
        assert state.transform_rate == pytest.approx(1.0)
        # e = 1 * 0.5 / mean|y_last| = 0.5 / 3.5 > 0.1 -> compute
        model(x)
        assert model.total_forward_calls == 9
        assert state.num_skipped == 0

    def test_threshold_forces_recompute(self):
        """Blocks scaling the input have k == scale^n; a tiny threshold never skips."""
        model = _affine_model(3, scale=2.0, delta=0.0)
        apply_easy_cache_hook(model, _config(threshold=1e-6, num_inference_steps=6))
        for i in range(6):
            model(torch.full((1, 4, 8), float(i + 1)))
        assert model.total_forward_calls == 6 * 3

    def test_estimate_accumulates_until_threshold(self):
        """With k == 1, the accumulated estimate crosses a mid threshold on step 4."""
        model = _affine_model(2, scale=1.0, delta=0.0)
        # identity blocks: k = mean|dy|/mean|dx| = 1, residual = 0.
        apply_easy_cache_hook(model, _config(threshold=0.25, num_inference_steps=8))
        state = _head_state(model)

        model(torch.ones(1, 4, 8))  # step 0 warmup
        model(torch.ones(1, 4, 8) * 2.0)  # step 1 initialize -> k = 1, y_last = 2
        assert state.transform_rate == pytest.approx(1.0)

        model(torch.ones(1, 4, 8) * 2.2)  # step 2: e = 1 * 0.2 / 2 = 0.1 < 0.25 -> skip
        assert state.accumulated_error == pytest.approx(0.1)
        assert state.num_skipped == 1
        model(torch.ones(1, 4, 8) * 2.4)  # step 3: acc = 0.2 -> skip
        assert state.accumulated_error == pytest.approx(0.2)
        assert state.num_skipped == 2
        model(torch.ones(1, 4, 8) * 2.6)  # step 4: acc = 0.3 >= 0.25 -> compute, acc reset
        assert state.num_skipped == 2
        assert state.num_computed == 3
        assert state.accumulated_error == 0.0
        assert model.total_forward_calls == 3 * 2

    def test_max_skip_steps_cap(self):
        model = _affine_model(2, delta=1.0)
        apply_easy_cache_hook(model, _config(max_skip_steps=1, num_inference_steps=8))
        x = torch.zeros(1, 4, 8)
        for _ in range(7):
            model(x)
        # steps: 0 compute, 1 compute, 2 skip, 3 compute (cap), 4 skip, 5 compute, 6 skip
        assert model.total_forward_calls == 4 * 2
        assert _head_state(model).num_skipped == 3

    def test_single_block_advances_schedule_on_skip(self):
        model = _affine_model(1, delta=1.0)
        apply_easy_cache_hook(model, _config(num_inference_steps=4))
        x = torch.zeros(1, 4, 8)

        model(x)  # step 0 warmup
        model(x)  # step 1 initialize
        model(x)  # step 2 skip
        assert model.total_forward_calls == 2
        model(x)  # step 3 cooldown -> run ends, state reset
        assert model.total_forward_calls == 3

        state = _head_state(model)
        assert state.step_index == 0
        assert state.cached_residual is None
        # A new run starts by computing, not by replaying stale state.
        model(x)
        assert model.total_forward_calls == 4

    def test_dual_stream_blocks_pass_through_on_skip(self):
        model = _ToyTransformer([_DualStreamBlock(1.0) for _ in range(3)])
        apply_easy_cache_hook(model, _config(num_inference_steps=5))
        x = torch.zeros(1, 4, 8)
        e = torch.ones(1, 2, 8)

        model(x, e)
        model(x, e)
        assert model.total_forward_calls == 6
        # Skipped step: hidden gets the residual once, encoder passes through.
        out = model.blocks[0](x, e)
        assert isinstance(out, tuple)
        assert torch.allclose(out[0], x + 3.0)
        assert torch.equal(out[1], e)
        mid = model.blocks[1](out[0], out[1])
        assert torch.equal(mid[0], out[0]) and torch.equal(mid[1], e)
        assert model.total_forward_calls == 6

    def test_shape_change_recomputes_and_clears_cache(self):
        model = _affine_model(2, delta=1.0)
        apply_easy_cache_hook(model, _config(num_inference_steps=8))
        model(torch.zeros(1, 4, 8))
        model(torch.zeros(1, 4, 8))
        model(torch.zeros(1, 4, 8))
        assert model.total_forward_calls == 4  # third step skipped

        out = model(torch.zeros(1, 6, 8))
        assert model.total_forward_calls == 6
        assert torch.allclose(out, torch.ones(1, 6, 8) * 2.0)

    def test_hidden_states_kwarg_and_unhooked_baseline_match(self):
        model = _affine_model(3, scale=1.5, delta=0.3)
        reference = _affine_model(3, scale=1.5, delta=0.3)
        apply_easy_cache_hook(model, _config(threshold=1e-9, num_inference_steps=4))
        x = torch.randn(1, 4, 8)
        for _ in range(4):
            out = model.blocks[0](hidden_states=x)
            ref = reference.blocks[0](hidden_states=x)
            assert torch.allclose(out, ref)

    def test_remove_hooks(self):
        model = _affine_model(3)
        apply_easy_cache_hook(model, _config())
        assert len(list(iter_easy_cache_hooks(model))) == 3
        remove_easy_cache_hook(model)
        assert list(iter_easy_cache_hooks(model)) == []
        # Re-applying after removal registers cleanly.
        apply_easy_cache_hook(model, _config())
        names = [
            name
            for _, block in model.named_children()
            for blk in block
            for name in (_EASY_CACHE_HEAD_HOOK, _EASY_CACHE_BLOCK_HOOK)
            if blk._hook_registry.get_hook(name) is not None
        ]
        assert names == [_EASY_CACHE_HEAD_HOOK, _EASY_CACHE_BLOCK_HOOK, _EASY_CACHE_BLOCK_HOOK]

    def test_true_cfg_alternates_branch_state(self):
        model = _affine_model(2, delta=1.0)
        model.do_true_cfg = True
        apply_easy_cache_hook(model, _config(num_inference_steps=6))
        head = model.blocks[0]._hook_registry.get_hook(_EASY_CACHE_HEAD_HOOK)
        x = torch.zeros(1, 4, 8)

        model(x)  # positive, step 0
        assert head.state_manager._context == "easycache_positive"
        model(x)  # negative, step 0
        assert head.state_manager._context == "easycache_negative"
        assert set(head.state_manager._states) == {"easycache_positive", "easycache_negative"}
        assert all(s.step_index == 1 for s in head.state_manager._states.values())

    # ---------------------------------------------------------------------------
    # Sequence-parallel statistic reduction
    # ---------------------------------------------------------------------------

    def test_packed_positional_block_skips_and_reuses_residual(self):
        """H3-style blocks (positional ``x``, tensor output) cache correctly."""
        model = _ToyPackedTransformer(n_blocks=3, delta=1.0)
        apply_easy_cache_hook(model, _config(warmup_steps=2, cooldown_steps=1, num_inference_steps=6))

        rows = torch.zeros(8, 16)
        t_emb = torch.ones(1, 4)
        cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

        out0 = model(rows, t_emb, cu_seqlens)
        out1 = model(rows, t_emb, cu_seqlens)
        assert model.total_forward_calls == 2 * 3
        torch.testing.assert_close(out1, rows + 3.0)

        # The input does not drift, so the estimate stays 0 and step 2 skips.
        out2 = model(rows, t_emb, cu_seqlens)
        assert model.total_forward_calls == 2 * 3
        torch.testing.assert_close(out2, out0)

    def test_packed_positional_block_matches_unhooked_baseline(self):
        model = _ToyPackedTransformer(n_blocks=2, delta=0.5)
        baseline = _ToyPackedTransformer(n_blocks=2, delta=0.5)
        apply_easy_cache_hook(model, _config(warmup_steps=4, cooldown_steps=1, num_inference_steps=4))

        t_emb = torch.ones(1, 4)
        cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
        for step in range(4):
            rows = torch.full((8, 16), float(step))
            torch.testing.assert_close(
                model(rows, t_emb, cu_seqlens),
                baseline(rows, t_emb, cu_seqlens),
            )
        # Warmup covers every step, so nothing was skipped.
        assert model.total_forward_calls == baseline.total_forward_calls


class TestSynchronizedMeanAbs:
    def test_local_only(self):
        a = torch.tensor([1.0, -3.0])
        b = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        assert synchronized_mean_abs([a, b]) == pytest.approx([2.0, 2.0])
        assert synchronized_mean_abs([]) == []

    def test_reduces_sums_and_counts_over_sp_group(self):
        """A 2-rank group whose peer holds identical data doubles sums and counts."""

        class _FakeGroup:
            world_size = 2
            calls = 0

            def all_reduce(self, tensor, op=None):
                self.calls += 1
                assert op == torch.distributed.ReduceOp.SUM
                return tensor * 2

        group = _FakeGroup()
        with patch.object(easycache_hook, "_get_sequence_parallel_group", return_value=group):
            means = synchronized_mean_abs([torch.tensor([1.0, 3.0])])
        assert group.calls == 1
        assert means == pytest.approx([2.0])

    def test_decision_uses_reduced_statistics(self):
        """Rank-local shards of different magnitude reach the same decision after reduction."""

        class _SumGroup:
            world_size = 2

            def __init__(self, peer_totals):
                self.peer_totals = peer_totals

            def all_reduce(self, tensor, op=None):
                return tensor + self.peer_totals.to(tensor.dtype)

        # Peer rank contributes sum|.| = [8, 8] over 8 elements each.
        group = _SumGroup(torch.tensor([[8.0, 8.0], [8.0, 8.0]]))
        with patch.object(easycache_hook, "_get_sequence_parallel_group", return_value=group):
            means = synchronized_mean_abs([torch.zeros(8), torch.zeros(8)])
        # local (0 over 8) + peer (8 over 8) -> 8 / 16 = 0.5
        assert means == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# Backend and selector
# ---------------------------------------------------------------------------


class TestEasyCacheBackend:
    def test_init(self):
        config = DiffusionCacheConfig(easy_threshold=0.3, easy_warmup_steps=2)
        backend = EasyCacheBackend(config)
        assert backend.config.easy_threshold == 0.3
        assert backend.config.easy_warmup_steps == 2
        assert backend.enabled is False
        assert backend.is_enabled() is False

    @patch("vllm_omni.diffusion.cache.easycache.backend.apply_easy_cache_hook")
    def test_enable(self, mock_apply_hook):
        mock_pipeline = Mock()
        mock_transformer = Mock()
        mock_transformer.__class__.__name__ = "SanaVideoTransformer3DModel"
        mock_pipeline.transformer = mock_transformer

        backend = EasyCacheBackend(DiffusionCacheConfig(easy_threshold=0.2, easy_max_skip_steps=3))
        backend.enable(mock_pipeline)

        assert backend.enabled is True
        mock_apply_hook.assert_called_once()
        transformer, config = mock_apply_hook.call_args[0]
        assert transformer is mock_transformer
        assert isinstance(config, EasyCacheConfig)
        assert config.threshold == 0.2
        assert config.max_skip_steps == 3
        assert config.transformer_type == "SanaVideoTransformer3DModel"

    def test_refresh_enables_when_not_registered(self):
        pipeline = Mock()
        pipeline.transformer = _affine_model(3)
        backend = EasyCacheBackend(DiffusionCacheConfig())

        backend.refresh(pipeline, num_inference_steps=20)
        assert backend.enabled is True
        assert backend._easycache_config.num_inference_steps == 20
        assert len(list(iter_easy_cache_hooks(pipeline.transformer))) == 3

    def test_refresh_updates_steps_and_resets_state(self):
        pipeline = Mock()
        model = _affine_model(3, delta=1.0)
        pipeline.transformer = model
        backend = EasyCacheBackend(DiffusionCacheConfig(easy_warmup_steps=1, easy_cooldown_steps=1))
        backend.refresh(pipeline, num_inference_steps=10)

        x = torch.zeros(1, 4, 8)
        model(x)
        model(x)
        model(x)
        state = _head_state(model)
        assert state.step_index == 3 and state.cached_residual is not None
        hooks_before = [id(h) for _, h in iter_easy_cache_hooks(model)]

        backend.refresh(pipeline, num_inference_steps=4)
        assert backend._easycache_config.num_inference_steps == 4
        # Hooks are reused, only their state is reset.
        assert [id(h) for _, h in iter_easy_cache_hooks(model)] == hooks_before
        state = _head_state(model)
        assert state.step_index == 0 and state.cached_residual is None

        # Before refresh: steps 0/1 computed, step 2 skipped.
        assert model.total_forward_calls == 2 * 3

        # The new schedule is honored: steps 0/1 compute, step 2 skips,
        # step 3 is the cooldown and computes.
        model(x)
        model(x)
        model(x)
        assert model.total_forward_calls == 4 * 3
        model(x)
        assert model.total_forward_calls == 5 * 3

    def test_refresh_reregisters_when_transformer_replaced(self):
        pipeline = Mock()
        first = _affine_model(2)
        pipeline.transformer = first
        backend = EasyCacheBackend(DiffusionCacheConfig())
        backend.refresh(pipeline, num_inference_steps=10)

        second = _affine_model(2)
        pipeline.transformer = second
        backend.refresh(pipeline, num_inference_steps=10)
        assert backend._transformer_ids == (id(second),)
        assert len(list(iter_easy_cache_hooks(second))) == 2

    def test_enable_hooks_every_declared_dit(self):
        first = _affine_model(2)
        second = _affine_model(2)
        pipeline = _FakeMultiDiTPipeline(first, second)

        backend = EasyCacheBackend(DiffusionCacheConfig())
        backend.enable(pipeline, num_inference_steps=10)

        assert backend.enabled is True
        assert backend._transformer_ids == (id(first), id(second))
        assert len(list(iter_easy_cache_hooks(first))) == 2
        assert len(list(iter_easy_cache_hooks(second))) == 2

    def test_refresh_resets_every_declared_dit(self):
        first = _affine_model(2, delta=1.0)
        second = _affine_model(2, delta=1.0)
        pipeline = _FakeMultiDiTPipeline(first, second)
        backend = EasyCacheBackend(DiffusionCacheConfig(easy_warmup_steps=1, easy_cooldown_steps=1))
        backend.refresh(pipeline, num_inference_steps=10)

        x = torch.zeros(1, 4, 8)
        for _ in range(3):
            first(x)
            second(x)
        assert _head_state(first).step_index == 3
        assert _head_state(second).step_index == 3

        backend.refresh(pipeline, num_inference_steps=4)
        assert _head_state(first).step_index == 0 and _head_state(first).cached_residual is None
        assert _head_state(second).step_index == 0 and _head_state(second).cached_residual is None

    def test_enable_without_dit_module_stays_disabled(self):
        class _NoDiTPipeline:
            _dit_modules = ["transformer"]

        backend = EasyCacheBackend(DiffusionCacheConfig())
        backend.enable(_NoDiTPipeline())
        assert backend.enabled is False

    def test_resolve_denoise_steps_passes_through_without_hook(self):
        assert resolve_denoise_steps(Mock(spec=[]), 50) == 50

    def test_resolve_denoise_steps_uses_pipeline_hook(self):
        """H3 counts sigma points; the block stack runs one forward per interval."""

        class _SigmaPointPipeline:
            def cache_denoise_steps(self, num_inference_steps):
                return max(int(num_inference_steps) - 1, 1)

        assert resolve_denoise_steps(_SigmaPointPipeline(), 50) == 49
        assert resolve_denoise_steps(_SigmaPointPipeline(), 1) == 1

    def test_resolve_denoise_steps_rejects_non_positive(self):
        class _BrokenPipeline:
            def cache_denoise_steps(self, num_inference_steps):
                del num_inference_steps
                return 0

        with pytest.raises(ValueError, match="must be positive"):
            resolve_denoise_steps(_BrokenPipeline(), 50)

    def test_cooldown_covers_the_last_step_of_a_sigma_point_schedule(self):
        """With the resolved forward count, the final step is forced to compute."""
        model = _affine_model(3, delta=1.0)
        pipeline = Mock()
        pipeline.transformer = model
        pipeline.cache_denoise_steps = lambda steps: steps - 1
        backend = EasyCacheBackend(DiffusionCacheConfig(easy_warmup_steps=1, easy_cooldown_steps=1))

        # A 5-sigma-point request denoises 4 intervals.
        backend.refresh(pipeline, resolve_denoise_steps(pipeline, 5))
        assert backend._easycache_config.first_cooldown_step == 3

        x = torch.zeros(1, 4, 8)
        for _ in range(4):
            model(x)
        state = _head_state(model)
        # Steps 0/1 compute (the estimator needs two full steps), step 2 skips,
        # and step 3 is the cooldown, which computes because the resolved count
        # matches the run. The run then ends exactly on that count and resets.
        assert state.step_index == 0 and state.num_computed == 0
        assert model.total_forward_calls == 3 * 3

    def test_get_easy_cache_backend(self):
        for name in ("easy_cache", "easycache"):
            backend = get_cache_backend(name, {"easy_threshold": 0.05, "easy_warmup_steps": 3})
            assert isinstance(backend, EasyCacheBackend)
            assert backend.config.easy_threshold == 0.05
            assert backend.config.easy_warmup_steps == 3

    def test_diffusion_cache_config_defaults(self):
        config = DiffusionCacheConfig.from_dict({})
        assert config.easy_threshold == 0.1
        assert config.easy_warmup_steps == 5
        assert config.easy_cooldown_steps == 1
        assert config.easy_max_skip_steps == 0
