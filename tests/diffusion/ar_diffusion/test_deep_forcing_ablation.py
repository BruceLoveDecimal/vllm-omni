# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Each Deep Forcing switch executes only when enabled, against explicit oracles."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.ar_diffusion.test_deep_forcing import absolute_keys, df, raw

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("rebase", [False, True], ids=["absolute", "rebase"])
@pytest.mark.parametrize("pc", [False, True], ids=["window", "pc"])
def test_factorial_mechanisms_execute_only_when_enabled(rebase, pc):
    config = df.DeepForcingConfig(
        window_frames=18,
        sink_frames=3,
        rebase_sink=rebase,
        pc_budget_frames=12 if pc else 0,
        pc_recent_frames=3,
    )
    state = df.DeepForcing(config, num_layers=2, tokens_per_frame=2, temporal_dim=4)
    generator = torch.Generator().manual_seed(72)
    raw_keys = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    canonical_keys = absolute_keys(raw_keys, torch.arange(120) // 2)
    values = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    # An exact token identifier in V permits independent reconstruction of the
    # selected attention rows without using the production selection indices.
    values[0, :, 0, 0] = torch.arange(120)
    with (
        patch.object(df, "rebase_temporal_keys", wraps=df.rebase_temporal_keys) as rotate,
        patch.object(df, "participative_indices", wraps=df.participative_indices) as select,
    ):
        for start in range(0, 60, 3):
            current = slice(start * 2, (start + 3) * 2)
            query = raw_keys[:, current]
            for commit in (False, True):
                with state.forward(start, 3, commit=commit):
                    for layer in range(2):
                        key, value = state.assemble(layer, query, canonical_keys[:, current], values[:, current])
                        ids = value[0, :, 0, 0].long()
                        assert len(ids.unique()) == len(ids)
                        torch.testing.assert_close(value, values[:, ids], atol=0, rtol=0)
                        if not rebase:
                            expected_key = canonical_keys[:, ids]
                            torch.testing.assert_close(key, expected_key, atol=0, rtol=0)
                            logits = torch.einsum("bqhd,bkhd->bhqk", query, expected_key) / 8**0.5
                            expected = torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), values[:, ids])
                            actual = F.scaled_dot_product_attention(
                                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                            ).transpose(1, 2)
                            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            for history in state.history:
                torch.testing.assert_close(history.key, canonical_keys[:, history.token_ids], atol=0, rtol=0)
            assert bool(state._query_history) == pc
        assert bool(rotate.call_count) == rebase
        assert bool(select.call_count) == pc
    state.close()


@pytest.mark.parametrize("window", [12, 18])
@pytest.mark.parametrize("sink", [0, 3])
def test_fixed_window_controls_match_explicit_token_ranges(window, sink):
    state = df.DeepForcing(
        df.DeepForcingConfig(window_frames=window, sink_frames=sink),
        num_layers=1,
        tokens_per_frame=2,
        temporal_dim=4,
    )
    generator = torch.Generator().manual_seed(83)
    raw_keys = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    keys = absolute_keys(raw_keys, torch.arange(120) // 2)
    values = torch.randn(1, 120, 2, 8, generator=generator, dtype=torch.float64)
    for start in range(0, 60, 3):
        end = start + 3
        tail_start = max(sink, end - window + sink)
        ids = torch.cat((torch.arange(min(sink, start) * 2), torch.arange(min(tail_start, start) * 2, end * 2)))
        with state.forward(start, 3, commit=True):
            key, value = state.assemble(
                0, raw_keys[:, start * 2 : end * 2], keys[:, start * 2 : end * 2], values[:, start * 2 : end * 2]
            )
            torch.testing.assert_close(key, keys[:, ids], atol=0, rtol=0)
            torch.testing.assert_close(value, values[:, ids], atol=0, rtol=0)
            assert key.shape[1] <= window * 2
    state.close()


def append(memory, start):
    with memory.forward(start, 2, commit=True):
        for layer in range(memory.num_layers):
            q = raw(start)
            memory.assemble(layer, q, absolute_keys(q, torch.arange(start, start + 2).repeat_interleave(2)), q)


@pytest.mark.parametrize("sink_rebase", [False, True])
@pytest.mark.parametrize("pc_rebase", [False, True])
def test_attention_matches_explicit_virtual_positions_and_keeps_canonical_keys(sink_rebase, pc_rebase):
    config = df.DeepForcingConfig(
        window_frames=12,
        sink_frames=2,
        pc_budget_frames=8,
        pc_recent_frames=2,
        rebase_sink=sink_rebase,
        rebase_pc=pc_rebase,
    )
    memory = df.DeepForcing(config, num_layers=2, tokens_per_frame=2, temporal_dim=4)
    checked = 0
    for start in range(0, 60, 2):
        originals = [(row.key.clone() if row else None) for row in memory.history]
        with memory.forward(start, 2, commit=False):
            for layer in range(2):
                q = raw(start)
                current = absolute_keys(q, torch.arange(start, start + 2).repeat_interleave(2))
                k, v = memory.assemble(layer, q, current, q)
                rows = memory._previous(layer)
                if rows is None or layer not in memory._selection:
                    continue
                checked += 1
                sink = rows.positions < 2
                recent = rows.positions >= start
                middle = ~sink & ~recent
                middle_frames = (int(middle.sum()) + 1) // 2 if pc_rebase else 0
                target = start - middle_frames + torch.arange(int(middle.sum())) // 2
                # The sink ends right before the earliest attended non-sink frame.
                if pc_rebase and middle.any():
                    first = int(target[0])
                else:
                    first = int(rows.positions[~sink][0])
                sink_shift = max(0, first - 2) if sink_rebase else 0
                expected_k = torch.cat(
                    (
                        absolute_keys(rows.key[:, sink], torch.full_like(rows.positions[sink], sink_shift)),
                        absolute_keys(rows.key[:, middle], target - rows.positions[middle])
                        if pc_rebase
                        else rows.key[:, middle],
                        rows.key[:, recent],
                        current,
                    ),
                    1,
                )
                expected_v = torch.cat((rows.value[:, sink], rows.value[:, middle], rows.value[:, recent], q), 1)
                torch.testing.assert_close(k, expected_k)
                torch.testing.assert_close(v, expected_v)
        for layer, old in enumerate(originals):
            if old is not None:
                assert torch.equal(old, memory.history[layer].key)
        append(memory, start)
    assert checked > 0


def lingbot_like(**kwargs):
    # LingBot's native 18-frame window and 9-frame sink, with 3-frame chunks.
    config = df.DeepForcingConfig.parse(kwargs, window_frames=18, sink_frames=9, chunk_frames=3)
    return df.DeepForcing(config, num_layers=1, tokens_per_frame=1, temporal_dim=4)


@pytest.mark.parametrize("pc,pc_rebase", [(False, False), (True, False), (True, True)])
def test_shifted_sink_never_shares_a_phase_with_attended_history(monkeypatch, pc, pc_rebase):
    settings = {"rebase_sink": True, "rebase_pc": pc_rebase}
    if pc:
        settings["pc_budget_frames"] = 15
    memory = lingbot_like(**settings)
    rotated = []
    original = df.rebase_temporal_keys
    monkeypatch.setattr(df, "rebase_temporal_keys", lambda k, d, t: (rotated.append(d.clone()), original(k, d, t))[1])
    shifted = 0
    for start in range(0, 90, 3):
        with memory.forward(start, 3, commit=True):
            x = torch.randn(1, 3, 1, 8, dtype=torch.float64)
            memory.assemble(0, x, x, x)
            rows = memory._previous(0)
        if rows is None:
            continue
        if not pc:
            # The sliding window attends to the sink and the recent tail only.
            tail_start = max(9, start + 3 - 18 + 9)
            rows = rows.select(torch.where((rows.positions < 9) | (rows.positions >= tail_start))[0])
        delta = rotated.pop()
        # Only the sink (and PC-rebased middle history) is rotated.
        sink_count = int((rows.positions < 9).sum())
        assert delta.numel() >= sink_count and (pc_rebase or delta.numel() == sink_count)
        virtual = rows.positions.clone()
        virtual[: delta.numel()] += delta
        sink, rest = virtual[:sink_count], virtual[sink_count:]
        upper = rest.min() if rest.numel() else torch.tensor(start)
        assert int(sink.max()) < int(upper), (start, sink.tolist(), rest.tolist())
        # Contiguous: the sink ends right before the earliest attended frame.
        assert int(sink.max()) + 1 == int(upper)
        shifted += int(delta[0] > 0)
        if not pc or pc_rebase:
            assert torch.equal(rest, rest.sort().values)
    assert shifted > 0


def test_pc_uses_the_full_window_and_compresses_only_on_overflow():
    memory = lingbot_like(pc_budget_frames=15)
    lengths = []
    for start in range(0, 36, 3):
        with memory.forward(start, 3, commit=True):
            x = torch.randn(1, 3, 1, 8)
            key, _ = memory.assemble(0, x, x, x)
            lengths.append(key.shape[1])
    # Native attention grows to the 18-frame window; PC then keeps the
    # 15-frame budget where the native window would evict a chunk.
    assert lengths == [3, 6, 9, 12, 15, 18, 15, 18, 15, 18, 15, 18]
