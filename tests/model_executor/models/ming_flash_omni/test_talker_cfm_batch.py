# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-request batching of the talker's CFM audio step.

The contract under test: batching requests into one CFM call must not change
any single request's output beyond floating-point reduction order. Noise is
drawn from each request's own generator before stacking, so the latent stream
a request sees is independent of who shares its batch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("x_transformers")

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (  # noqa: E402
    MingFlashOmniTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (  # noqa: E402
    CFM,
    DiT,
    MingAudioGenerator,
)
from vllm_omni.model_executor.models.ming_flash_omni.talker_request_state import (  # noqa: E402
    MingTalkerStateManager,
)

HIDDEN = 16
LATENT = 4
PATCH = 2
HIS_PATCH = 6
STEPS = 3


class _TinyAggregator(torch.nn.Module):
    """Batch-first stand-in for the real Aggregator (linear, deterministic)."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(LATENT, HIDDEN)

    def forward(self, x):
        return self.proj(x)


def _tiny_generator(seed: int = 0) -> MingAudioGenerator:
    torch.manual_seed(seed)
    dit = DiT(
        in_channels=LATENT,
        hidden_size=HIDDEN,
        depth=1,
        num_heads=2,
        llm_cond_dim=HIDDEN,
    )
    # eval() matters: DiTBlock defaults to dropout=0.1, so a train-mode module
    # is stochastic per call and serial-vs-batch comparison is meaningless.
    # Production runs the talker in eval mode.
    return MingAudioGenerator(
        config=SimpleNamespace(steps=STEPS),
        llm_config=SimpleNamespace(),
        model=None,
        cfm=CFM(dit, steps=STEPS).eval(),
        aggregator=_TinyAggregator().eval(),
        stop_head=torch.nn.Linear(HIDDEN, 2).eval(),
        audio_vae=None,
        patch_size=PATCH,
        his_patch_size=HIS_PATCH,
        latent_dim=LATENT,
        cfg_strength=2.0,
    )


def _row_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, 1, HIDDEN, generator=g)
    his_lat = torch.randn(1, HIS_PATCH, LATENT, generator=g)
    randn = torch.randn(1, PATCH, LATENT, generator=g)
    sde_rnd = torch.randn(STEPS, 1, PATCH, LATENT, generator=g)
    return hidden, his_lat, randn, sde_rnd


def test_batched_cfm_step_matches_per_request_calls():
    """Stacked rows with per-row knobs must reproduce the serial outputs."""
    gen = _tiny_generator()
    knobs = [(2.0, 0.25, 0.0), (1.5, 0.10, 0.5), (3.0, 0.40, 1.0)]
    rows = [_row_inputs(seed) for seed in (11, 22, 33)]

    serial = [
        gen.cfm_sample_step(
            hidden,
            his_lat,
            cfg=cfg,
            sigma=sigma,
            temperature=temp,
            randn_tensor=randn,
            sde_rnd=sde_rnd,
        )
        for (hidden, his_lat, randn, sde_rnd), (cfg, sigma, temp) in zip(rows, knobs)
    ]

    batched = gen.cfm_sample_step_batch(
        torch.cat([r[0] for r in rows], dim=0),
        torch.cat([r[1] for r in rows], dim=0),
        cfg=[k[0] for k in knobs],
        sigma=[k[1] for k in knobs],
        temperature=[k[2] for k in knobs],
        randn_tensor=torch.cat([r[2] for r in rows], dim=0),
        sde_rnd=torch.cat([r[3] for r in rows], dim=1),
    )

    for row, (gen_lat, embeds, stop_out) in enumerate(serial):
        torch.testing.assert_close(batched[0][row : row + 1], gen_lat, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batched[1][row : row + 1], embeds, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batched[2][row : row + 1], stop_out, rtol=1e-5, atol=1e-6)


def test_batch_knob_length_mismatch_is_rejected():
    gen = _tiny_generator()
    hidden, his_lat, randn, sde_rnd = _row_inputs(7)
    with pytest.raises(ValueError, match="per-request sampling knobs"):
        gen.cfm_sample_step_batch(
            hidden,
            his_lat,
            cfg=[2.0, 2.0],
            sigma=[0.25],
            temperature=[0.0],
            randn_tensor=randn,
            sde_rnd=sde_rnd,
        )


def _bare_talker(gen: MingAudioGenerator) -> MingFlashOmniTalkerForConditionalGeneration:
    talker = MingFlashOmniTalkerForConditionalGeneration.__new__(MingFlashOmniTalkerForConditionalGeneration)
    talker.audio_generator = gen
    talker.config = SimpleNamespace(steps=STEPS)
    talker.patch_size = PATCH
    talker.state_manager = MingTalkerStateManager()
    return talker


def _make_state(talker, req_id: str, seed: int):
    return talker.state_manager.create(
        req_id,
        his_lat=torch.zeros(1, HIS_PATCH, LATENT),
        seed=seed,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
    )


def test_request_stream_is_independent_of_batch_composition():
    """Same request + seed must produce the same latents solo and co-batched.

    This is the property that makes cross-request batching safe to enable
    unconditionally: the per-request generator is consulted before stacking,
    so a neighbor in the batch cannot shift another request's noise stream.
    """
    gen = _tiny_generator()

    solo_talker = _bare_talker(gen)
    solo_state = _make_state(solo_talker, "req-solo", seed=1234)
    solo_hidden = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(99))
    solo_gen_lat, _, solo_stop = solo_talker._talker_audio_step(solo_state, solo_hidden[-1])

    batch_talker = _bare_talker(gen)
    same = _make_state(batch_talker, "req-same", seed=1234)
    other = _make_state(batch_talker, "req-other", seed=5678)
    outputs = batch_talker._talker_audio_step_batch(
        [(same, solo_hidden[-1]), (other, torch.randn(HIDDEN, generator=torch.Generator().manual_seed(7)))]
    )

    torch.testing.assert_close(outputs[0][0], solo_gen_lat, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(outputs[0][2], solo_stop, rtol=1e-5, atol=1e-6)
    assert same.step == 1 and other.step == 1
    assert len(same.all_latents) == 1 and len(other.all_latents) == 1
    assert same.next_inputs_embed is not None and other.next_inputs_embed is not None


def test_run_pending_audio_steps_batches_active_rows():
    """Multiple audio-phase rows go through ONE batched call; bookkeeping is per-row."""
    gen = _tiny_generator()
    talker = _bare_talker(gen)
    talker.hidden_size = HIDDEN
    talker._pending_requests = [("req-a", True, 1), ("req-skip", False, 1), ("req-b", True, 1)]
    talker._pending_state_creations = set()
    talker._pending_prefill_done_updates = {}
    talker._results_queue = []
    talker._audio_queue = []
    for req_id, seed in (("req-a", 1), ("req-b", 2)):
        state = _make_state(talker, req_id, seed=seed)
        state.min_steps = 0
        state.max_steps = 100

    calls: list[int] = []
    real_batch = gen.cfm_sample_step_batch

    def counting_batch(hidden, his_lat, **kwargs):
        calls.append(int(his_lat.shape[0]))
        return real_batch(hidden, his_lat, **kwargs)

    gen.cfm_sample_step_batch = counting_batch

    talker._run_pending_audio_steps(torch.randn(3, HIDDEN))

    assert calls == [2], f"expected one batched call over 2 rows, got {calls}"
    assert [req_id for req_id, _ in talker._results_queue] == ["req-a", "req-skip", "req-b"]
    logits_by_req = dict(talker._results_queue)
    assert logits_by_req["req-skip"] is None
    for req_id in ("req-a", "req-b"):
        assert talker.state_manager.get(req_id).step == 1
        assert logits_by_req[req_id] is not None
