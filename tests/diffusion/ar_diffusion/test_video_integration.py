# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU integration of the real cell, attention hook, timetable and memory owners.

Only external vLLM types and codec/serving imports are stubbed. Schedule,
NoisyKV allocation, Wan attention, Euler and policy code execute unchanged.
All module replacements are fixture-scoped, including on import failure.
"""

import importlib.util
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.diffusion.ar_diffusion.test_video_memory import absolute_keys, raw, vm
from tests.diffusion.ar_diffusion.test_video_retrieval import append, pose

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).parents[3]
AR = "vllm_omni.experimental.ar_diffusion"
WAN = "vllm_omni.diffusion.models.causvid_wan"


@pytest.fixture
def runtime(monkeypatch):
    def stub(name, **attrs):
        if "." in name:
            parent = name.rsplit(".", 1)[0]
            if parent not in sys.modules:
                stub(parent)
        module = ModuleType(name)
        module.__path__ = [str(ROOT / name.replace(".", "/"))]
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name):
        parent = name.rsplit(".", 1)[0]
        if parent not in sys.modules:
            stub(parent)
        spec = importlib.util.spec_from_file_location(name, ROOT / (name.replace(".", "/") + ".py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    # Prevent package initializers from importing the CUDA runtime.
    for name in ("vllm", "vllm_omni", "vllm_omni.diffusion", AR, WAN):
        stub(name)
    stub("vllm.logger", init_logger=logging.getLogger)
    stub("vllm.sequence", IntermediateTensors=type("IntermediateTensors", (), {}))

    def single_rank_indices(count, rank, world):
        assert rank == 0 and world == 1
        return 0, count

    stub("vllm.distributed.utils", get_pp_indices=single_rank_indices)
    stub("vllm.v1.core.single_type_kv_cache_manager", SlidingWindowManager=object)

    @dataclass(frozen=True)
    class SlidingWindowSpec:
        pass

    stub("vllm.v1.kv_cache_interface", SlidingWindowSpec=SlidingWindowSpec)
    stub("vllm.v1.kv_cache_spec_registry", register_kv_cache_spec=lambda **kw: lambda cls: cls)
    monkeypatch.setitem(sys.modules, AR + ".video_memory", vm)
    schedule = load(AR + ".cell_schedule")
    load(AR + ".kv_cache.config")
    load(AR + ".kv_cache.paged")
    paged = load(AR + ".kv_cache.paged_attention")
    noisy = load(AR + ".kv_cache.noisy")
    flow = load(AR + ".flow_euler")
    load("vllm_omni.diffusion.attention.conditioning")
    load(AR + ".kv_cache.conditioning")
    load(AR + ".kv_cache.first_block")
    load(AR + ".kv_cache.segmented")
    state = load(AR + ".kv_cache.state")
    executor = stub(
        AR + ".cell_executor",
        CellAdapter=object,
        ARDiffusionCellContext=object,
        resolve_pp_rank_and_group=lambda: (0, None),
        run_cell_pipeline=None,
    )
    transformer = load(WAN + ".transformer")
    cell = load(WAN + ".wan_cell")
    stub("vllm_omni.diffusion.data", DiffusionOutput=object, OmniDiffusionConfig=object)
    stub(
        "vllm_omni.diffusion.media",
        **dict.fromkeys(
            (
                "DiffusionMediaOutput",
                "VideoMediaOutput",
                "VideoTensorEncoding",
                "VideoTensorLayout",
                "VideoTensorSpec",
                "VideoValueRange",
            ),
            object,
        ),
    )
    stub("vllm_omni.diffusion.model_loader.diffusers_loader", DiffusersPipelineLoader=object)
    stub(WAN + ".async_stream_decode", start_async_stream_decode=None)
    stub(WAN + ".weights", iter_causvid_diffusers_weights=None, load_causvid_generator_state=None)
    stub("vllm_omni.diffusion.request", OmniDiffusionRequest=object)
    stub("vllm_omni.diffusion.worker.request_batch", DiffusionRequestBatch=object)
    stub(AR + ".streaming_decode", WanStreamingDecoder=object)
    pipeline = load(WAN + ".pipeline_causvid_wan")
    return SimpleNamespace(
        schedule=schedule,
        paged=paged,
        noisy=noisy,
        flow=flow,
        state=state,
        transformer=transformer,
        cell=cell,
        pipeline=pipeline,
        executor=executor,
    )


class Projection(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class Attention(nn.Module):
    num_heads = num_kv_heads = 1
    head_dim = 8

    def __init__(self):
        super().__init__()
        self.to_qkv = Projection(8, 24, bias=False, dtype=torch.float64)
        self.norm_q = self.norm_k = self.dropout = nn.Identity()
        self.to_out = nn.Linear(8, 8, bias=False, dtype=torch.float64)

    @staticmethod
    def attn(q, k, v, _mask):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def rotary_embedding(x, cos, sin):
        pairs = torch.view_as_complex(x.unflatten(-1, (-1, 2)).contiguous())
        return torch.view_as_real(pairs * torch.complex(cos, sin)).flatten(-2)


class TinyTransformer:
    """Small weights around the production Wan attention hook; no checkpoint I/O."""

    start_layer = 0
    end_layer = num_layers = local_num_layers = 2
    is_layer_group_first = is_layer_group_last = True
    layer_groups = 1
    head_dim = dim = 8
    patch_size = (1, 1, 1)

    def __init__(self, runtime):
        self.runtime = runtime
        with torch.random.fork_rng():
            torch.manual_seed(55)
            self.layers = [Attention(), Attention()]
        self.fail = None
        self.observed = []

    def prepare_cell(self, _prompt):
        pass

    def seq_len_for_latent(self, shape):
        return shape[2] * shape[3] * shape[4]

    def embed(self, latent):
        return latent.flatten(2).transpose(1, 2)

    def block(self, index, hidden, *, chunk, step, latent, timestep, kv_ctx):
        if self.fail == index:
            raise RuntimeError("block failure")
        self.observed.append(kv_ctx)
        positions = torch.arange(chunk * 2, chunk * 2 + 2).repeat_interleave(2)
        angle = torch.zeros(1, 4, 1, 4, dtype=torch.float64)
        angle[..., :2] = positions[None, :, None, None] * torch.tensor([1, 0.01])
        rotary = angle.cos(), angle.sin()
        return hidden + self.runtime.transformer._wan_self_attn_with_kv(self.layers[index], hidden, rotary, kv_ctx)

    def unembed(self, tokens, **kw):
        if self.fail == "head":
            raise RuntimeError("head failure")
        return tokens.transpose(1, 2).reshape_as(kw["latent"])


def cell(runtime, config):
    return runtime.cell.WanCell(
        TinyTransformer(runtime),
        sampler=runtime.flow.FlowEuler(2),
        prompt_embeds=torch.empty(1, 1, 8),
        latent_shape=(1, 8, 2, 1, 2),
        seed=17,
        device=torch.device("cpu"),
        dtype=torch.float64,
        video_memory=config,
    )


def tick(adapter, req, chunk, step):
    hidden = None
    for layer in range(2):
        hidden = adapter.forward([(req, (chunk, step, layer))], [None], hidden=hidden)
    return hidden


def test_causvid_attention_hook_matches_explicit_dense_qkv(runtime):
    attn = Attention()
    x = raw(0)[:, :, 0]
    q, k, v = nn.functional.linear(x, attn.to_qkv.weight).chunk(3, -1)
    expected = F.softmax(q @ k.transpose(1, 2) / 8**0.5, -1) @ v
    expected = nn.functional.linear(expected, attn.to_out.weight)
    actual = runtime.transformer._wan_self_attn_with_kv(attn, x, None, None)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("pc", [False, True])
def test_causvid_real_cell_schedule_clean_only_bounded_and_isolated(runtime, pc):
    config = vm.VideoMemoryConfig(window_frames=8, sink_frames=2, pc_budget_frames=6 if pc else 0, pc_recent_frames=2)
    adapter = cell(runtime, config)
    control = cell(runtime, config)
    schedule = runtime.schedule.CellSchedule(
        chunks=20,
        num_denoise_steps=2,
        step_replicas=1,
        layer_groups=1,
        ordering=runtime.schedule.Ordering.SELF_FORCING,
        kv_history_chunks=0,
        blocks=2,
    )
    plan = runtime.schedule.build_cell_plan(schedule)
    spec = runtime.noisy.ARDiffusionNoisyKVSpec(2, 1, 8, 4, 4, 1)
    cache = runtime.noisy.NoisyKVCache(
        spec, dtype=torch.float64, device=torch.device("cpu"), layer_groups=1, max_batch_size=1
    )
    noisy = runtime.noisy.NoisyKVState(cache)
    noisy.begin_request("A", plan, chunk_tokens=4)
    # Poison the unused legacy pool: any accidental history read makes output
    # non-finite; a duplicate write would overwrite the sentinel.
    for pool in (*cache.pool.k_pools, *cache.pool.v_pools):
        pool.fill_(float("nan"))
    hidden = None
    for slot in range(plan.num_slots):
        task = plan.task(slot, 0)
        if task is None:
            continue
        chunk, step, layer = task
        assert not plan.sources(task, 0)
        tasks = (("A", task),)
        contexts = noisy.prepare(tasks)
        hidden = adapter.forward(tasks, contexts, hidden=hidden)
        assert all(torch.isfinite(tensor).all() for tensor in hidden.values())
        noisy.publish(tasks)
        noisy.evict(slot)
        assert noisy.resident_versions == 0
        memory = adapter._memories["A"]
        expected_frame = 2 * (chunk + int(step == 2 and layer == 1))
        assert memory.next_frame == expected_frame
        for rows in memory.history:
            if rows is not None:
                assert rows.key.shape[1] <= 16
        if layer == 1:
            hidden = None  # same boundary behavior as the serial executor
    for pool in (*cache.pool.k_pools, *cache.pool.v_pools):
        assert torch.isnan(pool).all()
    noisy.end_request("A")
    # A different request starts at frame zero, independent of A's 40 frames.
    for step in range(3):
        actual = tick(adapter, "B", 0, step)
        reference = tick(control, "B", 0, step)
        torch.testing.assert_close(actual["latent"], reference["latent"])
    assert adapter._memories["A"] is not adapter._memories["B"]
    assert all(isinstance(ctx, vm.VideoMemoryLayer) for ctx in adapter.transformer.observed)
    memories = list(adapter._memories.values())
    adapter.close()
    assert all(m._closed and all(r is None for r in m.history) for m in memories)
    assert not adapter._transactions and not adapter._memories
    assert len(adapter.finished["A"]) == 20


@pytest.mark.parametrize("failure", [1, "head"])
def test_causvid_cell_failure_closes_pending_transaction(runtime, failure):
    adapter = cell(runtime, vm.VideoMemoryConfig(window_frames=6, sink_frames=2))
    tick(adapter, "A", 0, 0)
    memory = adapter._memories["A"]
    adapter.transformer.fail = failure
    with pytest.raises(RuntimeError, match="failure"):
        tick(adapter, "A", 0, 2)
    assert memory.next_frame == 0 and memory._closed and not memory._pending
    assert not adapter._transactions and not adapter._memories


def test_causvid_policy_off_matches_first_chunk_and_uses_no_owner(runtime):
    default = cell(runtime, None)
    enabled = cell(runtime, vm.VideoMemoryConfig(window_frames=6, sink_frames=2))
    for step in range(3):
        a = tick(default, "A", 0, step)
        b = tick(enabled, "A", 0, step)
        torch.testing.assert_close(a["latent"], b["latent"])
    assert not default._memories


def pipeline(runtime, config=None):
    p = runtime.pipeline.CausVidWanPipeline.__new__(runtime.pipeline.CausVidWanPipeline)
    nn.Module.__init__(p)
    p._video_memory_config = config
    p.step_replicas = p.layer_groups = 1
    p.max_history_chunks = 6
    p._default_shift, p._default_denoise = 8.0, 2
    p.transformer = SimpleNamespace(num_layers=2, local_num_layers=2, num_heads=1, head_dim=8)
    p.max_chunk_tokens = p.block_size = 4
    p.od_config = SimpleNamespace(dtype=torch.float64)
    return p


@pytest.mark.parametrize("enabled", [False, True])
def test_causvid_pipeline_plan_and_memory_spec(runtime, enabled):
    config = vm.VideoMemoryConfig(window_frames=6, sink_frames=2) if enabled else None
    p = pipeline(runtime, config)
    req = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}, num_inference_steps=None))
    plan, _, _ = p._plan_for(req)
    assert plan.schedule.kv_history_chunks == (0 if enabled else 6)
    spec = p.ar_diffusion_noisy_kv_spec()
    assert spec.policy_bytes_per_request == (config.reservation_bytes(2, 4, 1, 8, 8) if enabled else 0)
    assert spec.max_history_chunks == (1 if enabled else 6)
    if enabled:
        req.sampling_params.extra_args = {"kv_history_chunks": 2}
        with pytest.raises(ValueError, match="owns history"):
            p._plan_for(req)


@pytest.mark.parametrize("batch", [1, 2])
def test_noisy_kv_reserves_policy_bytes_before_pool_allocation(runtime, batch):
    spec = runtime.noisy.ARDiffusionNoisyKVSpec(2, 1, 8, 4, 4, 1, policy_bytes_per_request=1234)
    args = dict(dtype=torch.float32, device=torch.device("cpu"), layer_groups=1, max_batch_size=batch)
    baseline = runtime.noisy.NoisyKVCache(replace(spec, policy_bytes_per_request=0), **args)
    total = baseline.reserved_bytes + batch * 1234
    cache = runtime.noisy.NoisyKVCache(spec, available_bytes=total, **args)
    assert cache.reserved_bytes == total
    assert cache.policy_reserved_bytes == batch * 1234
    assert cache.capacity == baseline.capacity
    with pytest.raises(RuntimeError, match="budget"):
        runtime.noisy.NoisyKVCache(spec, available_bytes=total - 1, **args)
    with pytest.raises(RuntimeError, match="budget"):
        runtime.noisy.NoisyKVCache(spec, available_bytes=batch * 1234, **args)


@pytest.mark.parametrize("bad", [-1, 0.5, True])
def test_noisy_spec_rejects_bad_policy_reservation(runtime, bad):
    with pytest.raises(ValueError, match="policy_bytes_per_request"):
        runtime.noisy.ARDiffusionNoisyKVSpec(2, 1, 8, 4, 4, 1, policy_bytes_per_request=bad)


@pytest.mark.parametrize("keep_cross", [(), ("text",)])
def test_runner_state_memory_is_per_session_branch_and_reset(runtime, keep_cross):
    cache = SimpleNamespace(
        kv_branches=[SimpleNamespace(name="positive"), SimpleNamespace(name="negative")],
        cross_attention_lengths={"text": 1},
        release_cross_attention=lambda *args: None,
        end_request=lambda *args: None,
        retain_cross_attention=lambda *args: None,
        begin_request=lambda name: SimpleNamespace(request_id=name),
    )

    def owner(name):
        return runtime.state.ARDiffusionKVState(
            cache, name, {b.name: SimpleNamespace(request_id=name + b.name) for b in cache.kv_branches}, num_layers=2
        )

    a, b = owner("A"), owner("B")
    config = vm.VideoMemoryConfig(window_frames=6, sink_frames=2)

    def memory(owner, branch="positive", cfg=config):
        return owner.video_memory(branch, cfg, tokens_per_frame=2, temporal_dim=4)

    original = memory(a)
    assert memory(a) is original
    assert memory(b) is not original and memory(a, "negative") is not original
    with pytest.raises(ValueError, match="cannot change"):
        memory(a, cfg=replace(config, window_frames=8))
    a.reset(keep_cross_attention=keep_cross)
    assert original._closed and memory(a) is not original
    a.close()
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        memory(a)
    assert not memory(b)._closed


@pytest.mark.parametrize("ratio", [0, 0.25, 1])
@pytest.mark.parametrize("rebase", [False, True])
def test_combined_long_rollout_deduplicates_and_bounds_all_layers(ratio, rebase):
    config = vm.VideoMemoryConfig(
        window_frames=12,
        sink_frames=2,
        retrieval_frames=2,
        bank_bytes=1 << 20,
        pc_budget_frames=8,
        pc_recent_frames=2,
        world_keep_ratio=ratio,
        rebase_sink=rebase,
        rebase_retrieval=rebase,
    )
    memory = vm.VideoMemory(config, num_layers=2, tokens_per_frame=2, temporal_dim=4)
    retrieved_count = 0
    for start in range(0, 100, 2):
        memory.set_pose(pose(0))
        with memory.forward(start, 2, commit=False):
            chunks = memory._retrieved
            retrieved_count += len(chunks)
            for layer in range(2):
                k, v = memory.assemble(
                    layer,
                    raw(start),
                    absolute_keys(raw(start), torch.arange(start, start + 2).repeat_interleave(2)),
                    raw(start),
                )
                assert k.shape == v.shape and k.shape[1] <= config.window_frames * 2
                previous = memory._previous(layer)
                if previous is not None:
                    for chunk in chunks:
                        assert not torch.isin(chunk.layers[layer].positions, previous.positions).any()
                if chunks:
                    assert all(chunk.start + chunk.frames <= start for chunk in chunks)
        append(memory, start, camera=pose(0))
        assert memory.bank_nbytes <= config.bank_bytes
        assert all(row.key.shape[1] <= (config.window_frames - config.retrieval_frames) * 2 for row in memory.history)
    assert retrieved_count > 0


@pytest.mark.parametrize(
    "settings",
    [
        {"enforce_eager": False},
        {"pipeline_parallel_size": 2},
        {"tensor_parallel_size": 2},
        {"ulysses_degree": 2},
        {"ring_degree": 2},
        {"stage": {"step_replicas": 2}},
        {"stage": {"max_batch_size": 2}},
        {"memory": {"retrieval_frames": 2, "bank_bytes": 1000}},
    ],
)
def test_causvid_rejects_unsupported_policy_before_loading_weights(runtime, settings):
    od = SimpleNamespace(
        model_config={"video_memory": settings.get("memory", {})},
        ar_diffusion_stage_config=settings.get("stage", {}),
        enforce_eager=settings.get("enforce_eager", True),
        parallel_config=SimpleNamespace(
            **{
                k: settings.get(k, 1)
                for k in ("pipeline_parallel_size", "tensor_parallel_size", "ulysses_degree", "ring_degree")
            }
        ),
    )
    with pytest.raises(ValueError, match="CausVid"):
        runtime.pipeline.CausVidWanPipeline(od_config=od)


def test_causvid_real_stage_hook_uses_policy_and_restores_context(runtime):
    stage_cls = runtime.transformer.StageWanTransformer
    stage = stage_cls.__new__(stage_cls)
    nn.Module.__init__(stage)
    stage.start_layer, stage.end_layer = 0, 1
    stage._cell_kv_ctx = None
    stage._text = torch.zeros(1, 1, 8)
    stage._embeddings, stage._rotary = {}, {}
    stage._embedding = lambda *args, **kwargs: (None, None, None)
    attn = Attention()
    attn.forward = lambda hs, rotary, meta: hs

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = attn

        def forward(self, hs, enc, timestep, rotary, *args):
            return self.attn1(hs, rotary, None)

    wan = nn.Module()
    wan.blocks = nn.ModuleList([Block()])
    # Temporal frequencies, followed by independent height/width frequencies.
    angles = torch.arange(32, dtype=torch.float64)[:, None] * torch.tensor([1, 1, 0.01, 0.01, 1, 1, 1, 1])
    wan.rope = SimpleNamespace(
        patch_size=(1, 1, 1), max_seq_len=32, attention_head_dim=8, freqs_cos=angles.cos(), freqs_sin=angles.sin()
    )
    stage._wan = wan
    stage._install_cell_attn_hooks()
    config = vm.VideoMemoryConfig(window_frames=6, sink_frames=2, rebase_sink=False)
    memory = vm.VideoMemory(config, num_layers=1, tokens_per_frame=2, temporal_dim=4)
    latent = torch.zeros(1, 8, 2, 1, 2, dtype=torch.float64)
    hidden = raw(0)[:, :, 0]
    for chunk in range(2):
        with memory.forward(chunk * 2, 2, commit=True):
            actual = stage.block(
                0,
                hidden,
                chunk=chunk,
                step=2,
                latent=latent,
                timestep=torch.tensor(0),
                kv_ctx=vm.VideoMemoryLayer(memory, 0),
            )
            q, k, v = attn.to_qkv(hidden)[0].chunk(3, -1)
            q, k, v = (x.unsqueeze(2) for x in (q, k, v))
            # Reference uses the real absolute RoPE but explicit dense attention.
            cos, sin = stage._rotary_emb(chunk, latent)
            q, k = (attn.rotary_embedding(x, cos, sin) for x in (q, k))
            if chunk:
                k = torch.cat((memory.history[0].key, k), 1)
                v = torch.cat((memory.history[0].value, v), 1)
            scores = torch.einsum("bthd,bshd->bhts", q, k) / 8**0.5
            expected = torch.einsum("bhts,bshd->bthd", scores.softmax(-1), v).flatten(2)
            torch.testing.assert_close(actual, attn.to_out(expected))
        assert stage._cell_kv_ctx is None

    # Exercise cleanup on an actual block exception.
    def fail(*args, **kwargs):
        raise RuntimeError("attention failure")

    attn.forward = fail
    with pytest.raises(RuntimeError, match="attention failure"), memory.forward(4, 2, commit=True):
        stage.block(
            0, hidden, chunk=2, step=2, latent=latent, timestep=torch.tensor(0), kv_ctx=vm.VideoMemoryLayer(memory, 0)
        )
    assert stage._cell_kv_ctx is None and memory.next_frame == 4


@pytest.mark.parametrize("ratio", [0, 0.25, 1])
def test_combined_attention_matches_explicit_virtual_positions_and_keeps_canonical_keys(ratio):
    config = vm.VideoMemoryConfig(
        window_frames=12,
        sink_frames=2,
        retrieval_frames=2,
        bank_bytes=1 << 20,
        pc_budget_frames=8,
        pc_recent_frames=2,
        world_keep_ratio=ratio,
        rebase_sink=True,
        rebase_retrieval=True,
    )
    memory = vm.VideoMemory(config, num_layers=2, tokens_per_frame=2, temporal_dim=4)
    checked = 0
    for start in range(0, 60, 2):
        memory.set_pose(pose(0))
        originals = [(row.key.clone() if row else None) for row in memory.history]
        bank = {k: tuple(row.key.clone() for row in c.layers) for k, c in memory.bank.items()}
        with memory.forward(start, 2, commit=False):
            for layer in range(2):
                q = raw(start)
                current = absolute_keys(q, torch.arange(start, start + 2).repeat_interleave(2))
                k, v = memory.assemble(layer, q, current, q)
                rows = memory._previous(layer)
                if rows is None or not memory._retrieved:
                    continue
                checked += 1
                sink = rows.positions < 2
                recent = rows.positions >= start
                middle = ~sink & ~recent
                # Assign independent packed regions ending just before current.
                middle_frames = (int(middle.sum()) + 1) // 2
                count = sum(chunk.layers[layer].positions.numel() for chunk in memory._retrieved)
                retrieval_frames = (count + 1) // 2
                sink_start = start - middle_frames - retrieval_frames - 2
                # Sink targets are its original 0,1 plus the shift.
                parts_k = [absolute_keys(rows.key[:, sink], torch.full_like(rows.positions[sink], sink_start))]
                parts_v = [rows.value[:, sink]]
                offset = 0
                for chunk in memory._retrieved:
                    r = chunk.layers[layer]
                    target = (
                        start - middle_frames - retrieval_frames + (offset + torch.arange(r.positions.numel())) // 2
                    )
                    parts_k.append(absolute_keys(r.key, target - r.positions))
                    parts_v.append(r.value)
                    offset += r.positions.numel()
                target = start - middle_frames + torch.arange(int(middle.sum())) // 2
                parts_k.extend(
                    (absolute_keys(rows.key[:, middle], target - rows.positions[middle]), rows.key[:, recent], current)
                )
                parts_v.extend((rows.value[:, middle], rows.value[:, recent], q))
                expected_k, expected_v = torch.cat(parts_k, 1), torch.cat(parts_v, 1)
                torch.testing.assert_close(k, expected_k)
                torch.testing.assert_close(v, expected_v)
                actual = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                scores = torch.einsum("bthd,bshd->bhts", q, expected_k) / 8**0.5
                expected = torch.einsum("bhts,bshd->bhtd", scores.softmax(-1), expected_v)
                torch.testing.assert_close(actual, expected)
        for layer, old in enumerate(originals):
            if old is not None:
                assert torch.equal(old, memory.history[layer].key)
        for key, old in bank.items():
            assert all(torch.equal(a, b.key) for a, b in zip(old, memory.bank[key].layers))
        append(memory, start, camera=pose(0))
    assert checked > 0
