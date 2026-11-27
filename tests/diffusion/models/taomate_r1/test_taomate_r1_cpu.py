# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the TaoMate-R1 streaming port."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.diffusion.models.taomate_h3.attention import _STREAM_CONTEXT, StreamMode
from vllm_omni.diffusion.models.taomate_h3.geometry import CanvasGeometry
from vllm_omni.diffusion.models.taomate_h3.lora import TaoMateLoRAError, prepare_taomate_lora, taomate_lora_targets
from vllm_omni.diffusion.models.taomate_h3.schedule import student_sigmas
from vllm_omni.diffusion.models.taomate_r1.chunks import (
    align_frame_count,
    bank_for_chunk,
    chunk_audio_ids,
    compact_chunk_layout,
    plan_session,
)
from vllm_omni.diffusion.models.taomate_r1.conditioning import (
    build_encoder_inputs,
    condition_labels,
    extract_reference_images,
    reference_size,
)
from vllm_omni.diffusion.models.taomate_r1.lora import read_taomate_r1_banks
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# ----------------------------------------------------------------------------
# session geometry and chunk plan


@pytest.mark.parametrize(
    ("num_frames", "aligned", "video_latents", "chunk_latents", "audio_latents"),
    [
        (107, 107, 32, [12, 10, 10], 178),
        (124, 124, 37, [12, 10, 10, 5], 207),
        (240, 243, 72, [12, 10, 10, 10, 10, 10, 10], 405),
        (22, 22, 7, [7], 37),
    ],
)
def test_plan_follows_the_release_chunking(num_frames, aligned, video_latents, chunk_latents, audio_latents) -> None:
    plan = plan_session(num_frames)
    assert plan.frame_count == aligned == align_frame_count(num_frames)
    assert plan.video_latent_t == video_latents
    assert plan.audio_latent_t == audio_latents
    assert [chunk.video_latent_count for chunk in plan.chunks] == chunk_latents
    # Chunks tile the session without gaps on every axis.
    assert plan.chunks[0].frame_start == plan.chunks[0].video_latent_start == plan.chunks[0].audio_latent_start == 0
    for left, right in zip(plan.chunks, plan.chunks[1:]):
        assert left.frame_stop == right.frame_start
        assert left.video_latent_stop == right.video_latent_start
        assert left.audio_latent_stop == right.audio_latent_start
    last = plan.chunks[-1]
    assert (last.frame_stop, last.video_latent_stop, last.audio_latent_stop) == (aligned, video_latents, audio_latents)


def test_three_chunks_select_cold_warm_warm_and_skip_the_terminal_commit() -> None:
    plan = plan_session(39 + 34 + 34)
    assert [chunk.bank for chunk in plan.chunks] == ["cold", "warm", "warm"]
    assert [plan.commits_clean_history(chunk.index) for chunk in plan.chunks] == [True, True, False]
    assert bank_for_chunk(0) == "cold" and bank_for_chunk(7) == "warm"
    with pytest.raises(IndexError):
        plan.commits_clean_history(3)


def test_chunk_audio_ids_are_channel_major() -> None:
    plan = plan_session(107)
    ids = chunk_audio_ids(plan.chunks[1], audio_latent_t=plan.audio_latent_t)
    assert ids.tolist() == list(range(65, 122)) + list(range(178 + 65, 178 + 122))


# ----------------------------------------------------------------------------
# compact chunk layout


def _session_packed(*, text_len=7, canvas=CanvasGeometry(height=864, width=480), refs=((6, 4), (4, 8)), frames=107):
    plan = plan_session(frames)
    blocks = [{"kind": "image", "latent_h": h, "latent_w": w} for h, w in refs]
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=text_len,
        latent_t=plan.video_latent_t,
        latent_h=canvas.latent_h,
        latent_w=canvas.latent_w,
        audio_t=plan.audio_latent_t,
        ref_blocks=blocks,
    )
    reference_rows = sum((h // 2) * (w // 2) for h, w in refs)
    return plan, packed, reference_rows, canvas


def test_compact_layout_gathers_global_positions_for_every_chunk() -> None:
    plan, packed, reference_rows, canvas = _session_packed()
    text_tags = torch.tensor([1, 1, 0, 0, 1, 1, 1])
    packed["token_tags"][packed["text_pos"]] = text_tags
    img_pos = packed["img_pos"].view(-1)
    for chunk in plan.chunks:
        layout = compact_chunk_layout(
            packed,
            reference_rows=reference_rows,
            chunk=chunk,
            frame_rows=canvas.frame_rows,
            audio_latent_t=plan.audio_latent_t,
        )
        nt, nr = 7, reference_rows
        na = 2 * chunk.audio_latent_count
        nv = chunk.video_latent_count * canvas.frame_rows
        used = nt + nr + na + nv
        assert layout["seq_len"] % 64 == 0 and layout["seq_len"] >= used
        assert layout["cu_seqlens"].tolist() == [0, used, layout["seq_len"]]
        # Text and references are the same rows in every chunk; targets move along the session.
        grid = packed["img_position_ids"]
        assert torch.equal(layout["img_position_ids"][:nt], grid[packed["text_pos"]])
        assert torch.equal(layout["img_position_ids"][nt : nt + nr], grid[img_pos[:nr]])
        video_global = img_pos[nr:][
            chunk.video_latent_start * canvas.frame_rows : chunk.video_latent_stop * canvas.frame_rows
        ]
        assert torch.equal(layout["img_position_ids"][nt + nr + na : used], grid[video_global])
        assert torch.equal(layout["token_tags"][:nt], text_tags)
        assert (layout["token_tags"][used:] == -1).all()
        assert layout["update_mask"].tolist() == [False] * nr + [True] * nv
        assert layout["audio_pos"].tolist() == list(range(nt + nr, nt + nr + na))
        assert layout["video_spans"][0]["start"] == nt + nr + na


def test_compact_layout_rejects_a_mismatched_canvas() -> None:
    plan, packed, reference_rows, canvas = _session_packed()
    with pytest.raises(ValueError, match="latent_grid"):
        compact_chunk_layout(
            packed,
            reference_rows=reference_rows,
            chunk=plan.chunks[0],
            frame_rows=canvas.frame_rows + 1,
            audio_latent_t=plan.audio_latent_t,
        )


# ----------------------------------------------------------------------------
# references


def test_reference_size_follows_the_release_rule() -> None:
    from vllm_omni.diffusion.models.taomate_r1.conditioning import resize_references, validate_reference_short_edge

    assert [image.size for image in resize_references([Image.new("RGB", (1024, 1536))], 768)] == [(768, 1152)]
    assert validate_reference_short_edge(768) == 768
    for bad in (100, 770, 8192):
        with pytest.raises(ValueError, match="multiple of 32"):
            validate_reference_short_edge(bad)
    assert reference_size(1024, 1536) == (2048, 3072)
    assert reference_size(1080, 1920) == (2048, 3648)
    assert reference_size(777, 513) == (3104, 2048)


def test_references_keep_order_and_reject_other_modalities() -> None:
    first, second = Image.new("RGB", (40, 30), "red"), Image.new("RGB", (30, 40), "blue")
    images = extract_reference_images({"image": [first, second]})
    assert [image.size for image in images] == [(40, 30), (30, 40)]
    assert condition_labels(2) == [("image", 1), ("image", 2)]
    with pytest.raises(OmniClientError, match="at least one image"):
        extract_reference_images({})
    with pytest.raises(OmniClientError, match="unsupported inputs"):
        extract_reference_images({"image": [first], "video": ["clip.mp4"]})
    with pytest.raises(OmniClientError, match="1 to 9"):
        extract_reference_images({"image": [first] * 10})


def test_encoder_inputs_describe_the_whole_session() -> None:
    plan = plan_session(107)
    images = [Image.new("RGB", (64, 96), "red"), Image.new("RGB", (96, 64), "blue")]
    prepared = build_encoder_inputs("<Picture 2> beside <Picture 1>", images, plan=plan, height=1344, width=768)
    assert prepared.condition_labels == [("image", 1), ("image", 2)]
    assert prepared.media.task == "ref2va"
    assert (prepared.media.latent_t, prepared.media.audio_t) == (plan.video_latent_t, plan.audio_latent_t)
    assert [tuple(image.shape) for image in prepared.media.images] == [(96, 64, 3), (64, 96, 3)]
    with pytest.raises(OmniClientError):
        build_encoder_inputs("  ", images, plan=plan, height=1344, width=768)


# ----------------------------------------------------------------------------
# LoRA banks


class _TinyArch:
    num_layers = 1
    token_refiner_num_layers = 1
    num_attention_heads = 2
    attention_head_dim = 4
    hidden_size = 8
    ffn_hidden_size = 6


def _tiny_targets() -> tuple[str, ...]:
    return taomate_lora_targets(num_blocks=1, num_refiner_blocks=1)


def _write_banks(root: Path, shapes: dict[str, tuple[int, int]], *, rank=4, alpha=4.0, seeds=(1, 2)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for role, seed in zip(("cold", "warm"), seeds):
        generator = torch.Generator().manual_seed(seed)
        tensors = {}
        for target, (out_features, in_features) in shapes.items():
            tensors[f"{target}.lora_a"] = torch.randn(rank, in_features, generator=generator)
            tensors[f"{target}.lora_b"] = torch.randn(out_features, rank, generator=generator)
        (root / role).mkdir(exist_ok=True)
        save_file(tensors, str(root / role / "adapter_model.safetensors"))
        (root / role / "config.json").write_text(json.dumps({"rank": rank, "alpha": alpha}))
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_variant": "Ref2VA",
                "rank": rank,
                "alpha": alpha,
                "banks": {"cold": "cold/adapter_model.safetensors", "warm": "warm/adapter_model.safetensors"},
            }
        )
    )
    return root


def test_bank_config_resolves_both_adapter_directories(tmp_path) -> None:
    shapes = {target: (4, 8) for target in _tiny_targets()}
    root = _write_banks(tmp_path / "r1", shapes)
    banks = read_taomate_r1_banks(root)
    assert banks == {"cold": (root / "cold").resolve(), "warm": (root / "warm").resolve()}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda config: config.update(model_variant="FL2VA"), "Ref2VA"),
        (lambda config: config["banks"].pop("warm"), "exactly the banks"),
        (lambda config: config["banks"].update(warm="../outside/adapter_model.safetensors"), "inside"),
        (lambda config: config["banks"].update(warm="warm/other.safetensors"), "adapter_model.safetensors"),
        (lambda config: config["banks"].update(warm="cold/adapter_model.safetensors"), "distinct"),
        (lambda config: config.update(rank=8), "differs from the root"),
    ],
)
def test_bank_config_rejects_unrelated_layouts(tmp_path, mutate, match) -> None:
    shapes = {target: (4, 8) for target in _tiny_targets()}
    root = _write_banks(tmp_path / "r1", shapes)
    config = json.loads((root / "config.json").read_text())
    mutate(config)
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(TaoMateLoRAError, match=match):
        read_taomate_r1_banks(root)


def test_bank_config_requires_the_root_directory(tmp_path) -> None:
    root = _write_banks(tmp_path / "r1", {target: (4, 8) for target in _tiny_targets()})
    with pytest.raises(TaoMateLoRAError, match="TaoMate-R1 root"):
        read_taomate_r1_banks(root / "cold")


def _r1_pipeline(**attributes):
    from vllm_omni.diffusion.models.taomate_r1.pipeline import TaoMateR1Pipeline

    pipeline = TaoMateR1Pipeline.__new__(TaoMateR1Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._tm_lora_manager = None
    pipeline._r1_active_bank = "cold"
    pipeline._r1_bank_dirs = {}
    pipeline._r1_prepared = {}
    pipeline._r1_sessions = {}
    pipeline._r1_dump_latents_dir = None
    pipeline._turbo_lora_specs = {}
    pipeline._native_distilled_lora_adapter_ids = set()
    for name, value in attributes.items():
        setattr(pipeline, name, value)
    return pipeline


@contextmanager
def _bank_manager_case(tmp_path, monkeypatch):
    """Two real TaoMate banks behind the production LoRA manager on tiny vLLM linears."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: lambda _layer, x, weight, bias=None: torch.nn.functional.linear(x, weight, bias),
    )
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(5)
        transformer = nn.Module()
        transformer.arch = _TinyArch()
        layers = {}
        for target in _tiny_targets():
            parent = transformer
            *path, leaf = target.split(".")
            for part in path:
                if not hasattr(parent, part):
                    parent.add_module(part, nn.Module())
                parent = getattr(parent, part)
            if leaf == "qkv_proj":
                layer = linear.QKVParallelLinear(8, 4, 2, bias=False, params_dtype=torch.bfloat16)
            elif leaf == "fc1":
                layer = linear.MergedColumnParallelLinear(8, [6, 6], bias=False, params_dtype=torch.bfloat16)
            elif leaf == "out_proj":
                layer = linear.RowParallelLinear(8, 8, bias=False, params_dtype=torch.bfloat16)
            else:
                layer = linear.RowParallelLinear(6, 8, bias=False, params_dtype=torch.bfloat16)
            layer.weight.data.normal_()
            parent.add_module(leaf, layer)
            layers[f"transformer.{target}"] = layer
        shapes = {target.removeprefix("transformer."): tuple(layer.weight.shape) for target, layer in layers.items()}
        root = _write_banks(tmp_path / "r1", shapes, rank=8, alpha=8.0)
        bank_dirs = read_taomate_r1_banks(root)
        prepared = {}
        for role, bank_dir in bank_dirs.items():
            model, peft, _ = prepare_taomate_lora(bank_dir, transformer=transformer, device=torch.device("cpu"))
            prepared[role] = (model, peft)
        pipeline = _r1_pipeline(transformer=transformer, _r1_bank_dirs=bank_dirs, _r1_prepared=dict(prepared))
        manager = DiffusionLoRAManager(
            pipeline, torch.device("cpu"), torch.bfloat16, lora_path=str(root), lora_scale=0.0
        )
        try:
            yield pipeline, manager, prepared, layers
        finally:
            manager.close()


def test_manager_pins_both_banks_and_follows_the_pipelines_active_bank(tmp_path, monkeypatch) -> None:
    with _bank_manager_case(tmp_path, monkeypatch) as (pipeline, manager, prepared, layers):
        assert pipeline._tm_lora_manager is manager and pipeline._r1_prepared == {}
        bank_ids = {role: request.lora_int_id for role, request in manager._fixed_bank_requests.items()}
        assert set(bank_ids) == {"cold", "warm"} and len(set(bank_ids.values())) == 2
        assert set(bank_ids.values()) <= manager._pinned_adapters
        assert all(not manager.remove_adapter(adapter_id) for adapter_id in bank_ids.values())
        name = "transformer.blocks.0.attn.out_proj"
        x = torch.randn(3, 8).bfloat16()

        def expected(role: str) -> torch.Tensor:
            weights = prepared[role][0].loras[name]
            out = torch.nn.functional.linear(x, layers[name].weight)
            return out.addmm_(x @ weights.lora_a.T, weights.lora_b.T, alpha=weights.scaling)

        layer = pipeline.get_submodule(name)
        with torch.inference_mode():
            # The worker's per-step reselection keeps whichever bank the pipeline selected,
            # whatever the request carries.
            manager.set_request_adapters(None, 1.0)
            assert manager._active_adapter_id == bank_ids["cold"]
            torch.testing.assert_close(layer(x)[0], expected("cold"), rtol=0, atol=0)
            pipeline._select_bank("warm")
            assert manager._active_adapter_id == bank_ids["warm"]
            torch.testing.assert_close(layer(x)[0], expected("warm"), rtol=0, atol=0)
            manager.set_request_adapters(None, 1.0)
            assert manager._active_adapter_id == bank_ids["warm"]
            pipeline._select_bank("cold")
            torch.testing.assert_close(layer(x)[0], expected("cold"), rtol=0, atol=0)
            assert not torch.equal(expected("cold"), expected("warm"))


def test_lora_hooks_map_requests_to_their_bank(tmp_path) -> None:
    from vllm_omni.lora.request import LoRARequest

    root = _write_banks(tmp_path / "r1", {target: (4, 8) for target in _tiny_targets()})
    banks = read_taomate_r1_banks(root)
    cold, warm = object(), object()

    class _Model:
        def __init__(self, tag):
            self.tag = tag

        def clone(self, lora_id):
            return (self.tag, lora_id)

    pipeline = _r1_pipeline(
        _r1_bank_dirs=banks, _r1_prepared={"cold": (_Model(cold), "pc"), "warm": (_Model(warm), "pw")}
    )
    assert pipeline.diffusion_lora_fixed_banks(str(root)) == {role: str(path) for role, path in banks.items()}
    request = LoRARequest(lora_name="static:warm", lora_int_id=7, lora_path=str(banks["warm"]))
    assert pipeline._load_diffusion_lora_adapter(lora_request=request, lora_path=str(banks["warm"]), dtype=None) == (
        (warm, 7),
        "pw",
    )
    stray = LoRARequest(lora_name="x", lora_int_id=8, lora_path=str(tmp_path))
    with pytest.raises(ValueError, match="fixed cold/warm banks"):
        pipeline._load_diffusion_lora_adapter(lora_request=stray, lora_path=str(tmp_path), dtype=None)


# ----------------------------------------------------------------------------
# step execution over three chunks


def _segmented_state(*, capacity: int, heads=1, head_dim=2):
    from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionSegmentedKVSpec
    from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

    pool = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=16, window_chunks=1, gpu_memory_fraction=0.01),
        num_layers=1,
        num_kv_heads=heads,
        head_size=head_dim,
        dtype=torch.bfloat16,
        block_size=64,
        max_model_len=1_000_000,
        available_bytes=512 * 1024**2,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        device=torch.device("cpu"),
        segmented=ARDiffusionSegmentedKVSpec(
            block_size=64,
            max_retained_tokens=capacity,
            max_retained_segments=5,
            max_pending_tokens=capacity,
            max_pending_segments=2,
            initial_staging_tokens=capacity * 2,
        ),
    )
    return ARDiffusionKVState(pool, "test", {"main": pool.begin_request("test")}, num_layers=1)


class _RecordingManager:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.selected: list[str] = []

    def set_request_adapter(self, request, scale=1.0):
        assert request is None
        self.selected.append(self.pipeline.diffusion_lora_active_bank())


class _FakeTransformer(nn.Module):
    """Returns a bank-dependent constant velocity and stages K/V like the streaming attention."""

    VELOCITY = {"cold": 0.25, "warm": -0.5}

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.calls: list[tuple[str, str, int]] = []

    def plan_local_embed(self, **_kwargs):
        return None

    def forward(self, **kwargs):
        context = _STREAM_CONTEXT.get()
        assert context is not None
        bank = self.pipeline.diffusion_lora_active_bank()
        session = next(iter(self.pipeline._r1_sessions.values()))
        img_pos = kwargs["img_pos_info"]["position_ids"]
        x = kwargs["x"][0]
        reference_rows = session.reference_rows
        # Anchors lead the visual rows of every forward.
        assert torch.equal(x[img_pos[:reference_rows]], session.anchors)
        c_start, c_len = context.condition_span
        m_start, m_len = context.media_span
        heads, dim = context.cache.cache.num_kv_heads, context.cache.cache.head_size
        rows = lambda count: torch.zeros(count, heads, dim, dtype=torch.bfloat16)  # noqa: E731
        context.cache.assemble(0, rows(c_len), rows(c_len), rows(m_len), rows(m_len))
        if context.mode is StreamMode.CLEAN_COMMIT:
            context.cache.stage(0)
        self.calls.append((bank, context.mode.value, m_len))
        velocity = self.VELOCITY[bank]
        video = torch.full((img_pos.numel(), 96), velocity)
        video[:reference_rows] = 0
        audio = torch.full((kwargs["audio_pos_info"]["position_ids"].numel(), 32), velocity)
        return video, audio


class _FakeVideoDecoder:
    def __init__(self):
        self.pushed: list[torch.Tensor] = []
        self.flushed = 0

    def push_device(self, latents):
        self.pushed.append(latents)
        frames = 34 if self.pushed[:-1] else 34
        start = 34 * (len(self.pushed) - 1)
        return start, np.zeros((frames, 2, 2, 3), np.uint8)

    @staticmethod
    def to_host(frames):
        return frames

    def flush(self):
        self.flushed += 1
        return 34 * len(self.pushed), np.zeros((5, 2, 2, 3), np.uint8)

    def reset(self):
        pass


class _FakeAudioDecoder:
    def __init__(self):
        self.appended: list[torch.Tensor] = []

    def append(self, latents):
        self.appended.append(latents)

    def decode_range(self, s0, s1):
        return np.zeros((s1 - s0, 2), np.float32)

    def skip_range(self, s1):
        pass

    def reset(self):
        pass


def test_three_chunk_session_runs_banks_commits_and_release_euler(tmp_path) -> None:
    from vllm_omni.diffusion.models.taomate_r1 import pipeline as r1
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    plan, packed, reference_rows, canvas = _session_packed(text_len=5, refs=((2, 4),), frames=107)
    chunk_rows = max(c.video_latent_count * canvas.frame_rows + 2 * c.audio_latent_count for c in plan.chunks)
    bound = _segmented_state(capacity=3 * chunk_rows + 12 * canvas.frame_rows)
    pipeline = _r1_pipeline(
        device=torch.device("cpu"),
        _ar_diffusion_kv_state=bound,
        _tm_log_timings=False,
        _tm_output_rank=True,
        _r1_dump_latents_dir=tmp_path / "dump",
    )
    transformer = _FakeTransformer(pipeline)
    pipeline.transformer = transformer
    manager = _RecordingManager(pipeline)
    pipeline._tm_lora_manager = manager
    video_decoder, audio_decoder = _FakeVideoDecoder(), _FakeAudioDecoder()
    session = r1._R1Session(
        session_id="test",
        canvas=canvas,
        seed=9111,
        plan=plan,
        packed=packed,
        reference_rows=reference_rows,
        anchors=torch.randn(reference_rows, 96),
        text_embeddings=torch.randn(5, 16),
        cache=lambda: bound.segmented("main"),
        video_decoder=video_decoder,
        audio_decoder=audio_decoder,
    )
    pipeline._r1_sessions["test"] = session
    state = StepRequestState(request_id="req", sampling=OmniDiffusionSamplingParams(num_frames=107), prompt="p")
    state.chunk_num_steps, state.total_chunks, state.chunk_index = 3, plan.total_chunks, 0
    state.step_index = state.step_in_chunk = 0
    state.extra = {"taomate_session_id": "test"}

    sigmas_video, sigmas_audio = student_sigmas()
    outputs = []
    while not state.request_denoise_completed:
        pipeline.prepare_next_chunk(state)
        chunk = plan.chunks[state.chunk_index]
        generator = torch.Generator().manual_seed(9111 + chunk.index * 97)
        noise_video = torch.randn(state.latents.shape, generator=generator)
        noise_audio = torch.randn(state.extra["audio_rows"].shape, generator=generator)
        assert torch.equal(state.latents, noise_video) and torch.equal(state.extra["audio_rows"], noise_audio)
        for _ in range(3):
            velocity = pipeline.denoise_step(None, states=[state])
            pipeline.step_scheduler(state, velocity)
        v = _FakeTransformer.VELOCITY[chunk.bank]
        expected_video = noise_video.clone()
        expected_audio = noise_audio.clone()
        for step in range(3):
            expected_video = expected_video + (sigmas_video[step] - sigmas_video[step + 1]) * torch.full_like(
                noise_video, v
            )
            expected_audio = expected_audio + (sigmas_audio[step] - sigmas_audio[step + 1]) * torch.full_like(
                noise_audio, v
            )
        torch.testing.assert_close(state.latents, expected_video, rtol=0, atol=0)
        torch.testing.assert_close(state.extra["audio_rows"], expected_audio, rtol=0, atol=0)
        outputs.append(pipeline.post_decode(state))

    facts = [output.output["metadata"]["stream"]["model"] for output in outputs]
    assert [(item["lora_bank"], item["clean_commit"]) for item in facts] == [
        ("cold", True),
        ("warm", True),
        ("warm", False),
    ]
    modes = [(bank, mode) for bank, mode, _ in transformer.calls]
    assert modes == (
        [("cold", "noisy")] * 3
        + [("cold", "clean_commit")]
        + [("warm", "noisy")] * 3
        + [("warm", "clean_commit")]
        + [("warm", "noisy")] * 3
    )
    assert set(manager.selected) == {"cold", "warm"}
    # Two commits: chunk 0 (also the video sink) and chunk 1 are the retained history.
    committed = [m_len for _, mode, m_len in transformer.calls if mode == "clean_commit"]
    assert bound.segmented("main").history_tokens == sum(committed)
    assert [output.finished for output in outputs] == [False, False, True]
    assert video_decoder.flushed == 1 and len(audio_decoder.appended) == 3
    assert [tuple(latent.shape[2:3]) for latent in video_decoder.pushed] == [(12,), (10,), (10,)]
    assert sum(item.shape[-1] for item in audio_decoder.appended) == plan.audio_latent_t
    dumps = [torch.load(path) for path in sorted((tmp_path / "dump" / "test").glob("chunk_*.pt"))]
    assert [(item["lora_bank"], item["clean_commit"]) for item in dumps] == [
        ("cold", True),
        ("warm", True),
        ("warm", False),
    ]
    assert torch.equal(dumps[0]["anchors"], session.anchors) and dumps[1]["anchors"] is None


def test_video_api_leaves_reference_resizing_to_the_pipeline() -> None:
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
    from vllm_omni.model_extras.registry import should_preserve_reference_image_size

    # /v1/videos would otherwise stretch every reference to the output canvas before the 2048-edge resize.
    assert should_preserve_reference_image_size("TaoMateR1Pipeline", model=None)
    metadata = get_diffusion_model_metadata("TaoMateR1Pipeline")
    assert metadata.supports_multimodal_inputs and metadata.max_multimodal_image_inputs == 9
    assert metadata.final_output_type == "video"
