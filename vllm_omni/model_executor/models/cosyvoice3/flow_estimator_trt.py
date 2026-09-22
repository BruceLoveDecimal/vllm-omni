# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TensorRT engine for the CosyVoice3 flow-decoder (CFM) DiT estimator.

The estimator is the per-step network the conditional flow-matching ODE solver
calls during code2wav (token -> mel). It dominates code2wav latency; the
upstream ``CausalConditionalCFM.forward_estimator`` already supports running it
through a TensorRT engine (it switches on ``self.estimator`` not being an
``nn.Module`` and drives it via ``acquire_estimator`` / ``execute_async_v3``).
This module builds that engine from the bundled ``flow.decoder.estimator*.onnx``
and wraps it so it can be dropped in for the torch estimator.

The estimator engine has 6 inputs ``x, mask, mu, t, spks, cond`` and one output.
``x/mask/mu/cond`` carry a dynamic time dim; ``t``/``spks`` are fixed, so only
the former need an optimization profile (TRT infers the latter from the ONNX).
Shapes mirror the CosyVoice runtime.

The bundled ONNX was traced with full attention, so upstream's streaming
chunk-causal mask (``DiT.forward(streaming=True)``) never reaches such an
engine. ``build_chunk_mask_flow_estimator_trt`` exports the repo's own DiT with
a seventh input, ``attn_mask`` (``(B, 1, T, T)`` bool), so the host builds the
same mask upstream would and the engine honours it; ``supports_attn_mask`` on
the wrapper tells the caller which kind of engine it holds.

Batch: the solver doubles every request into a CFG pair, so one request is a
batch of 2, which is all the bundled ONNX takes. The exported ONNX has a
dynamic batch dim, and with cross-request flow batching on its engine gets a
second profile for up to ``max_batch`` rows at a shorter maximum length (see
``_batched_profile_max_frames``); the wrapper keeps one context per profile
and picks the profile that fits each call.

Precision: TensorRT >= 11 dropped the weakly-typed FP16/INT8 builder flags, so
fp16 only comes from a STRONGLY_TYPED network built from an fp16 ONNX
(``*autocast_fp16*``, fp16 I/O). An fp32 ONNX is built fp32 + the TF32 matmul
flag. ``EXPLICIT_BATCH`` is implicit (no flag) and ``ITensor.dtype`` is
read-only, so neither is set here.
"""

from __future__ import annotations

import contextlib
import math
import os
import queue
import uuid

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import (
    _resolve_plan_path,
    _trt_logger,
)

logger = init_logger(__name__)

# Optimization-profile shapes for the time-dim inputs (channels, then the time
# dim's min/opt/max). One CFG pair (batch 2) up to 3000 frames matches
# CosyVoice's token2wav and is the whole profile of the bundled ONNX.
_TIME_INPUT_CHANNELS = (("x", 80), ("mask", 1), ("mu", 80), ("cond", 80))
_PAIR_ROWS = 2
_MIN_FRAMES, _OPT_FRAMES, _MAX_FRAMES = 4, 500, 3000
# The chunk-mask engine adds the query-key map, dynamic on both time dims.
ATTN_MASK_INPUT = "attn_mask"
# The batched profile's length is a whole number of the DiT's 50-frame blocks.
_BATCHED_FRAMES_MULTIPLE = 50


def _is_fp16_onnx(onnx_path: str) -> bool:
    """Heuristic: the project's fp16 estimator ONNX is exported strongly-typed
    and named ``*autocast_fp16*`` / ``*fp16*`` (vs ``*fp32*``)."""
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def _write_plan_atomically(engine_bytes, plan_path: str) -> None:
    tmp = f"{plan_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_created = False
    try:
        with open(tmp, "xb") as f:
            tmp_created = True
            f.write(engine_bytes)
        os.replace(tmp, plan_path)
        tmp_created = False
    except BaseException:
        if tmp_created:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Failed to remove temporary TensorRT plan %s", tmp, exc_info=True)
        raise


def _batched_profile_max_frames(max_batch: int) -> int:
    """The longest input of the batched profile.

    An execution context's activation memory is sized by the largest
    attention map any profile allows, ``rows * T * T``. Capping the batched
    profile at the single-pair profile's ``2 * 3000 * 3000`` keeps that where
    it was (1050 frames at 16 rows); the batched profile's own context is the
    one extra allocation. The streaming chunks batching is for are the prompt
    plus about 150 frames, and a longer call falls back to single-pair slices.
    """
    frames = int(_MAX_FRAMES * math.sqrt(_PAIR_ROWS / max_batch))
    return frames - frames % _BATCHED_FRAMES_MULTIPLE


def _profile_ranges(max_batch: int) -> list[tuple[int, int, int]]:
    """``(min batch, max batch, max frames)`` per profile, single pair first."""
    ranges = [(_PAIR_ROWS, _PAIR_ROWS, _MAX_FRAMES)]
    if max_batch > _PAIR_ROWS:
        ranges.append((2 * _PAIR_ROWS, max_batch, _batched_profile_max_frames(max_batch)))
    return ranges


def _convert_onnx_to_trt(
    onnx_path: str, plan_path: str, strongly_typed: bool, with_attn_mask: bool = False, max_batch: int = _PAIR_ROWS
) -> None:
    import tensorrt as trt

    logger.info(
        "Building flow-estimator TensorRT engine from %s (%s) ...",
        onnx_path,
        "strongly-typed/fp16" if strongly_typed else "fp32+TF32",
    )
    trt_logger = _trt_logger()
    builder = trt.Builder(trt_logger)
    # STRONGLY_TYPED takes precision from the ONNX graph (fp16 engine from an
    # fp16 ONNX) — this is the only way to get fp16 in TRT>=11, which dropped
    # the weakly-typed FP16 BuilderFlag. Otherwise EXPLICIT_BATCH is implicit,
    # so create the network with no flags.
    if strongly_typed:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    else:
        network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4 GiB
    if not strongly_typed:
        # fp32 ONNX: enable the best available reduced-precision matmul flag
        # (FP16 on older TRT, else TF32 on Ampere+/Hopper). For a strongly-typed
        # network the precision is fixed by the graph, so no flag is set.
        for _flag_name in ("FP16", "TF32"):
            _flag = getattr(trt.BuilderFlag, _flag_name, None)
            if _flag is not None:
                config.set_flag(_flag)
                break

    # Only the exported chunk-mask ONNX has a dynamic batch (``t`` and
    # ``spks`` included); the bundled one keeps the single-pair profile.
    for min_batch, profile_max_batch, max_frames in _profile_ranges(max_batch if with_attn_mask else _PAIR_ROWS):
        opt_frames = min(_OPT_FRAMES, max_frames)
        profile = builder.create_optimization_profile()
        for name, channels in _TIME_INPUT_CHANNELS:
            profile.set_shape(
                name,
                (min_batch, channels, _MIN_FRAMES),
                (profile_max_batch, channels, opt_frames),
                (profile_max_batch, channels, max_frames),
            )
        if with_attn_mask:
            profile.set_shape(
                ATTN_MASK_INPUT,
                (min_batch, 1, _MIN_FRAMES, _MIN_FRAMES),
                (profile_max_batch, 1, opt_frames, opt_frames),
                (profile_max_batch, 1, max_frames, max_frames),
            )
            profile.set_shape("t", (min_batch,), (profile_max_batch,), (profile_max_batch,))
            profile.set_shape("spks", (min_batch, 80), (profile_max_batch, 80), (profile_max_batch, 80))
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build flow-estimator engine from {onnx_path}")
    _write_plan_atomically(engine_bytes, plan_path)
    logger.info("Wrote flow-estimator TensorRT engine to %s", plan_path)


class TrtContextWrapper:
    """Pool of TensorRT execution contexts for the flow estimator.

    Exposes the ``acquire_estimator`` / ``release_estimator`` contract that
    ``CausalConditionalCFM.forward_estimator`` expects.
    """

    def __init__(
        self, engine, device: str | torch.device, io_dtype: torch.dtype = torch.float32, trt_concurrent: int = 1
    ):
        self.trt_engine = engine
        # Engine I/O dtype (fp16 for a strongly-typed fp16 engine). The flow runs
        # in fp32, so forward_estimator casts to/from this at the boundary.
        self.io_dtype = io_dtype
        # Whether the engine takes the chunk-causal query-key map as an input
        # (see ``build_chunk_mask_flow_estimator_trt``). A legacy engine runs
        # full attention whatever the caller's ``streaming`` flag says.
        self.supports_attn_mask = _engine_has_input(engine, ATTN_MASK_INPUT)
        self.input_names = _engine_input_names(engine)
        # The output buffer must match the engine's output dtype, which an
        # autocast-traced graph can leave different from its inputs.
        self.out_dtype = _engine_tensor_dtype(engine, "estimator_out", io_dtype)
        # ``(min batch, max batch, max frames)`` per optimization profile; the
        # caller slices a batch wider than any profile takes (see
        # ``max_batch_for``) and each call runs on a context of the profile
        # that fits it.
        self.profile_limits = _engine_profile_limits(engine)
        # Filled in by the model when it swaps the estimator, so the host can
        # build the mask with the DiT's block size.
        self.static_chunk_size = 0
        self._pools: list[queue.Queue] = []
        self._pool_of_context: dict[int, queue.Queue] = {}
        for index in range(len(self.profile_limits)):
            pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
            for _ in range(trt_concurrent):
                ctx = engine.create_execution_context()
                assert ctx is not None, "failed to create TRT execution context (out of memory?)"
                stream = torch.cuda.Stream(torch.device(device))
                if index > 0:
                    # A new context starts on profile 0.
                    ctx.set_optimization_profile_async(index, stream.cuda_stream)
                    stream.synchronize()
                pool.put([ctx, stream])
                self._pool_of_context[id(ctx)] = pool
            self._pools.append(pool)

    def max_batch_for(self, frames: int) -> int:
        """The widest estimator batch (2 rows per request, for CFG) the engine
        runs at ``frames`` mel frames, or 0 if no profile reaches that length."""
        return max((max_batch for _, max_batch, max_frames in self.profile_limits if frames <= max_frames), default=0)

    def _profile_for(self, batch: int, frames: int) -> int:
        for index, (min_batch, max_batch, max_frames) in enumerate(self.profile_limits):
            if min_batch <= batch <= max_batch and frames <= max_frames:
                return index
        # Nothing fits: profile 0's context rejects the shape with a clear error.
        return 0

    def acquire_estimator(self, batch: int = _PAIR_ROWS, frames: int = _MIN_FRAMES):
        return self._pools[self._profile_for(batch, frames)].get(), self.trt_engine

    def release_estimator(self, context, stream):
        self._pool_of_context[id(context)].put([context, stream])


def _engine_input_names(engine) -> frozenset[str]:
    try:
        import tensorrt as trt

        return frozenset(
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        )
    except Exception:
        return frozenset()


def _engine_has_input(engine, name: str) -> bool:
    return name in _engine_input_names(engine)


def _engine_tensor_dtype(engine, name: str, fallback: torch.dtype) -> torch.dtype:
    try:
        import tensorrt as trt

        dtype = engine.get_tensor_dtype(name)
        return {trt.float16: torch.float16, trt.float32: torch.float32, trt.bfloat16: torch.bfloat16}.get(
            dtype, fallback
        )
    except Exception:
        return fallback


def _engine_profile_limits(engine) -> list[tuple[int, int, int]]:
    """``(min batch, max batch, max frames)`` of ``x`` in each optimization profile."""
    limits = []
    try:
        for index in range(engine.num_optimization_profiles):
            min_shape, _opt_shape, max_shape = engine.get_tensor_profile_shape("x", index)
            limits.append((int(min_shape[0]), int(max_shape[0]), int(max_shape[2])))
    except Exception:
        return [(_PAIR_ROWS, _PAIR_ROWS, _MAX_FRAMES)]
    return limits


def _engine_io_dtype(engine, fallback: torch.dtype) -> torch.dtype:
    """The dtype of the engine's ``x`` input, which the caller casts to."""
    return _engine_tensor_dtype(engine, "x", fallback)


class _EstimatorWithMaskInput(torch.nn.Module):
    """Export view of the DiT: the attention map is an input, not derived."""

    def __init__(self, estimator: torch.nn.Module):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, mask, mu, t, spks, cond, attn_mask):
        out = self.estimator(x, mask, mu, t, spks, cond, attn_mask=attn_mask)
        # With an explicit map the DiT never reads ``mask``, and the exporter
        # would drop an unused input. Zeroing padded frames keeps the
        # six-input calling convention of the bundled ONNX (the caller drops
        # those frames anyway). The cast keeps the output in the input dtype
        # under autocast, so the engine's I/O is uniform.
        return (out * mask.to(out.dtype)).to(x.dtype)


@contextlib.contextmanager
def _fp32_attention_for_export():
    """Trace scaled-dot-product attention in fp32 under fp16 autocast.

    The softmax over a masked ``T x T`` score map loses precision in fp16 as
    ``T`` grows (the bundled engine keeps it in fp32 too: its error against
    the torch DiT is about 4x lower than an all-fp16 trace). Casting q/k/v up
    and the output back down is recorded into the graph, so the engine runs
    that one block in fp32 and everything else in fp16.
    """
    original = F.scaled_dot_product_attention

    def fp32_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        with torch.autocast(device_type=query.device.type, enabled=False):
            out = original(
                query.float(),
                key.float(),
                value.float(),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )
        return out.to(query.dtype)

    F.scaled_dot_product_attention = fp32_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def export_chunk_mask_estimator_onnx(estimator: torch.nn.Module, onnx_path: str, *, fp16: bool = True) -> str:
    """Export the repo's DiT to ONNX with ``attn_mask`` as a seventh input.

    Traced under fp16 autocast when ``fp16`` (the layout of the project's
    ``*autocast_fp16*`` ONNX, which TensorRT >= 11 builds strongly typed),
    with attention kept in fp32; plain fp32 otherwise. The result is written
    next to the model's other estimator ONNX files so the plan cache keys off
    it like any other.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Exporting the chunk-mask flow estimator needs the 'onnx' package (pip install onnx)"
        ) from exc

    device = next(estimator.parameters()).device
    was_training = estimator.training
    estimator.eval()
    wrapper = _EstimatorWithMaskInput(estimator).to(device).eval()
    # Traced at four rows, not the two every single-request call uses, so a
    # shape the trace freezes at the example batch breaks those calls instead
    # of going unnoticed (``test_export_runs_at_other_batch_sizes``).
    batch = 2 * _PAIR_ROWS
    frames = 64
    x = torch.randn(batch, 80, frames, device=device)
    mask = torch.ones(batch, 1, frames, device=device)
    mu = torch.randn(batch, 80, frames, device=device)
    t = torch.rand(batch, device=device)
    spks = torch.randn(batch, 80, device=device)
    cond = torch.randn(batch, 80, frames, device=device)
    attn_mask = torch.ones(batch, 1, frames, frames, dtype=torch.bool, device=device)
    dynamic_axes = {
        "x": {0: "batch", 2: "seq_len"},
        "mask": {0: "batch", 2: "seq_len"},
        "mu": {0: "batch", 2: "seq_len"},
        "t": {0: "batch"},
        "spks": {0: "batch"},
        "cond": {0: "batch", 2: "seq_len"},
        ATTN_MASK_INPUT: {0: "batch", 2: "seq_len", 3: "seq_len"},
        "estimator_out": {0: "batch", 2: "seq_len"},
    }
    tmp = f"{onnx_path}.tmp.{os.getpid()}"
    logger.info("Exporting chunk-mask flow estimator ONNX to %s (fp16=%s) ...", onnx_path, fp16)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16),
        _fp32_attention_for_export() if fp16 else contextlib.nullcontext(),
    ):
        torch.onnx.export(
            wrapper,
            (x, mask, mu, t, spks, cond, attn_mask),
            tmp,
            input_names=["x", "mask", "mu", "t", "spks", "cond", ATTN_MASK_INPUT],
            output_names=["estimator_out"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            dynamo=False,
        )
    os.replace(tmp, onnx_path)
    if was_training:
        estimator.train()
    return onnx_path


def build_chunk_mask_flow_estimator_trt(
    estimator: torch.nn.Module,
    onnx_dir: str,
    device: str | torch.device,
    *,
    fp16: bool = True,
    max_batch: int = _PAIR_ROWS,
) -> TrtContextWrapper:
    """Build/load a flow-estimator engine that takes the chunk-causal mask.

    The ONNX is exported from ``estimator`` (the loaded torch DiT) into
    ``onnx_dir`` on first use and cached there; the plan is cached like the
    legacy engine's, per ``max_batch``. ``max_batch`` above one CFG pair adds
    the batched profile. ``estimator.static_chunk_size`` is copied onto the
    wrapper so ``forward_estimator`` can build the same mask upstream would.
    """
    import tensorrt as trt

    tag = "autocast_fp16" if fp16 else "fp32"
    onnx_path = os.path.join(onnx_dir, f"flow.decoder.estimator.chunk_mask.dynamic_batch.{tag}.onnx")
    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) == 0:
        os.makedirs(onnx_dir, exist_ok=True)
        export_chunk_mask_estimator_onnx(estimator, onnx_path, fp16=fp16)
    plan_path = _resolve_plan_path(onnx_path, prefix=f"flow_estimator_chunk_mask_b{max_batch}")
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(onnx_path, plan_path, strongly_typed=fp16, with_attn_mask=True, max_batch=max_batch)

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize chunk-mask flow-estimator TensorRT engine {plan_path}")
    logger.info("Loaded chunk-mask flow-estimator TensorRT engine (%s)", plan_path)
    wrapper = TrtContextWrapper(engine, device=device, io_dtype=_engine_io_dtype(engine, torch.float32))
    if not wrapper.supports_attn_mask:
        raise RuntimeError(f"chunk-mask engine {plan_path} has no '{ATTN_MASK_INPUT}' input")
    wrapper.static_chunk_size = int(getattr(estimator, "static_chunk_size", 0))
    return wrapper


def build_flow_estimator_trt(onnx_path: str, device: str | torch.device) -> TrtContextWrapper:
    """Build/load the flow-estimator TRT engine and return a context-pool wrapper.

    An fp16 ONNX (``*autocast_fp16*``) is built as a strongly-typed network (the
    only way to get fp16 in TRT>=11); an fp32 ONNX is built fp32 + TF32.
    """
    import tensorrt as trt

    strongly_typed = _is_fp16_onnx(onnx_path)
    plan_path = _resolve_plan_path(onnx_path, prefix="flow_estimator")
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(onnx_path, plan_path, strongly_typed=strongly_typed)

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize flow-estimator TensorRT engine {plan_path}")
    logger.info("Loaded flow-estimator TensorRT engine (%s)", plan_path)
    io_dtype = _engine_io_dtype(engine, torch.float16 if strongly_typed else torch.float32)
    return TrtContextWrapper(engine, device=device, io_dtype=io_dtype)
