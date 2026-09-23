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
The upstream CosyVoice3 fp16 ONNX fixes classifier-free-guidance (CFG) batch to
2, while cross-request flow batching produces CFG batch ``2N``. The default
single-request path therefore keeps the original fixed-batch engine unchanged.
When cross-request batching is enabled, the ONNX graph I/O batch metadata is
made symbolic before TensorRT parses it, and one engine is built with two
optimization profiles: profile 0 keeps CFG batch 2 fixed, while profile 1 covers
CFG batches 4 through the stage scheduler limit. One context switches profiles
on demand, avoiding a second resident engine/context while preserving the
single-request specialization.

The bundled ONNX was traced with full attention, so upstream's streaming
chunk-causal mask (``DiT.forward(streaming=True)``) never reaches such an
engine. ``build_chunk_mask_flow_estimator_trt`` exports the repo's own DiT with
a seventh input, ``attn_mask`` (``(2, 1, T, T)`` bool), so the host builds the
same mask upstream would and the engine honours it; ``supports_attn_mask`` on
the wrapper tells the caller which kind of engine it holds. Its attention is
exported as the ONNX ``Attention`` op so TensorRT fuses it; an older TensorRT
that cannot gets the decomposed-attention export instead.

Precision: TensorRT >= 11 dropped the weakly-typed FP16/INT8 builder flags, so
fp16 only comes from a STRONGLY_TYPED network built from an fp16 ONNX
(``*autocast_fp16*``, fp16 I/O). An fp32 ONNX is built fp32 + the TF32 matmul
flag. ``EXPLICIT_BATCH`` is implicit (no flag) and ``ITensor.dtype`` is
read-only, so neither is set here.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import queue
import uuid
from contextlib import contextmanager

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import (
    _resolve_plan_path,
    _trt_logger,
)

logger = init_logger(__name__)

# Optimization-profile shapes for the original fixed-CFG-batch engine.
_DYNAMIC_INPUTS = ("x", "mask", "mu", "cond")
_MIN_SHAPES = ((2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4))
_OPT_SHAPES = ((2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500))
_MAX_SHAPES = ((2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000))
# The chunk-mask engine adds the query-key map, dynamic on both time dims.
ATTN_MASK_INPUT = "attn_mask"
_MASK_MIN_SHAPE, _MASK_OPT_SHAPE, _MASK_MAX_SHAPE = (2, 1, 4, 4), (2, 1, 500, 500), (2, 1, 3000, 3000)
# Part of the exported ONNX's cache key. Bump it whenever the DiT's forward or
# the export below changes, so engines traced from older code are not reused.
_CHUNK_MASK_EXPORT_VERSION = 1

# Bump when the dynamic-batch rewrite/profile contract changes so cached plans
# from older implementations cannot be loaded accidentally.
_DYNAMIC_BATCH_PLAN_VERSION = 4
_DYNAMIC_BATCH_IO_NAMES = ("x", "mask", "mu", "t", "spks", "cond", "estimator_out")
_MIN_T = 4
_OPT_T = 500
_SINGLE_MAX_T = 3000
_BATCH_MAX_T = 1024


def _is_fp16_onnx(onnx_path: str) -> bool:
    """Heuristic: the project's fp16 estimator ONNX is exported strongly-typed
    and named ``*autocast_fp16*`` / ``*fp16*`` (vs ``*fp32*``)."""
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def _set_onnx_cfg_batch_dynamic(model):
    """Make only the estimator graph I/O batch dimensions symbolic.

    CosyVoice3's current estimator export fixes CFG batch to 2 even though the
    graph operations are batch-generic.  TensorRT must see the symbolic batch
    before parsing; changing ``INetworkDefinition`` input shapes afterwards is
    too late for the time-embedding path, whose inferred shape is already fixed.
    """
    expected = set(_DYNAMIC_BATCH_IO_NAMES)
    found: set[str] = set()
    for value in (*model.graph.input, *model.graph.output):
        if value.name not in expected:
            continue
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 0:
            raise ValueError(f"ONNX tensor {value.name!r} has no batch dimension")
        dims[0].ClearField("dim_value")
        dims[0].dim_param = "cfg_batch"
        found.add(value.name)
    missing = expected - found
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"CosyVoice3 estimator ONNX is missing expected graph I/O: {names}")
    return model


def _dynamic_batch_onnx_bytes(onnx_path: str) -> bytes:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - packaging error on CUDA installs
        raise RuntimeError(
            "Dynamic CosyVoice3 TensorRT batching requires the 'onnx' package; "
            "install the CUDA requirements before enabling COSYVOICE3_BATCH_FLOW"
        ) from exc

    model = onnx.load(onnx_path)
    _set_onnx_cfg_batch_dynamic(model)
    return model.SerializeToString()


def _cfg_batch_profile(min_batch: int, opt_batch: int, max_batch: int, *, max_t: int):
    if not (2 <= min_batch <= opt_batch <= max_batch):
        raise ValueError(f"invalid CFG batch profile: min={min_batch}, opt={opt_batch}, max={max_batch}")
    if max_t < _OPT_T:
        raise ValueError(f"CFG batch profile max_t must be at least {_OPT_T}, got {max_t}")
    return {
        "x": ((min_batch, 80, _MIN_T), (opt_batch, 80, _OPT_T), (max_batch, 80, max_t)),
        "mask": ((min_batch, 1, _MIN_T), (opt_batch, 1, _OPT_T), (max_batch, 1, max_t)),
        "mu": ((min_batch, 80, _MIN_T), (opt_batch, 80, _OPT_T), (max_batch, 80, max_t)),
        "t": ((min_batch,), (opt_batch,), (max_batch,)),
        "spks": ((min_batch, 80), (opt_batch, 80), (max_batch, 80)),
        "cond": ((min_batch, 80, _MIN_T), (opt_batch, 80, _OPT_T), (max_batch, 80, max_t)),
    }


def _fixed_cfg_batch_profile():
    return _cfg_batch_profile(2, 2, 2, max_t=_SINGLE_MAX_T)


def _dynamic_batch_profile(max_cfg_batch: int):
    if max_cfg_batch < 4:
        raise ValueError(f"dynamic CFG batch must be at least 4, got {max_cfg_batch}")
    return _cfg_batch_profile(
        4,
        min(8, max_cfg_batch),
        max_cfg_batch,
        max_t=_BATCH_MAX_T,
    )


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


def _convert_onnx_to_trt(
    onnx_path: str,
    plan_path: str,
    strongly_typed: bool,
    with_attn_mask: bool = False,
    *,
    max_cfg_batch: int | None = None,
) -> None:
    import tensorrt as trt

    dynamic_batch = max_cfg_batch is not None
    logger.info(
        "Building flow-estimator TensorRT engine from %s (%s%s) ...",
        onnx_path,
        "strongly-typed/fp16" if strongly_typed else "fp32+TF32",
        f", dynamic CFG batch <= {max_cfg_batch}" if dynamic_batch else "",
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
    if dynamic_batch:
        model_bytes = _dynamic_batch_onnx_bytes(onnx_path)
    else:
        with open(onnx_path, "rb") as f:
            model_bytes = f.read()
    if not parser.parse(model_bytes):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    workspace = 1 << 32
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if dynamic_batch:
        runtime_resize = getattr(
            trt.PreviewFeature,
            "RUNTIME_ACTIVATION_RESIZE_10_10",
            None,
        )
        if runtime_resize is not None:
            config.set_preview_feature(runtime_resize, True)
    if not strongly_typed:
        # fp32 ONNX: enable the best available reduced-precision matmul flag
        # (FP16 on older TRT, else TF32 on Ampere+/Hopper). For a strongly-typed
        # network the precision is fixed by the graph, so no flag is set.
        for _flag_name in ("FP16", "TF32"):
            _flag = getattr(trt.BuilderFlag, _flag_name, None)
            if _flag is not None:
                config.set_flag(_flag)
                break

    if dynamic_batch and with_attn_mask:
        # The chunk-mask engine's attn_mask profile still pins the CFG pair;
        # widening it needs the mask bounds to follow each profile's batch.
        raise ValueError("dynamic CFG batching is not wired up for the chunk-mask estimator yet")
    if dynamic_batch:
        assert max_cfg_batch is not None
        # Profile 0 preserves the single-request specialization while profile 1
        # covers cross-request CFG batches. Keeping both in one engine avoids a
        # second resident copy of the weights and a second execution context.
        for shape_spec in (_fixed_cfg_batch_profile(), _dynamic_batch_profile(max_cfg_batch)):
            profile = builder.create_optimization_profile()
            for name, (mn, op, mx) in shape_spec.items():
                profile.set_shape(name, mn, op, mx)
            if not profile:
                raise RuntimeError("TensorRT produced an invalid flow-estimator optimization profile")
            config.add_optimization_profile(profile)
    else:
        profile = builder.create_optimization_profile()
        for name, mn, op, mx in zip(_DYNAMIC_INPUTS, _MIN_SHAPES, _OPT_SHAPES, _MAX_SHAPES):
            profile.set_shape(name, mn, op, mx)
        if with_attn_mask:
            profile.set_shape(ATTN_MASK_INPUT, _MASK_MIN_SHAPE, _MASK_OPT_SHAPE, _MASK_MAX_SHAPE)
        if not profile:
            raise RuntimeError("TensorRT produced an invalid flow-estimator optimization profile")
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build flow-estimator engine from {onnx_path}")
    _write_plan_atomically(engine_bytes, plan_path)
    logger.info("Wrote flow-estimator TensorRT engine to %s", plan_path)


_TRT_INPUT_NAMES = ("x", "mask", "mu", "t", "spks", "cond")
# Within one Euler solve only x and t change between estimator calls.
_TRT_DYNAMIC_INPUT_INDICES = (0, 3)


class _TrtEstimatorSession:
    """A fixed-shape TensorRT estimator binding reused across Euler steps."""

    def __init__(self, context, stream, engine, io_dtype: torch.dtype, inputs: tuple[torch.Tensor, ...]):
        # A chunk-mask engine takes the query-key map as a seventh input. It is
        # static within a solve, so the session binds it once and ``run`` keeps
        # taking only the six per-step tensors.
        if len(inputs) == len(_TRT_INPUT_NAMES) + 1:
            self.input_names = (*_TRT_INPUT_NAMES, ATTN_MASK_INPUT)
        elif len(inputs) == len(_TRT_INPUT_NAMES):
            self.input_names = _TRT_INPUT_NAMES
        else:
            raise ValueError(
                f"expected {len(_TRT_INPUT_NAMES)} or {len(_TRT_INPUT_NAMES) + 1} inputs, got {len(inputs)}"
            )

        self.context = context
        self.stream = stream
        self.io_dtype = io_dtype
        self._inputs = inputs
        self._shapes = tuple(tuple(tensor.shape) for tensor in inputs)
        self._input_buffers = tuple(self._make_input_buffer(tensor) for tensor in inputs)
        self._initialized = False
        self._engine_output = torch.empty_like(
            inputs[0],
            dtype=io_dtype,
            memory_format=torch.contiguous_format,
        )
        if self._engine_output.dtype == inputs[0].dtype:
            self._output = self._engine_output
        else:
            self._output = torch.empty_like(
                inputs[0],
                memory_format=torch.contiguous_format,
            )

        for name, buffer in zip(self.input_names, self._input_buffers, strict=True):
            context.set_input_shape(name, tuple(buffer.shape))

        bound_tensors = (*self._input_buffers, self._engine_output)
        for index, tensor in enumerate(bound_tensors):
            context.set_tensor_address(engine.get_tensor_name(index), tensor.data_ptr())

    def _make_input_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
        # ``attn_mask`` is bool; only the float inputs follow ``io_dtype``.
        dtype = self.io_dtype if tensor.is_floating_point() else tensor.dtype
        if tensor.dtype == dtype and tensor.is_contiguous():
            return tensor
        return torch.empty_like(
            tensor,
            dtype=dtype,
            memory_format=torch.contiguous_format,
        )

    def run(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Refresh the two dynamic inputs, retaining the staged conditioning."""
        for index, tensor in zip(_TRT_DYNAMIC_INPUT_INDICES, (x, t)):
            shape = self._shapes[index]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"TensorRT estimator input shape changed within a session: {tuple(tensor.shape)} != {shape}"
                )

        caller_stream = torch.cuda.current_stream(x.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            if not self._initialized:
                for index, tensor in enumerate(self._inputs):
                    if index in _TRT_DYNAMIC_INPUT_INDICES:
                        continue
                    buffer = self._input_buffers[index]
                    if tensor.data_ptr() != buffer.data_ptr():
                        buffer.copy_(tensor)

            for index, tensor in zip(_TRT_DYNAMIC_INPUT_INDICES, (x, t)):
                buffer = self._input_buffers[index]
                if tensor.data_ptr() != buffer.data_ptr():
                    buffer.copy_(tensor)

            assert self.context.execute_async_v3(self.stream.cuda_stream) is True
            self._initialized = True

            for tensor in (*self._input_buffers, self._engine_output):
                if tensor.is_cuda:
                    tensor.record_stream(self.stream)

        caller_stream.wait_stream(self.stream)
        if self._output is not self._engine_output:
            if self._engine_output.is_cuda:
                self._engine_output.record_stream(caller_stream)
            self._output.copy_(self._engine_output)
        if self._output.is_cuda:
            self._output.record_stream(caller_stream)
        return self._output


class TrtContextWrapper:
    """Pool of TensorRT execution contexts for the flow estimator.

    A normal engine has one fixed-CFG-batch profile. A cross-request engine has
    profile 0 for CFG batch 2 and profile 1 for CFG batches 4..2N. The same
    context switches profiles on its CUDA stream before input shapes are rebound.
    """

    def __init__(
        self,
        engine,
        device: str | torch.device,
        io_dtype: torch.dtype = torch.float32,
        trt_concurrent: int = 1,
        *,
        input_names: frozenset[str] = frozenset(),
        out_dtype: torch.dtype | None = None,
        static_chunk_size: int = 0,
    ):
        self.trt_engine = engine
        self.io_dtype = io_dtype
        self._device = torch.device(device)
        self._dynamic_profiles = int(getattr(engine, "num_optimization_profiles", 1)) > 1
        if self._dynamic_profiles and trt_concurrent != 1:
            raise ValueError("dynamic TensorRT profile switching requires trt_concurrent=1")
        # The inputs the engine declares; an exporter prunes unread ones.
        self.input_names = input_names
        # A legacy engine has no ``attn_mask`` input and runs full attention
        # whatever the caller's ``streaming`` flag says.
        self.supports_attn_mask = ATTN_MASK_INPUT in input_names
        # An autocast-traced graph can leave the output dtype unlike the inputs'.
        self.out_dtype = out_dtype or io_dtype
        # The DiT's block size, for building the mask on the host.
        self.static_chunk_size = static_chunk_size
        self._pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
        self._active_profile_by_context: dict[int, int] = {}
        for _ in range(trt_concurrent):
            ctx = engine.create_execution_context()
            assert ctx is not None, "failed to create TRT execution context (out of memory?)"
            stream = torch.cuda.Stream(self._device)
            self._active_profile_by_context[id(ctx)] = 0
            self._pool.put([ctx, stream])

    def _profile_for_shape(self, batch_size: int, sequence_length: int | None = None) -> int | None:
        if batch_size == 2:
            profile_index = 0
        elif self._dynamic_profiles and batch_size >= 4 and batch_size % 2 == 0:
            profile_index = 1
        else:
            return None

        try:
            min_shape, _, max_shape = self.trt_engine.get_tensor_profile_shape("x", profile_index)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0 if batch_size == 2 else None

        if not (int(min_shape[0]) <= batch_size <= int(max_shape[0])):
            return None
        if sequence_length is not None and not (int(min_shape[2]) <= sequence_length <= int(max_shape[2])):
            return None
        return profile_index

    def supports_estimator_shape(self, batch_size: int, sequence_length: int) -> bool:
        """Return whether an optimization profile accepts this CFG batch/length."""
        return self._profile_for_shape(batch_size, sequence_length) is not None

    def acquire_estimator(self, batch_size: int = 2, sequence_length: int | None = None):
        profile_index = self._profile_for_shape(batch_size, sequence_length)
        if profile_index is None:
            suffix = "" if sequence_length is None else f" at sequence length {sequence_length}"
            raise RuntimeError(f"TensorRT flow estimator does not support CFG batch {batch_size}{suffix}")

        [context, stream] = self._pool.get()
        context_id = id(context)
        prepared = False
        try:
            if self._active_profile_by_context[context_id] != profile_index:
                # Queue the profile switch on the same stream used by the
                # subsequent execute_async_v3(). CUDA stream ordering provides
                # the synchronization TensorRT requires between the profile
                # switch and enqueue without a host-side stream.synchronize().
                switched = context.set_optimization_profile_async(
                    profile_index,
                    stream.cuda_stream,
                )
                if not switched:
                    raise RuntimeError(
                        f"failed to select TensorRT optimization profile {profile_index} for CFG batch {batch_size}"
                    )
                self._active_profile_by_context[context_id] = profile_index
            prepared = True
            return [context, stream], self.trt_engine
        finally:
            if not prepared:
                self._pool.put([context, stream])

    def release_estimator(self, context, stream):
        if id(context) not in self._active_profile_by_context:
            raise RuntimeError("attempted to release an unknown TensorRT execution context")
        self._pool.put([context, stream])

    @contextmanager
    def estimation_session(self, *inputs: torch.Tensor):
        """Hold one pooled TRT context and fixed I/O binding for one flow solve."""
        if len(inputs) != len(_TRT_INPUT_NAMES):
            raise ValueError(f"expected {len(_TRT_INPUT_NAMES)} estimator inputs, got {len(inputs)}")
        batch_size, sequence_length = int(inputs[0].shape[0]), int(inputs[0].shape[2])
        if not self.supports_estimator_shape(batch_size, sequence_length):
            # Let CFM serialize unsupported request groups through CFG2.
            yield None
            return
        [context, stream], engine = self.acquire_estimator(batch_size, sequence_length)
        try:
            yield _TrtEstimatorSession(
                context=context,
                stream=stream,
                engine=engine,
                io_dtype=self.io_dtype,
                inputs=inputs,
            )
        finally:
            self.release_estimator(context, stream)


def _engine_tensor_dtype(engine, name: str) -> torch.dtype:
    import tensorrt as trt

    dtype = engine.get_tensor_dtype(name)
    torch_dtype = {trt.float16: torch.float16, trt.float32: torch.float32, trt.bfloat16: torch.bfloat16}.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"TensorRT flow estimator tensor '{name}' has unsupported dtype {dtype}")
    return torch_dtype


def _wrap_engine(engine, device: str | torch.device, static_chunk_size: int = 0) -> TrtContextWrapper:
    """Read the engine's inputs and I/O dtypes once and pool its contexts."""
    import tensorrt as trt

    names = (engine.get_tensor_name(i) for i in range(engine.num_io_tensors))
    input_names = frozenset(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    return TrtContextWrapper(
        engine,
        device=device,
        io_dtype=_engine_tensor_dtype(engine, "x"),
        input_names=input_names,
        out_dtype=_engine_tensor_dtype(engine, "estimator_out"),
        static_chunk_size=static_chunk_size,
    )


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
def _fp32_attention_for_export(estimator: torch.nn.Module):
    """Trace the DiT's masked attention in fp32 under fp16 autocast.

    The softmax over a masked ``T x T`` score map loses precision in fp16 as
    ``T`` grows (the bundled engine keeps it in fp32 too: its error against
    the torch DiT is about 4x lower than an all-fp16 trace). The casts are
    recorded into the graph, so the engine runs that block in fp32 and
    everything else in fp16. Scoped to ``estimator``'s own attention layers.
    """
    layers = [m for m in estimator.modules() if hasattr(m, "fp32_masked_attention")]
    for layer in layers:
        layer.fp32_masked_attention = True
    try:
        yield
    finally:
        for layer in layers:
            layer.fp32_masked_attention = False


# Custom domain the SDPA symbolic emits into; rewritten to the standard ONNX
# ``Attention`` op after export, since the TorchScript exporter stops below
# the opset (23) that defines it.
_EXPORT_ATTENTION_DOMAIN = "vllm_omni.export"
_ATTENTION_OPSET = 23


def _sdpa_as_attention_op(
    g, query, key, value, attn_mask=None, dropout_p=None, is_causal=None, scale=None, enable_gqa=None
):
    """TorchScript symbolic: SDPA -> one ``Attention`` node TensorRT fuses."""
    from torch.onnx import symbolic_helper

    if symbolic_helper._maybe_get_const(is_causal, "b"):
        raise NotImplementedError("causal SDPA is not exported as an Attention op")
    kwargs = {}
    if not symbolic_helper._is_none(scale):
        kwargs["scale_f"] = symbolic_helper._maybe_get_const(scale, "f")
    inputs = [query, key, value]
    if not symbolic_helper._is_none(attn_mask):
        inputs.append(attn_mask)
    out = g.op(f"{_EXPORT_ATTENTION_DOMAIN}::Attention", *inputs, **kwargs)
    out.setType(query.type())
    return out


@contextlib.contextmanager
def _sdpa_exported_as_attention_op():
    torch.onnx.register_custom_op_symbolic("aten::scaled_dot_product_attention", _sdpa_as_attention_op, 18)
    try:
        yield
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::scaled_dot_product_attention", 18)


def _promote_attention_nodes(onnx_path: str) -> None:
    """Move the exported ``Attention`` nodes into the default ONNX domain at
    opset 23. The rest of the graph is opset 18, whose ops keep their meaning
    at 23."""
    import onnx

    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.domain == _EXPORT_ATTENTION_DOMAIN:
            node.domain = ""
    others = [o for o in model.opset_import if o.domain not in ("", "ai.onnx", _EXPORT_ATTENTION_DOMAIN)]
    del model.opset_import[:]
    model.opset_import.extend([*others, onnx.helper.make_opsetid("", _ATTENTION_OPSET)])
    onnx.save(model, onnx_path)


def export_chunk_mask_estimator_onnx(
    estimator: torch.nn.Module, onnx_path: str, *, fp16: bool = True, fused_attention: bool = False
) -> str:
    """Export the repo's DiT to ONNX with ``attn_mask`` as a seventh input.

    Traced under fp16 autocast when ``fp16`` (the layout of the project's
    ``*autocast_fp16*`` ONNX, which TensorRT >= 11 builds strongly typed),
    with attention kept in fp32; plain fp32 otherwise. The result is written
    next to the model's other estimator ONNX files so the plan cache keys off
    it like any other.

    ``fused_attention`` (fp16 only) exports each masked SDPA as one ONNX
    ``Attention`` node (opset 23) instead of matmuls + softmax, still in fp32:
    TensorRT runs it as a fused MHA kernel, several times faster than the
    decomposed graph it cannot fuse. It stays fp32 because some layers of the
    released checkpoint reach attention scores past 1e6, which a fused fp16
    kernel overflows (it keeps the scores in the input dtype).
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
    fused = fp16 and fused_attention
    wrapper = _EstimatorWithMaskInput(estimator).to(device).eval()
    frames = 64
    x = torch.randn(2, 80, frames, device=device)
    mask = torch.ones(2, 1, frames, device=device)
    mu = torch.randn(2, 80, frames, device=device)
    t = torch.rand(2, device=device)
    spks = torch.randn(2, 80, device=device)
    cond = torch.randn(2, 80, frames, device=device)
    attn_mask = torch.ones(2, 1, frames, frames, dtype=torch.bool, device=device)
    dynamic_axes = {
        "x": {2: "seq_len"},
        "mask": {2: "seq_len"},
        "mu": {2: "seq_len"},
        "cond": {2: "seq_len"},
        ATTN_MASK_INPUT: {2: "seq_len", 3: "seq_len"},
        "estimator_out": {2: "seq_len"},
    }
    tmp = f"{onnx_path}.tmp.{os.getpid()}"
    logger.info(
        "Exporting chunk-mask flow estimator ONNX to %s (fp16=%s, fused_attention=%s) ...", onnx_path, fp16, fused
    )
    try:
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16),
            _fp32_attention_for_export(estimator) if fp16 else contextlib.nullcontext(),
            _sdpa_exported_as_attention_op() if fused else contextlib.nullcontext(),
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
        if fused:
            _promote_attention_nodes(tmp)
        os.replace(tmp, onnx_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
        if was_training:
            estimator.train()
    return onnx_path


def flow_checkpoint_fingerprint(model_dir: str, weight_file: str = "flow.pt") -> str:
    """A short digest identifying the flow checkpoint an ONNX was exported from.

    The exported ONNX bakes in the DiT weights, so its cache path must change
    whenever the checkpoint or the export code does: the digest covers
    ``_CHUNK_MASK_EXPORT_VERSION``, the resolved model directory and the
    size/mtime of its flow weights.
    """
    parts = [f"v{_CHUNK_MASK_EXPORT_VERSION}", os.path.realpath(model_dir)]
    weight_path = os.path.join(model_dir, weight_file)
    try:
        st = os.stat(weight_path)
        parts.append(f"{st.st_size}:{int(st.st_mtime)}")
    except OSError:
        parts.append("no-weights")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def chunk_mask_estimator_onnx_path(
    onnx_dir: str, *, fp16: bool, cache_key: str | None, fused_attention: bool = False
) -> str:
    tag = "autocast_fp16" if fp16 else "fp32"
    if fp16 and fused_attention:
        tag += "_fused_attn"
    suffix = f".{cache_key}" if cache_key else ""
    return os.path.join(onnx_dir, f"flow.decoder.estimator.chunk_mask.{tag}{suffix}.onnx")


def _build_chunk_mask_engine(estimator, onnx_dir, device, *, fp16, cache_key, fused_attention):
    import tensorrt as trt

    onnx_path = chunk_mask_estimator_onnx_path(
        onnx_dir, fp16=fp16, cache_key=cache_key, fused_attention=fused_attention
    )
    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) == 0:
        os.makedirs(onnx_dir, exist_ok=True)
        export_chunk_mask_estimator_onnx(estimator, onnx_path, fp16=fp16, fused_attention=fused_attention)
    plan_path = _resolve_plan_path(onnx_path, prefix="flow_estimator_chunk_mask")
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(onnx_path, plan_path, strongly_typed=fp16, with_attn_mask=True)

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize chunk-mask flow-estimator TensorRT engine {plan_path}")
    logger.info("Loaded chunk-mask flow-estimator TensorRT engine (%s)", plan_path)
    wrapper = _wrap_engine(engine, device, static_chunk_size=int(estimator.static_chunk_size))
    if not wrapper.supports_attn_mask:
        raise RuntimeError(f"chunk-mask engine {plan_path} has no '{ATTN_MASK_INPUT}' input")
    return wrapper


def build_chunk_mask_flow_estimator_trt(
    estimator: torch.nn.Module,
    onnx_dir: str,
    device: str | torch.device,
    *,
    fp16: bool = True,
    cache_key: str | None = None,
) -> TrtContextWrapper:
    """Build/load a flow-estimator engine that takes the chunk-causal mask.

    The ONNX is exported from ``estimator`` (the loaded torch DiT) into
    ``onnx_dir`` on first use and cached there; the plan is cached like the
    legacy engine's. ``cache_key`` (see ``flow_checkpoint_fingerprint``) is
    folded into the ONNX name so exports from different checkpoints never
    collide when ``onnx_dir`` is shared. ``estimator.static_chunk_size`` goes
    to the wrapper so ``forward_estimator`` can build the mask upstream would.

    An fp16 engine first tries fused attention (ONNX ``Attention``, opset 23);
    a TensorRT that cannot parse or build it gets the fp32-attention export.
    """
    if fp16:
        try:
            return _build_chunk_mask_engine(
                estimator, onnx_dir, device, fp16=True, cache_key=cache_key, fused_attention=True
            )
        except Exception as exc:
            logger.warning(
                "CosyVoice3 chunk-mask estimator: fused-attention engine unavailable (%s); "
                "using fp32 attention, which is slower",
                exc,
            )
    return _build_chunk_mask_engine(estimator, onnx_dir, device, fp16=fp16, cache_key=cache_key, fused_attention=False)


def build_flow_estimator_trt(
    onnx_path: str,
    device: str | torch.device,
    *,
    max_cfg_batch: int | None = None,
) -> TrtContextWrapper:
    """Build/load the flow-estimator TRT engine and return a context-pool wrapper.

    With cross-request batching disabled, this is the original fixed-CFG-batch
    engine. With max_cfg_batch > 2, one dynamic engine contains profile 0
    for CFG batch 2 and profile 1 for CFG batches 4 through max_cfg_batch.
    """
    import tensorrt as trt

    strongly_typed = _is_fp16_onnx(onnx_path)
    dynamic_batch = max_cfg_batch is not None and max_cfg_batch > 2
    if max_cfg_batch is not None and max_cfg_batch % 2 != 0:
        raise ValueError(f"max_cfg_batch must be even, got {max_cfg_batch}")

    if dynamic_batch:
        prefix = f"flow_estimator_dynamic_v{_DYNAMIC_BATCH_PLAN_VERSION}_b{max_cfg_batch}_t{_BATCH_MAX_T}"
    else:
        prefix = "flow_estimator"
    plan_path = _resolve_plan_path(onnx_path, prefix=prefix)

    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(
            onnx_path,
            plan_path,
            strongly_typed=strongly_typed,
            max_cfg_batch=max_cfg_batch if dynamic_batch else None,
        )

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize flow-estimator TensorRT engine {plan_path}")

    if dynamic_batch:
        logger.info(
            "Loaded flow-estimator TensorRT engine (CFG batch 2/T<=%d and CFG batch 4..%d/T<=%d, %s)",
            _SINGLE_MAX_T,
            max_cfg_batch,
            _BATCH_MAX_T,
            plan_path,
        )
    else:
        logger.info("Loaded flow-estimator TensorRT engine (%s)", plan_path)

    return _wrap_engine(engine, device)
