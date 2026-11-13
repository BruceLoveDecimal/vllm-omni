# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DiffSynth EntroPack packages of the AdaLN-pruned MiniMax-H3-Image DiT.

A package is one safetensors file plus a same-named JSON sidecar. Every linear
weight is a packed entropack tensor (``<module>.weight.entropack_quant_state``):

- INT8 layers also carry ``<module>.weight_quant_scale`` and hold ConvRot-rotated
  per-output-channel INT8 codes. They run as rotated W8A8 with dynamic per-token
  activations, the same arithmetic as the online ``convrot_int8`` method.
- The remaining layers hold lattice-coded BF16 weights and run unquantized.

Codes stored verbatim (``raw``) are read in place. Entropy-coded tensors are
decoded once while loading, which needs the optional ``entropack`` package;
nothing is decoded during inference.

The pruned DiT replaces the timestep embedder and every AdaLN projection by a
lookup table (``<module>.adaln_proj.table``) with one row per distilled step.
"""

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import ChannelQuantScaleParameter

from vllm_omni.quantization.convrot_config import regular_hadamard, rotate, rotation_group_size
from vllm_omni.quantization.int8_config import create_weight_parameter

try:
    import entropack
except ImportError:
    entropack = None

STATE_SUFFIX = ".weight.entropack_quant_state"
SCALE_SUFFIX = ".weight_quant_scale"
ADALN_TABLE_KEY = "blocks.0.adaln_proj.table"
PRUNED_DIT_CLASS = "MiniMaxH3ImageDiTPruned"
INT8_METHOD_PREFIX = "entropack_lossy_quant_int8"

# entropack's container layout: a uint64 JSON length, the JSON metadata, then
# the buffers, each starting on a 16-byte boundary.
_LENGTH_PREFIX_BYTES = 8
_ALIGNMENT = 16


def _aligned(size: int) -> int:
    return -(-size // _ALIGNMENT) * _ALIGNMENT


def unpack_entropack_state(state: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """The metadata of a packed entropack tensor and zero-copy views of its buffers."""
    if state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError(f"an entropack state must be a 1-D uint8 tensor, got {state.dtype} {list(state.shape)}")
    header_len = int(state[:_LENGTH_PREFIX_BYTES].view(torch.int64).item())
    header_end = _LENGTH_PREFIX_BYTES + header_len
    if header_len <= 0 or header_end > state.numel():
        raise ValueError("the entropack state is truncated: its header runs past the end of the tensor")
    metadata = json.loads(bytes(state[_LENGTH_PREFIX_BYTES:header_end].tolist()).decode("utf-8"))
    offset = _aligned(header_end)
    buffers: dict[str, torch.Tensor] = {}
    for name, dtype_name, shape in zip(
        metadata["buffer_names"], metadata["buffer_dtypes"], metadata["buffer_shapes"], strict=True
    ):
        dtype = getattr(torch, dtype_name)
        size = math.prod(shape) * dtype.itemsize
        if offset + size > state.numel():
            raise ValueError(f"the entropack state is truncated: buffer {name!r} runs past the end of the tensor")
        buffers[name] = state[offset : offset + size].view(dtype).view(shape)
        offset = _aligned(offset + size)
    return metadata, buffers


def decode_entropack_state(state: torch.Tensor) -> torch.Tensor:
    """The tensor a packed entropack state stores: INT8 codes or dense weights."""
    metadata, buffers = unpack_entropack_state(state)
    if metadata["compress_method"] == "raw":
        return buffers["data"]
    if entropack is None:
        raise ImportError(
            f"This EntroPack package holds {metadata['compress_method']!r} entropy-coded weights, which are "
            'decoded while loading by the entropack package: pip install "entropack[cuda13]" (or [cuda12]).'
        )
    # The decoder runs where the packed bytes live: on the accelerator when the
    # worker has one, otherwise through entropack's eager CPU lane.
    compressed = entropack.CompressedTensor.unpack(state.to(torch.get_default_device()))
    return entropack.decompress(compressed, compressed.scheme.make_config({}))


@dataclass(frozen=True)
class EntroPackDiTCheckpoint:
    """What the tensor names and the sidecar of an EntroPack DiT package declare."""

    int8_layers: frozenset[str]
    # None: the INT8 layers were quantized without ConvRot.
    rotation_group_size: int | None
    adaln_table_rows: int

    @classmethod
    def from_transformer_dir(cls, root: Path) -> "EntroPackDiTCheckpoint | None":
        """The package under ``root``; None when its weights are not an EntroPack package."""
        packages = []
        for path in sorted(root.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as reader:
                names = set(reader.keys())
                if not any(name.endswith(STATE_SUFFIX) for name in names):
                    continue
                if ADALN_TABLE_KEY not in names:
                    raise ValueError(f"{path} is an EntroPack package of an unpruned DiT, which is not supported")
                table_rows = reader.get_slice(ADALN_TABLE_KEY).get_shape()[0]
            packages.append((path, names, table_rows))
        if not packages:
            return None
        if len(packages) > 1:
            raise ValueError(f"{root} holds several EntroPack packages; keep exactly one")
        path, names, table_rows = packages[0]

        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"EntroPack package {path.name} needs its sidecar {sidecar.name} next to it")
        config = json.loads(sidecar.read_text(encoding="utf-8"))
        if not str(config.get("model_class", "")).endswith(PRUNED_DIT_CLASS):
            raise ValueError(f"{sidecar} does not describe a {PRUNED_DIT_CLASS} package")
        quant_config = config["quant_config"]
        entries = quant_config.get("configs", [quant_config])
        methods = {entry["method"] for entry in entries}
        unsupported = sorted(method for method in methods if "fp8" in method)
        if unsupported:
            raise ValueError(f"unsupported EntroPack methods in {sidecar.name}: {unsupported}")

        int8_layers = frozenset(name.removesuffix(SCALE_SUFFIX) for name in names if name.endswith(SCALE_SUFFIX))
        int8_entries = [entry for entry in entries if entry["method"].startswith(INT8_METHOD_PREFIX)]
        if bool(int8_layers) != bool(int8_entries):
            raise ValueError(f"{sidecar.name} and {path.name} disagree on whether the package has INT8 layers")
        rotation_groups: set[int | None] = set()
        for entry in int8_entries:
            # use_convrot / convrot_groupsize are runtime settings of
            # entropack's INT8 linear: only the sidecar records them.
            options = entry.get("backend_config_kwargs", {})
            if options.get("use_convrot", True):
                rotation_groups.add(int(options.get("convrot_groupsize", 256)))
            else:
                rotation_groups.add(None)
        if len(rotation_groups) > 1:
            raise ValueError(f"{sidecar.name} mixes ConvRot settings across its INT8 layers")
        rotation_group = rotation_groups.pop() if rotation_groups else None
        return cls(int8_layers=int8_layers, rotation_group_size=rotation_group, adaln_table_rows=table_rows)


class EntroPackInt8LinearMethod(LinearMethodBase):
    """Serialized rotated W8A8: per-output-channel INT8 codes and dynamic per-token INT8 activations."""

    def __init__(self, requested_rotation_group: int | None) -> None:
        self.requested_rotation_group = requested_rotation_group

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        weight = create_weight_parameter(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            weight_loader=weight_loader,
            params_dtype=torch.int8,
        )
        layer.register_parameter("weight", weight)
        scale = ChannelQuantScaleParameter(
            data=torch.empty(output_size_per_partition, dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

        hadamard = None
        if self.requested_rotation_group is not None:
            # The package rotated groups of the full input; a TP shard of the
            # input must therefore hold whole groups.
            group = rotation_group_size(input_size, self.requested_rotation_group)
            if group is not None:
                if input_size_per_partition % group:
                    raise ValueError(
                        f"a {input_size_per_partition}-wide input shard splits the package's ConvRot groups of {group}"
                    )
                hadamard = regular_hadamard(group).to(params_dtype)
        layer.register_buffer("convrot_hadamard", hadamard, persistent=False)

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.convrot_hadamard is not None:
            x = rotate(x, layer.convrot_hadamard)
        qx, x_scale, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
        return ops.cutlass_scaled_mm(
            qx, layer.weight.t(), scale_a=x_scale, scale_b=layer.weight_scale, out_dtype=x.dtype, bias=bias
        )


class EntroPackDiTQuantConfig(QuantizationConfig):
    """Model-local checkpoint adapter; does not change the global quant registry."""

    def __init__(self, checkpoint: EntroPackDiTCheckpoint) -> None:
        super().__init__()
        self.checkpoint = checkpoint

    def get_name(self) -> str:
        return "entropack"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EntroPackDiTQuantConfig":
        raise ValueError("EntroPack quantization is selected from the transformer's serialized weights")

    def get_quant_method(self, layer: nn.Module, prefix: str) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if prefix in self.checkpoint.int8_layers:
            return EntroPackInt8LinearMethod(self.checkpoint.rotation_group_size)
        return UnquantizedLinearMethod()


def entropack_dit_weights(
    weights: Iterable[tuple[str, torch.Tensor]], *, rope_theta: float
) -> Iterator[tuple[str, torch.Tensor]]:
    """Translate an EntroPack package's tensors to the native H3 DiT weight names."""
    for name, tensor in weights:
        if name.endswith(STATE_SUFFIX):
            yield name.removesuffix(".entropack_quant_state"), decode_entropack_state(tensor)
        elif name.endswith(SCALE_SUFFIX):
            yield name.removesuffix(SCALE_SUFFIX) + ".weight_scale", tensor
        elif name == "rope.inv_freq":
            # The package stores this buffer rounded to BF16. Rebuild the FP32
            # values the released checkpoint holds, as the Diffusers layout does.
            freq_dim = tensor.numel()
            exponents = torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32) / (2 * freq_dim)
            yield name, 1.0 / (rope_theta**exponents)
        else:
            yield name, tensor
