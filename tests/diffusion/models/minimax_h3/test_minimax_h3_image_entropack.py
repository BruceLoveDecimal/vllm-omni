# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3 import entropack_checkpoint
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.entropack_checkpoint import (
    EntroPackDiTCheckpoint,
    EntroPackDiTQuantConfig,
    EntroPackInt8LinearMethod,
    decode_entropack_state,
    entropack_dit_weights,
    unpack_entropack_state,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_SEED_HADAMARD = torch.tensor(
    [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]]
)


@pytest.fixture
def single_rank(monkeypatch):
    """vLLM weight parameters read the TP group while they are constructed."""
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


def _pack_raw(tensor: torch.Tensor) -> torch.Tensor:
    """A packed entropack state holding ``tensor`` verbatim, laid out as entropack 0.1.7 writes it."""
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    metadata = {
        "buffer_dtypes": [dtype_name],
        "buffer_names": ["data"],
        "buffer_shapes": [list(tensor.shape)],
        "compress_method": "raw",
        "dtype": dtype_name,
        "header": {"compress_method": "raw"},
        "shape": list(tensor.shape),
    }
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    start = -(-(8 + len(raw)) // 16) * 16
    payload = tensor.contiguous().view(-1).view(torch.uint8)
    state = torch.zeros(-(-(start + payload.numel()) // 16) * 16, dtype=torch.uint8)
    state[:8].view(torch.int64).fill_(len(raw))
    state[8 : 8 + len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    state[start : start + payload.numel()] = payload
    return state


def _entropack_convrot_int8(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 codes and row scales as entropack's ConvRot INT8 linear quantizes a weight."""
    matrix = _SEED_HADAMARD
    while matrix.shape[0] < group:
        matrix = torch.kron(matrix, _SEED_HADAMARD)
    matrix = matrix / math.sqrt(group)
    out_features, in_features = weight.shape
    rotated = torch.matmul(weight.float().reshape(out_features, in_features // group, group), matrix.t())
    rotated = rotated.reshape(out_features, in_features)
    scale = (rotated.abs().amax(dim=1) / 127.0).clamp(min=torch.finfo(torch.float32).eps)
    codes = (rotated / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return codes, scale


def _write_package(root, tensors: dict[str, torch.Tensor], *, methods: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "dit_8bit_int8.safetensors"))
    sidecar = {
        "model_class": "diffsynth.models.minimax_h3_image_dit.MiniMaxH3ImageDiTPruned",
        "quant_config": {"load_prequantized": True, "configs": methods},
    }
    (root / "dit_8bit_int8.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _package_tensors() -> dict[str, torch.Tensor]:
    codes = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
    return {
        "blocks.0.attn.out_proj.weight.entropack_quant_state": _pack_raw(codes),
        "blocks.0.attn.out_proj.weight_quant_scale": torch.rand(8),
        "condition_proj.weight.entropack_quant_state": _pack_raw(torch.randn(8, 6).to(torch.bfloat16)),
        "blocks.0.adaln_proj.table": torch.zeros(7, 3, 6, 8, dtype=torch.bfloat16),
    }


_INT8_METHOD = {
    "method": "entropack_lossy_quant_int8_raw",
    "backend_config_kwargs": {"use_convrot": True, "convrot_groupsize": 256},
}
_BF16_METHOD = {"method": "entropack_lossy_quant", "backend_config_kwargs": {"target_bpp": 8.5}}


def test_raw_state_decodes_to_the_stored_codes_without_entropack(monkeypatch):
    monkeypatch.setattr(entropack_checkpoint, "entropack", None)
    codes = torch.randint(-127, 128, (5, 12), dtype=torch.int8)
    state = _pack_raw(codes)

    metadata, buffers = unpack_entropack_state(state)

    assert metadata["compress_method"] == "raw"
    assert metadata["shape"] == [5, 12]
    assert torch.equal(buffers["data"], codes)
    assert torch.equal(decode_entropack_state(state), codes)


def test_truncated_state_is_rejected():
    state = _pack_raw(torch.zeros(4, 32, dtype=torch.int8))
    with pytest.raises(ValueError, match="truncated"):
        unpack_entropack_state(state[:-16])


def test_entropy_coded_state_needs_entropack(monkeypatch):
    monkeypatch.setattr(entropack_checkpoint, "entropack", None)
    state = _pack_raw(torch.zeros(4, 16, dtype=torch.int8))
    header_len = int(state[:8].view(torch.int64).item())
    text = bytes(state[8 : 8 + header_len].tolist()).decode("utf-8")
    coded = text.replace('"compress_method":"raw"', '"compress_method":"abc"')
    state[8 : 8 + header_len] = torch.frombuffer(bytearray(coded.encode("utf-8")), dtype=torch.uint8)

    with pytest.raises(ImportError, match="entropack"):
        decode_entropack_state(state)


def test_entropy_coded_state_decodes_through_entropack():
    entropack = pytest.importorskip("entropack")
    torch.manual_seed(0)
    weight = (torch.randn(64, 256) * 0.02).to(torch.bfloat16)
    compressed = entropack.compress(weight, entropack.EntroPackLossyConfig(target_bpp=8.5))
    assert compressed.compress_method == "lattice_rans"

    decoded = decode_entropack_state(compressed.entropack_quant_state.clone())

    assert decoded.dtype == torch.bfloat16
    assert decoded.shape == weight.shape
    relative_error = (decoded.float() - weight.float()).norm() / weight.float().norm()
    assert relative_error < 0.05


def test_checkpoint_reads_int8_layers_rotation_and_table_rows(tmp_path):
    _write_package(tmp_path, _package_tensors(), methods=[_INT8_METHOD, _BF16_METHOD])

    checkpoint = EntroPackDiTCheckpoint.from_transformer_dir(tmp_path)

    assert checkpoint.int8_layers == frozenset({"blocks.0.attn.out_proj"})
    assert checkpoint.rotation_group_size == 256
    assert checkpoint.adaln_table_rows == 7


def test_checkpoint_without_convrot_keeps_no_rotation(tmp_path):
    method = {"method": "entropack_lossy_quant_int8_raw", "backend_config_kwargs": {"use_convrot": False}}
    _write_package(tmp_path, _package_tensors(), methods=[method, _BF16_METHOD])

    assert EntroPackDiTCheckpoint.from_transformer_dir(tmp_path).rotation_group_size is None


def test_plain_transformer_dir_is_not_a_package(tmp_path):
    save_file({"condition_proj.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))

    assert EntroPackDiTCheckpoint.from_transformer_dir(tmp_path) is None
    assert EntroPackDiTCheckpoint.from_transformer_dir(tmp_path / "missing") is None


def test_package_without_sidecar_is_rejected(tmp_path):
    _write_package(tmp_path, _package_tensors(), methods=[_INT8_METHOD])
    (tmp_path / "dit_8bit_int8.json").unlink()

    with pytest.raises(ValueError, match="sidecar"):
        EntroPackDiTCheckpoint.from_transformer_dir(tmp_path)


def test_fp8_package_is_rejected(tmp_path):
    method = {"method": "entropack_lossy_quant_fp8_raw", "backend_config_kwargs": {}}
    _write_package(tmp_path, _package_tensors(), methods=[method])

    with pytest.raises(ValueError, match="unsupported EntroPack methods"):
        EntroPackDiTCheckpoint.from_transformer_dir(tmp_path)


def test_weights_are_translated_to_native_names():
    codes = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
    scale = torch.rand(8)
    table = torch.zeros(7, 3, 6, 8, dtype=torch.bfloat16)
    package = [
        ("blocks.0.attn.out_proj.weight.entropack_quant_state", _pack_raw(codes)),
        ("blocks.0.attn.out_proj.weight_quant_scale", scale),
        ("blocks.0.adaln_proj.table", table),
        ("rope.inv_freq", torch.zeros(16, dtype=torch.bfloat16)),
    ]

    translated = dict(entropack_dit_weights(package, rope_theta=10000.0))

    assert list(translated) == [
        "blocks.0.attn.out_proj.weight",
        "blocks.0.attn.out_proj.weight_scale",
        "blocks.0.adaln_proj.table",
        "rope.inv_freq",
    ]
    assert torch.equal(translated["blocks.0.attn.out_proj.weight"], codes)
    assert torch.equal(translated["blocks.0.attn.out_proj.weight_scale"], scale)
    assert translated["blocks.0.adaln_proj.table"] is table
    expected_rope = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    assert translated["rope.inv_freq"].dtype == torch.float32
    assert torch.equal(translated["rope.inv_freq"], expected_rope)


def _reference_int8_quant(x, scale=None, azp=None, symmetric=True):
    del scale, azp, symmetric
    x_scale = (x.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=torch.finfo(torch.float32).eps)
    return (x.float() / x_scale).round().clamp(-128, 127).to(torch.int8), x_scale, None


def _reference_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    out = (a.double() @ b.double()) * scale_a.double().reshape(-1, 1) * scale_b.double().reshape(1, -1)
    if bias is not None:
        out = out + bias.double()
    return out.to(out_dtype)


def test_int8_method_reproduces_the_entropack_convrot_linear(monkeypatch, single_rank):
    monkeypatch.setattr(entropack_checkpoint.ops, "scaled_int8_quant", _reference_int8_quant, raising=False)
    monkeypatch.setattr(entropack_checkpoint.ops, "cutlass_scaled_mm", _reference_scaled_mm, raising=False)
    torch.manual_seed(0)
    in_features, out_features = 512, 24
    weight = torch.randn(out_features, in_features) * 0.05
    # An outlier column, the case the rotation is for.
    weight[:, 7] *= 30.0
    codes, scale = _entropack_convrot_int8(weight, 256)

    layer = nn.Module()
    method = EntroPackInt8LinearMethod(256)
    method.create_weights(layer, in_features, [out_features], in_features, out_features, torch.float32)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (out_features,)
    layer.weight.data.copy_(codes)
    layer.weight_scale.data.copy_(scale)

    x = torch.randn(40, in_features)
    out = method.apply(layer, x)

    reference = x @ weight.t()
    relative_error = (out - reference).norm() / reference.norm()
    assert out.shape == (40, out_features)
    assert relative_error < 0.02


def test_int8_method_follows_the_package_group_for_narrow_inputs(single_rank):
    layer = nn.Module()
    # 96 inputs hold no group of 256 or 64; entropack falls back to 16.
    EntroPackInt8LinearMethod(256).create_weights(layer, 96, [4], 96, 4, torch.bfloat16)
    assert layer.convrot_hadamard.shape == (16, 16)

    unrotated = nn.Module()
    EntroPackInt8LinearMethod(None).create_weights(unrotated, 96, [4], 96, 4, torch.bfloat16)
    assert unrotated.convrot_hadamard is None


def test_int8_method_rejects_an_input_shard_that_splits_groups(single_rank):
    with pytest.raises(ValueError, match="ConvRot groups"):
        EntroPackInt8LinearMethod(256).create_weights(nn.Module(), 384, [4], 768, 4, torch.bfloat16)


class _FakeLinear(nn.Module):
    def __init__(
        self,
        *args,
        bias=False,
        params_dtype=None,
        quant_config=None,
        prefix="",
        total_num_heads=None,
        total_num_kv_heads=None,
        **kwargs,
    ):
        del args, kwargs
        super().__init__()
        self.prefix = prefix
        self.quant_config = quant_config
        self.weight = nn.Parameter(torch.empty(1, dtype=params_dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, dtype=params_dtype))
        else:
            self.register_parameter("bias", None)
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads


class _FakeAttention(nn.Module):
    def __init__(self, **kwargs):
        del kwargs
        super().__init__()

    def add_schedule_candidate_check(self, check):
        del check


def _pruned_model(monkeypatch) -> h3.MiniMaxH3DiTModel:
    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    arch = {
        "num_layers": 2,
        "token_refiner_num_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "ffn_hidden_size": 16,
        "latents_dim": 2,
        "audio_latents_dim": 2,
        "patch_size": (1, 2, 2),
        "text_dim": 6,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 8,
        "time_embed_dim": 4,
        "adaln_out_features": 18 * 8,
        "final_adaln_out_features": 2 * 8,
        "rope_inv_freq_len": 2,
        "audio_enabled": False,
    }
    od_config = SimpleNamespace(tf_model_config=arch, parallel_config=SimpleNamespace(ulysses_degree=1))
    checkpoint = EntroPackDiTCheckpoint(
        int8_layers=frozenset({"blocks.0.attn.qkv_proj"}), rotation_group_size=256, adaln_table_rows=7
    )
    return h3.MiniMaxH3DiTModel(od_config, quant_config=EntroPackDiTQuantConfig(checkpoint), adaln_table_rows=7)


def test_pruned_dit_has_tables_and_no_projection_weights(monkeypatch):
    model = _pruned_model(monkeypatch)

    names = {name for name, _ in model.named_parameters()}
    assert model.time_embedder is None
    assert model.final_layer.video_out is None
    assert not any(".adaln_proj.linear" in name or name.startswith("time_embedder") for name in names)
    assert model.blocks[0].adaln_proj.table.shape == (7, 3, 6, 8)
    assert model.blocks[1].adaln_proj.table.shape == (7, 3, 6, 8)
    assert model.final_layer.adaln_proj.table.shape == (7, 1, 2, 8)
    assert model.blocks[0].adaln_proj.table.dtype == torch.bfloat16


def test_adaln_table_returns_the_projection_layout(monkeypatch):
    model = _pruned_model(monkeypatch)
    table_module = model.blocks[0].adaln_proj
    table = torch.arange(7 * 3 * 6 * 8, dtype=torch.float32).reshape(7, 3, 6, 8).to(torch.bfloat16)
    table_module.table.data.copy_(table)

    chunks = table_module(torch.tensor([2, 6]))

    assert len(chunks) == 6
    for chunk_index, chunk in enumerate(chunks):
        assert chunk.shape == (2 * 3, 8)
        # Row r * modality_num + m of chunk c is table[timestep r, modality m, c].
        assert torch.equal(chunk[1], table[2, 1, chunk_index])
        assert torch.equal(chunk[3 + 2], table[6, 2, chunk_index])


def test_table_rows_follow_the_step_and_the_reference_level(monkeypatch):
    model = _pruned_model(monkeypatch)

    assert model._adaln_table_timestep_rows(torch.tensor([0.25]), 3).tolist() == [3]
    assert model._adaln_table_timestep_rows(torch.tensor([0.25, 0.999]), 5).tolist() == [5, 6]
    with pytest.raises(ValueError, match="pdd_step"):
        model._adaln_table_timestep_rows(torch.tensor([0.25]), None)
    with pytest.raises(ValueError, match="pdd_step"):
        model._adaln_table_timestep_rows(torch.tensor([0.25]), 6)
    with pytest.raises(ValueError, match="one target and one reference"):
        model._adaln_table_timestep_rows(torch.tensor([0.1, 0.2, 0.999]), 0)


def test_pruned_final_layer_loads_fused_pdd_heads(monkeypatch, tmp_path):
    model = _pruned_model(monkeypatch)
    heads = {"video_out.weight": torch.randn(6, 8, 8), "video_out.bias": torch.randn(6, 8)}
    save_file(heads, str(tmp_path / "pdd_fused_heads.safetensors"))

    assert model.final_layer.load_fused_pdd_heads(str(tmp_path / "pdd_fused_heads.safetensors")) == 6
    assert model.final_layer.fused_pdd_heads["video_out.weight"].dtype == torch.float32


def test_quant_config_routes_int8_layers_only(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    checkpoint = EntroPackDiTCheckpoint(
        int8_layers=frozenset({"blocks.0.attn.qkv_proj"}), rotation_group_size=256, adaln_table_rows=7
    )
    config = EntroPackDiTQuantConfig(checkpoint)
    layer = object.__new__(LinearBase)

    assert isinstance(config.get_quant_method(layer, "blocks.0.attn.qkv_proj"), EntroPackInt8LinearMethod)
    assert isinstance(config.get_quant_method(layer, "condition_proj"), UnquantizedLinearMethod)
    assert config.get_quant_method(nn.Identity(), "blocks.0.attn.qkv_proj") is None
