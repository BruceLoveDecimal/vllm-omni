# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released H3-World rank-32 DiffSynth LoRA, using H3's existing LoRA manager."""

from pathlib import Path

import torch
from safetensors import safe_open

H3_WORLD_FILENAME = "step-10000.safetensors"
_TARGETS = {
    f"{prefix}.{index}.attn.{projection}"
    for prefix, count in (("blocks", 50), ("token_refiner.blocks", 2))
    for index in range(count)
    for projection in ("qkv_proj", "out_proj")
}


def validate_world_checkpoint(checkpoint) -> dict[str, torch.Tensor]:
    expected = {
        target + f".lora_{side}.default.weight": (
            (32, 5376 if target.endswith("qkv_proj") else 7168)
            if side == "A"
            else (21504 if target.endswith("qkv_proj") else 5376, 32)
        )
        for target in _TARGETS
        for side in ("A", "B")
    }
    if set(checkpoint.keys()) != set(expected):
        raise ValueError("H3-World requires all 104 qkv/out LoRA pairs and no extra tensors")
    tensors = {}
    for name, shape in expected.items():
        if tuple(checkpoint.get_slice(name).get_shape()) != shape:
            raise ValueError(f"H3-World tensor {name} must have shape {shape}")
        tensors["transformer." + name] = checkpoint.get_tensor(name)
    return tensors


def load_h3_world_lora(*, partition, lora_request, lora_path, dtype, unsupported_offload_mode=None):
    path = Path(lora_path)
    if path.is_dir():
        path = path / H3_WORLD_FILENAME
    if path.name != H3_WORLD_FILENAME or not path.is_file():
        return None
    if partition != "fl2va":
        raise ValueError("H3-World requires the MiniMax-H3 FL2VA base checkpoint")
    if unsupported_offload_mode:
        raise ValueError(f"H3-World dynamic LoRA does not support {unsupported_offload_mode}")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        tensors = validate_world_checkpoint(checkpoint)
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    from .npu.lora import _NATIVE_WEIGHTS_MAPPER, _pack_h3_native_qkv

    # DiffSynth hotload uses B @ A with unit scaling; alpha must equal rank.
    helper = PEFTHelper.from_dict(
        {
            "r": 32,
            "lora_alpha": 32,
            "target_modules": r"^transformer\.(?:token_refiner\.blocks|blocks)\.\d+\.attn\.(?:qkv_proj|out_proj)$",
        }
    )
    model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_NATIVE_WEIGHTS_MAPPER,
    )
    # Native H3 stores grouped QKV; vLLM's QKVParallelLinear stores Q/K/V slices.
    _pack_h3_native_qkv(model)
    return model, helper
