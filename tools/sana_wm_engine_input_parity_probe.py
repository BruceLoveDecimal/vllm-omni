"""Probe SANA-WM engine-entered transformer input parity.

This script runs the vLLM-Omni Sana-WM native smoke path once, dumps the
Stage-1 transformer step-0 inputs/output, then rebuilds a standalone pipeline
with the same loader/context and re-calls ``transformer.forward`` on the
dumped inputs.

It answers one narrow question: does a byte-identical transformer input produce
the same ``noise_pred`` outside the engine-entered pipeline path?
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm.config import LoadConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader


WORKDIR = Path(os.environ.get("SANA_WM_PROBE_WORKDIR", "/root/autodl-tmp/probe-vllm-omni-feat-sana-wm"))
MODEL_ROOT = Path(
    os.environ.get(
        "SANA_WM_E2E_MODEL",
        "/root/autodl-tmp/hf-cache/hub/models--Efficient-Large-Model--SANA-WM_bidirectional/"
        "snapshots/90e0ff3b8f1f9b54a92b4b707edeaa27073aec84",
    )
)
DUMP_PATH = Path(
    os.environ.get(
        "SANA_WM_ENGINE_INPUT_DUMP",
        "/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs_recall.pt",
    )
)
PROMPT = os.environ.get("SANA_WM_PROBE_PROMPT", "A slow forward camera move through a quiet city street.")
NUM_FRAMES = int(os.environ.get("SANA_WM_PROBE_NUM_FRAMES", "9"))
STEPS = int(os.environ.get("SANA_WM_PROBE_STEPS", "20"))
SEED = int(os.environ.get("SANA_WM_PROBE_SEED", "0"))
FPS = int(os.environ.get("SANA_WM_PROBE_FPS", "16"))


def _set_env_defaults() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("SANA_WM_CAM_TRITON", "1")
    # Keep this unset while dumping: otherwise SanaWmPipeline.forward routes to
    # the NVlabs in-process backend instead of _run_native_smoke_backend.
    os.environ.pop("VLLM_OMNI_SANA_WM_OFFICIAL_REPO", None)
    os.environ.pop("VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI", None)
    if str(WORKDIR) not in sys.path:
        sys.path.insert(0, str(WORKDIR))


def _make_image() -> Image.Image:
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OUTPUT_HEIGHT, SANA_WM_OUTPUT_WIDTH

    return Image.new("RGB", (SANA_WM_OUTPUT_WIDTH, SANA_WM_OUTPUT_HEIGHT), (96, 128, 160))


def _make_camera(num_frames: int) -> np.ndarray:
    c2w = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    c2w[:, 2, 3] = -0.055 * np.arange(num_frames, dtype=np.float32)
    return c2w


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _diff(label: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().to(torch.float32).cpu()
    b = b.detach().to(torch.float32).cpu()
    if a.shape != b.shape:
        print(f"{label}: SHAPE_MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
        return {"cos": float("nan"), "mae": float("inf"), "max_abs": float("inf")}
    delta = a - b
    dot = float((a.flatten() * b.flatten()).sum())
    denom = float(a.flatten().norm() * b.flatten().norm())
    cos = dot / denom if denom else float("nan")
    mae = float(delta.abs().mean())
    max_abs = float(delta.abs().max())
    print(
        f"{label}: cos={cos:+.8f} mae={mae:.8f} max_abs={max_abs:.6f} "
        f"a_norm={float(a.norm()):.3f} b_norm={float(b.norm()):.3f}",
        flush=True,
    )
    return {"cos": cos, "mae": mae, "max_abs": max_abs}


def _vllm_config() -> VllmConfig:
    return VllmConfig()


def _od_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=str(MODEL_ROOT),
        model_class_name="SanaWmPipeline",
        enforce_eager=True,
    )


def phase_dump() -> None:
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OUTPUT_HEIGHT, SANA_WM_OUTPUT_WIDTH
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DUMP_PATH.exists():
        DUMP_PATH.unlink()
    os.environ["SANA_WM_DUMP_ENGINE_STEP0_INPUTS"] = str(DUMP_PATH)

    print(f"[probe] dump: frames={NUM_FRAMES} steps={STEPS} path={DUMP_PATH}", flush=True)
    omni = Omni(model=str(MODEL_ROOT), model_class_name="SanaWmPipeline", enforce_eager=True)
    extra_args = {
        "sana_wm_native_smoke": True,
        "sana_wm_output_type": "latent",
        "sana_wm_sampling_algo": "flow_euler_ltx",
        "sana_wm_native_smoke_max_tokens": 100000,
    }
    output = omni.generate(
        {
            "prompt": PROMPT,
            "multi_modal_data": {"image": _make_image()},
            "sana_wm": {
                "camera": {"poses": _make_camera(NUM_FRAMES)},
                "num_frames": NUM_FRAMES,
                "intrinsics": {
                    "fx": SANA_WM_OUTPUT_WIDTH / 2,
                    "fy": SANA_WM_OUTPUT_WIDTH / 2,
                    "cx": SANA_WM_OUTPUT_WIDTH / 2,
                    "cy": SANA_WM_OUTPUT_HEIGHT / 2,
                },
            },
        },
        OmniDiffusionSamplingParams(
            height=SANA_WM_OUTPUT_HEIGHT,
            width=SANA_WM_OUTPUT_WIDTH,
            num_frames=NUM_FRAMES,
            seed=SEED,
            fps=FPS,
            num_inference_steps=STEPS,
            guidance_scale=1.0,
            guidance_scale_provided=True,
            extra_args=extra_args,
        ),
    )
    req = output[0] if isinstance(output, list) else output
    if isinstance(req, OmniRequestOutput) and req.error is not None:
        raise RuntimeError(str(req.error))
    shutdown = getattr(omni, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del omni
    _cleanup_cuda()
    if not DUMP_PATH.exists():
        raise RuntimeError(f"dump not found: {DUMP_PATH}")
    print(f"[probe] dump done: {DUMP_PATH.stat().st_size // 1024} KiB", flush=True)


def _load_standalone_pipeline() -> Any:
    od_config = _od_config()
    vllm_config = _vllm_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[probe] recall: loading standalone pipeline with DiffusersPipelineLoader", flush=True)
    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_vllm_config(vllm_config),
    ):
        loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
        pipe = loader.load_model(
            od_config=od_config,
            load_device=str(device),
            load_format="default",
            device=device,
        )
    pipe.to(device)
    pipe.eval()
    transformer = pipe.transformer
    try:
        qkv = transformer.get_loaded_tensor("transformer.blocks.0.attn.qkv.weight")
        conv = transformer.get_loaded_tensor("transformer.blocks.0.mlp.point_conv.conv.weight")
        print(
            "[probe] post-load stored weight norms: "
            f"block0.qkv={float(qkv.float().norm()):.3f} "
            f"block0.mlp.point_conv={float(conv.float().norm()):.3f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[probe] stored weight norm check skipped: {type(exc).__name__}: {exc}", flush=True)
    return pipe, od_config, vllm_config, device


def phase_recall() -> None:
    print(f"[probe] recall: loading dump {DUMP_PATH}", flush=True)
    dump = torch.load(DUMP_PATH, map_location="cpu", weights_only=False)
    for key in ("latents", "model_timestep", "prompt_embeds", "plucker", "raymap", "spatial_raymap", "noise_pred"):
        value = dump.get(key)
        if isinstance(value, torch.Tensor):
            print(f"[probe] dump[{key}]: shape={tuple(value.shape)} dtype={value.dtype} norm={float(value.float().norm()):.3f}", flush=True)
        else:
            print(f"[probe] dump[{key}]: {type(value).__name__}", flush=True)

    pipe, od_config, vllm_config, device = _load_standalone_pipeline()
    transformer = pipe.transformer
    args = {
        "latents": dump["latents"].to(device).contiguous(),
        "timestep": dump["model_timestep"].to(device).contiguous(),
        "encoder_hidden_states": dump["prompt_embeds"].to(device).contiguous(),
        "plucker": dump["plucker"].to(device).contiguous() if isinstance(dump.get("plucker"), torch.Tensor) else None,
        "raymap": dump["raymap"].to(device).contiguous() if isinstance(dump.get("raymap"), torch.Tensor) else None,
        "spatial_raymap": (
            dump["spatial_raymap"].to(device).contiguous()
            if isinstance(dump.get("spatial_raymap"), torch.Tensor)
            else None
        ),
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_vllm_config(vllm_config),
        torch.inference_mode(),
    ):
        noise_pred = transformer(
            args["latents"],
            args["timestep"],
            encoder_hidden_states=args["encoder_hidden_states"],
            plucker=args["plucker"],
            raymap=args["raymap"],
            spatial_raymap=args["spatial_raymap"],
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[probe] standalone forward: {time.time() - start:.2f}s dtype={noise_pred.dtype}", flush=True)
    try:
        qkv_norm = float(transformer.blocks[0].attn.qkv.weight.float().norm())
        conv_norm = float(transformer.blocks[0].mlp.point_conv.conv.weight.float().norm())
        print(
            "[probe] materialized weight norms: "
            f"block0.qkv={qkv_norm:.3f} block0.mlp.point_conv={conv_norm:.3f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[probe] materialized weight norm check skipped: {type(exc).__name__}: {exc}", flush=True)

    engine_noise = dump["noise_pred"].to(device)
    print("[probe] === engine-entered output vs standalone recall ===", flush=True)
    _diff("full", noise_pred, engine_noise)
    if noise_pred.ndim == 5:
        _diff("frame0", noise_pred[:, :, :1], engine_noise[:, :, :1])
        if noise_pred.shape[2] > 1:
            _diff("generated", noise_pred[:, :, 1:], engine_noise[:, :, 1:])
            _diff("frame1", noise_pred[:, :, 1:2], engine_noise[:, :, 1:2])
            _diff(f"frame{noise_pred.shape[2] - 1}", noise_pred[:, :, -1:], engine_noise[:, :, -1:])


def main() -> None:
    _set_env_defaults()
    phase = os.environ.get("SANA_WM_PROBE_PHASE", "all").lower()
    if phase in {"all", "dump"}:
        phase_dump()
        _cleanup_cuda()
    if phase in {"all", "recall"}:
        phase_recall()


if __name__ == "__main__":
    main()
