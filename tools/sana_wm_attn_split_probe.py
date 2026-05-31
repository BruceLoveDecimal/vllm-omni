from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_OMNI_SANA_WM_OFFICIAL_REPO", "/root/autodl-tmp/NVlabs-Sana")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

WORKDIR = Path("/root/autodl-tmp/probe-vllm-omni-feat-sana-wm")
OFFICIAL_REPO = Path("/root/autodl-tmp/NVlabs-Sana")
MODEL_ROOT = Path(
    "/root/autodl-tmp/hf-cache/hub/models--Efficient-Large-Model--SANA-WM_bidirectional/"
    "snapshots/90e0ff3b8f1f9b54a92b4b707edeaa27073aec84"
)
OUTDIR = Path("/root/autodl-tmp/stage1_longseq_probe")
PREFIX_OFF_STEPS = OUTDIR / "official_321f_20step_controlled"
PROMPT = "A slow forward camera move through a quiet city street."
HEIGHT = 704
WIDTH = 1280
NUM_FRAMES = 321
SPATIAL_SHAPE = (41, 22, 40)
STEPS = tuple(int(s) for s in os.environ.get("SANA_WM_ATTN_SPLIT_STEPS", "15,18").split(",") if s.strip())
BLOCKS = tuple(int(s) for s in os.environ.get("SANA_WM_ATTN_SPLIT_BLOCKS", "0,15,18,19").split(",") if s.strip())
PROJ_CHECK_BLOCKS = tuple(
    int(s) for s in os.environ.get("SANA_WM_PROJ_CHECK_BLOCKS", "0").split(",") if s.strip()
)

if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_image() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), (96, 128, 160))


def make_camera(num_frames: int) -> np.ndarray:
    c2w = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    c2w[:, 2, 3] = -0.055 * np.arange(num_frames, dtype=np.float32)
    return c2w


def make_intrinsics(num_frames: int) -> np.ndarray:
    intr = np.array([WIDTH / 2.0, WIDTH / 2.0, WIDTH / 2.0, HEIGHT / 2.0], dtype=np.float32)
    return np.repeat(intr[None, :], num_frames, axis=0)


def load_official_module() -> Any:
    path = OFFICIAL_REPO / "inference_video_scripts/inference_sana_wm.py"
    spec = importlib.util.spec_from_file_location("nvlabs_inference_sana_wm", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def official_config(module: Any) -> Any:
    from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import resolve_sana_wm_local_paths

    paths = resolve_sana_wm_local_paths(MODEL_ROOT)
    config = module.pyrallis.parse(config_class=module.InferenceConfig, config_path=str(paths.config), args=[])
    config.vae.vae_pretrained = str(paths.root)
    return config, paths


def build_official_camera(module: Any, config: Any) -> dict[str, torch.Tensor]:
    image = make_image()
    _, src_size, resized_size, crop_offset = module.resize_and_center_crop(image)
    intr = module.transform_intrinsics_for_crop(make_intrinsics(NUM_FRAMES), src_size, resized_size, crop_offset)
    return module.prepare_camera(
        make_camera(NUM_FRAMES),
        intr,
        target_size=(HEIGHT, WIDTH),
        vae_stride=config.vae.vae_stride,
    )


def build_native_camera() -> dict[str, torch.Tensor]:
    from vllm_omni.diffusion.models.sana_wm.camera_control import (
        SanaWmCameraCondition,
        build_plucker_condition,
    )

    return build_plucker_condition(
        SanaWmCameraCondition(
            poses=make_camera(NUM_FRAMES),
            intrinsics={"fx": WIDTH / 2.0, "fy": WIDTH / 2.0, "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0},
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
        ),
        vae_stride=(8, 32, 32),
    )


def load_step(idx: int) -> dict[str, Any]:
    return torch.load(f"{PREFIX_OFF_STEPS}_step{idx}.pt", map_location="cpu", weights_only=True)


def to_cpu(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.is_floating_point() and t.dtype == torch.float32:
        t = t.to(torch.bfloat16)
    return t.cpu()


class Recorder:
    def __init__(self, side: str) -> None:
        self.side = side
        self.mode = "full"
        self.step = -1
        self.records: dict[str, dict[str, torch.Tensor]] = {}
        self.proj_params: dict[int, dict[str, torch.Tensor]] = {}
        self.proj_checks: dict[str, dict[str, Any]] = {}

    def key(self, block_idx: int) -> str:
        return f"{self.side}:{self.mode}:step{self.step}:block{block_idx}"

    def capture(self, block_idx: int, data: dict[str, torch.Tensor]) -> None:
        if block_idx not in BLOCKS:
            return
        self.records[self.key(block_idx)] = {
            name: to_cpu(value)
            for name, value in data.items()
            if torch.is_tensor(value)
        }
        print(
            f"[attn-split] captured {self.key(block_idx)} "
            + " ".join(f"{k}={tuple(v.shape)}" for k, v in self.records[self.key(block_idx)].items()),
            flush=True,
        )


def metric(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    d = a - b
    # Accumulate the cosine in fp64; fp32 reduction over 80M+ elements can
    # overshoot 1.0 enough to make the diagnostic misleading.
    a64 = a.flatten().double()
    b64 = b.flatten().double()
    dot = float((a64 * b64).sum())
    denom = float(a64.norm() * b64.norm())
    return {
        "mae": float(d.abs().mean()),
        "rmse": float(torch.sqrt((d * d).mean())),
        "max_abs": float(d.abs().max()),
        "cosine": dot / denom if denom else float("nan"),
        "native_norm": float(a.norm()),
        "official_norm": float(b.norm()),
    }


def compare_tensor(native: torch.Tensor, official: torch.Tensor) -> dict[str, Any]:
    if tuple(native.shape) != tuple(official.shape):
        return {"shape_mismatch": {"native": list(native.shape), "official": list(official.shape)}}
    out: dict[str, Any] = {"shape": list(native.shape), "global": metric(native, official)}
    if native.ndim == 3 and native.shape[1] % SPATIAL_SHAPE[0] == 0:
        spatial = native.shape[1] // SPATIAL_SHAPE[0]
        n4 = native.reshape(native.shape[0], SPATIAL_SHAPE[0], spatial, native.shape[2])
        o4 = official.reshape(official.shape[0], SPATIAL_SHAPE[0], spatial, official.shape[2])
        out["frame0"] = metric(n4[:, :1], o4[:, :1])
        out["generated"] = metric(n4[:, 1:], o4[:, 1:])
        out["last_frame"] = metric(n4[:, -1:], o4[:, -1:])
    return out


def compare_records(native: dict[str, torch.Tensor], official: dict[str, torch.Tensor]) -> dict[str, Any]:
    stages = sorted(set(native) & set(official))
    report = {stage: compare_tensor(native[stage], official[stage]) for stage in stages}
    attn_mae = report.get("attn_in", {}).get("generated", report.get("attn_in", {}).get("global", {})).get("mae")
    if attn_mae is not None and attn_mae > 0:
        for stage, values in report.items():
            scope = values.get("generated", values.get("global", {}))
            if "mae" in scope:
                scope["mae_vs_attn_in_mae"] = float(scope["mae"] / attn_mae)
    return report


def patch_official_attn(pipe: Any, recorder: Recorder) -> None:
    from diffusion.model.nets import sana_gdn_blocks as off_gdn
    from diffusion.model.nets import sana_gdn_blocks_triton as off_gdn_triton
    from diffusion.model.nets import sana_gdn_camctrl_blocks as off_cam

    def wrapped(self, x, mask=None, HW=None, rotary_emb=None, block_mask=None, camera_conditions=None, chunk_size=None, **kwargs):
        block_idx = getattr(self, "_probe_block_idx", -1)
        is_softmax = "Softmax" in type(self).__name__
        if is_softmax:
            main_raw = off_gdn._forward_softmax_attn(
                self,
                x,
                HW,
                rotary_emb,
                frame_causal=False,
                apply_output_gate=False,
                chunk_size=chunk_size,
                **kwargs,
            )
            cam_raw = None
            cam_contrib = 0
            if camera_conditions is not None:
                cam_raw = off_cam._forward_cam_branch_softmax(
                    self,
                    x,
                    HW,
                    camera_conditions,
                    rotary_emb,
                    frame_causal=False,
                    chunk_size=chunk_size,
                    **kwargs,
                )
                cam_contrib = self.out_proj_cam(cam_raw)
        else:
            precomputed_gates = self._compute_frame_gates(x, HW) if HW is not None else None
            main_raw = off_gdn_triton.BidirectionalGDNTriton.forward(
                self,
                x,
                mask=mask,
                HW=HW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
                apply_output_gate=False,
                chunk_size=chunk_size,
                precomputed_gates=precomputed_gates,
                **kwargs,
            )
            cam_raw = None
            cam_contrib = 0
            if camera_conditions is not None:
                cam_raw = self._forward_cam_branch(
                    x,
                    HW,
                    camera_conditions,
                    rotary_emb,
                    chunk_size=chunk_size,
                    precomputed_gates=precomputed_gates,
                    **kwargs,
                )
                cam_contrib = self.out_proj_cam(cam_raw)
        combined = main_raw + cam_contrib
        gate = F.silu(F.linear(x, self.output_gate.weight, self.output_gate.bias).to(torch.float32))
        pre_proj = combined * gate
        attn_out = self.proj(pre_proj.to(self.proj.weight.dtype))
        data = {
            "attn_in": x,
            "main_raw": main_raw,
            "combined": combined,
            "output_gate": gate,
            "pre_proj": pre_proj,
            "attn_out": attn_out,
        }
        if cam_raw is not None:
            data["cam_raw"] = cam_raw
            data["cam_contrib"] = cam_contrib
        recorder.capture(block_idx, data)
        return attn_out

    for i, block in enumerate(pipe.model.blocks):
        block.attn._probe_block_idx = i
        block.attn.forward = MethodType(wrapped, block.attn)


def patch_native_attn(transformer: Any, recorder: Recorder) -> None:
    import vllm_omni.diffusion.models.sana_wm.sana_wm_transformer as swt

    def wrapped(self, hidden_states, spatial_shape=None, rotary_emb=None, camera_conditions=None):
        block_idx = getattr(self, "block_idx", -1)
        if spatial_shape is not None and self.use_gdn:
            cam_disabled = os.environ.get("VLLM_OMNI_SANA_WM_DISABLE_CAM_BRANCH", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if camera_conditions is None or self.q_proj_cam is None or cam_disabled:
                main_raw, _ = self._forward_gdn_raw(hidden_states, spatial_shape, rotary_emb)
                gate = F.silu(swt._linear_output(self.output_gate(hidden_states)).float())
                pre_proj = main_raw * gate
                attn_out = swt._linear_output(self.proj(pre_proj.to(swt._linear_weight_dtype(self.proj))))
                recorder.capture(block_idx, {
                    "attn_in": hidden_states,
                    "main_raw": main_raw,
                    "combined": main_raw,
                    "output_gate": gate,
                    "pre_proj": pre_proj,
                    "attn_out": attn_out,
                })
                return attn_out
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
            main_raw, _ = self._forward_gdn_raw(
                hidden_states,
                spatial_shape,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_raw = self._forward_cam_branch(
                hidden_states,
                spatial_shape,
                camera_conditions,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_contrib = swt._linear_output(self.out_proj_cam(cam_raw))
            combined = main_raw + cam_contrib.to(main_raw.dtype)
            gate = F.silu(swt._linear_output(self.output_gate(hidden_states)).float())
            pre_proj = combined * gate
            attn_out = swt._linear_output(self.proj(pre_proj.to(swt._linear_weight_dtype(self.proj))))
            recorder.capture(block_idx, {
                "attn_in": hidden_states,
                "main_raw": main_raw,
                "cam_raw": cam_raw,
                "cam_contrib": cam_contrib,
                "combined": combined,
                "output_gate": gate,
                "pre_proj": pre_proj,
                "attn_out": attn_out,
            })
            return attn_out

        if spatial_shape is not None:
            cam_disabled = os.environ.get("VLLM_OMNI_SANA_WM_DISABLE_CAM_BRANCH", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            main_raw = self._forward_softmax_raw(hidden_states, spatial_shape, rotary_emb)
            cam_raw = None
            cam_contrib = 0
            if camera_conditions is not None and self.q_proj_cam is not None and not cam_disabled:
                cam_raw = self._forward_softmax_cam_branch(
                    hidden_states,
                    spatial_shape,
                    camera_conditions,
                    rotary_emb,
                )
                cam_contrib = swt._linear_output(self.out_proj_cam(cam_raw))
                combined = main_raw + cam_contrib.to(main_raw.dtype)
            else:
                combined = main_raw
            gate = F.silu(swt._linear_output(self.output_gate(hidden_states)).float())
            pre_proj = combined * gate
            attn_out = swt._linear_output(self.proj(pre_proj.to(swt._linear_weight_dtype(self.proj))))
            data = {
                "attn_in": hidden_states,
                "main_raw": main_raw,
                "combined": combined,
                "output_gate": gate,
                "pre_proj": pre_proj,
                "attn_out": attn_out,
            }
            if cam_raw is not None:
                data["cam_raw"] = cam_raw
                data["cam_contrib"] = cam_contrib
            recorder.capture(block_idx, data)
            return attn_out
        return self.__class__.forward(self, hidden_states, spatial_shape, rotary_emb, camera_conditions)

    for i, block in enumerate(transformer.blocks):
        block.attn.block_idx = i
        block.attn.forward = MethodType(wrapped, block.attn)


def run_official() -> tuple[Recorder, dict[int, torch.Tensor]]:
    module = load_official_module()
    config, paths = official_config(module)
    device = torch.device("cuda")
    pipe = module.SanaWMPipeline(
        config=config,
        model_path=str(paths.stage1_dit),
        device=device,
        refiner=None,
        offload_vae=True,
        offload_refiner=False,
    )
    pipe.model.eval()
    recorder = Recorder("official")
    patch_official_attn(pipe, recorder)
    for block_idx in set(BLOCKS) | set(PROJ_CHECK_BLOCKS):
        proj = pipe.model.blocks[block_idx].attn.proj
        recorder.proj_params[block_idx] = {
            "weight": proj.weight.detach().cpu(),
            "bias": proj.bias.detach().cpu() if getattr(proj, "bias", None) is not None else torch.zeros(
                proj.weight.shape[0],
                dtype=proj.weight.dtype,
            ),
        }
    camera = build_official_camera(module, config)
    cond, cond_mask, _neg, _neg_mask = pipe._encode_prompts(PROMPT, "")
    prompt_dump = torch.load(
        OUTDIR / "official_321f_20step_controlled_step0_full.pt",
        map_location="cpu",
        weights_only=True,
    )
    cond = prompt_dump["prompt_embeds"].to(device=device, dtype=pipe.weight_dtype)
    raymap = camera["raymap"].unsqueeze(0).to(device=device, dtype=pipe.weight_dtype)
    chunk_plucker = camera["chunk_plucker"].unsqueeze(0).to(device=device, dtype=pipe.weight_dtype)
    model_kwargs = {
        "data_info": {
            "img_hw": torch.tensor([[HEIGHT, WIDTH]], dtype=torch.float, device=device),
            "condition_frame_info": {0: 0.0},
        },
        "mask": cond_mask.to(device),
        "camera_conditions": raymap,
        "chunk_plucker": chunk_plucker,
    }
    outputs: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for step_idx in STEPS:
            step = load_step(step_idx)
            recorder.mode = "full"
            recorder.step = step_idx
            start = time.perf_counter()
            out = pipe.model(
                step["latent_in"].to(device=device, dtype=pipe.weight_dtype),
                step["timestep_per_frame"].to(device=device, dtype=torch.float32),
                cond,
                **model_kwargs,
            )
            torch.cuda.synchronize()
            print(
                f"[attn-split] official step{step_idx} full forward {time.perf_counter() - start:.1f}s "
                f"noise_norm={out.float().norm().item():.2f}",
                flush=True,
            )
            outputs[step_idx] = to_cpu(out)
    del pipe
    cleanup_cuda()
    return recorder, outputs


def build_native_transformer() -> tuple[Any, dict[str, torch.Tensor], torch.Tensor]:
    from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
    from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import SanaWmPipeline
    import vllm_omni.diffusion.models.sana_wm.sana_wm_transformer as swt

    swt.VllmRMSNorm = None
    swt.Attention = None
    od_config = SimpleNamespace(model=str(MODEL_ROOT), revision=None, quantization_config=None)
    pipe = SanaWmPipeline(od_config=od_config)
    paths = pipe.resolve_checkpoint(include_refiner=False)
    pipe.sana_wm_config = SanaWmConfig.from_yaml(paths.config)
    pipe.transformer.config = pipe.sana_wm_config
    state = load_file(str(paths.stage1_dit), device="cpu")
    pipe.load_weights(state.items())
    del state
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prompt_dump = torch.load(
        OUTDIR / "official_321f_20step_controlled_step0_full.pt",
        map_location="cpu",
        weights_only=True,
    )
    pipe.transformer.materialize(
        device=device,
        dtype=dtype,
        latent_channels=128,
        prompt_channels=int(prompt_dump["prompt_embeds"].shape[-1]),
    )
    cleanup_cuda()
    pipe.transformer.to(device=device, dtype=dtype)
    pipe.transformer.eval()
    camera = build_native_camera()
    prompt = prompt_dump["prompt_embeds"].squeeze(1).to(device=device, dtype=dtype)
    cam = {k: v.to(device=device, dtype=dtype) for k, v in camera.items() if torch.is_tensor(v)}
    return pipe.transformer, cam, prompt


def run_native(official_recorder: Recorder) -> tuple[Recorder, dict[int, torch.Tensor]]:
    import vllm_omni.diffusion.models.sana_wm.sana_wm_transformer as swt

    transformer, cam, prompt = build_native_transformer()
    recorder = Recorder("native")
    patch_native_attn(transformer, recorder)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    outputs: dict[int, torch.Tensor] = {}
    raymap = cam["raymap"]
    camera_conditions = raymap if raymap.ndim == 3 else raymap.unsqueeze(0)
    rotary_emb = transformer.rope(SPATIAL_SHAPE, device)
    with torch.inference_mode():
        for step_idx in STEPS:
            step = load_step(step_idx)
            recorder.mode = "full"
            recorder.step = step_idx
            start = time.perf_counter()
            out = transformer(
                step["latent_in"].to(device=device, dtype=dtype),
                step["timestep_per_frame"].to(device=device, dtype=torch.float32),
                encoder_hidden_states=prompt,
                plucker=cam["chunk_plucker"],
                raymap=cam["raymap"],
                spatial_raymap=cam["spatial_raymap"],
            )
            torch.cuda.synchronize()
            print(
                f"[attn-split] native step{step_idx} full forward {time.perf_counter() - start:.1f}s "
                f"noise_norm={out.float().norm().item():.2f}",
                flush=True,
            )
            outputs[step_idx] = to_cpu(out)

            recorder.mode = "on_official_attn_in"
            recorder.step = step_idx
            for block_idx in BLOCKS:
                off_key = f"official:full:step{step_idx}:block{block_idx}"
                attn_in = official_recorder.records[off_key]["attn_in"].to(device=device, dtype=dtype)
                _ = transformer.blocks[block_idx].attn(
                    attn_in,
                    SPATIAL_SHAPE,
                    rotary_emb,
                    camera_conditions,
                )
                del attn_in
                cleanup_cuda()
            for block_idx in PROJ_CHECK_BLOCKS:
                off_key = f"official:full:step{step_idx}:block{block_idx}"
                iso_key = f"native:on_official_attn_in:step{step_idx}:block{block_idx}"
                if off_key not in official_recorder.records or iso_key not in recorder.records:
                    continue
                off_pre = official_recorder.records[off_key]["pre_proj"].to(device=device, dtype=dtype)
                off_attn_out = official_recorder.records[off_key]["attn_out"]
                official_params = official_recorder.proj_params[block_idx]
                off_w = official_params["weight"].to(device=device)
                off_b = official_params["bias"].to(device=device)
                native_proj = transformer.blocks[block_idx].attn.proj
                native_w = native_proj.weight.detach()
                native_b = (
                    native_proj.bias.detach()
                    if getattr(native_proj, "bias", None) is not None
                    else torch.zeros(native_w.shape[0], device=device, dtype=native_w.dtype)
                )
                official_flinear = F.linear(off_pre.to(off_w.dtype), off_w, off_b)
                native_flinear = F.linear(off_pre.to(native_w.dtype), native_w, native_b)
                native_module = swt._linear_output(
                    native_proj(off_pre.to(swt._linear_weight_dtype(native_proj)))
                )
                recorder.proj_checks[f"step{step_idx}:block{block_idx}"] = {
                    "native_weight_vs_official_weight": metric(native_w.detach().cpu(), official_params["weight"]),
                    "native_bias_vs_official_bias": metric(native_b.detach().cpu(), official_params["bias"]),
                    "official_flinear_vs_official_module": metric(to_cpu(official_flinear), off_attn_out),
                    "native_flinear_vs_official_flinear_on_official_preproj": metric(
                        to_cpu(native_flinear),
                        to_cpu(official_flinear),
                    ),
                    "native_module_vs_official_flinear_on_official_preproj": metric(
                        to_cpu(native_module),
                        to_cpu(official_flinear),
                    ),
                    "native_module_vs_native_flinear_on_official_preproj": metric(
                        to_cpu(native_module),
                        to_cpu(native_flinear),
                    ),
                }
                del off_pre, official_flinear, native_flinear, native_module
                cleanup_cuda()
    return recorder, outputs


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    official_rec, official_noise = run_official()
    native_rec, native_noise = run_native(official_rec)
    report: dict[str, Any] = {
        "steps": list(STEPS),
        "blocks": list(BLOCKS),
        "official_main_path": "BidirectionalGDNTriton.forward",
        "projection_checks": {},
        "noise_pred": {},
        "full_native_vs_official": {},
        "native_on_official_attn_in_vs_official": {},
    }
    for step_idx in STEPS:
        report["noise_pred"][str(step_idx)] = compare_tensor(native_noise[step_idx], official_noise[step_idx])
        for block_idx in BLOCKS:
            off_key = f"official:full:step{step_idx}:block{block_idx}"
            full_key = f"native:full:step{step_idx}:block{block_idx}"
            iso_key = f"native:on_official_attn_in:step{step_idx}:block{block_idx}"
            report["full_native_vs_official"][f"step{step_idx}:block{block_idx}"] = compare_records(
                native_rec.records[full_key],
                official_rec.records[off_key],
            )
            report["native_on_official_attn_in_vs_official"][f"step{step_idx}:block{block_idx}"] = compare_records(
                native_rec.records[iso_key],
                official_rec.records[off_key],
            )
    report["projection_checks"] = native_rec.proj_checks
    tag = os.environ.get("SANA_WM_ATTN_SPLIT_TAG", f"blocks{'_'.join(str(i) for i in BLOCKS)}")
    out = OUTDIR / f"attn_split_probe_321f_steps{'_'.join(str(i) for i in STEPS)}_{tag}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    compact: dict[str, Any] = {"noise_pred": {}, "isolated_attn": {}, "full_attn": {}}
    for step_idx in STEPS:
        compact["noise_pred"][str(step_idx)] = report["noise_pred"][str(step_idx)].get(
            "generated", report["noise_pred"][str(step_idx)]["global"]
        )
        for block_idx in BLOCKS:
            key = f"step{step_idx}:block{block_idx}"
            compact["isolated_attn"][key] = {
                stage: values.get("generated", values.get("global", {}))
                for stage, values in report["native_on_official_attn_in_vs_official"][key].items()
                if stage in {"attn_in", "main_raw", "cam_raw", "cam_contrib", "combined", "output_gate", "pre_proj", "attn_out"}
            }
            compact["full_attn"][key] = {
                stage: values.get("generated", values.get("global", {}))
                for stage, values in report["full_native_vs_official"][key].items()
                if stage in {"attn_in", "main_raw", "cam_raw", "cam_contrib", "combined", "output_gate", "pre_proj", "attn_out"}
            }
    compact["projection_checks"] = report["projection_checks"]
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)
    print(f"[attn-split] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
