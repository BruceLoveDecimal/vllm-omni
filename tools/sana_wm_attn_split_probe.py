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
TEACHER_FORCE_BLOCKS = tuple(
    int(s) for s in os.environ.get("SANA_WM_TEACHER_FORCE_BLOCKS", "0").split(",") if s.strip()
)
RAW_CHECK_BLOCKS = tuple(
    int(s) for s in os.environ.get("SANA_WM_RAW_CHECK_BLOCKS", "").split(",") if s.strip()
)
BLOCK_STAGE_BLOCKS = tuple(
    int(s) for s in os.environ.get("SANA_WM_BLOCK_STAGE_BLOCKS", "").split(",") if s.strip()
)
USE_OFFICIAL_CAM_PREP_IN_NATIVE = os.environ.get("SANA_WM_USE_OFFICIAL_CAM_PREP_IN_NATIVE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
USE_OFFICIAL_MAIN_GDN_IN_NATIVE = os.environ.get("SANA_WM_USE_OFFICIAL_MAIN_GDN_IN_NATIVE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
COMPACT_STAGES = {
    "attn_in",
    "main_raw",
    "cam_raw",
    "cam_contrib",
    "combined",
    "output_gate",
    "pre_proj",
    "attn_out",
    "q_pre",
    "k_pre",
    "v_pre",
    "k_conv",
    "q_inv_rms",
    "k_inv_rms",
    "beta",
    "decay",
    "cam_q_raw",
    "cam_k_raw",
    "cam_v_raw",
    "cam_q_trans",
    "cam_k_trans",
    "cam_v_trans",
    "cam_inflation",
    "cam_beta",
    "cam_scan_out",
    "cam_raw_probe",
}

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
        self.prompt_mask: torch.Tensor | None = None

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


def _load_block_stage_tensor(side_prefix: Path, block_idx: int, stage: str) -> torch.Tensor | None:
    prefix = Path(f"{side_prefix}_block{block_idx}")
    if stage == "input":
        path = Path(f"{prefix}_input.pt")
        return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
    if stage == "post_cross_attn":
        path = Path(f"{prefix}_post_cross_attn.pt")
        return torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
    if stage in {"x_msa_in", "attn_output", "post_attn_residual"}:
        path = Path(f"{prefix}_post_attn.pt")
        if not path.exists():
            return None
        data = torch.load(path, map_location="cpu", weights_only=True)
        if stage == "attn_output":
            return data.get("attn_output", data.get("attn_out"))
        return data.get(stage)
    if stage in {"x_mlp_in", "mlp_out", "block_output"}:
        path = Path(f"{prefix}_output.pt")
        if not path.exists():
            return None
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data.get(stage)
    if stage in {"shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"}:
        path = Path(f"{prefix}_modulation.pt")
        if not path.exists():
            return None
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data.get(stage)
    return None


def compare_block_stage_dumps() -> dict[str, Any]:
    block_dump_prefix = os.environ.get("SANA_WM_BLOCK_STAGE_DUMP_PREFIX", "")
    if not block_dump_prefix or not BLOCK_STAGE_BLOCKS:
        return {}
    stages = [
        "input",
        "shift_msa",
        "scale_msa",
        "gate_msa",
        "shift_mlp",
        "scale_mlp",
        "gate_mlp",
        "x_msa_in",
        "attn_output",
        "post_attn_residual",
        "post_cross_attn",
        "x_mlp_in",
        "mlp_out",
        "block_output",
    ]
    native_prefix = Path(f"{block_dump_prefix}_native")
    official_prefix = Path(f"{block_dump_prefix}_official")
    report: dict[str, Any] = {}
    for block_idx in BLOCK_STAGE_BLOCKS:
        block_report: dict[str, Any] = {}
        for stage in stages:
            native = _load_block_stage_tensor(native_prefix, block_idx, stage)
            official = _load_block_stage_tensor(official_prefix, block_idx, stage)
            if native is None or official is None:
                continue
            block_report[stage] = compare_tensor(native, official)
        report[f"block{block_idx}"] = block_report
    return report


def patch_official_attn(pipe: Any, recorder: Recorder) -> None:
    from diffusion.model.nets import sana_gdn_blocks as off_gdn
    from diffusion.model.nets import sana_gdn_blocks_triton as off_gdn_triton
    from diffusion.model.nets import sana_gdn_camctrl_blocks as off_cam
    from diffusion.model.nets.sana_camctrl_blocks import _prepare_ray_apply_fns as off_prepare_ray_apply_fns
    from diffusion.model.ops.fused_cam_gdn import (
        _invert_SE3 as off_invert_SE3,
        _prepare_ucpe_rope_tables as off_prepare_ucpe_rope_tables,
        _process_camera_conditions_raymats_only as off_process_camera_conditions_raymats_only,
        cam_prep_func as off_cam_prep_func,
    )
    from diffusion.model.ops.fused_gdn import fused_qk_inv_rms as off_fused_qk_inv_rms
    from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise as off_cam_scan_bidi_chunkwise

    def wrapped(self, x, mask=None, HW=None, rotary_emb=None, block_mask=None, camera_conditions=None, chunk_size=None, **kwargs):
        block_idx = getattr(self, "_probe_block_idx", -1)
        is_softmax = "Softmax" in type(self).__name__
        raw_data: dict[str, torch.Tensor] = {}

        def maybe_capture_official_cam_raw(precomputed_gates: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
            if block_idx not in RAW_CHECK_BLOCKS or camera_conditions is None or HW is None or is_softmax:
                return {}
            B, N, _hidden_size = x.shape
            T, H_sp, W_sp = HW
            spatial_tokens = H_sp * W_sp
            dtype_orig = x.dtype
            cam_heads = self.cam_heads
            cam_head_dim = self.cam_head_dim

            qkv_w = torch.cat([self.q_proj_cam.weight, self.k_proj_cam.weight, self.v_proj_cam.weight])
            qkv_b = torch.cat([self.q_proj_cam.bias, self.k_proj_cam.bias, self.v_proj_cam.bias])
            qkv_cam = F.linear(x, qkv_w, qkv_b)
            q_raw, k_raw, v_raw = qkv_cam.chunk(3, dim=-1)
            if self.conv_k_cam is not None:
                k_raw = self._apply_temporal_short_conv(k_raw, self.conv_k_cam, HW)
            if self.conv_q_cam is not None:
                q_raw = self._apply_temporal_short_conv(q_raw, self.conv_q_cam, HW)
            if self.conv_v_cam is not None:
                v_raw = self._apply_temporal_short_conv(v_raw, self.conv_v_cam, HW)

            q_raw_heads = q_raw.contiguous().view(B, N, cam_heads, cam_head_dim).contiguous()
            k_raw_heads = k_raw.contiguous().view(B, N, cam_heads, cam_head_dim).contiguous()
            v_raw_heads = v_raw.contiguous().view(B, N, cam_heads, cam_head_dim).contiguous()

            raymats = off_process_camera_conditions_raymats_only(camera_conditions, B, HW, self.patch_size)
            raymats = raymats.reshape(B, -1, 4, 4)
            P = raymats
            P_T = P.transpose(-1, -2).contiguous()
            P_inv = off_invert_SE3(P).contiguous()

            if rotary_emb is not None:
                head_dim = cam_head_dim
                orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
                orig_h_size = head_dim // 6
                new_head_dim = head_dim // 2
                new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
                new_h_size = new_head_dim // 6
                new_w_size = new_head_dim // 6
                t_part = rotary_emb[..., :new_t_size]
                h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
                w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
                rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
                rope_cos, rope_sin = off_prepare_ucpe_rope_tables(rotary_emb_cam, N, cam_head_dim // 2, x.device)
            else:
                rotary_emb_cam = None
                rope_cos = torch.ones(N, cam_head_dim // 2, device=x.device, dtype=torch.float32)
                rope_sin = torch.zeros(N, cam_head_dim // 2, device=x.device, dtype=torch.float32)

            q_norm_w = self.q_norm_cam.weight.float().contiguous()
            k_norm_w = self.k_norm_cam.weight.float().contiguous()
            k_scale = (cam_head_dim**-0.5) * (spatial_tokens**-0.5)
            norm_eps_val = float(
                getattr(
                    self.q_norm_cam,
                    "eps",
                    getattr(self.q_norm_cam, "variance_epsilon", 1e-6),
                )
            )
            q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = off_cam_prep_func(
                q_raw_heads,
                k_raw_heads,
                v_raw_heads,
                q_norm_weight=q_norm_w,
                k_norm_weight=k_norm_w,
                proj_q=P_T,
                proj_kv=P_inv,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                k_scale=k_scale,
                norm_eps=norm_eps_val,
            )
            inflation_sq = inflation_sq.view(B, cam_heads, 1, N)
            beta, decay = precomputed_gates
            frame_inflation_sq = inflation_sq.view(B, cam_heads, T, spatial_tokens).mean(dim=-1)
            if beta.ndim == 3:
                beta_cam = beta / frame_inflation_sq.clamp_min(1.0)
            else:
                beta_cam = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

            if getattr(self, "fp32_attention", True):
                q_cam_trans = q_cam_trans.float()
                k_cam_trans = k_cam_trans.float()
                v_cam_trans = v_cam_trans.float()
                beta_cam = beta_cam.float()
                decay = decay.float()
            if beta_cam.ndim == 3:
                beta_scan = beta_cam.unsqueeze(-1).expand(B, cam_heads, T, spatial_tokens).contiguous()
            else:
                beta_scan = beta_cam.contiguous()
            decay_scan = decay.contiguous()
            q_cam_trans = q_cam_trans.contiguous()
            k_cam_trans = k_cam_trans.contiguous()
            v_cam_trans = v_cam_trans.contiguous()

            scan_out = off_cam_scan_bidi_chunkwise(q_cam_trans, k_cam_trans, v_cam_trans, beta_scan, decay_scan)
            scan_for_inverse = scan_out
            if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
                scan_for_inverse = scan_for_inverse.to(dtype_orig)
            _, _, apply_fn_o = off_prepare_ray_apply_fns(
                head_dim=cam_head_dim,
                P=P,
                P_T=P_T,
                P_inv=P_inv,
                rotary_emb=rotary_emb_cam,
            )
            cam_raw_probe = apply_fn_o(scan_for_inverse.transpose(-1, -2)).transpose(-1, -2).contiguous()
            cam_raw_probe = cam_raw_probe.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
            return {
                "cam_q_raw": q_raw_heads,
                "cam_k_raw": k_raw_heads,
                "cam_v_raw": v_raw_heads,
                "cam_q_trans": q_cam_trans,
                "cam_k_trans": k_cam_trans,
                "cam_v_trans": v_cam_trans,
                "cam_inflation": frame_inflation_sq,
                "cam_beta": beta_scan,
                "cam_scan_out": scan_out,
                "cam_raw_probe": cam_raw_probe,
            }

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
            if block_idx in RAW_CHECK_BLOCKS:
                B, N, C = x.shape
                H_heads, D_head = self.heads, self.dim
                qkv_pre = self.qkv(x).reshape(B, N, 3, H_heads, D_head)
                qkv_conv = qkv_pre.clone()
                if self.conv_k is not None:
                    k_raw = qkv_conv[:, :, 1].contiguous().reshape(B, N, C)
                    k_conv = self._apply_temporal_short_conv(k_raw, self.conv_k, HW)
                    qkv_conv[:, :, 1].copy_(k_conv.reshape(B, N, H_heads, D_head))
                q_inv_rms, k_inv_rms = off_fused_qk_inv_rms(
                    qkv_conv.contiguous(),
                    eps=float(getattr(self.q_norm, "eps", 1e-5)),
                )
                beta, decay = precomputed_gates
                raw_data = {
                    "q_pre": qkv_pre[:, :, 0],
                    "k_pre": qkv_pre[:, :, 1],
                    "v_pre": qkv_pre[:, :, 2],
                    "k_conv": qkv_conv[:, :, 1],
                    "q_inv_rms": q_inv_rms,
                    "k_inv_rms": k_inv_rms,
                    "beta": beta,
                    "decay": decay,
                }
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
            raw_data.update(maybe_capture_official_cam_raw(precomputed_gates))
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
        data.update(raw_data)
        if cam_raw is not None:
            data["cam_raw"] = cam_raw
            data["cam_contrib"] = cam_contrib
        recorder.capture(block_idx, data)
        return attn_out

    for i, block in enumerate(pipe.model.blocks):
        block.attn._probe_block_idx = i
        block.attn.forward = MethodType(wrapped, block.attn)


def patch_native_attn(transformer: Any, recorder: Recorder) -> None:
    from diffusion.model.nets.sana_camctrl_blocks import _prepare_ray_apply_fns as off_prepare_ray_apply_fns
    from diffusion.model.ops.fused_cam_gdn import (
        _invert_SE3 as off_invert_SE3,
        _prepare_ucpe_rope_tables as off_prepare_ucpe_rope_tables,
        _process_camera_conditions_raymats_only as off_process_camera_conditions_raymats_only,
        cam_prep_func as off_cam_prep_func,
    )
    from diffusion.model.ops.fused_gdn import (
        fused_bigdn_func as off_fused_bigdn_func,
        fused_qk_inv_rms as off_fused_qk_inv_rms,
        prepare_rope_tables as off_prepare_rope_tables,
    )
    from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise as off_cam_scan_bidi_chunkwise
    from vllm_omni.diffusion.models.sana_wm.fused_gdn import fused_qk_inv_rms
    import vllm_omni.diffusion.models.sana_wm.sana_wm_transformer as swt

    def wrapped(self, hidden_states, spatial_shape=None, rotary_emb=None, camera_conditions=None):
        block_idx = getattr(self, "block_idx", -1)
        raw_data: dict[str, torch.Tensor] = {}

        def maybe_capture_raw_gdn(beta: torch.Tensor, decay: torch.Tensor) -> dict[str, torch.Tensor]:
            if block_idx not in RAW_CHECK_BLOCKS:
                return {}
            B, N, _hidden_size = hidden_states.shape
            qkv_linear = swt._linear_output(self.qkv(hidden_states))
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            query, key_pre, value = qkv_linear.split([q_size, kv_size, kv_size], dim=-1)
            key_conv = key_pre
            if self.conv_k is not None:
                key_conv = self._bidirectional_temporal_short_conv(key_pre, self.conv_k, spatial_shape)
            qkv_conv = torch.stack(
                (
                    query.reshape(B, N, self.num_heads, self.head_dim),
                    key_conv.reshape(B, N, self.num_heads, self.head_dim),
                    value.reshape(B, N, self.num_heads, self.head_dim),
                ),
                dim=2,
            ).contiguous()
            q_inv_rms, k_inv_rms = fused_qk_inv_rms(
                qkv_conv,
                eps=float(getattr(self.q_norm, "eps", 1e-5)),
            )
            return {
                "q_pre": qkv_conv[:, :, 0],
                "k_pre": key_pre.reshape(B, N, self.num_heads, self.head_dim),
                "v_pre": qkv_conv[:, :, 2],
                "k_conv": qkv_conv[:, :, 1],
                "q_inv_rms": q_inv_rms,
                "k_inv_rms": k_inv_rms,
                "beta": beta,
                "decay": decay,
            }

        def native_main_raw_with_official_gdn(
            beta: torch.Tensor,
            decay: torch.Tensor,
        ) -> torch.Tensor:
            if spatial_shape is None:
                raise ValueError("official main-GDN ablation requires spatial_shape.")
            B, N, _hidden_size = hidden_states.shape
            frames, height, width = spatial_shape
            spatial_tokens = height * width
            qkv_linear = swt._linear_output(self.qkv(hidden_states))
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            query, key, value = qkv_linear.split([q_size, kv_size, kv_size], dim=-1)
            if self.conv_k is not None:
                key = self._bidirectional_temporal_short_conv(key, self.conv_k, spatial_shape)
            qkv = torch.stack(
                (
                    query.reshape(B, N, self.num_heads, self.head_dim),
                    key.reshape(B, N, self.num_heads, self.head_dim),
                    value.reshape(B, N, self.num_heads, self.head_dim),
                ),
                dim=2,
            ).contiguous()
            q_inv_rms, k_inv_rms = off_fused_qk_inv_rms(qkv, eps=float(getattr(self.q_norm, "eps", 1e-5)))
            q_norm_weight = (
                self.q_norm.weight.float().contiguous()
                if isinstance(getattr(self.q_norm, "weight", None), torch.Tensor)
                else torch.ones(q_size, device=hidden_states.device, dtype=torch.float32)
            )
            k_norm_weight = (
                self.k_norm.weight.float().contiguous()
                if isinstance(getattr(self.k_norm, "weight", None), torch.Tensor)
                else torch.ones(q_size, device=hidden_states.device, dtype=torch.float32)
            )
            rope_cos, rope_sin = off_prepare_rope_tables(rotary_emb, N, self.head_dim, hidden_states.device)
            out = off_fused_bigdn_func(
                qkv,
                q_inv_rms,
                k_inv_rms,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                beta=beta.contiguous(),
                decay=decay.contiguous(),
                F=frames,
                S=spatial_tokens,
                k_scale=(self.head_dim**-0.5) * (spatial_tokens**-0.5),
                eps=self.eps,
            )
            return out.reshape(B, N, q_size)

        def maybe_capture_native_cam_raw(beta: torch.Tensor, decay: torch.Tensor) -> dict[str, torch.Tensor]:
            if (
                block_idx not in RAW_CHECK_BLOCKS
                or camera_conditions is None
                or spatial_shape is None
                or self.q_proj_cam is None
            ):
                return {}
            B, N, _hidden_size = hidden_states.shape
            frames, height, width = spatial_shape
            spatial_tokens = height * width
            rotary_emb_freqs = self._ucpe_rotary_freqs(rotary_emb)
            apply_fn_q, apply_fn_kv, apply_fn_o = swt.prepare_prope_fns(
                head_dim=self.cam_head_dim,
                camera_conditions=camera_conditions,
                HW=spatial_shape,
                patch_size=self.patch_size,
                rotary_emb=rotary_emb_freqs,
            )

            q_cam = swt._linear_output(self.q_proj_cam(hidden_states))
            k_cam = swt._linear_output(self.k_proj_cam(hidden_states))
            v_cam = swt._linear_output(self.v_proj_cam(hidden_states))
            if self.conv_k_cam is not None:
                k_cam = self._bidirectional_temporal_short_conv(k_cam, self.conv_k_cam, spatial_shape)
            if self.conv_q_cam is not None:
                q_cam = self._bidirectional_temporal_short_conv(q_cam, self.conv_q_cam, spatial_shape)
            if self.conv_v_cam is not None:
                v_cam = self._bidirectional_temporal_short_conv(v_cam, self.conv_v_cam, spatial_shape)

            q_raw_heads = q_cam.reshape(B, N, self.cam_heads, self.cam_head_dim)
            k_raw_heads = k_cam.reshape(B, N, self.cam_heads, self.cam_head_dim)
            v_raw_heads = v_cam.reshape(B, N, self.cam_heads, self.cam_head_dim)

            q_cam = self.q_norm_cam(q_cam)
            k_cam = self.k_norm_cam(k_cam)
            q_cam = q_cam.reshape(B, N, self.cam_heads, self.cam_head_dim).transpose(1, 2)
            k_cam = k_cam.reshape(B, N, self.cam_heads, self.cam_head_dim).transpose(1, 2)
            v_cam = v_cam.reshape(B, N, self.cam_heads, self.cam_head_dim).transpose(1, 2)
            q_cam = F.relu(q_cam)
            k_cam = F.relu(k_cam)
            k_scale = (self.cam_head_dim**-0.5) * (spatial_tokens**-0.5)
            k_cam = k_cam * k_scale

            pre_ucpe_k_norm = torch.linalg.vector_norm(k_cam, dim=-1, keepdim=True).clamp_min(1e-6)
            q_cam_trans_bhnd = apply_fn_q(q_cam)
            kv_cam_trans = apply_fn_kv(torch.cat([k_cam, v_cam], dim=1))
            k_cam_trans_bhnd, v_cam_trans_bhnd = torch.chunk(kv_cam_trans, chunks=2, dim=1)
            post_ucpe_k_norm = torch.linalg.vector_norm(k_cam_trans_bhnd, dim=-1, keepdim=True).clamp_min(1e-6)
            inflation_sq = (post_ucpe_k_norm / pre_ucpe_k_norm) ** 2
            frame_inflation_sq = inflation_sq.squeeze(-1).reshape(
                B,
                self.cam_heads,
                frames,
                spatial_tokens,
            ).mean(dim=-1)
            if beta.shape[1] != self.cam_heads:
                repeat_factor = self.cam_heads // beta.shape[1]
                beta_cam = beta.repeat_interleave(repeat_factor, dim=1)
                decay_cam = decay.repeat_interleave(repeat_factor, dim=1)
            else:
                beta_cam = beta
                decay_cam = decay
            beta_cam = beta_cam / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

            q_rot_bhdn = q_cam_trans_bhnd.permute(0, 1, 3, 2).contiguous()
            k_rot_bhdn = k_cam_trans_bhnd.permute(0, 1, 3, 2).contiguous()
            v_bhdn = v_cam_trans_bhnd.permute(0, 1, 3, 2).contiguous()
            scan_out = self._bidi_single_path(
                q_rot_bhdn,
                k_rot_bhdn,
                v_bhdn,
                beta_cam,
                decay_cam,
                spatial_tokens=spatial_tokens,
            )
            cam_raw_probe = apply_fn_o(scan_out.transpose(-1, -2).contiguous())
            cam_raw_probe = cam_raw_probe.transpose(-1, -2).contiguous()
            cam_raw_probe = cam_raw_probe.permute(0, 3, 1, 2).reshape(B, N, self.cam_dim)
            return {
                "cam_q_raw": q_raw_heads,
                "cam_k_raw": k_raw_heads,
                "cam_v_raw": v_raw_heads,
                "cam_q_trans": q_rot_bhdn,
                "cam_k_trans": k_rot_bhdn,
                "cam_v_trans": v_bhdn,
                "cam_inflation": frame_inflation_sq,
                "cam_beta": beta_cam,
                "cam_scan_out": scan_out,
                "cam_raw_probe": cam_raw_probe,
            }

        def native_cam_raw_with_official_prep(
            beta: torch.Tensor,
            decay: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if camera_conditions is None or spatial_shape is None or self.q_proj_cam is None:
                raise ValueError("official cam-prep ablation requires camera conditions and cam projections.")
            B, N, _hidden_size = hidden_states.shape
            T, H_sp, W_sp = spatial_shape
            spatial_tokens = H_sp * W_sp
            dtype_orig = hidden_states.dtype
            cam_heads = self.cam_heads
            cam_head_dim = self.cam_head_dim

            q_raw = swt._linear_output(self.q_proj_cam(hidden_states))
            k_raw = swt._linear_output(self.k_proj_cam(hidden_states))
            v_raw = swt._linear_output(self.v_proj_cam(hidden_states))
            if self.conv_k_cam is not None:
                k_raw = self._bidirectional_temporal_short_conv(k_raw, self.conv_k_cam, spatial_shape)
            if self.conv_q_cam is not None:
                q_raw = self._bidirectional_temporal_short_conv(q_raw, self.conv_q_cam, spatial_shape)
            if self.conv_v_cam is not None:
                v_raw = self._bidirectional_temporal_short_conv(v_raw, self.conv_v_cam, spatial_shape)
            q_raw_heads = q_raw.contiguous().view(B, N, cam_heads, cam_head_dim).contiguous()
            k_raw_heads = k_raw.contiguous().view(B, N, cam_heads, cam_head_dim).contiguous()
            v_raw_heads = v_raw.contiguous().view(B, N, cam_heads, cam_head_dim).contiguous()

            raymats = off_process_camera_conditions_raymats_only(camera_conditions, B, spatial_shape, self.patch_size)
            raymats = raymats.reshape(B, -1, 4, 4)
            P = raymats
            P_T = P.transpose(-1, -2).contiguous()
            P_inv = off_invert_SE3(P).contiguous()

            if rotary_emb is not None:
                head_dim = cam_head_dim
                orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
                orig_h_size = head_dim // 6
                new_head_dim = head_dim // 2
                new_t_size = new_head_dim // 2 - 2 * (new_head_dim // 6)
                new_h_size = new_head_dim // 6
                new_w_size = new_head_dim // 6
                t_part = rotary_emb[..., :new_t_size]
                h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
                w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
                rotary_emb_cam = torch.cat([t_part, h_part, w_part], dim=-1)
                rope_cos, rope_sin = off_prepare_ucpe_rope_tables(rotary_emb_cam, N, cam_head_dim // 2, hidden_states.device)
            else:
                rotary_emb_cam = None
                rope_cos = torch.ones(N, cam_head_dim // 2, device=hidden_states.device, dtype=torch.float32)
                rope_sin = torch.zeros(N, cam_head_dim // 2, device=hidden_states.device, dtype=torch.float32)

            q_norm_w = self.q_norm_cam.weight.float().contiguous()
            k_norm_w = self.k_norm_cam.weight.float().contiguous()
            k_scale = (cam_head_dim**-0.5) * (spatial_tokens**-0.5)
            norm_eps_val = float(
                getattr(
                    self.q_norm_cam,
                    "eps",
                    getattr(self.q_norm_cam, "variance_epsilon", 1e-6),
                )
            )
            q_cam_trans, k_cam_trans, v_cam_trans, inflation_sq = off_cam_prep_func(
                q_raw_heads,
                k_raw_heads,
                v_raw_heads,
                q_norm_weight=q_norm_w,
                k_norm_weight=k_norm_w,
                proj_q=P_T,
                proj_kv=P_inv,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                k_scale=k_scale,
                norm_eps=norm_eps_val,
            )
            inflation_sq = inflation_sq.view(B, cam_heads, 1, N)
            frame_inflation_sq = inflation_sq.view(B, cam_heads, T, spatial_tokens).mean(dim=-1)
            if beta.ndim == 3:
                beta_cam = beta / frame_inflation_sq.clamp_min(1.0)
            else:
                beta_cam = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

            if getattr(self, "fp32_attention", True):
                q_cam_trans = q_cam_trans.float()
                k_cam_trans = k_cam_trans.float()
                v_cam_trans = v_cam_trans.float()
                beta_cam = beta_cam.float()
                decay = decay.float()
            if beta_cam.ndim == 3:
                beta_scan = beta_cam.unsqueeze(-1).expand(B, cam_heads, T, spatial_tokens).contiguous()
            else:
                beta_scan = beta_cam.contiguous()
            decay_scan = decay.contiguous()
            q_cam_trans = q_cam_trans.contiguous()
            k_cam_trans = k_cam_trans.contiguous()
            v_cam_trans = v_cam_trans.contiguous()

            scan_out = off_cam_scan_bidi_chunkwise(q_cam_trans, k_cam_trans, v_cam_trans, beta_scan, decay_scan)
            scan_for_inverse = scan_out
            if getattr(self, "fp32_attention", True) and dtype_orig != torch.float32:
                scan_for_inverse = scan_for_inverse.to(dtype_orig)
            _, _, apply_fn_o = off_prepare_ray_apply_fns(
                head_dim=cam_head_dim,
                P=P,
                P_T=P_T,
                P_inv=P_inv,
                rotary_emb=rotary_emb_cam,
            )
            cam_raw_probe = apply_fn_o(scan_for_inverse.transpose(-1, -2)).transpose(-1, -2).contiguous()
            cam_raw = cam_raw_probe.reshape(B, self.cam_dim, -1).permute(0, 2, 1)
            if block_idx not in RAW_CHECK_BLOCKS:
                return cam_raw, {}
            return cam_raw, {
                "cam_q_raw": q_raw_heads,
                "cam_k_raw": k_raw_heads,
                "cam_v_raw": v_raw_heads,
                "cam_q_trans": q_cam_trans,
                "cam_k_trans": k_cam_trans,
                "cam_v_trans": v_cam_trans,
                "cam_inflation": frame_inflation_sq,
                "cam_beta": beta_scan,
                "cam_scan_out": scan_out,
                "cam_raw_probe": cam_raw,
            }

        if spatial_shape is not None and self.use_gdn:
            cam_disabled = os.environ.get("VLLM_OMNI_SANA_WM_DISABLE_CAM_BRANCH", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if camera_conditions is None or self.q_proj_cam is None or cam_disabled:
                precomputed_gates = None
                if block_idx in RAW_CHECK_BLOCKS or USE_OFFICIAL_MAIN_GDN_IN_NATIVE:
                    precomputed_gates = self._compute_frame_gates(hidden_states, spatial_shape)
                    raw_data = maybe_capture_raw_gdn(*precomputed_gates)
                if USE_OFFICIAL_MAIN_GDN_IN_NATIVE:
                    if precomputed_gates is None:
                        precomputed_gates = self._compute_frame_gates(hidden_states, spatial_shape)
                    main_raw = native_main_raw_with_official_gdn(*precomputed_gates)
                else:
                    main_raw, _ = self._forward_gdn_raw(
                        hidden_states,
                        spatial_shape,
                        rotary_emb,
                        precomputed_gates=precomputed_gates,
                    )
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
                    **raw_data,
                })
                return attn_out
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
            raw_data = maybe_capture_raw_gdn(beta, decay)
            if USE_OFFICIAL_MAIN_GDN_IN_NATIVE:
                main_raw = native_main_raw_with_official_gdn(beta, decay)
            else:
                main_raw, _ = self._forward_gdn_raw(
                    hidden_states,
                    spatial_shape,
                    rotary_emb,
                    precomputed_gates=(beta, decay),
                )
            if USE_OFFICIAL_CAM_PREP_IN_NATIVE:
                cam_raw, cam_raw_data = native_cam_raw_with_official_prep(beta, decay)
                raw_data.update(cam_raw_data)
            else:
                raw_data.update(maybe_capture_native_cam_raw(beta, decay))
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
                **raw_data,
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
    block_dump_prefix = os.environ.get("SANA_WM_BLOCK_STAGE_DUMP_PREFIX", "")
    old_block_dump = os.environ.get("SANA_WM_DUMP_BLOCK0")
    old_block_idxs = os.environ.get("SANA_WM_DUMP_BLOCK_IDXS")
    if block_dump_prefix and BLOCK_STAGE_BLOCKS:
        os.environ["SANA_WM_DUMP_BLOCK0"] = f"{block_dump_prefix}_official"
        os.environ["SANA_WM_DUMP_BLOCK_IDXS"] = ",".join(str(i) for i in BLOCK_STAGE_BLOCKS)
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
    recorder.prompt_mask = cond_mask.detach().cpu()
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
    if old_block_dump is None:
        os.environ.pop("SANA_WM_DUMP_BLOCK0", None)
    else:
        os.environ["SANA_WM_DUMP_BLOCK0"] = old_block_dump
    if old_block_idxs is None:
        os.environ.pop("SANA_WM_DUMP_BLOCK_IDXS", None)
    else:
        os.environ["SANA_WM_DUMP_BLOCK_IDXS"] = old_block_idxs
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
    block_dump_prefix = os.environ.get("SANA_WM_BLOCK_STAGE_DUMP_PREFIX", "")
    old_block_dump = os.environ.get("SANA_WM_DUMP_BLOCK0")
    old_block_idxs = os.environ.get("SANA_WM_DUMP_BLOCK_IDXS")
    if block_dump_prefix and BLOCK_STAGE_BLOCKS:
        os.environ["SANA_WM_DUMP_BLOCK0"] = f"{block_dump_prefix}_native"
        os.environ["SANA_WM_DUMP_BLOCK_IDXS"] = ",".join(str(i) for i in BLOCK_STAGE_BLOCKS)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    outputs: dict[int, torch.Tensor] = {}
    prompt_mask = (
        official_recorder.prompt_mask.to(device=device, dtype=torch.float32)
        if official_recorder.prompt_mask is not None
        else None
    )
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
                encoder_attention_mask=prompt_mask,
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
    if old_block_dump is None:
        os.environ.pop("SANA_WM_DUMP_BLOCK0", None)
    else:
        os.environ["SANA_WM_DUMP_BLOCK0"] = old_block_dump
    if old_block_idxs is None:
        os.environ.pop("SANA_WM_DUMP_BLOCK_IDXS", None)
    else:
        os.environ["SANA_WM_DUMP_BLOCK_IDXS"] = old_block_idxs
    return recorder, outputs


def run_native_teacher_force(official_recorder: Recorder) -> dict[str, torch.Tensor]:
    if not TEACHER_FORCE_BLOCKS:
        return {}
    transformer, cam, prompt = build_native_transformer()
    patch_native_attn(transformer, Recorder("native_teacher_force_unused"))
    device = torch.device("cuda")
    dtype = torch.bfloat16
    prompt_mask = (
        official_recorder.prompt_mask.to(device=device, dtype=torch.float32)
        if official_recorder.prompt_mask is not None
        else None
    )
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for step_idx in STEPS:
            step = load_step(step_idx)
            for block_idx in TEACHER_FORCE_BLOCKS:
                official_key = f"official:full:step{step_idx}:block{block_idx}"
                if official_key not in official_recorder.records:
                    continue
                original_forward = transformer.blocks[block_idx].attn.forward
                forced_attn_out = official_recorder.records[official_key]["attn_out"].to(device=device, dtype=dtype)

                def forced_forward(self, hidden_states, spatial_shape=None, rotary_emb=None, camera_conditions=None):
                    del self, hidden_states, spatial_shape, rotary_emb, camera_conditions
                    return forced_attn_out

                transformer.blocks[block_idx].attn.forward = MethodType(
                    forced_forward,
                    transformer.blocks[block_idx].attn,
                )
                start = time.perf_counter()
                out = transformer(
                    step["latent_in"].to(device=device, dtype=dtype),
                    step["timestep_per_frame"].to(device=device, dtype=torch.float32),
                    encoder_hidden_states=prompt,
                    encoder_attention_mask=prompt_mask,
                    plucker=cam["chunk_plucker"],
                    raymap=cam["raymap"],
                    spatial_raymap=cam["spatial_raymap"],
                )
                torch.cuda.synchronize()
                transformer.blocks[block_idx].attn.forward = original_forward
                print(
                    f"[attn-split] native teacher-force step{step_idx} block{block_idx} attn_out "
                    f"forward {time.perf_counter() - start:.1f}s noise_norm={out.float().norm().item():.2f}",
                    flush=True,
                )
                outputs[f"step{step_idx}:block{block_idx}:attn_out"] = to_cpu(out)
                del forced_attn_out
                cleanup_cuda()
    del transformer
    cleanup_cuda()
    return outputs


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    official_rec, official_noise = run_official()
    native_rec, native_noise = run_native(official_rec)
    teacher_force_noise = run_native_teacher_force(official_rec)
    block_stage = compare_block_stage_dumps()
    report: dict[str, Any] = {
        "steps": list(STEPS),
        "blocks": list(BLOCKS),
        "raw_check_blocks": list(RAW_CHECK_BLOCKS),
        "block_stage_blocks": list(BLOCK_STAGE_BLOCKS),
        "native_uses_official_cam_prep": USE_OFFICIAL_CAM_PREP_IN_NATIVE,
        "native_uses_official_main_gdn": USE_OFFICIAL_MAIN_GDN_IN_NATIVE,
        "official_main_path": "BidirectionalGDNTriton.forward",
        "projection_checks": {},
        "teacher_force_noise": {},
        "block_stage": block_stage,
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
    for key, value in teacher_force_noise.items():
        step_idx = int(key.split(":", 1)[0].removeprefix("step"))
        report["teacher_force_noise"][key] = compare_tensor(value, official_noise[step_idx])
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
                if stage in COMPACT_STAGES
            }
            compact["full_attn"][key] = {
                stage: values.get("generated", values.get("global", {}))
                for stage, values in report["full_native_vs_official"][key].items()
                if stage in COMPACT_STAGES
            }
    compact["projection_checks"] = report["projection_checks"]
    compact["block_stage"] = {
        block: {
            stage: values.get("generated", values.get("global", {}))
            for stage, values in stages.items()
        }
        for block, stages in report["block_stage"].items()
    }
    compact["teacher_force_noise"] = {
        key: values.get("generated", values.get("global", {}))
        for key, values in report["teacher_force_noise"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)
    print(f"[attn-split] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
