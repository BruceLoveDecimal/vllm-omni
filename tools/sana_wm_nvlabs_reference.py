#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the NVlabs/Sana SANA-WM reference and emit a video usable as --reference.

This is a STANDALONE tool — it deliberately lives outside the shipped model code
so the vLLM-Omni integration does not carry an NVlabs runner. It shells out to
the official ``inference_video_scripts/inference_sana_wm.py`` from an NVlabs/Sana
checkout, mirroring the first frame / camera action / intrinsics that
``examples/offline_inference/sana_wm/sana_wm.py`` uses, so the two outputs are
comparable. Feed the result to the example's ``--reference`` to get MAE/PSNR/SSIM.

Note: NVlabs reads resolution / sampling steps / guidance_scale / seed from the
release ``config.yaml`` (only ``--num_frames`` and the camera are CLI args). Make
sure the example run uses values matching that config (see the printed command).

Usage:
    python tools/sana_wm_nvlabs_reference.py \
        --repo /root/autodl-tmp/NVlabs-Sana \
        --demo forward_push --num-frames 161 \
        --output /root/autodl-tmp/nvlabs_ref.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SANA_WM_OFFICIAL_SCRIPT = "inference_video_scripts/inference_sana_wm.py"
SANA_WM_STAGE1_DIT_FILE = "dit/sana_wm_1600m_720p.safetensors"
DEFAULT_REPO = os.environ.get("VLLM_OMNI_SANA_WM_OFFICIAL_REPO", "/root/autodl-tmp/NVlabs-Sana")
DEFAULT_HF_CACHE = os.environ.get("HF_HOME", "/root/autodl-tmp/hf-cache")
SANA_WM_REPO_ID = "models--Efficient-Large-Model--SANA-WM_bidirectional"

# Mirror the example's demo presets so action/prompt line up exactly.
DEMOS = {
    "forward_push": {
        "prompt": "A slow forward camera move through a quiet city street at golden hour.",
        "action": lambda nf: f"w-{nf - 1}",
    },
    "orbit_left": {
        "prompt": "A smooth orbit to the left around a stone fountain in a garden.",
        "action": lambda nf: f"jw-{nf - 1}",
    },
    "pan_right": {
        "prompt": "A steady pan to the right across a misty mountain valley.",
        "action": lambda nf: f"l-{nf - 1}",
    },
    "crane_dolly": {
        "prompt": "A cinematic crane-up then dolly-forward over a calm lake at dawn.",
        "action": lambda nf: f"i-{(nf - 1) // 2},w-{nf - 1 - (nf - 1) // 2}",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Produce an NVlabs SANA-WM reference video (for the example's --reference).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo", default=DEFAULT_REPO, help="NVlabs/Sana checkout containing inference_sana_wm.py.")
    p.add_argument("--snapshot", default=None,
                   help="SANA-WM HF snapshot dir (with config.yaml/dit/vae/refiner). Auto-resolved from HF cache.")
    p.add_argument("--demo", choices=sorted(DEMOS), default=None, help="Preset prompt + camera action.")
    p.add_argument("--image", default=None, help="First-frame image. If omitted, the example's placeholder is synthesized.")
    p.add_argument("--prompt", default=None, help="Prompt text or path to a .txt file.")
    p.add_argument("--action", default=None, help='Camera action DSL, e.g. "w-160". Mutually exclusive with --camera.')
    p.add_argument("--camera", default=None, help="Path to .npy camera-to-world poses (F,4,4).")
    p.add_argument("--intrinsics", default=None, help="Path to .npy intrinsics. If omitted, deterministic pinhole is used.")
    p.add_argument("--translation-speed", "--translation_speed", type=float, default=0.055)
    p.add_argument("--rotation-speed-deg", "--rotation_speed_deg", type=float, default=1.2)
    p.add_argument("--num-frames", "--num_frames", type=int, default=161)
    p.add_argument("--height", type=int, default=704, help="Used to synthesize the placeholder frame + intrinsics.")
    p.add_argument("--width", type=int, default=1280, help="Used to synthesize the placeholder frame + intrinsics.")
    p.add_argument("--no-refiner", "--no_refiner", action="store_true", help="Run Stage-1 only (NVlabs --no_refiner).")
    p.add_argument("--refiner-root", default=None, help="Override refiner root (default <snapshot>/refiner).")
    p.add_argument("--python", default=None, help="Python executable for the NVlabs subprocess (default: this one).")
    p.add_argument("--timeout", type=float, default=None, help="Subprocess timeout (seconds).")
    p.add_argument("--output", default="nvlabs_ref.mp4", help="Where to copy the produced reference mp4.")
    return p.parse_args()


def _synth_first_frame(width: int, height: int) -> "Image.Image":
    from PIL import Image

    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    rgb = np.stack([(0.30 + 0.5 * xx), (0.40 + 0.4 * yy), (0.55 + 0.3 * (1 - xx))], axis=-1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")


def _resolve_snapshot(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    hub = Path(DEFAULT_HF_CACHE) / "hub" / SANA_WM_REPO_ID / "snapshots"
    snaps = sorted(hub.glob("*")) if hub.is_dir() else []
    snaps = [s for s in snaps if (s / "config.yaml").is_file()]
    if not snaps:
        raise SystemExit(
            f"Could not auto-resolve a SANA-WM snapshot under {hub}. Pass --snapshot explicitly."
        )
    return snaps[-1]


def _read_prompt(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    p = Path(value)
    return p.read_text(encoding="utf-8").strip() if p.is_file() else value


def _write_config_with_local_vae(source_config: Path, local_model_root: Path, out_dir: Path) -> Path:
    """Point vae.vae_pretrained at the local snapshot so NVlabs does not re-download."""
    out = out_dir / "config.local_vae.yaml"
    try:
        import yaml

        cfg = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
        vae = cfg.setdefault("vae", {})
        if isinstance(vae, dict):
            vae["vae_pretrained"] = str(local_model_root)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        # Best-effort line patch if yaml is unavailable.
        lines, patched = [], False
        for line in source_config.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("vae_pretrained:"):
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}vae_pretrained: {local_model_root}")
                patched = True
            else:
                lines.append(line)
        if not patched:
            lines.extend(["vae:", f"  vae_pretrained: {local_model_root}"])
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _find_output_video(output_dir: Path) -> Path:
    candidates: list[Path] = []
    for suffix in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
        candidates.extend(output_dir.rglob(suffix))
    if not candidates:
        raise RuntimeError(f"NVlabs run produced no video under {output_dir}.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()

    # Resolve demo preset (prompt/action) where not explicit.
    preset = DEMOS.get(args.demo) if args.demo else None
    prompt = _read_prompt(args.prompt, (preset or {}).get("prompt", "A slow forward camera move."))
    if args.action is None and args.camera is None:
        args.action = preset["action"](args.num_frames) if preset else f"w-{args.num_frames - 1}"
    if args.action is not None and args.camera is not None:
        raise SystemExit("Pass only one of --action or --camera.")

    repo = Path(args.repo)
    script = repo / SANA_WM_OFFICIAL_SCRIPT
    if not script.is_file():
        raise SystemExit(f"NVlabs script not found: {script}\nPass --repo to an NVlabs/Sana checkout.")
    snapshot = _resolve_snapshot(args.snapshot)
    config = snapshot / "config.yaml"
    stage1_dit = snapshot / SANA_WM_STAGE1_DIT_FILE
    refiner_root = Path(args.refiner_root) if args.refiner_root else snapshot / "refiner"
    refiner_gemma = refiner_root / "text_encoder"
    for pth, name in [(config, "config.yaml"), (stage1_dit, "stage1 dit")]:
        if not pth.exists():
            raise SystemExit(f"Missing {name} in snapshot: {pth}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sana_wm_nvlabs_") as tmp_str:
        tmp = Path(tmp_str)
        image_path = tmp / "input.png"
        prompt_path = tmp / "prompt.txt"
        out_dir = tmp / "output"
        out_dir.mkdir()

        # First frame: load or synthesize the SAME placeholder as the example.
        if args.image:
            from PIL import Image

            Image.open(args.image).convert("RGB").resize((args.width, args.height)).save(image_path)
        else:
            _synth_first_frame(args.width, args.height).save(image_path)
            print(f"[info] No --image; synthesized the example's {args.width}x{args.height} placeholder.")
        prompt_path.write_text(prompt, encoding="utf-8")

        config_local = _write_config_with_local_vae(config, snapshot, tmp)

        # Deterministic pinhole intrinsics matching the example default.
        intrinsics_path = None
        if args.intrinsics:
            intrinsics_path = Path(args.intrinsics)
        elif args.action is not None:
            intrinsics_path = tmp / "intrinsics.npy"
            np.save(intrinsics_path, np.asarray(
                [args.width / 2, args.width / 2, args.width / 2, args.height / 2], dtype=np.float32))

        cmd = [
            args.python or sys.executable,
            str(script),
            "--image", str(image_path),
            "--prompt", str(prompt_path),
            "--num_frames", str(args.num_frames),
            "--output_dir", str(out_dir),
            "--config", str(config_local),
            "--model_path", str(stage1_dit),
        ]
        if args.action:
            cmd += ["--action", args.action]
        else:
            cmd += ["--camera", str(args.camera)]
        if intrinsics_path is not None:
            cmd += ["--intrinsics", str(intrinsics_path)]
        cmd += ["--translation_speed", str(args.translation_speed)]
        cmd += ["--rotation_speed_deg", str(args.rotation_speed_deg)]
        if args.no_refiner:
            cmd += ["--no_refiner"]
        else:
            cmd += ["--refiner_root", str(refiner_root), "--refiner_gemma_root", str(refiner_gemma)]

        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        # NVlabs imports FLA kernels wrapped in torch.compile; disable Dynamo to
        # avoid the inductor template issue on the Blackwell image (reference run).
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        env.setdefault("HF_HOME", DEFAULT_HF_CACHE)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        print("\n" + "=" * 60)
        print("NVlabs SANA-WM reference")
        print(f"  repo      : {repo}")
        print(f"  snapshot  : {snapshot}")
        print(f"  frames    : {args.num_frames}   refiner: {not args.no_refiner}")
        print(f"  camera    : {'poses=' + args.camera if args.camera else 'action=' + args.action}")
        print(f"  cmd       : {' '.join(cmd)}")
        print("  note      : resolution / steps / guidance_scale / seed come from config.yaml")
        print("=" * 60 + "\n", flush=True)

        completed = subprocess.run(cmd, cwd=str(repo), env=env, text=True, timeout=args.timeout, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"NVlabs reference failed (exit {completed.returncode}).")

        produced = _find_output_video(out_dir)
        shutil.copy2(produced, out_path)
        print(f"\nSaved NVlabs reference video to {out_path}")

    print("\nNext: compute metrics against it with the example (match num_frames + the config's size/steps/cfg/seed):")
    cam = f"--camera {args.camera}" if args.camera else f"--action {args.action}"
    print(
        "  python examples/offline_inference/sana_wm/sana_wm.py \\\n"
        f"    {cam} --num_frames {args.num_frames} --height {args.height} --width {args.width} \\\n"
        f"    --reference {out_path} --metrics-json vs_nvlabs.json --output ours.mp4"
    )


if __name__ == "__main__":
    main()
