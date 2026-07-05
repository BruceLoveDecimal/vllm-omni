# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVlabs SANA-WM reference Stage-1 e2e generation.

This is the *reference* (official NVlabs) generation used as the baseline for
the native vLLM-Omni implementation. It does NOT vendor the NVlabs code — it
imports the official ``inference_video_scripts/inference_sana_wm.py`` from the
checkout pointed at by ``VLLM_OMNI_SANA_WM_OFFICIAL_REPO`` (the same mechanism
``sana_wm_transformer._import_nvlabs_cam_scan_bidi`` uses), so this file stays
small and the heavyweight reference implementation lives upstream.

Run standalone to produce a reference clip:

    VLLM_OMNI_SANA_WM_OFFICIAL_REPO=/path/to/NVlabs-Sana \
    SANA_WM_E2E_MODEL=/path/to/SANA-WM_bidirectional/snapshots/<rev> \
    python -m tests.e2e.accuracy.sana_wm.run_sana_wm_reference \
        --num-frames 161 --steps 60 --cfg 5.0 --seed 42 --output ref.npy

``refiner=None`` keeps the comparison on the Stage-1 DiT + SANA VAE decode,
matching the native ``SanaWmPipeline`` (``sana_wm_output_type="np"``) path.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

PROMPT = "A slow forward camera move through a quiet city street."
HEIGHT = 704
WIDTH = 1280
FPS = 16
SEED = 42
CFG_SCALE = 5.0
TRANSLATION_SPEED = 0.055


def official_repo() -> Path | None:
    repo = os.environ.get("VLLM_OMNI_SANA_WM_OFFICIAL_REPO", "")
    if not repo:
        return None
    path = Path(repo)
    return path if (path / "inference_video_scripts" / "inference_sana_wm.py").is_file() else None


def load_official_module() -> ModuleType:
    """Import the upstream ``inference_sana_wm.py`` as a module (no vendoring)."""
    repo = official_repo()
    if repo is None:
        raise RuntimeError(
            "VLLM_OMNI_SANA_WM_OFFICIAL_REPO must point at an NVlabs-Sana checkout "
            "containing inference_video_scripts/inference_sana_wm.py."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    script = repo / "inference_video_scripts" / "inference_sana_wm.py"
    spec = importlib.util.spec_from_file_location("_sana_wm_official_ref", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load NVlabs reference from {script}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_camera(num_frames: int) -> np.ndarray:
    """Forward dolly (matches NVlabs ``w`` action).

    SANA-WM uses the OpenCV convention ``+X right, +Y down, +Z forward``
    (``inference_sana_wm.action_string_to_c2w``), so a forward camera move
    accumulates **+translation_speed** on the z axis. (A negative sign here
    dollies *backward* — the apparent motion looks reversed.)
    """
    c2w = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    c2w[:, 2, 3] = TRANSLATION_SPEED * np.arange(num_frames, dtype=np.float32)
    return c2w


def make_intrinsics(num_frames: int) -> np.ndarray:
    intr = np.array([WIDTH / 2.0, WIDTH / 2.0, WIDTH / 2.0, HEIGHT / 2.0], dtype=np.float32)
    return np.repeat(intr[None, :], num_frames, axis=0)


def _first_frame_image(image_path: str | None) -> Any:
    from PIL import Image

    if image_path and Path(image_path).is_file():
        return Image.open(image_path).convert("RGB")
    repo = official_repo()
    demo = repo / "asset" / "sana_wm" / "demo_0.png" if repo else None
    if demo and demo.is_file():
        return Image.open(demo).convert("RGB")
    return Image.new("RGB", (WIDTH, HEIGHT), (96, 128, 160))


def generate_reference(
    *,
    num_frames: int,
    steps: int,
    cfg: float = CFG_SCALE,
    seed: int = SEED,
    fps: int = FPS,
    prompt: str = PROMPT,
    image_path: str | None = None,
) -> np.ndarray:
    """Run the NVlabs reference Stage-1 + SANA VAE decode. Returns (T,H,W,3) uint8."""
    import torch

    from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import resolve_sana_wm_local_paths

    module = load_official_module()
    model_root = os.environ["SANA_WM_E2E_MODEL"]
    paths = resolve_sana_wm_local_paths(model_root)
    config = module.pyrallis.parse(config_class=module.InferenceConfig, config_path=str(paths.config), args=[])
    config.vae.vae_pretrained = str(paths.root)

    pipe = module.SanaWMPipeline(
        config=config,
        model_path=str(paths.stage1_dit),
        device=torch.device("cuda"),
        refiner=None,
        offload_vae=True,
        offload_refiner=False,
    )
    image = _first_frame_image(image_path)
    cropped, src_size, resized_size, crop_offset = module.resize_and_center_crop(image)
    intrinsics = module.transform_intrinsics_for_crop(make_intrinsics(num_frames), src_size, resized_size, crop_offset)
    c2w = make_camera(num_frames)
    params = module.GenerationParams(
        num_frames=num_frames,
        fps=fps,
        step=steps,
        cfg_scale=cfg,
        flow_shift=None,
        seed=seed,
        negative_prompt="",
        sampling_algo="flow_euler_ltx",
    )
    with torch.inference_mode():
        result = pipe.generate(cropped, prompt, c2w, intrinsics, params)
    video = np.asarray(result["video"])  # (T, H, W, 3) uint8
    del pipe
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.accelerator.empty_cache()
    return video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVlabs SANA-WM reference Stage-1 generation.")
    parser.add_argument("--num-frames", type=int, default=161)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--cfg", type=float, default=CFG_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--image", default=None, help="First-frame image (default: NVlabs demo_0.png).")
    parser.add_argument("--output", required=True, help="Output .npy path for the (T,H,W,3) uint8 video.")
    args = parser.parse_args()

    video = generate_reference(
        num_frames=args.num_frames,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        fps=args.fps,
        image_path=args.image,
    )
    np.save(args.output, video)
    print(f"reference video {video.shape} {video.dtype} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
