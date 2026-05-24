# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge to the official Sana-WM inference entrypoint.

The public HF release currently ships weights/configs, while the model card
points to NVlabs/Sana's ``inference_video_scripts/inference_sana_wm.py`` as the
reference runner. Keeping this bridge separate lets vLLM-Omni execute a real
GPU smoke run as soon as that official runner is present, without pretending
that the custom Stage-1 DiT has already been ported internally.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

SANA_WM_OFFICIAL_REPO_ENV = "VLLM_OMNI_SANA_WM_OFFICIAL_REPO"
SANA_WM_OFFICIAL_SCRIPT = "inference_video_scripts/inference_sana_wm.py"
SANA_WM_OFFICIAL_BACKEND_ERROR = (
    "Sana-WM internal Stage-1 DiT forward is not ported yet, and the official "
    f"runner was not found. Set {SANA_WM_OFFICIAL_REPO_ENV} to an NVlabs/Sana "
    f"checkout containing {SANA_WM_OFFICIAL_SCRIPT} to run GPU e2e inference."
)


@dataclass(frozen=True)
class SanaWmOfficialRunResult:
    frames: np.ndarray
    command: tuple[str, ...]
    stdout: str
    stderr: str
    video_path: str


class SanaWmLocalPathsLike(Protocol):
    root: Path
    config: Path
    stage1_dit: Path
    refiner_text_encoder_dir: Path


def get_sana_wm_official_repo_path(
    *,
    explicit_repo: str | os.PathLike[str] | None = None,
    od_config: Any | None = None,
) -> Path | None:
    """Resolve the optional official Sana repo path.

    Resolution order is per-request/per-config explicit value, config attribute,
    then environment variable. ``None`` means no official backend was requested.
    """

    config_repo = getattr(od_config, "sana_wm_official_repo", None) if od_config is not None else None
    raw = explicit_repo or config_repo or os.environ.get(SANA_WM_OFFICIAL_REPO_ENV)
    if raw in (None, ""):
        return None
    return Path(raw).expanduser()


def find_sana_wm_official_script(repo_path: str | os.PathLike[str] | None = None) -> Path | None:
    repo = get_sana_wm_official_repo_path(explicit_repo=repo_path)
    if repo is None:
        return None
    script = repo / SANA_WM_OFFICIAL_SCRIPT
    return script if script.is_file() else None


def require_sana_wm_official_script(repo_path: str | os.PathLike[str] | None = None) -> Path:
    script = find_sana_wm_official_script(repo_path)
    if script is None:
        raise FileNotFoundError(SANA_WM_OFFICIAL_BACKEND_ERROR)
    return script


def is_sana_wm_official_backend_requested(od_config: Any | None = None) -> bool:
    return get_sana_wm_official_repo_path(od_config=od_config) is not None


def is_sana_wm_official_backend_available(od_config: Any | None = None) -> bool:
    repo = get_sana_wm_official_repo_path(od_config=od_config)
    return repo is not None and find_sana_wm_official_script(repo) is not None


def build_sana_wm_official_command(
    *,
    script: str | os.PathLike[str],
    image_path: str | os.PathLike[str],
    prompt_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    num_frames: int,
    release_paths: SanaWmLocalPathsLike,
    include_refiner: bool,
    action: str | None = None,
    camera_path: str | os.PathLike[str] | None = None,
    intrinsics_path: str | os.PathLike[str] | None = None,
    translation_speed: float | None = None,
    rotation_speed_deg: float | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build the official Sana-WM CLI invocation from normalized inputs."""

    if bool(action) == bool(camera_path):
        raise ValueError("Exactly one of action or camera_path is required for Sana-WM official backend.")

    cmd = [
        python_executable or sys.executable,
        str(script),
        "--image",
        str(image_path),
        "--prompt",
        str(prompt_path),
        "--num_frames",
        str(num_frames),
        "--output_dir",
        str(output_dir),
        "--config",
        str(release_paths.config),
        "--model_path",
        str(release_paths.stage1_dit),
    ]
    if action:
        cmd.extend(["--action", action])
    if camera_path is not None:
        cmd.extend(["--camera", str(camera_path)])
    if intrinsics_path is not None:
        cmd.extend(["--intrinsics", str(intrinsics_path)])
    if translation_speed is not None:
        cmd.extend(["--translation_speed", str(translation_speed)])
    if rotation_speed_deg is not None:
        cmd.extend(["--rotation_speed_deg", str(rotation_speed_deg)])

    if include_refiner:
        cmd.extend(
            [
                "--refiner_root",
                str(release_paths.root / "refiner"),
                "--refiner_gemma_root",
                str(release_paths.refiner_text_encoder_dir),
            ]
        )
    else:
        cmd.append("--no_refiner")
    return cmd


def run_sana_wm_official_backend(
    *,
    prompt: Mapping[str, Any],
    release_paths: SanaWmLocalPathsLike,
    include_refiner: bool,
    repo_path: str | os.PathLike[str] | None = None,
    timeout_s: float | None = None,
) -> SanaWmOfficialRunResult:
    """Run the official Sana-WM CLI and return decoded video frames."""

    script = require_sana_wm_official_script(repo_path)
    repo = script.parents[1]
    payload = prompt["additional_information"]["sana_wm"]
    image = _extract_image(prompt)

    with tempfile.TemporaryDirectory(prefix="vllm_omni_sana_wm_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        input_image_path = tmpdir / "input.png"
        prompt_path = tmpdir / "prompt.txt"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        _save_image(image, input_image_path)
        prompt_path.write_text(str(prompt.get("prompt") or ""), encoding="utf-8")

        camera_path = None
        camera = payload.get("camera")
        if camera is not None:
            camera_path = tmpdir / "camera.npy"
            np.save(camera_path, np.asarray(camera["poses"], dtype=np.float32))

        intrinsics_path = None
        if payload.get("intrinsics") is not None:
            intrinsics_path = tmpdir / "intrinsics.npy"
            np.save(intrinsics_path, _intrinsics_to_array(payload["intrinsics"]))

        cmd = build_sana_wm_official_command(
            script=script,
            image_path=input_image_path,
            prompt_path=prompt_path,
            output_dir=output_dir,
            num_frames=int(payload["num_frames"]),
            release_paths=release_paths,
            include_refiner=include_refiner,
            action=payload.get("action"),
            camera_path=camera_path,
            intrinsics_path=intrinsics_path,
            translation_speed=payload.get("translation_speed"),
            rotation_speed_deg=payload.get("rotation_speed_deg"),
        )

        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        completed = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Official Sana-WM backend failed with exit code "
                f"{completed.returncode}.\nSTDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
            )

        video_path = _find_output_video(output_dir)
        frames = _load_video_frames(video_path)
        return SanaWmOfficialRunResult(
            frames=frames,
            command=tuple(cmd),
            stdout=completed.stdout,
            stderr=completed.stderr,
            video_path=str(video_path),
        )


def _extract_image(prompt: Mapping[str, Any]) -> Any:
    mm_data = prompt.get("multi_modal_data") or {}
    if "image" in prompt:
        return prompt["image"]
    image = mm_data.get("image") if isinstance(mm_data, Mapping) else None
    if isinstance(image, list):
        return image[0]
    if image is None:
        raise ValueError("Sana-WM official backend requires a first-frame image.")
    return image


def _save_image(image: Any, path: Path) -> None:
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        Image.open(image).convert("RGB").save(path)
        return
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path)
        return

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.float().numpy()
    else:
        array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"Sana-WM image must be HWC/CHW RGB-like data, got shape {array.shape}.")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        if (float(np.max(array)) if array.size else 0.0) <= 1.0:
            array = array * 255.0
        array = array.clip(0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    Image.fromarray(array, mode="RGB").save(path)


def _intrinsics_to_array(intrinsics: Any) -> np.ndarray:
    if isinstance(intrinsics, Mapping):
        return np.asarray(
            [
                float(intrinsics["fx"]),
                float(intrinsics["fy"]),
                float(intrinsics["cx"]),
                float(intrinsics["cy"]),
            ],
            dtype=np.float32,
        )
    return np.asarray(intrinsics, dtype=np.float32)


def _find_output_video(output_dir: Path) -> Path:
    candidates: list[Path] = []
    for suffix in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
        candidates.extend(output_dir.rglob(suffix))
    if not candidates:
        raise FileNotFoundError(f"Official Sana-WM backend did not write a video under {output_dir}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_video_frames(video_path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio

        frames = iio.imread(video_path)
    except ImportError as exc:
        raise ImportError("imageio is required to decode the official Sana-WM mp4 output.") from exc
    except Exception:
        import imageio

        reader = imageio.get_reader(video_path)
        try:
            frames = np.stack([frame for frame in reader], axis=0)
        finally:
            reader.close()

    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected decoded video frames with shape (T,H,W,C), got {frames.shape}.")
    return frames


__all__ = [
    "SANA_WM_OFFICIAL_BACKEND_ERROR",
    "SANA_WM_OFFICIAL_REPO_ENV",
    "SANA_WM_OFFICIAL_SCRIPT",
    "SanaWmOfficialRunResult",
    "build_sana_wm_official_command",
    "find_sana_wm_official_script",
    "get_sana_wm_official_repo_path",
    "is_sana_wm_official_backend_available",
    "is_sana_wm_official_backend_requested",
    "require_sana_wm_official_script",
    "run_sana_wm_official_backend",
]
