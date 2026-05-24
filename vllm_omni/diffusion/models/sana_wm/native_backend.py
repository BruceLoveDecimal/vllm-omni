# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process SANA-WM native reference backend.

The public SANA-WM release currently keeps the full Stage-1 DiT/GDN/refiner
implementation in NVlabs/Sana. This backend executes that Python model
in-process instead of shelling out to the official CLI, so vLLM-Omni owns the
request normalization, checkpoint resolution, and output handling while the
large reference modules are used directly for Stage-1 math.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol

import numpy as np

from vllm_omni.diffusion.models.sana_wm.camera_control import (
    action_string_to_c2w,
    intrinsics_to_vec4_array,
)
from vllm_omni.diffusion.models.sana_wm.official_backend import (
    SANA_WM_OFFICIAL_REPO_ENV,
    SANA_WM_OFFICIAL_SCRIPT,
    _extract_image,
)

SANA_WM_NATIVE_BACKEND_ERROR = (
    "Sana-WM in-process native backend requires an NVlabs/Sana checkout containing "
    f"{SANA_WM_OFFICIAL_SCRIPT}. Set {SANA_WM_OFFICIAL_REPO_ENV} or pass "
    "`sana_wm.official_repo_path`."
)
SANA_WM_FORCE_CLI_ENV = "VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI"


class SanaWmLocalPathsLike(Protocol):
    root: Path
    config: Path
    stage1_dit: Path
    refiner_root: Path
    refiner_text_encoder_dir: Path


@dataclass(frozen=True)
class SanaWmNativeRunResult:
    frames: np.ndarray
    backend: str
    repo_path: str
    model_path: str
    include_refiner: bool
    num_frames: int
    fps: int
    sampling_steps: int
    cfg_scale: float
    used_default_intrinsics: bool


def should_force_sana_wm_cli_backend() -> bool:
    return os.environ.get(SANA_WM_FORCE_CLI_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def get_sana_wm_native_repo_path(
    *,
    explicit_repo: str | os.PathLike[str] | None = None,
    od_config: Any | None = None,
) -> Path | None:
    config_repo = getattr(od_config, "sana_wm_official_repo", None) if od_config is not None else None
    raw = explicit_repo or config_repo or os.environ.get(SANA_WM_OFFICIAL_REPO_ENV)
    if raw in (None, ""):
        return None
    return Path(raw).expanduser()


def find_sana_wm_native_script(repo_path: str | os.PathLike[str] | None = None) -> Path | None:
    repo = get_sana_wm_native_repo_path(explicit_repo=repo_path)
    if repo is None:
        return None
    script = repo / SANA_WM_OFFICIAL_SCRIPT
    return script if script.is_file() else None


def require_sana_wm_native_script(repo_path: str | os.PathLike[str] | None = None) -> Path:
    script = find_sana_wm_native_script(repo_path)
    if script is None:
        raise FileNotFoundError(SANA_WM_NATIVE_BACKEND_ERROR)
    return script


def is_sana_wm_native_backend_available(od_config: Any | None = None) -> bool:
    repo = get_sana_wm_native_repo_path(od_config=od_config)
    return repo is not None and find_sana_wm_native_script(repo) is not None


class _OfficialConfig(dict):
    """Tiny subset of mmcv.Config used by Sana import-time helpers."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def fromfile(cls, filename: str | os.PathLike[str]) -> "_OfficialConfig":
        import json

        path = Path(filename)
        if path.suffix == ".json":
            return cls(json.loads(path.read_text(encoding="utf-8")))
        raise ValueError(f"Unsupported config file for SANA-WM mmcv shim: {path}.")

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def copy(self) -> "_OfficialConfig":
        return type(self)(self)

    def merge_from_dict(self, options: Mapping[str, Any]) -> None:
        for key, value in options.items():
            self[key] = self._wrap(value)


class _OfficialRegistry:
    """Minimal mmcv.Registry replacement for official SANA-WM modules."""

    def __init__(self, name: str):
        self.name = name
        self.module_dict: dict[str, Any] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.module_dict

    def __repr__(self) -> str:
        return f"_OfficialRegistry(name={self.name!r}, items={list(self.module_dict)})"

    def get(self, key: str) -> Any | None:
        return self.module_dict.get(key)

    def register_module(
        self,
        module: Any | None = None,
        *,
        name: str | list[str] | tuple[str, ...] | None = None,
        force: bool = False,
    ) -> Any:
        def _register(obj: Any) -> Any:
            names = [name] if isinstance(name, str) or name is None else list(name)
            for raw_name in names:
                key = raw_name or obj.__name__
                if not force and key in self.module_dict and self.module_dict[key] is not obj:
                    raise KeyError(f"{key!r} is already registered in {self.name}.")
                self.module_dict[key] = obj
            return obj

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg: Any, default_args: Mapping[str, Any] | None = None) -> Any:
        return _official_build_from_cfg(cfg, self, default_args=default_args)


def _official_build_from_cfg(
    cfg: Any,
    registry: _OfficialRegistry,
    default_args: Mapping[str, Any] | None = None,
) -> Any:
    if isinstance(cfg, str):
        args = {"type": cfg}
    elif isinstance(cfg, Mapping):
        args = dict(cfg)
    else:
        raise TypeError(f"SANA-WM mmcv shim expected dict or str cfg, got {type(cfg)!r}.")

    default_args = dict(default_args or {})
    for key, value in default_args.items():
        args.setdefault(key, value)
    obj_type = args.pop("type", None)
    if obj_type is None:
        raise KeyError(f"`type` is required to build from registry {registry.name}.")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type!r} is not registered in {registry.name}.")
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"Invalid registry type {obj_type!r}.")
    return obj_cls(**args)


def _install_mmcv_shim() -> None:
    try:
        import mmcv  # noqa: F401

        return
    except ImportError:
        pass

    def _get_dist_info() -> tuple[int, int]:
        try:
            import torch.distributed as dist
        except ImportError:
            return 0, 1
        if not dist.is_available() or not dist.is_initialized():
            return 0, 1
        return dist.get_rank(), dist.get_world_size()

    def _mkdir_or_exist(path: str | os.PathLike[str]) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def _dump(obj: Any, filename: str | os.PathLike[str]) -> None:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def _load(filename: str | os.PathLike[str]) -> Any:
        with open(filename, "rb") as f:
            return pickle.load(f)

    try:
        import torch.nn as nn

        batch_norm = nn.modules.batchnorm._BatchNorm
        instance_norm = nn.modules.instancenorm._InstanceNorm
    except Exception:
        batch_norm = object
        instance_norm = object

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = _OfficialConfig
    mmcv.Registry = _OfficialRegistry
    mmcv.build_from_cfg = _official_build_from_cfg
    mmcv.mkdir_or_exist = _mkdir_or_exist
    mmcv.dump = _dump
    mmcv.load = _load

    runner = types.ModuleType("mmcv.runner")
    runner.get_dist_info = _get_dist_info
    runner.OPTIMIZER_BUILDERS = _OfficialRegistry("optimizer_builders")
    runner.OPTIMIZERS = _OfficialRegistry("optimizers")

    class _DefaultOptimizerConstructor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("mmcv optimizer construction is unavailable in the SANA-WM inference shim.")

    def _build_optimizer(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("mmcv optimizer construction is unavailable in the SANA-WM inference shim.")

    runner.DefaultOptimizerConstructor = _DefaultOptimizerConstructor
    runner.build_optimizer = _build_optimizer

    utils = types.ModuleType("mmcv.utils")
    utils._BatchNorm = batch_norm
    utils._InstanceNorm = instance_norm

    logging_mod = types.ModuleType("mmcv.utils.logging")
    logging_mod.logger_initialized = {}
    utils.logging = logging_mod

    mmcv.runner = runner
    mmcv.utils = utils
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.runner"] = runner
    sys.modules["mmcv.utils"] = utils
    sys.modules["mmcv.utils.logging"] = logging_mod


def _install_qwen_vl_utils_shim() -> None:
    try:
        import qwen_vl_utils  # noqa: F401

        return
    except ImportError:
        pass

    qwen_vl_utils = types.ModuleType("qwen_vl_utils")

    def _process_vision_info(*args: Any, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        raise ImportError("qwen-vl-utils is required only when using the Qwen-VL text encoder.")

    qwen_vl_utils.process_vision_info = _process_vision_info
    sys.modules["qwen_vl_utils"] = qwen_vl_utils


def _install_official_import_shims() -> None:
    _install_mmcv_shim()
    _install_qwen_vl_utils_shim()


@lru_cache(maxsize=4)
def _load_official_module(script_path_str: str) -> ModuleType:
    script_path = Path(script_path_str)
    repo = script_path.parents[1]
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    _install_official_import_shims()

    module_name = "_vllm_omni_sana_wm_inprocess"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SANA-WM native backend from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _to_pil_image(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")

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
        raise ValueError(f"Sana-WM image must be RGB-like HWC/CHW data, got shape {array.shape}.")
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
    return Image.fromarray(array, mode="RGB")


def _resolve_num_frames(module: ModuleType, payload: Mapping[str, Any], c2w_full: np.ndarray) -> int:
    requested = int(payload["num_frames"])
    num_frames = min(requested, int(c2w_full.shape[0]))
    snap = getattr(module, "_snap_num_frames", None)
    if callable(snap):
        return int(snap(num_frames, stride=8, upper_bound=int(c2w_full.shape[0])))
    return num_frames


def _resolve_c2w(payload: Mapping[str, Any]) -> np.ndarray:
    if payload.get("action"):
        return action_string_to_c2w(
            str(payload["action"]),
            translation_speed=float(payload.get("translation_speed", 0.05)),
            rotation_speed_deg=float(payload.get("rotation_speed_deg", 1.2)),
        )
    camera = payload.get("camera")
    if not isinstance(camera, Mapping) or camera.get("poses") is None:
        raise ValueError("Sana-WM native backend requires action or camera.poses.")
    c2w = np.asarray(camera["poses"], dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Sana-WM camera poses must have shape (F, 4, 4), got {c2w.shape}.")
    return c2w


def _sampling_attr(sampling_params: Any, name: str, default: Any) -> Any:
    value = getattr(sampling_params, name, None)
    return default if value is None else value


def run_sana_wm_native_backend(
    *,
    prompt: Mapping[str, Any],
    release_paths: SanaWmLocalPathsLike,
    include_refiner: bool,
    repo_path: str | os.PathLike[str] | None = None,
    sampling_params: Any | None = None,
) -> SanaWmNativeRunResult:
    """Run SANA-WM through the official Python modules without a subprocess."""

    script = require_sana_wm_native_script(repo_path)
    module = _load_official_module(str(script))
    payload = prompt["additional_information"]["sana_wm"]
    image = _to_pil_image(_extract_image(prompt))
    c2w_full = _resolve_c2w(payload)
    num_frames = _resolve_num_frames(module, payload, c2w_full)
    c2w = c2w_full[:num_frames]

    cropped, src_size, resized_size, crop_offset = module.resize_and_center_crop(image)
    used_default_intrinsics = payload.get("intrinsics") is None
    intrinsics_source = intrinsics_to_vec4_array(
        payload.get("intrinsics"),
        num_frames=num_frames,
        height=image.height,
        width=image.width,
    )
    intrinsics = module.transform_intrinsics_for_crop(intrinsics_source, src_size, resized_size, crop_offset)

    config = module.pyrallis.parse(
        config_class=module.InferenceConfig,
        config_path=str(release_paths.config),
        args=[],
    )
    # The public config points VAE loading at the HF repo id. Use the already
    # resolved snapshot so diffusers does not fetch the full Sana-WM repo again.
    config.vae.vae_pretrained = str(release_paths.root)
    seed = int(_sampling_attr(sampling_params, "seed", 42) or 42)
    fps = int(_sampling_attr(sampling_params, "fps", payload.get("fps", 16)) or 16)
    steps = int(_sampling_attr(sampling_params, "num_inference_steps", 60) or 60)
    guidance_provided = bool(getattr(sampling_params, "guidance_scale_provided", False))
    cfg_scale = float(getattr(sampling_params, "guidance_scale", 5.0) if guidance_provided else 5.0)
    extra_args = getattr(sampling_params, "extra_args", {}) or {}
    sampling_algo = str(extra_args.get("sana_wm_sampling_algo", "flow_euler_ltx"))
    flow_shift = extra_args.get("sana_wm_flow_shift", None)

    refiner = None
    if include_refiner:
        refiner = module.RefinerSettings(
            root=release_paths.refiner_root,
            gemma_root=release_paths.refiner_text_encoder_dir,
            sink_size=int(extra_args.get("sana_wm_sink_size", 1)),
            seed=int(extra_args.get("sana_wm_refiner_seed", seed)),
        )

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = module.SanaWMPipeline(
        config=config,
        model_path=str(release_paths.stage1_dit),
        device=device,
        refiner=refiner,
        offload_vae=bool(extra_args.get("sana_wm_offload_vae", False)),
        offload_refiner=bool(extra_args.get("sana_wm_offload_refiner", False)),
    )
    params = module.GenerationParams(
        num_frames=num_frames,
        fps=fps,
        step=steps,
        cfg_scale=cfg_scale,
        flow_shift=flow_shift,
        seed=seed,
        negative_prompt=str(prompt.get("negative_prompt") or ""),
        sampling_algo=sampling_algo,
    )
    output = pipeline.generate(cropped, str(prompt.get("prompt") or ""), c2w, intrinsics, params)
    frames = np.asarray(output["video"])
    return SanaWmNativeRunResult(
        frames=frames,
        backend="native_inprocess_official",
        repo_path=str(script.parents[1]),
        model_path=str(release_paths.stage1_dit),
        include_refiner=include_refiner,
        num_frames=num_frames,
        fps=fps,
        sampling_steps=steps,
        cfg_scale=cfg_scale,
        used_default_intrinsics=used_default_intrinsics,
    )
