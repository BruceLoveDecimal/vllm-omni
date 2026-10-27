# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 VLA pipeline for vllm-omni.

Entry point for ``DiffusionEngine.step() → pipeline.forward(req)``. Mirrors the
DreamZero contract and the π0 pipeline: the pipeline owns ALL preprocessing. It
reads the raw robot observation from
``req.sampling_params.extra_args["robot_obs"]`` (delivered by the OpenPI
realtime serving layer), builds model inputs, runs flow-matching denoising, and
returns ``DiffusionOutput(output={"actions": ndarray})``.

π0.5 is stateless across calls (no KV reuse, first-order Markov), so
``session_id`` / ``reset`` from the OpenPI protocol are accepted but ignored.

The post-processing order is load-bearing and matches LeRobot::

    unnormalize → absolute actions → to_cpu

``AbsoluteActionsProcessorStep`` must run *after* unnormalization, because a
relative-action checkpoint's ``norm_stats`` are computed in relative space.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.config import SUPPORTED_DTYPE_NAMES, Pi05Config
from vllm_omni.diffusion.models.pi05.cuda_graph_pi05 import Pi05CUDAGraphs
from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction
from vllm_omni.diffusion.models.pi05.processor_pi05 import Pi05Processor, observed_camera_keys, stack_model_inputs
from vllm_omni.diffusion.models.pi05_pipeline_config import PI05_PIPELINE as PI05_PIPELINE
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.component_config import resolve_component_quant_config

logger = init_logger(__name__)

# π0.5 pins the PaliGemma tokenizer (LeRobot hardcodes it too).
DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"

# float16 is absent deliberately, not by oversight: nothing here has been
# validated against it. The float32/bfloat16 trade-off is in the deploy config
# and recipes/lerobot/Pi05.md.
SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)

# The two lists guard different entry points — what the checkpoint declares
# versus the dtype actually cast to — so they must not drift apart.
assert {str(dtype).split(".")[-1] for dtype in SUPPORTED_DTYPES} == set(SUPPORTED_DTYPE_NAMES)


def _checkpoint_declared_keys(model_dir: str) -> set[str]:
    """The keys the checkpoint's config.json actually carries."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def _comparable(value):
    """Compare yaml and checkpoint values without tripping on list/tuple."""
    return list(value) if isinstance(value, (list, tuple)) else value


def _pi05_post_process(x):
    """Module-level identity post-process (picklable across the orchestrator's
    multiprocess boundary — a local closure is not)."""
    return x


def get_pi05_post_process_func(od_config: OmniDiffusionConfig):
    """π0.5 returns actions directly; post-processing is identity."""
    del od_config
    return _pi05_post_process


def _pi05_pre_process(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
    """Key the request by the cameras its observation carries.

    The request scheduler batches only requests with equal keys, so a batch
    shares one view layout, as RFC #6867 asks of R2. A missing camera is
    already a masked slot, so mixed layouts would batch correctly too; that
    stays off until it has its own parity coverage.
    """
    robot_obs = (request.sampling_params.extra_args or {}).get("robot_obs")
    if isinstance(robot_obs, dict):
        request.batch_compatibility_key = ("pi05_cameras", observed_camera_keys(robot_obs))
    return request


def get_pi05_pre_process_func(od_config: OmniDiffusionConfig):
    """Engine-side batch keying; see :func:`_pi05_pre_process`."""
    del od_config
    return _pi05_pre_process


@dataclass
class _Pi05Row:
    """One request of a batch, ready to stack."""

    robot_obs: dict
    # ``Pi05Processor.build_model_inputs`` output for this observation alone.
    inputs: tuple[Any, ...]
    num_steps: int | None
    generator: torch.Generator | None


_LEROBOT_FLOAT32_IN_BFLOAT16 = (
    "vision_tower",
    "multi_modal_projector",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


def _to_bfloat16_for_inference(model: Pi05ForActionPrediction) -> None:
    """Apply LeRobot's mixed-precision inference layout explicitly."""
    inner_model = model.paligemma_with_expert
    for name, param in inner_model.named_parameters():
        target_dtype = (
            torch.float32 if any(selector in name for selector in _LEROBOT_FLOAT32_IN_BFLOAT16) else torch.bfloat16
        )
        param.data = param.data.to(dtype=target_dtype)
    for buffer in inner_model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.bfloat16)

    for name, param in model.named_parameters():
        if not name.startswith("paligemma_with_expert."):
            param.data = param.data.to(dtype=torch.float32)
    for name, buffer in model.named_buffers():
        if not name.startswith("paligemma_with_expert.") and buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.float32)


def _set_inference_dtype(model: Pi05ForActionPrediction, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        model.to(dtype=torch.float32)
    elif dtype == torch.bfloat16:
        _to_bfloat16_for_inference(model)
    else:
        raise ValueError(f"Unsupported π0.5 inference dtype: {dtype!r}.")


class Pi05Pipeline(nn.Module):
    """π0.5 VLA pipeline: raw robot obs → continuous action chunk.

    Registered as ``"Pi05Pipeline"`` in the diffusion registry.
    """

    # A request batch runs as one ``sample_actions`` call (``forward``).
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_dir = self._resolve_model_dir(od_config.model)
        self.config = self._build_config(od_config)

        custom_args = od_config.custom_pipeline_args or {}
        self.tokenizer_source = str(custom_args.get("tokenizer", self._resolve_tokenizer_source()))

        self._torch_dtype = self._resolve_dtype(od_config)
        self._device = self._resolve_device(od_config)

        self.tokenizer = self._load_tokenizer()
        self.model = self._initialize_model()
        # Sized for one observation, the deployed default (``max_num_seqs: 1``).
        # A larger batch gets a workspace of its own size in ``sample_actions``.
        self.model.kv_workspace = self.model.new_kv_workspace(batch_size=1)
        # ``Pi05CUDAGraphs`` replays ``sample_actions``; read by the model
        # runner (inductor cudagraphs off under compile) and the offloader.
        self.enable_cuda_graph_decode = (
            bool(getattr(od_config, "enable_cuda_graph_decode", True))
            and not od_config.enforce_eager
            and self._device.type == "cuda"
        )
        self.model.enable_cuda_graph_decode = self.enable_cuda_graph_decode
        self._install_cuda_graphs(od_config)

        self.processor = Pi05Processor(self.config, self.tokenizer, self._device)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_model_dir(model: str | None) -> str | None:
        """Return a local directory for ``model``; download an HF repo id if needed."""
        if not model:
            return None
        if os.path.isdir(model):
            return model
        # Via repo_utils' shared HfApi rather than huggingface_hub directly, so
        # the download carries vLLM-Omni's user agent like every other repo access.
        from vllm_omni.transformers_utils.repo_utils import hf_api

        return hf_api().snapshot_download(
            repo_id=model,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )

    def _build_config(self, od_config: OmniDiffusionConfig) -> Pi05Config:
        """Read the config from the checkpoint, then let the deploy yaml override it."""
        checkpoint_config = Pi05Config.from_pretrained(self.model_dir) if self.model_dir else None
        if checkpoint_config is None:
            return Pi05Config.from_model_config(od_config.model_config)
        if not od_config.model_config:
            return checkpoint_config

        resolved = {
            item.name: getattr(checkpoint_config, item.name) for item in dataclass_fields(Pi05Config) if item.init
        }
        # Only a key the checkpoint actually declares can disagree with the yaml.
        # Comparing against the dataclass default instead would report every
        # serving-only field, such as policy_server_config, on every start.
        declared_keys = _checkpoint_declared_keys(self.model_dir)
        for key, value in od_config.model_config.items():
            if key in resolved and key in declared_keys and _comparable(value) != _comparable(resolved[key]):
                logger.warning(
                    "Pi05Pipeline: the deploy config sets %s=%r, overriding %r from the checkpoint.",
                    key,
                    value,
                    resolved[key],
                )
        resolved.update(od_config.model_config)
        return Pi05Config.from_model_config(resolved)

    def _resolve_tokenizer_source(self) -> str:
        """Prefer the checkpoint dir if it ships tokenizer files; else PaliGemma."""
        if self.model_dir and os.path.isdir(self.model_dir):
            if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")):
                return self.model_dir
        return DEFAULT_PI05_TOKENIZER

    @staticmethod
    def _resolve_dtype(od_config: OmniDiffusionConfig) -> torch.dtype:
        """Resolve the dtype the weights are actually cast to.

        This is the load-bearing check, not ``Pi05Config.dtype``: the cast in
        :meth:`_initialize_model` reads the *top-level* ``OmniDiffusionConfig``
        field, so a guard on the model config alone would let an unsupported
        dtype through. See :data:`SUPPORTED_DTYPES` for why float16 is excluded.
        """
        dt = od_config.dtype
        resolved = dt if isinstance(dt, torch.dtype) else getattr(torch, str(dt).split(".")[-1], None)
        if resolved not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported π0.5 dtype: {dt!r}. Supported: "
                f"{', '.join(sorted(str(d).split('.')[-1] for d in SUPPORTED_DTYPES))}."
            )
        return resolved

    @staticmethod
    def _resolve_device(od_config: OmniDiffusionConfig) -> torch.device:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        try:
            return get_local_device()
        except Exception:  # noqa: BLE001
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        # padding_side="right" is part of the π0.5 spec: the prefix is a fixed
        # 200-token block whose live tokens must start at index 0.
        return AutoTokenizer.from_pretrained(self.tokenizer_source, padding_side="right")

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))

    def _initialize_model(self) -> Pi05ForActionPrediction:
        if not self.has_real_checkpoint():
            expected = os.path.join(self.model_dir or "<missing-model-dir>", "model.safetensors")
            raise FileNotFoundError(f"π0.5 serving requires checkpoint weights at {expected}.")
        # Only the PaliGemma language model is quantized; a component config
        # can still narrow it, under the ``language_model`` prefix.
        quant_config = resolve_component_quant_config(self.od_config.quantization_config, "language_model")
        # Built on the device: an online-quantized weight starts on ``meta``
        # and is materialized there as it loads, which ``.to(device)`` cannot move.
        with torch.device(self._device):
            model = Pi05ForActionPrediction(self.config, quant_config=quant_config)
        _set_inference_dtype(model, self._torch_dtype)
        self._load_checkpoint(model)
        model.process_weights_after_loading()
        model.eval()
        return model

    def _install_cuda_graphs(self, od_config: OmniDiffusionConfig) -> None:
        """Pick the ``sample_actions`` execution path.

        The stage's ``enforce_eager`` (deploy yaml, or ``--enforce-eager`` on the
        CLI, which takes precedence) keeps the eager baseline. Otherwise the
        optimized path is set up here, at init: the fused Triton kernels are
        enabled, and unless ``enable_cuda_graph_decode`` is off the CUDA graphs
        are captured over them and installed. With ``torch_compile`` the capture
        waits for ``setup_compile``, so the graphs record the compiled layers.
        A failure of either raises rather than silently serving eagerly.
        """
        if od_config.enforce_eager:
            logger.info("Pi05Pipeline: enforce_eager is set; sample_actions runs eagerly.")
            return
        if self._device.type != "cuda":
            logger.warning(
                "Pi05Pipeline: CUDA graphs need a CUDA device, got %s; sample_actions runs eagerly.",
                self._device,
            )
            return
        try:
            self.model.enable_fused_kernels()
        except RuntimeError as exc:
            raise RuntimeError(
                "π0.5's optimized path needs its fused kernels. Set `enforce_eager: true` in the deploy "
                "config, or pass --enforce-eager, to serve eagerly."
            ) from exc
        if not self.enable_cuda_graph_decode:
            logger.info("Pi05Pipeline: enable_cuda_graph_decode is off; sample_actions runs without CUDA graphs.")
            return
        if self.config.torch_compile and current_omni_platform.supports_torch_inductor():
            logger.info("Pi05Pipeline: torch_compile is set; the CUDA graphs are captured in setup_compile.")
            return
        self._capture_cuda_graphs()

    def _capture_cuda_graphs(self) -> None:
        graphs = Pi05CUDAGraphs(self.model)
        start = time.perf_counter()
        graphs.capture()
        self.model.cuda_graphs = graphs
        logger.info(
            "Pi05Pipeline: captured the sample_actions CUDA graphs over the fused kernels in %.2f s; "
            "sample_actions runs its optimized path.",
            time.perf_counter() - start,
        )

    def setup_compile(self) -> None:
        """The model runner's compile hook, called unless the stage is ``enforce_eager``.

        Compiles only with the deploy config's ``torch_compile``: the Gemma
        decoder layers, one ``torch.compile`` each (``regionally_compile``;
        ``diffusion_compile_granularity`` has no other granularity to offer
        here), then captures the CUDA graphs over them. Inductor's own
        cudagraphs stay off under the graphs this pipeline captures, so the
        two never stack. Compilation is lazy and happens in the capture's
        warm-up. The batch-1 denoising step runs the fused kernels, not the
        expert layers, so on the served path compiling changes the prefix pass.
        """
        if not self.config.torch_compile:
            logger.info("Pi05Pipeline: torch_compile is off in the deploy config; the layers run uncompiled.")
            return
        compile_kwargs: dict[str, Any] = {"dynamic": self.od_config.diffusion_compile_dynamic}
        if self.enable_cuda_graph_decode:
            compile_kwargs["options"] = {"triton.cudagraphs": False, "triton.cudagraph_trees": False}
        regionally_compile(self.model, **compile_kwargs)
        if self.enable_cuda_graph_decode:
            self._capture_cuda_graphs()

    def _load_checkpoint(self, model: Pi05ForActionPrediction) -> None:
        import safetensors.torch

        path = os.path.join(self.model_dir, "model.safetensors")
        logger.info("Pi05Pipeline: loading π0.5 weights from %s", path)
        try:
            state = safetensors.torch.load_file(path)
            model.load_weights(state.items())
        except Exception as exc:
            raise RuntimeError(f"Failed to load complete π0.5 checkpoint {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Framework weight-loading hook
    # ------------------------------------------------------------------
    def load_weights(self, weights=()):  # noqa: D401
        """No-op for the diffusion loader: π0.5 self-loads its checkpoint."""
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _prepare_row(self, request: OmniDiffusionRequest) -> _Pi05Row | DiffusionOutput:
        """One request's model inputs, or the output it gets without running the model."""
        sampling = request.sampling_params
        robot_obs = (getattr(sampling, "extra_args", None) or {}).get("robot_obs")
        num_steps = getattr(sampling, "num_inference_steps", None)

        if robot_obs is None:
            # Dummy warmup path (no obs): return zeros so engine warmup/capture
            # doesn't crash. Mirrors DreamZero's dummy-run handling.
            prompt = getattr(request, "prompt", None) or ""
            prompt = prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")
            if prompt == "dummy run" or num_steps == 1:
                logger.info("Pi05Pipeline: dummy warmup request without robot_obs — returning zeros.")
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.config.chunk_size, self.config.action_dim),
                            dtype=np.float32,
                        )
                    },
                )
            return DiffusionOutput(
                error="Pi05Pipeline.forward requires sampling_params.extra_args['robot_obs'].",
            )

        try:
            if num_steps is not None and (
                isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)) or int(num_steps) < 1
            ):
                raise ValueError(f"num_inference_steps must be a positive integer, got {num_steps!r}.")
            generator = getattr(sampling, "generator", None)
            if isinstance(generator, list):
                if len(generator) != 1:
                    raise ValueError(f"A π0.5 request takes one generator, got {len(generator)}.")
                generator = generator[0]
            inputs = self.processor.build_model_inputs(robot_obs)
        except ValueError as exc:
            return DiffusionOutput(error=str(exc))
        return _Pi05Row(
            robot_obs=robot_obs,
            inputs=inputs,
            num_steps=None if num_steps is None else int(num_steps),
            generator=generator,
        )

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> list[DiffusionOutput]:
        """Serve every observation in ``req`` through one batched ``sample_actions``.

        The request scheduler only batches requests whose step count and camera
        layout match (``get_pi05_pre_process_func``). A request whose input is
        invalid gets its own error output; the others still run. Each row draws
        its noise from its own request's generator, so its actions match that
        request served alone.
        """
        outputs: list[DiffusionOutput | None] = []
        groups: dict[int | None, list[tuple[int, _Pi05Row]]] = {}
        for index, request in enumerate(req.requests):
            row = self._prepare_row(request)
            if isinstance(row, DiffusionOutput):
                outputs.append(row)
            else:
                outputs.append(None)
                groups.setdefault(row.num_steps, []).append((index, row))

        # One group in practice: ``num_inference_steps`` is part of the
        # scheduler's batch key. Grouping keeps a direct caller correct too.
        for num_steps, members in groups.items():
            rows = [row for _, row in members]
            images, image_masks, lang_tokens, lang_masks = stack_model_inputs([row.inputs for row in rows])
            actions = self.model.sample_actions(
                images=images,
                image_masks=image_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                num_steps=num_steps,
                generator=[row.generator for row in rows],
            )
            for batch_row, (index, row) in enumerate(members):
                chunk = self.processor.build_model_outputs(actions[batch_row : batch_row + 1], row.robot_obs)
                outputs[index] = DiffusionOutput(output={"actions": chunk})
        return outputs
