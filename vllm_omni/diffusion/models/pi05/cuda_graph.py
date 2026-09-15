# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph execution for the π0.5 flow-matching kernel.

π0.5 inference has two fixed-shape phases: one prefix pass (three 224×224
camera slots + a 200-token prompt through SigLIP and Gemma-2B) and
``num_inference_steps`` denoising steps that each push 50 action tokens through
the Gemma-300M action expert. Every shape is pinned by the deploy config, so
both phases are captured once per batch size and replayed:

* **prefix graph** — inputs are the camera images, image masks and prompt
  tokens/masks; outputs are the prefix padding mask and the per-layer
  ``(k, v)`` cache, which stay in the graph's own memory.
* **denoise graph** — inputs are ``x_t`` and the timestep; it reads the prefix
  graph's outputs directly (they are static tensors at fixed addresses) and
  writes ``v_t``.

The denoising loop is where the wall-clock goes: ~18 layers × ~30 tiny kernels
per step, which eager PyTorch cannot launch fast enough to keep the GPU busy.
Replaying the whole step as one graph removes that launch overhead. The prefix
pass is graphed for the same reason at a smaller gain.

Replaying a captured graph runs the identical kernels eager mode ran, so the
action chunk is bit-identical to the eager one. Outputs are static buffers: a
returned tensor is valid until the next replay of the graph that produced it,
which is all the Euler loop in ``Pi05ForActionPrediction.sample_actions`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _PrefixGraph:
    graph: torch.cuda.CUDAGraph
    images: list[torch.Tensor]
    image_masks: list[torch.Tensor]
    lang_tokens: torch.Tensor
    lang_masks: torch.Tensor
    prefix_pad_masks: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class _DenoiseGraph:
    graph: torch.cuda.CUDAGraph
    x_t: torch.Tensor
    timestep: torch.Tensor
    v_t: torch.Tensor


def serving_shaped_inputs(model, batch_size: int) -> dict[str, object]:
    """Inputs at the deployed shapes, as ``sample_actions`` keyword arguments.

    Zero images with every camera slot marked present, an all-live prompt and
    Gaussian noise: only shapes and dtypes matter for warm-up and capture.
    """
    config = model.config
    device = model.action_in_proj.weight.device
    vision_dtype = next(model.paligemma_with_expert.paligemma.model.vision_tower.parameters()).dtype
    height, width = config.image_resolution
    num_cameras = int(config.max_cameras)
    prompt_len = int(config.tokenizer_max_length)
    return {
        "images": [
            torch.zeros(batch_size, 3, height, width, dtype=vision_dtype, device=device) for _ in range(num_cameras)
        ],
        "image_masks": [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(num_cameras)],
        "lang_tokens": torch.zeros(batch_size, prompt_len, dtype=torch.long, device=device),
        "lang_masks": torch.ones(batch_size, prompt_len, dtype=torch.bool, device=device),
        "noise": torch.randn(batch_size, model.action_horizon, model.action_dim, dtype=torch.float32, device=device),
    }


class Pi05CudaGraphRunner:
    """Replay ``encode_prefix`` / ``denoise_step`` as CUDA graphs.

    Installed on the model as ``model.cuda_graph_runner``; ``sample_actions``
    then routes both phases through here. Graphs are captured lazily per batch
    size on first use, or ahead of time via :meth:`capture`.
    """

    def __init__(self, model):
        self.model = model
        # A private pool keeps this runner's graph memory apart from vLLM's
        # global pool; the prefix and denoise graphs share it and never run
        # concurrently.
        self.pool = torch.cuda.graph_pool_handle()
        self._prefix: dict[int, _PrefixGraph] = {}
        self._denoise: dict[int, _DenoiseGraph] = {}

    # ------------------------------------------------------------------
    # Model-facing interface (same signatures as Pi05ForActionPrediction)
    # ------------------------------------------------------------------
    def encode_prefix(self, images, image_masks, lang_tokens, lang_masks):
        bsize = lang_tokens.shape[0]
        entry = self._prefix.get(bsize)
        if entry is None:
            entry = self._capture_prefix(images, image_masks, lang_tokens, lang_masks)
            self._prefix[bsize] = entry

        for static, live in zip(entry.images, images):
            static.copy_(live)
        for static, live in zip(entry.image_masks, image_masks):
            static.copy_(live)
        entry.lang_tokens.copy_(lang_tokens)
        entry.lang_masks.copy_(lang_masks)
        entry.graph.replay()
        return entry.prefix_pad_masks, entry.past_key_values

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        bsize = x_t.shape[0]
        prefix = self._prefix.get(bsize)
        if prefix is None or past_key_values is not prefix.past_key_values:
            raise RuntimeError(
                "Pi05CudaGraphRunner.denoise_step must consume the past_key_values returned by "
                "Pi05CudaGraphRunner.encode_prefix for the same batch size; the denoise graph reads "
                "the prefix graph's static outputs in place."
            )
        entry = self._denoise.get(bsize)
        if entry is None:
            entry = self._capture_denoise(prefix, x_t, timestep)
            self._denoise[bsize] = entry

        entry.x_t.copy_(x_t)
        entry.timestep.copy_(timestep)
        entry.graph.replay()
        return entry.v_t

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture(self, batch_size: int = 1) -> None:
        """Capture both graphs for ``batch_size`` ahead of the first request."""
        inputs = serving_shaped_inputs(self.model, batch_size)
        x_t = inputs["noise"]
        timestep = torch.ones(batch_size, dtype=torch.float32, device=x_t.device)

        prefix_pad_masks, past_key_values = self.encode_prefix(
            inputs["images"], inputs["image_masks"], inputs["lang_tokens"], inputs["lang_masks"]
        )
        self.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)

    def _capture_prefix(self, images, image_masks, lang_tokens, lang_masks) -> _PrefixGraph:
        static_images = [img.clone() for img in images]
        static_image_masks = [mask.clone() for mask in image_masks]
        static_lang_tokens = lang_tokens.clone()
        static_lang_masks = lang_masks.clone()

        def run():
            return self.model.encode_prefix(static_images, static_image_masks, static_lang_tokens, static_lang_masks)

        graph, (prefix_pad_masks, past_key_values) = self._capture(run)
        logger.info("Pi05CudaGraphRunner: captured prefix graph for batch size %d.", lang_tokens.shape[0])
        return _PrefixGraph(
            graph=graph,
            images=static_images,
            image_masks=static_image_masks,
            lang_tokens=static_lang_tokens,
            lang_masks=static_lang_masks,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
        )

    def _capture_denoise(self, prefix: _PrefixGraph, x_t, timestep) -> _DenoiseGraph:
        static_x_t = x_t.clone()
        static_timestep = timestep.clone()

        def run():
            return self.model.denoise_step(prefix.prefix_pad_masks, prefix.past_key_values, static_x_t, static_timestep)

        graph, v_t = self._capture(run)
        logger.info("Pi05CudaGraphRunner: captured denoise graph for batch size %d.", x_t.shape[0])
        return _DenoiseGraph(graph=graph, x_t=static_x_t, timestep=static_timestep, v_t=v_t)

    def _capture(self, run):
        """Warm up ``run`` eagerly, then record it into a new graph."""
        with torch.inference_mode():
            # The eager pass triggers lazy work that must not land in the
            # capture: cuBLAS workspace allocation, autotuning, and any pending
            # torch.compile compilation of the per-layer functions.
            run()
            torch.accelerator.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                outputs = run()
        return graph, outputs
