# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-layer reference K/V cache for Wan2.2-Animate-2.

Animate-2 conditions generation on a driving video through *in-context
reference attention*: once per segment the transformer runs a dedicated
"extract" forward over the driving-video latents and stores each layer's
pre-RoPE key/value tensors.  Every denoising step then attends jointly over
the generated tokens and the frame-aligned slice of these cached tensors.

The cache is deliberately a plain Python object owned by the pipeline rather
than a submodule: attaching it to the ``nn.Module`` tree would leak segment
state into ``state_dict()``, FSDP flat parameters and offload bookkeeping.
"""

from __future__ import annotations

import torch


class ReferenceKVCache:
    """Layer-indexed store of pre-RoPE reference key/value tensors.

    Entries are written once per segment by
    ``WanAnimate2Transformer3DModel.extract_reference`` and read by every
    denoising forward of that segment.  Both tensors are ``[B, S, H, D]`` with
    ``H`` the *local* (tensor-parallel) head count, so no cross-rank
    communication is needed between the two phases.
    """

    def __init__(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.num_layers = num_layers
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    def __len__(self) -> int:
        return self.num_layers

    @torch._dynamo.disable
    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Record layer ``layer_idx``'s pre-RoPE key/value tensors.

        Hidden from Dynamo: indexing a Python list with the layer number makes
        it specialise the block graph per layer, so a 40-layer model blows the
        recompile limit and silently falls back to eager part-way through the
        run.  This is bookkeeping with no arithmetic, so a graph break here
        costs nothing and keeps one compiled graph shared across
        ``_repeated_blocks``.
        """
        self._check_index(layer_idx)
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch at layer {layer_idx}: {key.shape} vs {value.shape}")
        if key.ndim != 4:
            raise ValueError(f"expected [B, S, H, D] reference key at layer {layer_idx}, got {tuple(key.shape)}")
        self._keys[layer_idx] = key
        self._values[layer_idx] = value

    @torch._dynamo.disable
    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached ``(key, value)`` pair for ``layer_idx``.

        Hidden from Dynamo for the same reason as :meth:`store`.
        """
        self._check_index(layer_idx)
        key = self._keys[layer_idx]
        value = self._values[layer_idx]
        if key is None or value is None:
            raise RuntimeError(
                f"reference K/V for layer {layer_idx} was never populated; "
                "call extract_reference() once per segment before denoising"
            )
        return key, value

    def is_populated(self) -> bool:
        """True when every layer has been written."""
        return all(key is not None for key in self._keys)

    def release(self) -> None:
        """Drop all references so the segment's cache memory can be reclaimed."""
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers

    def _check_index(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer index {layer_idx} out of range for {self.num_layers}-layer cache")
