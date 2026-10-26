# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned, mask-preserving mixed video/text KV storage."""

import torch


class HyNomasKVCache:
    """Reserve a contiguous history plus scratch tail for one world, per segment.

    Denoising overwrites the uncommitted tail. Only the separate cache-write
    forward advances length. Commits keep only visible tokens, so committed
    history never needs a key mask.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("KV capacity must be positive.")
        self.capacity = capacity
        self.length = 0
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None

    def stage(
        self, keys: tuple[torch.Tensor, ...], values: tuple[torch.Tensor, ...], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key, value = keys[0], values[0]
        count = sum(part.shape[1] for part in keys)
        end = self.length + count
        if (
            len(keys) != len(values)
            or any(k.shape[0] != 1 or k.shape != v.shape or k.shape[2:] != key.shape[2:] for k, v in zip(keys, values))
            or mask.shape != (1, count)
        ):
            raise ValueError("Preallocated Hy-Nomas KV requires one world and matching K/V/mask shapes.")
        if end > self.capacity:
            raise ValueError("Hy-Nomas KV capacity exceeded.")
        if self.key is None:
            shape = (1, end, *key.shape[2:])
            self.key = key.new_empty(shape)
            self.value = value.new_empty(shape)
            self.mask = mask.new_empty((1, end))
        elif end > self.key.shape[1]:
            # Grow only on the first forward of a larger segment prefix, not
            # every denoising step. Do not reserve future segments on the GPU.
            for name in ("key", "value", "mask"):
                previous = getattr(self, name)
                expanded = previous.new_empty((1, end, *previous.shape[2:]))
                expanded[:, : self.length].copy_(previous[:, : self.length])
                setattr(self, name, expanded)
        assert self.value is not None and self.mask is not None
        start = self.length
        for key_part, value_part in zip(keys, values):
            stop = start + key_part.shape[1]
            self.key[:, start:stop].copy_(key_part.detach())
            self.value[:, start:stop].copy_(value_part.detach())
            start = stop
        self.mask[:, self.length : end].copy_(mask)
        return self.key[:, :end], self.value[:, :end], self.mask[:, :end]

    def commit(self, mask: torch.Tensor, visible_prefix: int | None = None) -> None:
        """Append the staged tokens that ``mask`` keeps visible.

        ``visible_prefix`` states that exactly the first tokens are visible,
        avoiding a device sync and gather.
        """
        end = self.length + mask.shape[1]
        if self.mask is None or self.key is None or self.value is None or end > self.capacity:
            raise ValueError("Stage KV before committing within its capacity.")
        if visible_prefix is None:
            index = mask[0].nonzero().squeeze(1)
            visible_prefix = index.numel()
            if visible_prefix != mask.shape[1]:
                for tensor in (self.key, self.value):
                    tensor[:, self.length : self.length + visible_prefix] = tensor[:, self.length : end].index_select(
                        1, index
                    )
        elif not 0 <= visible_prefix <= mask.shape[1]:
            raise ValueError("visible_prefix must lie within the staged tokens.")
        self.mask[:, self.length : self.length + visible_prefix] = True
        self.length += visible_prefix

    def evict(self, tokens: int) -> None:
        if not 0 <= tokens <= self.length:
            raise ValueError("Invalid Hy-Nomas KV eviction length.")
        remaining = self.length - tokens
        if remaining:
            for tensor in (self.key, self.value, self.mask):
                assert tensor is not None
                # Overlapping copy_ is invalid; compact only at segment commits,
                # with one layer/tensor scratch allocation at a time.
                tensor[:, :remaining].copy_(tensor[:, tokens : self.length].clone())
        self.length = remaining
