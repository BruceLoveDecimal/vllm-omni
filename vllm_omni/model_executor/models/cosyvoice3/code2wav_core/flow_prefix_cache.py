# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, worker-local prefix reuse for chunk-causal packed Flow.

Each Euler step has its own CFG K/V and vector field. Only complete attention
chunks are reusable. Exact input comparisons also cover the lookahead encoder,
noise, speaker and prompt mel; request IDs alone never authorize a cache hit.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class FlowPrefixStep:
    # Each tensor has shape (2, frames, channels): conditional, unconditional.
    kv: tuple[torch.Tensor, ...]
    velocity: torch.Tensor


@dataclass
class FlowPrefixEntry:
    signature: torch.Tensor
    steps: list[FlowPrefixStep]

    @property
    def nbytes(self) -> int:
        tensors = [self.signature]
        for step in self.steps:
            tensors.extend((*step.kv, step.velocity))
        return sum(t.numel() * t.element_size() for t in tensors)


class FlowPrefixCache:
    def __init__(self, max_bytes: int, max_entries: int = 32):
        self.max_bytes = max(0, max_bytes)
        self.max_entries = max_entries
        self.entries: OrderedDict[int, FlowPrefixEntry] = OrderedDict()
        self.nbytes = 0
        self.schedule: tuple | None = None
        self._serial = 0
        self.hit_frames = 0
        self.query_frames = 0

    def reset(self, schedule: tuple) -> None:
        if schedule != self.schedule:
            self.entries.clear()
            self.nbytes = 0
            self.schedule = schedule

    def lookup(self, signatures: list[torch.Tensor], chunk_size: int) -> list[tuple[FlowPrefixEntry | None, int]]:
        """One device-to-host synchronization for all candidate comparisons.

        A changed lookahead or continuation can shorten a hit to an earlier
        complete chunk. This also permits reference-prefix reuse across requests.
        Keep at least one query frame so FA3 never receives an empty request.
        """
        candidates = list(self.entries.items())
        counts, pairs = [], []
        for row, signature in enumerate(signatures):
            for key, entry in candidates:
                length = min(signature.shape[0] - 1, entry.signature.shape[0])
                length = length // chunk_size * chunk_size
                if length <= 0:
                    continue
                equal = (signature[:length] == entry.signature[:length]).all(dim=-1)
                counts.append(equal.to(torch.int32).cumprod(0).sum())
                pairs.append((row, key))
        matches: list[tuple[FlowPrefixEntry | None, int]] = [(None, 0) for _ in signatures]
        if counts:
            for (row, key), length in zip(pairs, torch.stack(counts).tolist()):
                length = int(length) // chunk_size * chunk_size
                if length > matches[row][1]:
                    matches[row] = (self.entries[key], length)
                    self.entries.move_to_end(key)
        self.hit_frames += sum(length for _, length in matches)
        self.query_frames += sum(s.shape[0] - length for s, (_, length) in zip(signatures, matches))
        return matches

    def put(self, entry: FlowPrefixEntry, *, replaces: FlowPrefixEntry | None = None) -> None:
        size = entry.nbytes
        if size > self.max_bytes or size == 0 or self.max_entries <= 0:
            return
        if replaces is not None:
            for key, old in self.entries.items():
                if old is replaces:
                    del self.entries[key]
                    self.nbytes -= old.nbytes
                    break
        while self.entries and (self.nbytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
            _, old = self.entries.popitem(last=False)
            self.nbytes -= old.nbytes
        self.entries[self._serial] = entry
        self._serial += 1
        self.nbytes += size
