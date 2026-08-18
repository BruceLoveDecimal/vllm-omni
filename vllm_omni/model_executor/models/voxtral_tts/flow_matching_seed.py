# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def generators_from_tts_local_seed(
    batch_size: int,
    sampling_extra_args: Sequence[Any] | None,
    *,
    device: torch.device,
) -> list[torch.Generator | None] | None:
    extras = list(sampling_extra_args) if sampling_extra_args is not None else []
    generators: list[torch.Generator | None] = []
    any_seeded = False
    for index in range(batch_size):
        extra = extras[index] if index < len(extras) else None
        seed = extra.get("tts_local_seed") if isinstance(extra, Mapping) else None
        if seed is None:
            generators.append(None)
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)
        any_seeded = True
    return generators if any_seeded else None


def sample_flow_matching_noise(
    batch_size: int,
    n_codebooks: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generators: Sequence[torch.Generator | None] | None = None,
) -> torch.Tensor:
    if generators is None or all(generator is None for generator in generators):
        return torch.randn(batch_size, n_codebooks, device=device, dtype=dtype)
    if len(generators) != batch_size:
        raise ValueError(f"Expected {batch_size} noise generators, got {len(generators)}")
    noise = torch.empty(batch_size, n_codebooks, device=device, dtype=dtype)
    fill_flow_matching_noise(noise, generators)
    return noise


def fill_flow_matching_noise(
    noise: torch.Tensor,
    generators: Sequence[torch.Generator | None] | None = None,
    *,
    actual_size: int | None = None,
) -> None:
    row_count = noise.shape[0] if actual_size is None else actual_size
    if generators is None or all(generator is None for generator in generators[:row_count]):
        if actual_size is None:
            noise.normal_()
        else:
            noise[:row_count].normal_()
        return
    for row in range(row_count):
        generator = generators[row] if row < len(generators) else None
        if generator is None:
            noise[row].normal_()
        else:
            noise[row].normal_(generator=generator)
