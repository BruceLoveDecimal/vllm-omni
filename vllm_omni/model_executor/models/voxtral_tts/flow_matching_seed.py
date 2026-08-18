# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def resolve_tts_local_seeds(
    batch_size: int,
    sampling_extra_args: Sequence[Any] | None,
) -> list[int | None]:
    extras = list(sampling_extra_args) if sampling_extra_args is not None else []
    seeds: list[int | None] = []
    for index in range(batch_size):
        extra = extras[index] if index < len(extras) else None
        seed = extra.get("tts_local_seed") if isinstance(extra, Mapping) else None
        seeds.append(int(seed) if seed is not None else None)
    return seeds


def resolve_flow_matching_generators(
    seeds: Sequence[int | None],
    req_ids: Sequence[str] | None,
    *,
    device: torch.device,
    cache: dict[str, tuple[int, torch.Generator]],
    active_req_ids: Sequence[str] | None = None,
) -> list[torch.Generator | None] | None:
    if all(seed is None for seed in seeds):
        return None
    if active_req_ids is not None:
        live = set(active_req_ids)
        for stale_id in [req_id for req_id in cache if req_id not in live]:
            del cache[stale_id]

    generators: list[torch.Generator | None] = []
    for index, seed in enumerate(seeds):
        if seed is None:
            generators.append(None)
            continue
        req_id = req_ids[index] if req_ids is not None and index < len(req_ids) else f"__seed_{seed}_{index}"
        cached = cache.get(req_id)
        if cached is None or cached[0] != seed or cached[1].device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            cache[req_id] = (int(seed), generator)
        generators.append(cache[req_id][1])
    return generators


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
