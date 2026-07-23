# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

"""Request-local Gaussian-Shading noise generation for Mage-Flow."""

import hashlib

import numpy as np
import torch

DEFAULT_GS_KEY = 20260720
DEFAULT_GS_PAYLOAD = "MageFlow"
_MESSAGE_BITS = 256


def _key_to_int(value: int | str) -> int:
    if isinstance(value, int):
        return abs(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty Gaussian-Shading key")
    if text.lstrip("-").isdigit():
        return abs(int(text))
    return int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")


def _payload_to_bits(
    payload: str,
    num_bits: int = _MESSAGE_BITS,
) -> np.ndarray:
    bits: list[int] = []
    counter = 0
    while len(bits) < num_bits:
        digest = hashlib.sha256(f"{payload}:{counter}".encode()).digest()
        for byte in digest:
            bits.extend((byte >> bit_index) & 1 for bit_index in range(8))
        counter += 1
    return np.asarray(bits[:num_bits], dtype=np.int64)


def _pad_and_positions(
    size: int,
    key: int | str,
    num_bits: int = _MESSAGE_BITS,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_key_to_int(key))
    pad = rng.integers(0, 2, size=size).astype(np.int64)
    positions = rng.integers(0, num_bits, size=size).astype(np.int64)
    return pad, positions


def make_gaussian_shading_noise(
    shape: tuple[int, int, int, int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    key: int | str = DEFAULT_GS_KEY,
) -> torch.Tensor:
    """Generate the distribution-preserving noise used by official Mage."""
    batch, channels, height, width = shape
    if batch != 1:
        raise ValueError("Mage-Flow Gaussian-Shading currently supports batch size 1")
    size = channels * height * width
    message = _payload_to_bits(DEFAULT_GS_PAYLOAD)
    pad, positions = _pad_and_positions(size, key)
    target_half = (message[positions] ^ pad).astype(np.float64)

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed) & 0x7FFFFFFF)
    uniform = torch.rand(
        size,
        generator=cpu_generator,
        dtype=torch.float64,
        device="cpu",
    )
    half = torch.from_numpy(target_half)
    quantile = ((half + uniform) / 2.0).clamp(1e-6, 1.0 - 1e-6)
    noise = torch.special.ndtri(quantile).reshape(shape)
    return noise.to(device=device, dtype=dtype)


def make_initial_noise(
    shape: tuple[int, int, int, int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    enable_watermark: bool,
) -> torch.Tensor:
    """Generate noise without touching global RNG state."""
    if enable_watermark:
        return make_gaussian_shading_noise(
            shape,
            seed=generator.initial_seed(),
            device=device,
            dtype=dtype,
        )
    return torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
