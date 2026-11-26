# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-R1 opening (cold) and continuation (warm) LoRA banks.

The release directory holds a root ``config.json`` naming both banks::

    {"model_variant": "Ref2VA", "rank": 128, "alpha": 128,
     "banks": {"cold": "cold/adapter_model.safetensors",
               "warm": "warm/adapter_model.safetensors"}}

Each bank directory carries its own ``config.json`` and the same native
208-target A/B inventory as TaoMate-H3, so ``prepare_taomate_lora`` loads it
unchanged. The diffusion LoRA manager pins both banks and asks the pipeline
which one is active (``diffusion_lora_fixed_banks`` /
``diffusion_lora_active_bank``).
"""

from __future__ import annotations

import json
from pathlib import Path

from vllm_omni.diffusion.models.taomate_h3.lora import TAOMATE_LORA_WEIGHTS_FILE, TaoMateLoRAError

from .chunks import BANKS, Bank

TAOMATE_R1_MODEL_VARIANT = "ref2va"


def read_taomate_r1_banks(root: str | Path) -> dict[Bank, Path]:
    """Validate the release layout and return each bank's adapter directory."""
    root = Path(root).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise TaoMateLoRAError(
            f"TaoMate-R1 adapter config not found at {config_path}; point --lora-path at the TaoMate-R1 root"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaoMateLoRAError(f"cannot read TaoMate-R1 adapter config: {exc}") from exc
    if not isinstance(config, dict):
        raise TaoMateLoRAError("TaoMate-R1 adapter config must be a JSON object")
    if "banks" not in config and (root / TAOMATE_LORA_WEIGHTS_FILE).is_file():
        raise TaoMateLoRAError(
            f"{root} is a single adapter bank; point --lora-path at the TaoMate-R1 root that holds cold/ and warm/"
        )
    variant = str(config.get("model_variant", "")).lower()
    if variant != TAOMATE_R1_MODEL_VARIANT:
        raise TaoMateLoRAError(f"TaoMate-R1 adapter targets model_variant {config.get('model_variant')!r}, not Ref2VA")
    banks = config.get("banks")
    if not isinstance(banks, dict) or set(banks) != set(BANKS):
        raise TaoMateLoRAError(f"TaoMate-R1 adapter config must name exactly the banks {list(BANKS)}")
    resolved: dict[Bank, Path] = {}
    for role in BANKS:
        relative = banks[role]
        if not isinstance(relative, str) or not relative:
            raise TaoMateLoRAError(f"TaoMate-R1 bank {role!r} path must be a non-empty string")
        weights = (root / relative).resolve()
        if root not in weights.parents:
            raise TaoMateLoRAError(f"TaoMate-R1 bank {role!r} must stay inside {root}")
        if weights.name != TAOMATE_LORA_WEIGHTS_FILE:
            raise TaoMateLoRAError(f"TaoMate-R1 bank {role!r} must point at {TAOMATE_LORA_WEIGHTS_FILE}")
        if not weights.is_file():
            raise TaoMateLoRAError(f"TaoMate-R1 bank {role!r} weights not found at {weights}")
        bank_config = weights.parent / "config.json"
        if bank_config.is_file():
            try:
                bank = json.loads(bank_config.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise TaoMateLoRAError(f"cannot read TaoMate-R1 {role} bank config: {exc}") from exc
            for key in ("rank", "alpha"):
                if key in config and key in bank and bank[key] != config[key]:
                    raise TaoMateLoRAError(
                        f"TaoMate-R1 {role} bank {key}={bank[key]!r} differs from the root {key}={config[key]!r}"
                    )
        else:
            raise TaoMateLoRAError(f"TaoMate-R1 bank {role!r} lacks config.json next to its weights")
        resolved[role] = weights.parent
    if resolved["cold"] == resolved["warm"]:
        raise TaoMateLoRAError("TaoMate-R1 cold and warm banks must be distinct directories")
    return resolved


__all__ = ["TAOMATE_R1_MODEL_VARIANT", "read_taomate_r1_banks"]
