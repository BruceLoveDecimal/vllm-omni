# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated per-request dots.tts controls, shared by serving and offline prompts."""

import importlib.util
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def is_meanflow(hf_config: Any) -> bool:
    return bool((getattr(hf_config, "meanflow", None) or {}).get("enabled", False))


class DotsTTSRequestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    template_name: Literal["tts", "instruction_tts", "text_to_audio"] = "tts"
    normalize_text: bool = Field(default=False, strict=True)

    num_steps: int = Field(default=int(os.environ.get("DOTS_TTS_DIT_NUM_STEPS", "10")), gt=0, strict=True)
    guidance_scale: float = Field(default=1.2, ge=0)
    speaker_scale: float = Field(default=1.5, ge=0)
    eos_threshold: float = Field(default=0.8, ge=0, le=1)
    ode_method: Literal["euler", "midpoint", "rk4"] = "euler"

    @field_validator("ode_method")
    @classmethod
    def require_solver_dependency(cls, value: str) -> str:
        if value != "euler" and importlib.util.find_spec("torchdiffeq") is None:
            raise ValueError("midpoint/rk4 require torchdiffeq; install with: pip install torchdiffeq==0.2.5")
        return value

    @classmethod
    def for_model(cls, values: dict | None, hf_config: Any) -> "DotsTTSRequestConfig":
        """Resolve checkpoint defaults without baking FM defaults into prompts."""
        meanflow = is_meanflow(hf_config)
        defaults = {"num_steps": int(os.environ.get("DOTS_TTS_DIT_NUM_STEPS", "4" if meanflow else "10"))}
        config = cls.model_validate(defaults | (values or {}))
        if meanflow and config.ode_method != "euler":
            raise ValueError("dots.tts MeanFlow supports only ode_method=euler")
        # MeanFlow checkpoints bake guidance into their learned field. Like
        # upstream, accept guidance_scale but do not apply an external CFG pass.
        return config
