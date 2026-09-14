# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM-H3: camera-controlled causal video world model on MiniMax-H3."""

from .pipeline_solarwm_h3 import SolarWMH3Pipeline, get_solarwm_h3_post_process_func

__all__ = ["SolarWMH3Pipeline", "get_solarwm_h3_post_process_func"]
