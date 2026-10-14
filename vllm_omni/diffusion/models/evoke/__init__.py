# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alaya-EVOKE real-time world model.

Submodules are imported explicitly: the interaction registry imports
``evoke.camera`` and must not pull in the pipeline (and its heavy deps) with it.
"""
