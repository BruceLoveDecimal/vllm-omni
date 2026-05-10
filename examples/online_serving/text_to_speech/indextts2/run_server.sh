#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

MODEL="${1:-${INDEXTTS2_MODEL:-}}"
PORT="${PORT:-8091}"

if [[ -z "${MODEL}" ]]; then
  echo "Usage: $0 /path/to/IndexTTS2/checkpoint"
  echo "or set INDEXTTS2_MODEL=/path/to/IndexTTS2/checkpoint"
  exit 1
fi

vllm serve "${MODEL}" \
  --omni \
  --trust-remote-code \
  --port "${PORT}"
