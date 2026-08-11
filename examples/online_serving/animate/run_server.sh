#!/bin/bash
# Wan2.2-Animate (character animation) online serving startup script.
#
# Animation mode only: a reference character image plus pre-processed pose and
# face videos. Replacement mode is not supported.

MODEL="${MODEL:-Wan-AI/Wan2.2-Animate-14B-Diffusers}"
PORT="${PORT:-8092}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
TP="${TP:-1}"

echo "Starting Wan2.2-Animate server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
echo "Tensor parallel size: $TP"

# No --cache-backend here: the face adapter is injected between backbone
# blocks, so a generic block-level cache would skip the injection. The pipeline
# is registered in _NO_CACHE_ACCELERATION to make that explicit.
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$MODEL" --omni \
    --tensor-parallel-size "$TP" \
    --flow-shift "$FLOW_SHIFT" \
    --vae-use-slicing --vae-use-tiling \
    --port "$PORT"
