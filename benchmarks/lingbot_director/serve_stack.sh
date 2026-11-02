#!/usr/bin/env bash
# Serve every model of the Director pipeline, each on its own GPU and port.
#
#   WORLD_GPU=0 STANDBY_GPU=1 VLM_GPU=2 EDIT_GPU=3 SEGMENT_GPU=3 \
#     WORLD_WIDTH=480 WORLD_HEIGHT=832 LOG_DIR=/tmp/director \
#     bash benchmarks/lingbot_director/serve_stack.sh
#
# Leave STANDBY_GPU empty to run one world; restarts then stop the running
# segment first. Every *_MODEL may be a local path. facebook/sam3.1 is gated, so
# export HF_TOKEN or point SEGMENT_MODEL at a local copy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORLD_MODEL="${WORLD_MODEL:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.8-27B}"
VLM_NAME="${VLM_NAME:-Qwen/Qwen3.8-27B}"
# The VLM is its own process, so it may use a newer vLLM than the Omni stages.
VLM_SERVE="${VLM_SERVE:-vllm serve}"
# Extra vLLM flags, for example --tensor-parallel-size 3 with VLM_GPU=2,3,4.
VLM_ARGS="${VLM_ARGS:-}"
EDIT_MODEL="${EDIT_MODEL:-Qwen/Qwen-Image-Edit-2511}"
SEGMENT_MODEL="${SEGMENT_MODEL:-facebook/sam3.1}"
WORLD_GPU="${WORLD_GPU:-0}"
STANDBY_GPU="${STANDBY_GPU:-}"
VLM_GPU="${VLM_GPU:-1}"
EDIT_GPU="${EDIT_GPU:-2}"
SEGMENT_GPU="${SEGMENT_GPU:-3}"
WORLD_WIDTH="${WORLD_WIDTH:-832}"
WORLD_HEIGHT="${WORLD_HEIGHT:-480}"
LOG_DIR="${LOG_DIR:-/tmp/lingbot-director}"
mkdir -p "$LOG_DIR"

# The world's cache geometry is fixed at load time.
WORLD_DEPLOY="$LOG_DIR/world-${WORLD_WIDTH}x${WORLD_HEIGHT}.yaml"
sed -e "s/ar_diffusion_height: 480/ar_diffusion_height: ${WORLD_HEIGHT}/" \
    -e "s/ar_diffusion_width: 832/ar_diffusion_width: ${WORLD_WIDTH}/" \
    "$ROOT/benchmarks/lingbot_world/configs/single_gpu_eager.yaml" > "$WORLD_DEPLOY"

serve() {
  local name="$1" gpu="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$@" > "$LOG_DIR/$name.log" 2>&1 &
  echo "$!" > "$LOG_DIR/$name.pid"
  echo "$name: pid $! on GPU $gpu, log $LOG_DIR/$name.log"
}

serve world "$WORLD_GPU" vllm serve "$WORLD_MODEL" --omni --deploy-config "$WORLD_DEPLOY" --port 8000
if [[ -n "$STANDBY_GPU" ]]; then
  serve standby "$STANDBY_GPU" vllm serve "$WORLD_MODEL" --omni --deploy-config "$WORLD_DEPLOY" --port 8004
fi
# shellcheck disable=SC2086
serve vlm "$VLM_GPU" $VLM_SERVE "$VLM_MODEL" --served-model-name "$VLM_NAME" \
  --limit-mm-per-prompt '{"image": 6}' --max-model-len 32768 --port 8001 $VLM_ARGS
serve edit "$EDIT_GPU" vllm serve "$EDIT_MODEL" --omni --port 8002
serve segment "$SEGMENT_GPU" vllm serve "$SEGMENT_MODEL" --omni --port 8003
