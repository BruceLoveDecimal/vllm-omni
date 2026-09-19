#!/usr/bin/env bash
# Drive the AuK perf matrix on one GPU box.
#
# For every (checkpoint, arm): serve the arm's checkout, warm it with probe
# requests, then run `vllm-omni bench serve --omni` for every
# (workload, task, concurrency, repeat). A sidecar samples nvidia-smi and the
# stage-1 peak-memory gauge while the server is up. Results land in
# $OUT/<ckpt>/<arm>/ as bench JSONs plus a memory CSV per server run.
#
# Env knobs (defaults suit the RunPod layout after copying /workspace to local disk):
#   PY, CLIENT, ARMS_DIR, CKPT_BASE, CKPT_FLASH, DATASET, OUT
#   ARMS, CKPTS, TASKS, WORKLOADS, CONCS, REPEATS, NUM_PROMPTS, NUM_WARMUPS, PORT
set -uo pipefail

PY=${PY:-/root/envs/vllm029/bin/python}
CLIENT=${CLIENT:-/root/arms/client}
ARMS_DIR=${ARMS_DIR:-/root/arms}
CKPT_BASE=${CKPT_BASE:-/root/ckpts/auk-omni}
CKPT_FLASH=${CKPT_FLASH:-/root/ckpts/auk-omni-flash}
DATASET=${DATASET:-/root/seed-tts-eval}
OUT=${OUT:-/root/auk-bench/results}
PORT=${PORT:-8000}
HOST=127.0.0.1

ARMS=${ARMS:-"base E B V EB EV BV EBV"}
CKPTS=${CKPTS:-"flash base"}
TASKS=${TASKS:-"voice_clone default_voice"}
WORKLOADS=${WORKLOADS:-"uniform mixed"}
CONCS=${CONCS:-"1 2 4 8 16"}
REPEATS=${REPEATS:-3}
NUM_PROMPTS=${NUM_PROMPTS:-20}
NUM_WARMUPS=${NUM_WARMUPS:-2}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1200}

# Stage-0 needs these on sm_120; harmless on H100.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

SERVER_PID=""
SIDECAR_PID=""

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

ckpt_path() {
  case "$1" in
    base) echo "$CKPT_BASE" ;;
    flash) echo "$CKPT_FLASH" ;;
    *) echo "unknown ckpt $1" >&2; return 1 ;;
  esac
}

start_server() {
  local arm=$1 ckpt=$2 logfile=$3
  local checkout="$ARMS_DIR/$arm"
  [ -d "$checkout/vllm_omni" ] || { log "missing checkout $checkout"; return 1; }
  # setsid: own process group, so stop_server can kill the whole stage tree
  # without pattern-matching process names.
  PYTHONPATH="$checkout" setsid "$PY" -m vllm_omni.entrypoints.cli.main serve "$(ckpt_path "$ckpt")" \
    --omni --host "$HOST" --port "$PORT" --trust-remote-code \
    >"$logfile" 2>&1 &
  SERVER_PID=$!
  log "server pid=$SERVER_PID arm=$arm ckpt=$ckpt log=$logfile"
}

wait_health() {
  local t0=$SECONDS
  while (( SECONDS - t0 < HEALTH_TIMEOUT )); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      log "server healthy after $((SECONDS - t0))s"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "server died during startup"
      return 1
    fi
    sleep 3
  done
  log "server health timeout"
  return 1
}

stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  # The stage engine cores can outlive the serve process; wait until the GPU
  # is actually free before the next arm allocates.
  for _ in $(seq 1 30); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -KILL "$pid" 2>/dev/null
    done
    sleep 2
  done
  sleep 3
}

start_sidecar() {
  local csv=$1
  echo "ts,gpu_mem_used_mib,stage1_peak_mb" >"$csv"
  (
    while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
      peak=$(curl -sf "http://$HOST:$PORT/metrics" 2>/dev/null | awk '/^vllm_omni:peak_memory_mb\{.*stage="1"/ {print $2; exit}')
      printf '%s,%s,%s\n' "$(date +%s.%N)" "${used:-}" "${peak:-}" >>"$csv"
      sleep 0.25
    done
  ) &
  SIDECAR_PID=$!
}

stop_sidecar() {
  [ -n "$SIDECAR_PID" ] || return 0
  kill "$SIDECAR_PID" 2>/dev/null
  wait "$SIDECAR_PID" 2>/dev/null
  SIDECAR_PID=""
}

probe() {
  # Warm every graph/bucket the runs will touch: both tasks, three durations.
  local model=$1 ref
  ref=$(ls "$DATASET"/en/prompt-wavs/*.wav 2>/dev/null | head -1)
  for dur in 2.0 5.0 10.0; do
    curl -sf -o /dev/null "http://$HOST:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"input\":\"Warm up request for the benchmark.\",\"voice\":\"default\",\"task_type\":\"CustomVoice\",\"duration_seconds\":$dur,\"response_format\":\"wav\"}" \
      || log "probe (instruct, ${dur}s) failed"
    if [ -n "$ref" ]; then
      "$PY" - "$ref" "$model" "$dur" "http://$HOST:$PORT/v1/audio/speech" <<'PYEOF' || log "probe (clone, ${dur}s) failed"
import base64, json, sys, urllib.request
ref, model, dur, url = sys.argv[1:]
data = "data:audio/wav;base64," + base64.b64encode(open(ref, "rb").read()).decode()
body = {"model": model, "input": "Warm up request for the benchmark.", "voice": "default",
        "task_type": "Base", "ref_audio": data, "ref_text": "warm up", "duration_seconds": float(dur),
        "response_format": "wav"}
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=300).read()
PYEOF
    fi
  done
}

run_bench() {
  local model=$1 task=$2 workload=$3 conc=$4 rep=$5 outdir=$6
  local dataset_name extra fname
  case "$task" in
    voice_clone) dataset_name=seed-tts; extra='{"duration_seconds":5.0,"voice":"default","task_type":"Base","seed":0}' ;;
    default_voice) dataset_name=seed-tts-text; extra='{"duration_seconds":5.0,"voice":"default","task_type":"CustomVoice","seed":0}' ;;
    *) log "unknown task $task"; return 1 ;;
  esac
  fname="${task}_${workload}_c${conc}_r${rep}.json"
  if [ -s "$outdir/$fname" ]; then
    log "skip existing $fname"
    return 0
  fi
  local mode=""
  [ "$workload" = mixed ] && mode=text
  local t0 t1
  t0=$(date +%s.%N)
  VLLM_OMNI_BENCH_TTS_DURATION_MODE="$mode" PYTHONPATH="$CLIENT" "$PY" -m vllm_omni.entrypoints.cli.main bench serve --omni \
    --host "$HOST" --port "$PORT" --model "$model" \
    --backend openai-audio-speech --endpoint /v1/audio/speech \
    --dataset-name "$dataset_name" --dataset-path "$DATASET" --seed-tts-locale en --seed 0 \
    --num-prompts "$NUM_PROMPTS" --num-warmups "$NUM_WARMUPS" \
    --max-concurrency "$conc" --request-rate inf \
    --percentile-metrics ttft,e2el,audio_rtf,audio_ttfp,audio_duration --metric-percentiles 50,90,99 \
    --extra-body "$extra" --trust-remote-code \
    --save-result --result-dir "$outdir" --result-filename "$fname" \
    >"$outdir/${fname%.json}.log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  printf '%s,%s,%s,%s\n' "$fname" "$t0" "$t1" "$rc" >>"$outdir/runs.csv"
  log "bench $fname rc=$rc ($(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f", b - a}')s)"
  return $rc
}

trap 'stop_sidecar; stop_server' EXIT
# A signal must still run the EXIT trap so the server and sidecar go with us.
trap 'exit 143' TERM INT

mkdir -p "$OUT"
for ckpt in $CKPTS; do
  for arm in $ARMS; do
    outdir="$OUT/$ckpt/$arm"
    mkdir -p "$outdir"
    if [ -f "$outdir/DONE" ]; then
      log "skip $ckpt/$arm (DONE)"
      continue
    fi
    model=$(ckpt_path "$ckpt")
    start_server "$arm" "$ckpt" "$outdir/server.log" || continue
    if ! wait_health; then
      stop_server
      continue
    fi
    start_sidecar "$outdir/memory.csv"
    sleep 3
    idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
    echo "idle_after_start_mib=$idle" >"$outdir/static.txt"
    probe "$model"
    idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
    echo "idle_after_probe_mib=$idle" >>"$outdir/static.txt"
    failed=0
    for workload in $WORKLOADS; do
      for task in $TASKS; do
        for conc in $CONCS; do
          for rep in $(seq 1 "$REPEATS"); do
            run_bench "$model" "$task" "$workload" "$conc" "$rep" "$outdir" || failed=$((failed + 1))
            # A dead server fails everything after it; bail out to the next arm.
            kill -0 "$SERVER_PID" 2>/dev/null || { log "server gone"; break 4; }
          done
        done
      done
    done
    stop_sidecar
    stop_server
    [ "$failed" -eq 0 ] && touch "$outdir/DONE"
    log "finished $ckpt/$arm failed=$failed"
  done
done
