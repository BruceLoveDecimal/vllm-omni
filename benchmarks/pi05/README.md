# π0.5 Benchmarks

Latency benchmarks for π0.5 (`lerobot/pi05_base`) serving, used for the measurements in [`recipes/lerobot/Pi05.md`](../../recipes/lerobot/Pi05.md).

| Script | Measures | Needs |
| --- | --- | --- |
| `e2e_latency_pi05.py` | Client E2E round trip over the OpenPI websocket: msgpack pack, send, engine, denoise, unpack. Reports p50/p95/min/max and the device memory peak sampled from `nvidia-smi`. | A running server |
| `bench_pi05.py` | Model-level `sample_actions` latency and output shape over a batch size × camera-view sweep, with no serving stack; `--compare-sequential` also times each batch as back-to-back batch-1 calls. Writes `pi05_sweep.csv` and `pi05_sweep.md`. | A GPU and the checkpoint |
| `batch_parity_pi05.py` | Request-batch correctness: max \|batched − alone\| per dtype and batch size through `Pi05Pipeline.forward`, and whether a lone request still matches the pre-batching call. Random weights on CPU by default; `--model` for a checkpoint. | CPU (random weights), or a GPU and the checkpoint |

## Client E2E latency

Start a server with the recipe's command, then run the client per camera count.
It serves the optimized path in the deploy config's `float32`; pass `--enforce-eager` to the server for the eager baseline and `--dtype bfloat16` for bfloat16.

```bash
# optimized path (CUDA graphs + fused kernels)
vllm serve lerobot/pi05_base --omni --port 8000 --deploy-config vllm_omni/deploy/pi05.yaml --disable-log-stats
# eager baseline
vllm serve lerobot/pi05_base --omni --port 8000 --deploy-config vllm_omni/deploy/pi05.yaml --enforce-eager --disable-log-stats

python benchmarks/pi05/e2e_latency_pi05.py --host 127.0.0.1 --port 8000 --views 3 --warmup 3 --iters 30 --out e2e.json
```

## Model-level latency

The script builds the model through `Pi05Pipeline`, as the server does, from the deploy config's `model_config`.
It runs the optimized path by default; pass `--enforce-eager` for the eager baseline and `--dtype bfloat16` for bfloat16.

```bash
# optimized path (CUDA graphs + fused kernels)
python benchmarks/pi05/bench_pi05.py --bs 1 --views 3 --warmup 5 --iters 30 --out pi05_bench/
# eager baseline
python benchmarks/pi05/bench_pi05.py --enforce-eager --bs 1 --views 3 --warmup 5 --iters 30 --out pi05_bench_eager/
# batched sample_actions against the same rows run as back-to-back batch-1 calls
python benchmarks/pi05/bench_pi05.py --enforce-eager --bs 1 2 4 8 16 32 --views 3 --compare-sequential --out pi05_bench_batch/
```

## Request-batch parity

```bash
# random weights at the real widths, two layers per tower, on CPU
python benchmarks/pi05/batch_parity_pi05.py --out batch_parity.json
# the checkpoint, eager, on a GPU
python benchmarks/pi05/batch_parity_pi05.py --model lerobot/pi05_base --batch-sizes 1 2 4 8 16 32 --out batch_parity.json
```

It exits non-zero when a float32 row exceeds `--atol` (default `1e-4`); bfloat16 rows are reported only.

## Reference results

### Client E2E latency

RTX 5080 16 GB, batch size 1, 10 denoising steps, p50/p95 of 30 round trips after 3 warmup, untimed server:

| dtype | path | p50 ms, 1 / 2 / 3 cameras | p95 ms, 1 / 2 / 3 cameras | device peak (GiB) |
| --- | --- | ---: | ---: | ---: |
| float32 | eager | 263.3 / 264.2 / 264.8 | 272.5 / 272.2 / 267.3 | 14.67 |
| float32 | **CUDA graphs + fused kernels** | **240.5 / 241.9 / 242.4** | 243.1 / 245.9 / 243.8 | 14.95 |
| bfloat16 | eager | 166.4 / 166.7 / 167.7 | 167.4 / 168.2 / 190.6 | 8.71 |
| bfloat16 | **CUDA graphs + fused kernels** | **118.0 / 118.1 / 118.2** | 118.9 / 120.1 / 119.6 | 9.07 |

### Model-level latency

RTX 5080 16 GB, batch size 1, 3 cameras, 10 steps, 5 warmup:

| dtype | PR 6950 (eager) | eager path | optimized path (CUDA graph + fused Triton kernels) |
| --- | ---: | ---: | ---: |
| float32 | 264.0 ms | 260.4 ms | 237.7 ms |
| bfloat16 | 173.0 ms | 168.3 ms | 114.0 ms |
