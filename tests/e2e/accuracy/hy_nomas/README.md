# Native Hy-Nomas E2E measurements

Run each branch independently. On `feat/nomas_native_dit`, select `dit`; on
`feat/nomas_native_decoder`, select `decoder`. Each case launches reference and
native backends sequentially with identical weights and requests. The other
component remains on its default reference implementation.

Prerequisites: one idle CUDA GPU with enough memory, the recipe's pinned Hy-Nomas
reference source and checkpoint, matching vLLM dependencies, FlashAttention,
PyAV, torchmetrics, pynvml and pytest. The three-turn case reserves 48 latent
frames and four prompt segments; this capacity stays identical across backends.

```bash
export HY_NOMAS_MODEL=/path/to/Hy-Nomas-Diffusers
export HY_NOMAS_RUN_COMPARISON=1
export HY_NOMAS_COMPONENT=decoder # use dit on the DiT branch
export HY_NOMAS_RESULTS=/path/to/new-results-directory
python -m pytest -o addopts='' \
  tests/e2e/accuracy/hy_nomas/test_native_similarity.py \
  -m 'full_model and diffusion and cuda' --run-level full_model -v -s
```

The default workload is 704x1280, eight denoising steps, seed 42, two warmups and
five measured repetitions. The video API case generates 121 frames. Session
cases generate three turns of 49 frames, change the prompt on turn two and enable
camera movement on turn three. The final turn closes and flushes the session.
The last case repeats the session workload with component CPU offload enabled.

Small preflight (not a target-resolution performance result):

```bash
HY_NOMAS_HEIGHT=256 HY_NOMAS_WIDTH=256 HY_NOMAS_FRAMES=9 \
HY_NOMAS_STEPS=4 HY_NOMAS_WARMUPS=0 HY_NOMAS_REPEATS=1 \
python -m pytest -o addopts='' \
  tests/e2e/accuracy/hy_nomas/test_native_similarity.py \
  -k video-False --run-level full_model -v -s
```

Use `-k session-False` or `-k session-True` to select the session cases. Repeat
core cases with `HY_NOMAS_SEED=123` and `2026`, using a fresh results directory.
For source archives without `.git`, set `HY_NOMAS_SOURCE_REVISION` to the archived
commit and retain a manifest of any uncommitted test changes alongside results.

## Measurement boundary

- Video API: `/v1/videos` submission through completion polling and download.
  Poll every 100 ms to limit latency quantization (the shared helper's default
  remains two seconds for other tests).
  Both backends use the default H.264/YUV420 export. Hy-Nomas rejects codec
  overrides in model extras. Metrics include lossy encoding and color conversion;
  they are not raw-model parity. Use the session case for raw RGB fidelity.
- Session: the production `AsyncOmni` engine and AR tick consumer, with raw RGB
  frames before video encoding. This covers persistent turns and final flush;
  it does not measure WebSocket transport, live event arrival, or playback pacing.
- Timing excludes local artifact writes and metric computation. Warmup results
  are retained separately. Startup is not counted in steady-state latency.
- GPU memory is device-wide NVML usage sampled every 50 ms, not an allocator
  high-water mark. Short peaks can be missed. CPU RSS sums the runner's process
  tree and can double-count shared pages. Use the same idle machine for both.
- The first measured video is retained as `video_0.npy`. Later repeats retain
  hashes; more than one unique hash means repeatability needs investigation.
- `frames.csv` contains every frame's RGB SSIM, PSNR and exact-match flag.
  `runs.jsonl` includes per-turn timing, frame offsets and withheld-frame counts.
  Generation boundaries and stable-output emission boundaries are reported
  separately: a withheld decoder tail makes their frame indices different.
  `summary.json` reports quality and performance separately. Infinite PSNR means
  identical frames. No clipping, resizing or frame alignment is applied.

This is initially a measurement experiment. No new quality tolerance is assumed:
without explicit `HY_NOMAS_MIN_SSIM` and `HY_NOMAS_MIN_PSNR`, pytest success only
means generation, structural checks and metric collection succeeded. Both
variables enable minimum-per-frame gates after evidence-based calibration.
Existing exact-equality accuracy tests are unchanged. This opt-in measurement
suite is not automatically added to a GPU CI schedule.
