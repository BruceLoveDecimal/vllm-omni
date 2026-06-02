# SANA-WM acceleration benchmark: cuda_graph vs torch.compile vs eager

Why SANA-WM defaults to **eager** execution, with measured data.

## TL;DR

- The bespoke `cuda_graph.py` (per-frame-bucket CUDA-graph denoiser) was **removed**:
  unverified, default-off, rigid exact-frame buckets that didn't even cover the
  production 161-frame length, and not on any production path.
- Reusing the shared diffusion `torch.compile` path (`regionally_compile(dynamic=True)`
  on the DiT blocks, like flux/wan/ltx2) was made to **engage and run**, then
  **benchmarked** — and found **net-negative for SANA-WM**:
  - **~8% slower** than eager (compute-bound; GDN/attention run eager, refiner not compiled).
  - **perturbs the output** vs eager (PSNR ≈ 24.5 dB, not bit-identical).
- ⇒ **eager is the default.** `cuda_graph` is gone; `torch.compile` is opt-in only
  (`--no-enforce-eager`) for experiments.

## Setup

| | |
|---|---|
| Hardware | 2× NVIDIA A800-SXM4-80GB (Ampere sm_80), TP=2 |
| Env | vllm 0.22.0, torch 2.11.0+cu130, diffusers 0.38.0, transformers 5.9.0 |
| Pipeline | `SanaWmTwoStagesPipeline` (native Stage-1 DiT + in-process LTX-2 refiner) |
| Config | 704×1280, **161 frames**, **60** DiT steps, **cfg=5.0**, seed=42, fps=16, refiner steps=3 |
| Attention backend | FLASH_ATTN (A800 platform default) |
| Demo | `forward_push` (action `w-160`), synthesized placeholder first frame |
| Reference | NVlabs `inference_sana_wm.py` (defaults: step=60, cfg=5.0, seed=42, flow_euler_ltx) via `tools/sana_wm_nvlabs_reference.py` |

NVlabs and our pipeline use the **same** first frame, deterministic pinhole
intrinsics, camera action, frame count, steps, cfg, and seed, so the comparison
is apples-to-apples (the one residual difference: NVlabs is the official impl;
ours is the native vLLM-Omni port).

## Results

| Path | Latency | vs eager (deviation) | vs NVlabs (frame-0 aligned) |
|------|---------|----------------------|-----------------------------|
| **eager**          | **258.2 s** | — (baseline)                                   | MAE 26.88 / PSNR 16.78 dB / SSIM-Y 0.695 |
| compiled (cold)    | 294.8 s     | —                                              | — |
| **compiled (warm)**| **278.6 s** | MAE 7.27 / PSNR 24.55 dB / max\|Δ\| 227         | MAE 26.56 / PSNR 16.87 dB / SSIM-Y 0.704 |

- *cold* = first invocation, includes one-time TorchInductor codegen.
- *warm* = second invocation reusing the inductor FX cache (steady-state).
- vs-NVlabs uses auto frame alignment, which picked `drop_pred_frame0` (our 161
  frames include the conditioning frame 0; NVlabs emits 160), then compares the
  160 common frames.

### Interpretation

1. **Compile is slower, not faster.** Warm-compiled (278.6 s) > eager (258.2 s) by
   ~8%. SANA-WM is compute-bound (1.6B DiT at 704×1280 + LTX-2 refiner). The
   GDN/UCPE self-attention runs eager (see fix #3 below) and the refiner is not
   compiled, so only MLP/norm/cross-attention are left to fuse — too little to
   offset the graph-break + guard overhead.
2. **Compile changes the output.** compiled-vs-eager PSNR ≈ 24.5 dB (not
   bit-identical), with localized max pixel diff 227/255. Inductor's fused
   reduction order differs from eager; the tiny per-step deltas are amplified
   across the 60-step denoise loop + refiner into visible regional differences.
   So torch.compile is **not numerically transparent** here.
3. **vs NVlabs** (≈ PSNR 16.8 dB, SSIM 0.70 after alignment + matched refiner
   steps) is a moderate gap — the native port is close-ish but not matched to the
   official impl. This is a separate correctness track from the compile question.

## Fixes landed along the way (all on `feat/sana_wm_integration`)

1. **Removed `cuda_graph.py`** + its pipeline wiring and the 4 denoiser unit tests.
2. **Materialize the Stage-1 transformer at load** (`pipeline_sana_wm.load_weights`,
   GPU runtime only). The DiT blocks (`SanaWmBlock`) were built lazily on first
   forward — *after* the diffusion runner applies `regionally_compile` at load —
   so regional compile silently skipped ("classes not found"). Materializing at
   load lets the shared compile path actually find and compile the blocks.
   (Also fixed `_repeated_blocks`: `["blocks"]` → `["SanaWmBlock"]`, the class name
   `regionally_compile` matches on.)
3. **`@torch.compiler.disable` on `SanaWmSelfAttention.forward`.** The GDN/UCPE
   branch dispatches into a hand-written Triton kernel whose launch grid is
   computed from runtime shapes; under torch.compile those go symbolic and the
   grid degrades to a float → `'float' object cannot be interpreted as an integer`
   at Triton launch. Running the attention eagerly (graph break) gives concrete
   int shapes; the rest of the block still compiles.
4. **Frame-0 metric alignment** in `examples/offline_inference/sana_wm/sana_wm.py`
   (`--frame-align auto`): absorbs the conditioning-frame / off-by-one convention
   (our 161 vs NVlabs 160) before computing MAE/PSNR/SSIM.

## How to reproduce

```bash
# 1) NVlabs reference (uses its own defaults: 60 steps / cfg 5 / seed 42)
python tools/sana_wm_nvlabs_reference.py --demo forward_push --num-frames 161 \
    --output nvlabs_ref.mp4

# 2) eager (default)
python examples/offline_inference/sana_wm/sana_wm.py --demo forward_push \
    --tensor-parallel-size 2 --num_frames 161 --num_inference_steps 60 \
    --guidance_scale 5.0 --seed 42 --inprocess-refiner-steps 3 \
    --reference nvlabs_ref.mp4 --frame-align auto \
    --save-frames-npy eager.npy --metrics-json eager.json --output eager.mp4

# 3) compiled (opt-in; expected slower + output deviation)
python examples/offline_inference/sana_wm/sana_wm.py --demo forward_push \
    --no-enforce-eager --tensor-parallel-size 2 --num_frames 161 \
    --num_inference_steps 60 --guidance_scale 5.0 --seed 42 --inprocess-refiner-steps 3 \
    --save-frames-npy compiled.npy --metrics-json compiled.json --output compiled.mp4

# compiled-vs-eager deviation: compare compiled.npy vs eager.npy (MAE/PSNR)
```

Env for offline / China hosts: `HF_HOME=<cache> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
and `VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER=Efficient-Large-Model/gemma-2-2b-it`
(huggingface.co and github.com are unreachable from the SeetaCloud boxes; weights
are pre-cached on the shared `/root/autodl-tmp` disk).

## Caveats

- mp4 references carry H.264 codec noise; use `--save-frames-npy` + an `.npy`
  reference for a lossless compile-vs-eager comparison (the deviation numbers above
  use the lossless npy).
- Latencies are single-run (cold/warm); ±~10 s run-to-run variance on these boxes.
- The vs-NVlabs gap is a correctness question (native port vs official) tracked
  separately from acceleration.
