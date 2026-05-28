# Sana-WM Integration — Progress Audit

> **Audit date:** 2026-05-28 (revision 13 — camera-module parity + Stage-1 forward probe + per-token timestep root cause)
> **Branch:** `feat/sana_wm`
> **Implementation HEAD:** `c57ba6bc chore(sana-wm): localise §6.13 root cause to per-token timesteps`
> **Pushed to:** `fork/feat/sana_wm` (`BruceLoveDecimal/vllm-omni`)
> **Spec (single source of truth):**
> [`sana_wm_integration.md`](sana_wm_integration.md)
> **Tracking issue:**
> [vllm-project/vllm-omni#3656](https://github.com/vllm-project/vllm-omni/issues/3656)

---

## 0. Revision-9 Critical-Path Items — Updated Status

Revision-9 blockers are mostly closed. The active correctness blocker is now
the newly identified per-token/per-frame timestep sampling contract.

| # | Item | Status |
|---|---|---|
| 1 | GDN Triton — long-sequence + multi-card parity vs. NVlabs | ⚠️ **Open** — main-branch GDN math verified equivalent (see §6.10); divergence is upstream of recurrence |
| 2 | First-frame VAE encode for I2V conditioning | ✅ **Closed** — commit `f7e59121` A.1 |
| 3 | NVlabs flow-DPM solver | ✅ **Closed** — commit `f7e59121` A.2 (`DPMSolverMultistepScheduler`) |
| 4 | UCPE branch decomposition + numeric Plücker reference test | ✅ **Verified 2026-05-28** — UCPE math (`ucpe.py`) and native raw camera branch match NVlabs `prepare_prope_fns` + `BidirectionalGDNUCPESinglePathLiteLA._forward_cam_branch` at fp32 `~1e-7` max abs. See §6.12a. |
| 6 | vLLM parallel linear weight loading | ✅ **Fixed 2026-05-28** — `use_official_backend` gating tightened to require explicit `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`. Loaded weight norms verified on GPU. See §6.11. |
| 7 | Stage-1 latent magnitude vs LTX-2 refiner | ✅ **Fixed 2026-05-28** — cam branch rewritten as `BidirectionalGDNUCPESinglePathLiteLA` (single-path + apply_fn_o + RMS renorm). Latent in normal range now. See §6.12. |
| 8 | Per-token timestep sampling contract | ⚠️ **Breaks 1+2+3 + VAE-norm landed (2026-05-28)** — model side per-frame `(B, 1, F)` fp32 timestep + native `step_flow_euler_per_token` on the scheduler + `condition_mask` re-enabled by default + LTX-2 VAE per-channel normalisation in encode AND decode (closes §6.13e). New default MAE=47.35, PSNR=13.14, SSIM-Y=+0.025 (was 102.48 / 6.57 / -0.017 at session start). SSIM first time positive. Per-token step reveals the cam branch is load-bearing: cam-off jumps to MAE=129. See §6.13b/c/d/e. |
| 5 | TP layers → HSDP+USP → CUDA Graphs → Cache-DiT (ordered DAG) | ⚠️ **Partial** — TP + CUDA Graphs done; HSDP+USP CPU-static only; Cache-DiT not registered |

**Implication for reference alignment:** The UCPE / camera-control module is now
numerically aligned with NVlabs (§6.12a), but full Stage-1 forward parity is
blocked by the timestep contract (§6.12b): NVlabs' real path accepts a
per-frame timestep tensor, while native Stage-1 currently only runs the scalar
timestep path. The 9-frame harness therefore remains in the ~90-100 MAE regime
after the loader and cam-branch fixes.

---

## 1. TL;DR

Overall progress:

- **Single-GPU production-native inference: ~92–95%.** The four correctness blockers (§0.3 items 2–4) landed in commits `f7e59121`/`291397fc`. GDN production-shape parity (item 1) remains. CUDA Graph and `_sp_plan` are wired. Cache-DiT is the last single-GPU throughput item.
- **Highly-available multi-card (TP/SP/HSDP + quant + live server): ~80–85%.** TP parallel layers are fully migrated across all attention/FFN/projection sites. `_sp_plan` is declared. Real USP and HSDP sweeps remain GPU-gated. Cache-DiT is not yet registered. Online serving endpoint is mapped but not live-server validated.

Summary of recent commits since revision 10:

```text
c57ba6bc  chore(sana-wm): localise §6.13 root cause to per-token timesteps  ← HEAD
7aa39678  feat(sana-wm): cam branch — single-path + apply_fn_o + RMS renorm
597801b4  fix(sana-wm): tighten use_official_backend gating to require USE_OFFICIAL_CLI
bc5db3be  feat(sana-wm): add cam-branch escape hatch + document loader bug §6.11
d34133d8  feat(sana-wm): port UCPE camera branch — replace SDPA with GDN+UCPE
```

**A.1** — VAE encode first frame: `_preprocess_first_frame` + `_vae_encode_first_frame` in `pipeline_sana_wm.py`; replaces `torch.randn` placeholder.
**A.2** — Real scheduler: `SanaWmFlowMatchScheduler` wraps `DPMSolverMultistepScheduler` with `flow_shift=9.8`; the shifted-Euler `SanaWmFlowDpmScheduler` is kept for backward-compat only.
**A.3** — UCPE camera: reopened on 2026-05-27, then fixed by replacing the SDPA placeholder with the NVlabs-style GDN+UCPE camera branch. Direct GPU parity against NVlabs `prepare_prope_fns` + `BidirectionalGDNUCPESinglePathLiteLA._forward_cam_branch` is verified in §6.12a.
**B.1** — CUDA Graph: `SanaWmCudaGraphDenoiser` per-bucket replay wired into pipeline denoising loop; gated by `VLLM_OMNI_SANA_WM_CUDAGRAPH=1`.
**C.1** — TP closure: `SanaWmCameraEmbedder.raymap.proj` now uses `ColumnParallelLinear` when TP layers available. Conv3d `plucker.proj` kept as-is (no parallel equivalent).

---

## 2. Method

For each spec line item the audit:

1. Confirms whether the file exists on `feat/sana_wm`.
2. Reads the file's intent (docstring, exported surface, key class/function signatures).
3. Cross-checks against [`sana_wm_integration.md`](sana_wm_integration.md) §§ "Confirmed Release Inventory", "Concrete Architecture", "Updated Implementation Plan".
4. Cross-checks against [`.claude/skills/add-diffusion-model/references/intergration.md`](../../.claude/skills/add-diffusion-model/references/intergration.md) — practical contributor checklist (revision 11 addition).

Status icons:
- ✅ Implementation matches spec / checklist.
- ⚠️ Present but partial or blocked.
- ❌ Missing.

---

## 3. File Inventory (HEAD `94dfaa3f`)

```text
vllm_omni/diffusion/models/sana_wm/
  __init__.py                         169 LOC   ✅ public surface; re-exports all pipeline/transformer/scheduler classes
  config.py                           165 LOC   ✅ SanaWmConfig + from_yaml
  sana_wm_transformer.py             1771 LOC   ✅ Stage-1 DiT — vLLM parallel layers + UCPE + fused GDN + _sp_plan
  pipeline_sana_wm.py                 837 LOC   ✅ Stage-1: VAE encode, FlowMatch DPM, UCPE, CUDA Graph, 3-backend dispatch
  pipeline_sana_wm_two_stages.py      476 LOC   ⚠️ in-process refiner; reference alignment still open
  camera_control.py                   352 LOC   ✅ Plücker / raymap + spatial_raymap + all camera schemas
  weight_mapping.py                    49 LOC   ✅ Stage-1 prefix remap helper
  gated_deltanet_triton.py            408 LOC   ✅ fused GDN wrapper + PyTorch fallback
  fused_gdn.py                        275 LOC   ✅ vendored NVlabs fused QK/RMS + BiGDN entry point
  fused_gdn_chunkwise.py             2273 LOC   ✅ vendored NVlabs chunkwise Triton kernels
  cuda_graph.py                       261 LOC   ✅ SanaWmCudaGraphDenoiser per-bucket replay (committed)
  official_backend.py                 401 LOC   ✅ NVlabs CLI bridge w/ external refiner root + PYTHONPATH wiring
  native_backend.py                   489 LOC   ✅ direct-import NVlabs Python modules (no subprocess)
  scheduling_sana_wm.py               112 LOC   ✅ SanaWmFlowMatchScheduler (DPM-Solver++); SanaWmFlowDpmScheduler kept for compat

vllm_omni/model_executor/stage_input_processors/sana_wm.py  309 LOC   ✅ request payload schema validator
recipes/Efficient-Large-Model/SANA-WM-bidirectional.md      265 LOC   ✅ checkpoint layout + 3 backends + GPU bands
examples/offline_inference/sana_wm/sana_wm.py               144 LOC   ✅ offline example w/ camera + action + in-process refiner

tests/diffusion/models/sana_wm/  (L1 CPU-only module tests — not the required L2 expansion tests)
  conftest.py                          34 LOC
  test_sana_wm_camera_control.py      139 LOC   ✅ numerical Plücker/raymap reference tests (B.2)
  test_sana_wm_cfg_parallel_adaptation.py  28 LOC   ✅ CFG mixin + combine contract
  test_sana_wm_correctness_blockers.py  927 LOC   ✅ A.1–A.3 + CUDA graph unit tests (B.2)
  test_sana_wm_gdn_triton.py          193 LOC   ✅ multi-shape/dtype fused-vs-reference parity
  test_sana_wm_hsdp.py                 28 LOC   ✅ HSDP block-matching + _sp_plan contract
  test_sana_wm_pipeline.py            126 LOC   ✅ component declarations + scheduler + cuda-graph default
  test_sana_wm_scaffold.py           1080 LOC   ✅ ~35+ functions incl. serving payload tests (C.3)
  test_sana_wm_two_stages.py          165 LOC   ✅ refiner slots + prompt encode + in-process step
  test_sana_wm_vllm_infra.py          179 LOC   ✅ TP-layer coverage matrix + _sp_plan contract

tests/e2e/accuracy/test_sana_wm_video_e2e.py  167 LOC   ✅ reference-alignment harness (opt-in)
tests/examples/offline_inference/test_sana_wm_cfg_parallel_parity.py  25 LOC   ✅ cfg/perf contract
tests/dfx/perf/tests/test_sana_wm_vllm_omni.json  149 LOC   ✅ official/in-process/cfg2 benchmark entries

tests/e2e/offline_inference/test_sana_wm_expansion.py    ❌  MISSING — required L2 offline expansion test
tests/e2e/online_serving/test_sana_wm_expansion.py       ❌  MISSING — required L2 online expansion test
vllm_omni/deploy/sana_wm.yaml                            ❌  MISSING — deploy config
```

---

## 4. Phase-by-Phase Status

| Phase | Scope | Status | Coverage |
|---|---|---|---|
| **P0** | scaffold, registry, pipeline classes, processor stub | All entries present; compileall passes | ✅ **100%** |
| **P0.5** | official CLI backend bridge | `official_backend.py` 401 LOC; e2e test gated by `SANA_WM_E2E=1` | ✅ **100%** |
| **P1** | Stage-1 weight load + image/camera input packing | Parallel layers migrated; `_sp_plan` declared; latest GPU run `59 passed` | ✅ **~95%** |
| **P2** | Triton GDN + UCPE camera → offline native e2e | Plücker ✅; UCPE ✅ (A.3); fused-GDN e2e ✅; full 321-frame parity pending GPU | ⚠️ **~80%** |
| **P3** | LTX-2 refiner attach + dual text encoder | Gemma3 + LTX2TextConnectors + transformer load real; 2-step validated on GPU; quality threshold still open | ⚠️ **~92%** |
| **P4** | Online serving + recipes + accuracy | Recipe ✅; e2e paths ✅; serving endpoint ✅; MAE=69.64 measured; PSNR/SSIM gate not wired | ⚠️ **~75%** |
| **P5** | SP/USP/CFG-parallel/HSDP + dfx perf + Cache-DiT | TP ✅; CUDA Graph ✅; CFG/HSDP CPU-static ✅; USP/multi-GPU ❌; Cache-DiT not registered | ⚠️ **~45%** |

---

## 5. What's Strong

- **Correctness blockers A.1–A.3 closed.** First-frame VAE encode, production DPM-Solver++ scheduler, and UCPE camera branch all landed in a single well-structured commit with 27 + 5 new tests.
- **TP parallel layers complete.** Every attention/FFN/projection path in Stage-1 now uses `QKVParallelLinear` / `ColumnParallelLinear` / `RowParallelLinear` when the vLLM layer stack is available, with `nn.Linear` fallback only for CPU/unit-test environments. `attention_y_norm` uses `_maybe_make_vllm_rms_norm`. `quant_config` threaded through.
- **`_sp_plan` declared.** `SequenceParallelInput` / `SequenceParallelOutput` boundaries registered on every block and at `final_layer`.
- **CUDA Graph wired.** `SanaWmCudaGraphDenoiser` per-bucket replay is committed and gated by env/extra-arg flag. Eager fallback on capture failure.
- **Test coverage is broad for the scope.** 2,899 LOC of L1 CPU-static tests covering scheduler API, spatial_raymap, VAE encode pipeline, UCPE forward, GDN parity (multi-shape/dtype), HSDP, CFG mixin, vLLM infra contract, and serving payload building.
- **Spec discipline.** `SanaWmConfig.from_yaml` is the single config source; interface declarations (`support_image_input`, `_dit_modules`, etc.) match spec exactly.

---

## 6. What's Weak

Issues are ordered by severity. Items marked **[intergration.md]** are new findings from the practical contributor checklist.

### 6.1 Missing L2 E2E Expansion Tests [intergration.md §2] ❌ CRITICAL

`tests/e2e/offline_inference/test_sana_wm_expansion.py` and
`tests/e2e/online_serving/test_sana_wm_expansion.py` **do not exist**.

Per `docs/contributing/ci/CI_5levels.md`, every new model PR must include L2 expansion tests. All other models in the repo have them (`test_wan22_expansion.py`, `test_longcat_image_expansion.py`, etc.). The current `tests/e2e/accuracy/test_sana_wm_video_e2e.py` is an L3/accuracy harness, not the required L2 offline/online expansion test.

The PR description must also paste the actual text output of both tests in a markdown code block (not screenshots).

### 6.2 Framework Code Modified [intergration.md §3] ⚠️ NEEDS REVIEWER APPROVAL

Three framework files outside `models/sana_wm/` were modified:

| File | Change | Issue |
|---|---|---|
| `vllm_omni/diffusion/utils/hf_utils.py` | `_looks_like_sana_wm()` + `is_diffusion_model` updated | Model detection should not live in framework |
| `vllm_omni/diffusion/data.py` | SANA-WM layout detection + `model_class_name` assignment in `OmniDiffusionConfig` | Model-specific logic in core config |
| `vllm_omni/diffusion/diffusion_engine.py` | SANA-WM specific skip in `_dummy_run()` | Model-specific behavior in engine |

**Root cause:** SANA-WM uses a non-standard HF layout (`config.yaml` instead of `model_index.json`). The standard fix is to add a `model_index.json` to the HF repo (with `"_class_name": "SanaWmTwoStagesPipeline"`) and upload renamed weights. This makes the model self-describing and removes the need for all three framework modifications. Compare: BAGEL also has no `model_index.json` but follows the same `_looks_like_bagel` pattern — this is a precedent, but the reviewer will likely ask for a `model_index.json` on HF.

`model_metadata.py` and `registry.py` changes are normal model registration and do not need justification.

### 6.3 Transformer `load_weights()` > 30 Lines [intergration.md §1]

The transformer `load_weights()` is ~40 lines. The intergration.md target is ≤ 30 (ideally ~10). The extra lines are error reporting (`SanaWmStage1LoadReport`, unmapped/duplicate tracking, raise on invalid keys). This reporting is useful during development but may be over-engineered for a final PR. Consider simplifying.

### 6.4 Defensive try/except in TP Helpers [intergration.md §5]

`_get_tp_rank()`, `_get_tp_world_size()`, and `_vllm_tp_group_ready()` each wrap vLLM internals in a bare `except Exception: return default`. These catch things that shouldn't throw in normal operation and make errors silent. Consider removing the inner try/except from `_get_tp_rank`/`_get_tp_world_size` (they should only be called when vLLM is importable anyway) and narrowing `_vllm_tp_group_ready` to catch only `RuntimeError` (which is what vLLM actually raises when the process group is not initialized).

### 6.5 No Deploy YAML [intergration.md §9]

`vllm_omni/deploy/sana_wm.yaml` does not exist. Per the ming-flash pattern, every model should have a deploy config at `vllm_omni/deploy/{model_name}.yaml`. This is also where CI expansion tests point via `get_deploy_config_path()`.

### 6.6 Cache-DiT Not Registered

`CUSTOM_DIT_ENABLERS` in `vllm_omni/diffusion/cache/cache_dit_backend.py` has no entry for `SanaWmPipeline` or `SanaWmTwoStagesPipeline`. The CUDA Graph denoiser is wired (B.1), but Cache-DiT (step-level computation caching) is a separate feature and remains unregistered. Whether it makes sense alongside the GDN recurrence needs a design decision.

### 6.7 Reference-Alignment Threshold Is Still Permissive

`SANA_WM_E2E_REFERENCE_MAX_MAE=255.0` is a sanity guard. The spec's `PSNR ≥ 30 / SSIM-Y ≥ 0.93` quality gate still needs to be wired. With A.1–A.3 now closed, the next GPU run should show a meaningful MAE drop from 69.64.

### 6.8 GDN Multi-Step Parity at Full Production Shape

The fused GDN path runs in the pipeline; small/multi-shape parity is covered. Multi-step reference-alignment vs the official NVlabs path at full 704×1280 / 321 frames is still unmeasured.

### 6.9 Online Serving Is Mapped But Not Live-Server Validated

`/v1/videos/generations` and `/sync` aliases are covered by unit tests. A live `vllm serve` + `curl` smoke on GPU is still pending.

### 6.10 Cam Branch Uses SDPA Instead of GDN+UCPE; Never Called From `forward()` ✅ FIXED

**Update 2026-05-27 (later same day):** P0 implementation landed in this
branch. ucpe.py ported with 7/7 standalone unit tests passing on the
remote GPU venv (energy-preservation delta = 2.38e-07 at identity poses,
4.77e-07 with RoPE); `SanaWmSelfAttention` rewritten to do main +
cam → shared output_gate + proj; `camera_conditions` routed through
the transformer; tests/diffusion/models/sana_wm/test_sana_wm_ucpe.py
added.

**Update 2026-05-28 (overnight GPU run):** E2E reference-alignment harness
re-run on remote GPU produced MAE=98.31 / PSNR=6.86 / SSIM=0.011 — almost
identical to the pre-UCPE result (MAE=95.82). Trace instrumentation
inside `_forward_gdn_raw` and `_forward_cam_branch` revealed two
pre-existing bugs that the UCPE port cannot mask:

1. **`QKVParallelLinear.weight` is all zeros for every GDN block.**
   Trace at the start of `_forward_gdn_raw` showed
   `qkv_w_norm=0.0` and `qkv_proj_norm=0.0` while `hidden_norm`
   was at the bf16 fp32-fp16-mixed overflow boundary (norm = inf).
   With weight=0 and only bias contributing, all 15 GDN blocks return
   `bias` for QKV → bounded constant per channel through the recurrence
   → bounded constant output. The 5 softmax blocks plus the residual
   stream are doing all the "denoising work", explaining why the
   pre-UCPE MAE was already in the 90s — the model was operating
   purely on bias paths even before UCPE was reopened.

2. **`out_proj_cam.weight` is also zero** (consistent with the
   `init_cam_from_base=True` warm-start that NVlabs does at load time
   but our weight loader does not). With weight=0, `out_proj_cam(cam_raw)`
   returns just the bias regardless of how meaningful `cam_raw` is.
   The UCPE cam branch IS being invoked correctly (cam_raw_norm shows
   non-zero values pre-projection) but its contribution is masked.

The UCPE port itself is correct — `ucpe.py` unit tests pass with
1e-7 energy-preservation precision, `_forward_cam_branch` executes,
and the dual-branch fusion ordering matches NVlabs. The reference
alignment cannot improve until the QKV / out_proj_cam weight-loading
bugs are fixed. That is now §6.11 / §7 item 11 below.

Verified by deep-dive vs NVlabs `sana_gdn_camctrl_blocks.py` on 2026-05-27.

### 6.11 vLLM Parallel Linear Weight Loading Broken — `QKVParallelLinear.weight=0` ✅ FIXED 2026-05-28

**Root cause located.** [`SanaWmPipeline.__init__`](../../../vllm_omni/diffusion/models/sana_wm/pipeline_sana_wm.py#L287)
set `self.use_official_backend = is_sana_wm_official_backend_requested(od_config)`,
and that helper returned `True` whenever the `VLLM_OMNI_SANA_WM_OFFICIAL_REPO`
env var was set — regardless of whether the user actually wanted the
CLI bridge or merely wanted the reference-alignment harness to be
able to invoke the CLI bridge for the *reference* run.

When `use_official_backend=True`, `SanaWmPipeline.load_weights`
short-circuited to `return set()` without ever loading any
checkpoint tensor onto the native Stage-1 model. The native path
then ran with zero-initialised `QKVParallelLinear.weight` (and
zero-initialised `out_proj_cam.weight`), which is why the
GPU-run trace showed `qkv_w_norm=0.0` for every GDN block.

**Fix.** Tighten the gating so the CLI bridge is enabled only when
the user *explicitly* opts in via `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`
*and* the repo path is provided:

```python
self.use_official_backend = (
    is_sana_wm_official_backend_requested(od_config)
    and should_force_sana_wm_cli_backend()
)
```

This matches the long-standing recipe convention (both env vars
required to activate the CLI bridge) and lets `OFFICIAL_REPO` be
set for the reference run without disabling native weight loading
on the prediction-path pipeline instance.

**Verification.** GPU trace 2026-05-28 (post-fix, same harness):

```
[WEIGHT CHECK] block0 qkv.weight norm=7.2072e+02   # source ckpt norm: 7.2004e+02 ✓
[WEIGHT CHECK] block0 out_proj_cam.weight norm=1.0782e+01   # was 0.0 before fix ✓
```

The native Stage-1 model is now exercising the loaded weights.
The 9-frame e2e MAE moved from `98.31` (zero-weight era) to `95.25`
(loaded-weight era) — small drop because a *new* downstream
blocker has been exposed (see §6.12).

### 6.12 Stage-1 Latent Magnitude 10× Too Large for LTX-2 Refiner ✅ FIXED 2026-05-28

**Resolution.** The 10× magnitude inflation was localised to the cam
branch via a `VLLM_OMNI_SANA_WM_DISABLE_CAM_BRANCH=1` isolation run
(`[-10.5, 9.75]` main-only vs `[-59, 61]` with cam). The cam branch
was missing **three** pieces required by the SANA-WM 1600M release
config `camctrl_type: BidirectionalGDNUCPESinglePathLiteLABothTriton`:

1. **Single-path delta-rule recurrence** (numerator only, no Z
   denominator, no `num/den` divide). NVlabs
   `torch_recurrent_cam_single_path_delta_rule`. We were running the
   full bidirectional GDN with denominator divide.
2. **`apply_fn_o` inverse output transform** — applied after the
   recurrence to bring the output from per-ray frame back to world
   frame.
3. **`_downscale_to_reference_rms` (PostUCPERenorm)** — clips the
   UCPE-transformed Q/K/V back to their pre-UCPE per-token RMS
   envelope. Without this, the per-ray 4×4 projection inflates
   magnitudes by ~6-10×.

After porting all three plus the existing β-discount logic,
`STAGE1_STEPS=1` latent dropped to `[-12.3, 11.8]` — within the
LTX-2 refiner input distribution. fp32 attention discipline added
on the recurrence accumulator to match NVlabs `fp32_attention=True`.

### 6.12a NVlabs Camera-Control Module Parity ✅ VERIFIED 2026-05-28

**Scope.** Ran a direct GPU module-alignment harness on
`sana-wm-seeta` using:

- native vLLM-Omni checkout synced to
  `/root/autodl-tmp/vllm-omni-align-c57ba6bc-20260528`
- NVlabs checkout at `/root/autodl-tmp/NVlabs-Sana`
- clean venv `/root/autodl-tmp/venvs/vllm-omni-clean-pytest/bin/python3`
- `TORCHDYNAMO_DISABLE=1 GDN_DISABLE_COMPILE=1`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- torch: `2.11.0+cu130`

The harness compared:

1. NVlabs `sana_camctrl_blocks.prepare_prope_fns("UCPE", ...)`
   against native `vllm_omni.diffusion.models.sana_wm.ucpe.prepare_prope_fns`
   for `apply_q`, `apply_kv`, and `apply_o`.
2. NVlabs
   `BidirectionalGDNUCPESinglePathLiteLA._forward_cam_branch`
   against native `SanaWmSelfAttention._forward_cam_branch`.

Both modules used identical random camera-branch weights, identical
inputs, identical `(beta, decay)` gates, identical camera conditions,
and identical rotary embeddings when enabled. Shape:
`B=2, T=3, H=2, W=3, N=18, C=64, heads=4, head_dim=16`.
`patch_size=(1, 1, 1)` was used so the NVlabs intrinsics contract
matches the native latent-pixel camera-condition contract.

| dtype | RoPE | conv kernel | UCPE apply-fn max abs | raw cam-branch max abs | raw cam-branch rel mean abs |
|---|---:|---:|---:|---:|---:|
| fp32 | off | 0 | `2.38e-7` | `1.79e-7` | `8.01e-8` |
| fp32 | off | 4 | `2.38e-7` | `1.19e-7` | `8.91e-8` |
| fp32 | on | 0 | `2.38e-7` | `1.49e-7` | `1.17e-7` |
| fp32 | on | 4 | `2.38e-7` | `1.79e-7` | `1.01e-7` |
| bf16 | off | 0 | `1.56e-2` | `1.17e-2` | `3.83e-3` |
| bf16 | off | 4 | `1.56e-2` | `3.91e-3` | `2.85e-3` |
| bf16 | on | 0 | `1.56e-2` | `3.91e-3` | `3.44e-3` |
| bf16 | on | 4 | `1.56e-2` | `4.88e-3` | `3.97e-3` |

**Conclusion.** The native camera-control module is numerically aligned
with the NVlabs implementation at the UCPE-closure and raw camera-branch
levels. fp32 differences are only roundoff (`~1e-7` max abs); bf16
differences are within expected quantisation error. This closes the
question of whether the remaining §6.13 MAE gap is caused by the
camera branch's UCPE/QKV/O transform, single-path recurrence,
PostUCPERenorm, or identity short-conv behavior. It is not.

**Caveat.** This is module-level parity, not a full denoising parity
test. It intentionally bypasses checkpoint loading, `out_proj_cam`,
the shared `output_gate + proj`, text conditioning, scheduler stepping,
and LTX-2 refinement. Those surfaces remain covered by the 9-frame
reference-alignment harness and the §6.13 sampling-contract work.

### 6.12b NVlabs Stage-1 Full Forward Probe ⚠️ BLOCKED 2026-05-28

**Scope.** Ran a direct Stage-1 forward probe on `sana-wm-seeta` using
the public `Efficient-Large-Model/SANA-WM_bidirectional` snapshot:

- NVlabs checkout: `/root/autodl-tmp/NVlabs-Sana`
- native checkout: `/root/autodl-tmp/vllm-omni-align-c57ba6bc-20260528`
- clean venv: `/root/autodl-tmp/venvs/vllm-omni-clean-pytest/bin/python3`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- torch: `2.11.0+cu130`
- model config: `config.yaml`
- Stage-1 weights: `dit/sana_wm_1600m_720p.safetensors`

Synthetic but production-shaped miniature input:
`latents=(1,128,2,2,3)`, `prompt=(1,300,2304)`,
`raymap=(1,2,20)`, `chunk_plucker=(1,48,2,2,3)`,
`spatial_raymap=(1,3,2,2,3)`.

| Path | Timestep input | Result |
|---|---|---|
| NVlabs Stage-1 | `(B,1,F) = (1,1,2)` with `[0, 999]` | ✅ forward succeeds; output `(1,128,2,2,3)`, bf16, `min=-3.5`, `max=5.56`, `std=1.54` |
| NVlabs Stage-1 | scalar `(1,)` | ❌ fails in the unexercised scalar branch: `UnboundLocalError: x_sa` |
| native Stage-1 | `(B,1,F) = (1,1,2)` | ❌ fails at block modulation: `2240` vs `4480`, because timestep embedding is flattened as batch scalar embeddings |
| native Stage-1 | scalar `(1,)` | ✅ forward succeeds; output `(1,128,2,2,3)`, bf16, `min=-1.91`, `max=2.42`, `std=0.64` |

Weight loading is not the blocker: native consumed all Stage-1 weights
(`872/872` loaded, `872/872` materialized, `0` unapplied/unmapped), and
NVlabs loaded with only `pos_embed` missing as expected.

For visibility only, comparing native scalar output against NVlabs
per-frame output on the same tensors gives:

| Metric | Value |
|---|---:|
| max abs | `5.0934` |
| mean abs | `1.1852` |
| RMSE | `1.4696` |
| relative mean abs | `0.9719` |
| cosine | `0.3163` |

**Conclusion.** This is not yet a meaningful same-contract numeric
parity failure. It confirms that NVlabs' exercised Stage-1 forward path
is per-frame/per-token timestep aware, while the native Stage-1 forward
can only execute a scalar-timestep contract. The next alignment step is
therefore the §6.13 refactor, not more camera/GDN tuning.

### 6.13 Persistent ~90 MAE Baseline — Cam Hurts Slightly ❌ NEW

**Symptom (2026-05-28, post §6.10/§6.11/§6.12 all closed).**

| Configuration | MAE | PSNR | SSIM-Y |
|---|---|---|---|
| Loader fix + main-only (`DISABLE_CAM_BRANCH=1`) | 91.76 | 7.25 | 0.076 |
| Loader fix + new cam (single-path + apply_fn_o + RMS) | 102.48 | 6.57 | -0.017 |

Main-branch only at MAE=91.76 is the best result so far — and adding
the algorithmically-correct cam branch makes it ~10 MAE worse, even
though the latent magnitude is now in the normal range. The cam
branch is structurally matched to NVlabs but is not yet improving
output quality.

Both numbers are still in the "uncorrelated to reference" regime
(MAE > 80 means the decoded video has essentially no overlap with
the official Stage-1 → refiner output). This is a sign that
multiple non-cam-branch alignment items still dominate the
remaining error budget.

**Localisation experiments (2026-05-28 evening, all on remote GPU):**

| Configuration | MAE | Notes |
|---|---|---|
| Loader fix + main-only | 91.76 | baseline |
| Loader fix + new cam (single-path+RMS+apply_fn_o) | 102.48 | cam ~+10 MAE |
| Above + disable `plucker_proj` external residual | 103.59 | slightly worse — embedder not the cause |
| Clean first_latent (no add_noise) + hard-restore frame 0 | 134.31 | much worse — model expects noised conditioning |

**Root cause of the persistent gap (identified 2026-05-28):**
NVlabs' `LTXFlowEuler.sample` (see `diffusion/scheduler/flow_euler_sampler.py`)
uses a fundamentally different sampling contract than what our
pipeline implements:

1. **Per-token timesteps.** NVlabs constructs a
   ``(B, 1, F)`` timestep tensor where the conditioning frame's
   timestep is forced to 0 and other frames carry the current
   sampling sigma. This is passed as the ``timestep`` argument to
   the model so the per-frame timestep embedding modulates each
   frame differently.

   ```python
   condition_mask = torch.zeros_like(latents)  # 1,C,F,H,W
   for frame_idx in condition_frame_info: condition_mask[:, :, frame_idx] = 1
   timestep = t.expand(condition_mask_input.shape).float()
   timestep = torch.min(timestep, (1 - condition_mask_input) * 1000.0)
   noise_pred = self.model(latent_model_input, timestep[:, :1, :, 0, 0], ...)
   ```

2. **Per-token scheduler.step**, with ``per_token_timesteps``
   forwarded to a flow-matching scheduler that supports per-token
   step sizes.

3. **Hard masking after step.** Only update non-conditioning tokens:
   ```python
   tokens_to_denoise_mask = t / 1000 - 1e-6 < (1.0 - condition_mask)
   latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
   ```

4. **Motion-continuity noise** (`add_noise_to_image_conditioning_latents`)
   injected to the conditioning frame before each step.

Our pipeline does a simple "noise the first frame to t=t0, run model
with scalar timestep, scheduler.step on the full latent" loop. The
hard-conditioning experiment (item 4 in the table above) confirmed
the model REQUIRES per-frame sigma signalling — when we put a clean
first frame without the per-token timestep signal, the model
mis-interprets it and corrupts ALL frames via attention.

**Acceptance criterion:**

```text
[ ] Refactor `SanaWmTimestepEmbedder` + `t_block` to accept and
    output per-frame timesteps. Shape: timestep (B, F) →
    timestep_modulation (B, F, 6*hidden_size).
[ ] Block forward unpacks per-frame modulation and applies
    shift/scale per spatial-token group.
[ ] Pipeline builds the (B, F) timestep tensor with frame 0 = 0
    (or the configured `condition_frame_info` value) and other
    frames = current sigma.
[ ] Replace the wrapped `DPMSolverMultistepScheduler` step with a
    per-token-timestep aware step (LTX flow-matching style).
[ ] Add `tokens_to_denoise_mask` to preserve conditioning frames.
[ ] e2e MAE drops below 30 on the 9-frame harness.
```

Cross-ref: integration.md §3 (scheduler), §4a (embedder), §4b (UCPE branch).

### 6.13b Per-Frame Timestep Contract — Partial Fix Landed 2026-05-28 ⚠️ IN PROGRESS

The three breaks from §6.13a were partially closed in commit
`<pending>`:

* **Break 1 (model timestep rank)** ✅ — `SanaWmTimestepEmbedder` +
  `t_block` now flatten/unflatten the timestep axes so both scalar
  `(B,)` and per-frame `(B, 1, F)` contracts produce the right
  `time_embed` / `timestep_modulation` shapes.
* **Break 2 (block + final-layer modulation must be per frame)** ✅
  — `SanaWmBlock.forward` and `SanaWmFinalLayer.forward` now
  dispatch to `_forward_frame_aware` when `t.ndim > 2`, mirroring
  NVlabs' `forward_frame_aware` modulation shapes
  `(B, F, 6, D)` for blocks and `(B, F, 2, D)` for the final layer.
* **Break 3 (per-token scheduler step + mask preservation)** ⚠️
  Partial. The pipeline now builds a `(B, 1, F)` timestep, forces
  frame 0 to 0, and places the CLEAN VAE-encoded first latent at
  frame 0 (no `add_noise`). A `condition_mask` `torch.where` post-
  step restore is implemented, but is OFF by default — see
  empirical finding below.

**Empirical 9-frame reference-alignment harness** (`STAGE1_STEPS=3`,
`REFINER_STEPS=3`, real DPMSolver wrapper, no scheduler swap yet):

| Configuration | MAE | PSNR | SSIM-Y |
|---|---|---|---|
| Scalar t + main-only (loader fix only) | 91.76 | 7.25 | 0.076 |
| Scalar t + main + cam (post §6.12) | 102.48 | 6.57 | -0.017 |
| Per-frame t + main + cam + mask ON | 110.76 | 6.10 | 0.049 |
| Per-frame t + main-only + mask ON | 101.18 | 6.84 | 0.083 |
| Per-frame bf16 t + main + cam + mask OFF | 82.32 | 8.20 | 0.016 |
| Per-frame bf16 t + main-only + mask OFF | 80.10 | 8.55 | 0.011 |
| **Per-frame fp32 t + main + cam + mask OFF (new default)** | **80.74** | **8.34** | **0.020** |
| Per-frame fp32 t + main-only + mask OFF | 79.99 | 8.56 | 0.011 |

The fp32-mask-OFF entry is the current default. Headline: the
per-frame timestep contract drops MAE from `102.48` to `80.74`
(−21.74) and PSNR rises from `6.57` → `8.34` dB. The fp32 upgrade
alone is worth `−1.58` MAE (commit `39168e08`): the previous
default silently cast `timestep.to(latents.dtype)` to bf16 before
the sinusoidal embedder, while NVlabs `LTXFlowEuler.sample` keeps
the timestep in fp32 throughout.

Removing `condition_mask` post-step restore drops a further `~28`
MAE in our wrapped-DPMSolver setup because the mask creates a hard
discontinuity that contaminates the multistep solver's stored
model-output history. In NVlabs' LTXFlowEuler the mask is a no-op
safety belt (the per-token sigma for the conditioning token is
already 0 so the step doesn't change it); ours is not yet a true
per-token scheduler.

**Cam branch.** With per-frame fp32 t and no mask, cam-on (`80.74`)
and cam-off (`79.99`) are within `~0.75` MAE. Cam is no longer
dominant — the §6.12 structural correctness holds but UCPE
attention is marginally not pulling its weight in this regime.
Diagnosable separately after the scheduler step lands.

> **Important caveat.** MAE ~80 / PSNR ~8 dB / SSIM-Y ~0.02 is
> still firmly in the "weakly correlated" regime. The model side
> contract change is structurally right (PSNR direction, multiple
> independent ablations consistent), but should not be read as
> "near aligned". Break 3 (per-token flow-matching Euler step + an
> active `tokens_to_denoise_mask`) is the main lever for the next
> chunk of headroom — see §6.13c.

**Remaining work (still §6.13a "Break 3"):**
- Port LTX flow-matching Euler step with `per_token_timesteps`
  argument support, applied per-token over the flattened
  `(B, FHW, C)` token axis. NVlabs reference:
  `flow_euler_sampler.py:178-188`.
- Add `tokens_to_denoise_mask = t/1000 - 1e-6 < (1 - condition_mask)`
  post-step. With a true per-token scheduler this becomes the
  no-op safety belt; with the current wrapper it would still hurt.
- Optional motion-continuity term
  `add_noise_to_image_conditioning_latents` — gated by
  `image_cond_noise_scale > 0`; SANA-WM public config keeps it at
  `0` so it can ship later.

### 6.13e LTX-2 VAE Per-Channel Normalisation ✅ FIXED 2026-05-28 late

**Symptom** (post-§6.13d): MAE/PSNR improved (80.74→69.06 MAE,
8.34→9.68 PSNR) under the per-token Euler step, but SSIM-Y
*flipped negative* (0.020 → -0.059). Per-frame and per-seed
diagnostic ruled out alignment / noise: the seed had zero effect
on output, and all 8 frames showed the same -0.05 to -0.10
SSIM-Y band.

**Root cause** localised by visual side-by-side dump of pred[0]
vs ref[0]:

```
pred[0] RGB mean:  R=110.6  G=219.8  B=208.4   (cyan/green-heavy)
ref[0]  RGB mean:  R=94.1   G=114.3  B=138.2   (neutral)
```

A systematic per-channel bias of +105 on G and +70 on B,
consistent across all frames — clearly not noise, clearly a
decoder-side bias.

Reading NVlabs `diffusion/model/builder.py:vae_encode/vae_decode`
revealed the LTX-2 diffusers VAE (`AutoencoderKLLTX2Video`,
identified by `"LTX2VAE_diffusers" in name`) ships with **per-
channel `latents_mean` and `latents_std` tensors** (128 channels)
that NVlabs applies in BOTH encode and decode:

```python
# encode
z = (z - latents_mean) * scaling_factor / latents_std

# decode
latent = latent * latents_std / scaling_factor + latents_mean
samples = vae.decode(latent, temb=None, return_dict=False)[0]
```

Our pipeline was using the simpler diffusers convention with
`* scaling_factor` on encode and *no* corresponding scaling on
decode. That broke the round-trip identity and produced a
systematic colour shift via the per-channel `latents_mean` drift.

**Fix** (commit `<pending>`): new helpers
`_vae_normalize_latent` / `_vae_denormalize_latent` on
`SanaWmPipeline` matching the NVlabs formulae exactly. Encode
path replaces the old `* scaling_factor` with full per-channel
normalisation. Decode path drops the conditional zero-timestep
arg in favour of NVlabs' unconditional `temb=None` and applies
denormalisation before `vae.decode`.

**GPU verification on the 9-frame harness** (`STAGE1_STEPS=3`,
`REFINER_STEPS=3`):

| Configuration | MAE | PSNR | SSIM-Y |
|---|---|---|---|
| §6.13d default (per-token step, missing VAE norm) | 69.06 | 9.68 | -0.059 |
| **§6.13e new default (per-token step + VAE per-channel norm)** | **47.35** | **13.14** | **+0.025** |

* MAE -21.71 (down 31%)
* PSNR +3.46 dB (up 36%)
* SSIM-Y crosses zero into positive territory — the structure is
  no longer anti-correlated.

Pred RGB after fix: `R=84.0 G=142.9 B=191.4` vs ref
`R=94.1 G=114.3 B=138.2`. The +105 G bias is gone (now +29);
B bias is reduced (was +70, now +53). A subtle blue cast
remains, attributable to remaining content drift rather than
decoder bias.

Cumulative session progress (2026-05-28):

```
102.48 → 95.82 → 95.25 → 91.76 → 82.32 → 80.74 → 69.06 → 47.35  (MAE)
  6.57 →  7.37 →  7.04 →  7.25 →  8.20 →  8.34 →  9.68 → 13.14  (PSNR)
-0.017 → 0.005 → -.003 →  .076 →  .016 →  .020 → -.059 → +.025  (SSIM-Y)
```

Still firmly in "weakly correlated" regime (PSNR ~13 dB is far
from a typical 25+ dB target), but the trajectory is now
consistently positive across all three metrics for the first
time in the session.

### 6.13d Break 3 Landed — Per-Token Flow-Matching Euler Step ✅ 2026-05-28

Break 3 implemented per the §6.13c design. New method
`step_flow_euler_per_token` on `SanaWmFlowMatchScheduler` reuses the
existing DPMSolver `use_flow_sigmas=True` schedule to look up
`sigma_cur`/`sigma_next`, builds a per-token `dt = sigma_next -
sigma_cur` (zero for conditioning tokens), and applies
`prev = sample + dt * (sign * noise_pred)`. The pipeline switches
to this step under the per-frame contract by default, and
re-enables `condition_mask` preservation now that it is a genuine
no-op safety belt rather than a discontinuity.

**Sign convention verified on GPU.** Both `+noise_pred` and
`-noise_pred` were measured. Default is `-noise_pred` (matching
NVlabs `LTXFlowEuler.sample`); `VLLM_OMNI_SANA_WM_FLIP_FLOW_SIGN=1`
swaps to the alternative.

| Sign | MAE | PSNR | SSIM-Y |
|---|---|---|---|
| `+noise_pred` (DPM v_pred convention) | 112.49 | 6.01 | 0.047 |
| **`-noise_pred` (NVlabs convention, default)** | **69.06** | **9.68** | -0.059 |

**Sigma lookup correctness.** First implementation hand-computed
`sigma_cur_token = per_token_t / num_train_timesteps` which is the
unshifted t/1000 mapping. Under `use_flow_sigmas=True` with
`flow_shift=9.8` the scheduler's actual sigmas are heavily shifted
(`shift * t / (1 + (shift-1) * t)`), so the hand-computed sigma
disagreed with the scheduler's `sigma_next` lookup. Fix: take both
sigmas from `self._sched.sigmas[cur_idx]` and
`self._sched.sigmas[cur_idx + 1]`; conditioning tokens are
explicitly clamped to sigma=0. GPU debug:

```
[STEP DEBUG] idx=0 sigma_cur=0.9999 sigma_next=0.9513 dt_mean=-2.43e-02
[STEP DEBUG] idx=1 sigma_cur=0.9513 sigma_next=0.8303 dt_mean=-6.05e-02
[STEP DEBUG] idx=2 sigma_cur=0.8303 sigma_next=0.0000 dt_mean=-4.15e-01
```

(`dt_mean` is over all tokens including the 50% conditioning
tokens at frame 0 of a 2-latent-frame layout.)

**Headline 9-frame harness numbers (`STAGE1_STEPS=3`,
`REFINER_STEPS=3`):**

| Configuration | MAE | PSNR | SSIM-Y |
|---|---|---|---|
| Per-frame fp32 + DPM step + mask off + cam on (§6.13b default) | 80.74 | 8.34 | 0.020 |
| **Per-token step + mask on + cam on (§6.13d NEW DEFAULT)** | **69.06** | **9.68** | -0.059 |
| Per-token step + mask off + cam on | 69.65 | 9.59 | -0.069 |
| Per-token step + mask on + cam off | 129.05 | 4.85 | 0.028 |

**Headline observations:**

1. Per-token step alone (cam-on path) drops MAE `80.74 → 69.06`
   (−11.7) and lifts PSNR `8.34 → 9.68` dB (+1.34) vs the §6.13b
   default. That is a real, beyond-noise improvement.
2. Mask ON vs Mask OFF differ by only `~0.6` MAE under the
   per-token step — the safety belt is a genuine no-op now, as
   predicted in §6.13c.
3. **The cam branch is no longer near-neutral.** Under per-token
   step + mask, disabling cam balloons MAE from `69.06 → 129.05`
   (+60). Under the prior DPM step regime cam-on and cam-off were
   within `~0.75` MAE. The DPM multistep extrapolation was
   smoothing over the cam contribution; per-token Euler exposes
   that contribution as load-bearing. This validates §6.10/§6.12
   structural work retroactively.

**Caveats** (do not over-read into the headline number):

* SSIM-Y is now **negative** (`-0.059` to `-0.069`). MAE went
  down and PSNR went up while spatial structure correlation went
  the other way. Hypotheses to investigate before claiming
  alignment progress:
  - the per-token Euler is integrating an off-by-something
    velocity that brings the *average* pixel intensity closer to
    reference while corrupting the spatial layout;
  - the cam branch is now over-contributing because §6.12
    `_downscale_to_reference_rms` was calibrated against the
    DPM-step regime;
  - the refiner stage was tuned against an upstream Stage-1
    latent distribution that we still don't quite produce.
* MAE 69 / PSNR 9.68 / SSIM −0.06 is still firmly in the
  "weakly correlated" regime. We are not "approaching alignment".

**Remaining work after §6.13d:**

* Diagnose negative SSIM (single-block parity test against NVlabs
  forward path, or visual side-by-side of decoded frames).
* Port softmax-UCPE for the every-4th softmax block (integration.md
  §4a + §4b).
* Camera embedder structural alignment (integration.md §4a).
* Once SSIM turns positive, re-tighten
  `SANA_WM_E2E_REFERENCE_MAX_MAE` and `MIN_PSNR`/`MIN_SSIM_Y`
  gates in `test_sana_wm_video_e2e.py`.

### 6.13c Break 3 Design — Native Per-Token Flow-Matching Euler Step

Break 3 from §6.13a (per-token scheduler step + active
`tokens_to_denoise_mask`) is the next implementation chunk. The
plan, before opening the editor, is captured here so reviewers can
push back on the design before implementation.

**Choice: native step, not full scheduler swap.** Three options
were considered:

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| A — Add `step_flow_euler_per_token(...)` method on the existing `SanaWmFlowMatchScheduler` and reuse DPMSolver's sigma schedule | Full control, ~50–80 LOC, no diffusers version dependency, keeps the wrapper class as the single source of truth | Have to verify that DPMSolver's `use_flow_sigmas=True` sigma table matches NVlabs' FlowMatchEulerDiscrete sigma table | **Pick** |
| B — Drop the wrapper and use diffusers `FlowMatchEulerDiscreteScheduler` directly | Less new code | `per_token_timesteps` is a newer-version API; we lock our diffusers floor higher | reject |
| C — Vendor NVlabs' full sampler module | Exact parity | Too much surface area; pulls in their flow-matching infra | reject |

**API.**

```python
# scheduling_sana_wm.py
def step_flow_euler_per_token(
    self,
    noise_pred: torch.Tensor,         # (B, C, F, H, W) — model output
    timestep: torch.Tensor,           # scalar current step
    latents: torch.Tensor,            # (B, C, F, H, W)
    per_token_timesteps: torch.Tensor,# (B, FHW) per-token t, 0 for conditioning
) -> torch.Tensor:                    # (B, C, F, H, W) prev_sample
```

**Math.** NVlabs' `LTXFlowEuler.sample` passes `-noise_pred` to
`FlowMatchEulerDiscreteScheduler.step`, whose update rule is
`prev = sample - (sigma_next - sigma) * model_output`. Substituted:
`prev = sample + (sigma_next - sigma) * noise_pred`. Per-token:

```text
sigma_token_t      = per_token_timesteps / 1000          # (B, FHW)
sigma_token_next   = where(is_conditioning, sigma_token_t,
                           sigma_next_scalar)             # (B, FHW)
dt                 = sigma_token_next - sigma_token_t     # (B, FHW), negative
prev_flat          = latents_flat + dt * (-noise_pred_flat)
```

Where `is_conditioning = per_token_timesteps < 1e-6`.
Conditioning tokens get `dt = 0` and the step is a no-op for them
by construction, so the `torch.where(condition_mask > 0.5, latents,
stepped)` safety belt becomes the no-op it is in NVlabs.

**Sigma source.** The existing `_sched = DPMSolverMultistepScheduler(use_flow_sigmas=True, flow_shift=...)` already
produces `sigmas` of shape `(num_steps + 1,)` (terminal sigma is 0)
matching the FlowMatchEuler schedule under flow-shift. We index
into `_sched.sigmas` by finding the current step in
`_sched.timesteps`. No need to instantiate a second scheduler.

**Pipeline wiring** (in `_run_native_smoke_backend`):

1. Build `per_token_timesteps` from the existing `(B, 1, F)`
   model timestep by broadcasting over `H*W` for each frame and
   flattening: `per_token = model_timestep.expand(B, 1, F).broadcast(...).reshape(B, F*H*W)`.
2. Replace `scheduler.step(noise_pred, timestep, latents)` with
   `scheduler.step_flow_euler_per_token(noise_pred, timestep,
   latents, per_token_timesteps)` — only under the
   `use_per_frame_timestep` path.
3. Re-enable `condition_mask` `torch.where` post-step. With the
   per-token step this is genuinely a no-op safety belt; switch
   the default for `VLLM_OMNI_SANA_WM_ENABLE_COND_MASK` to ON
   (and drop the env var or invert it to disable for ablation).

**Risks and verification plan.**

| Risk | A/B test | Expected signal |
|---|---|---|
| Sign convention wrong (`-noise_pred` vs `+noise_pred`) | one-line flip, re-run 9-frame harness | MAE either drops substantially or balloons; easy to tell |
| `use_flow_sigmas` table not equal to NVlabs FlowMatchEuler | print first 4 sigmas vs NVlabs reference run | mismatched ≥ 2nd decimal would explain residual gap |
| Removing DPMSolver `solver_order=2` extrapolation costs accuracy on the few real-denoise frames | re-run with `solver_order=1` (Euler) wrapper for comparison | should be small (~1 MAE) |

**Estimate.** 0.5–1 person day: ~60 LOC scheduler method, ~10 LOC
pipeline change, ~3 GPU runs to verify sign + sigma + final MAE.

**Acceptance.** Same as the §6.13a "Fix implication" bullets,
plus: 9-frame harness MAE drops below `60` (intermediate gate;
the eventual `≤ 30` target also needs softmax-UCPE and the
embedder work that are *not* on this commit).

---

Env-var escape hatches kept for debug:

```
VLLM_OMNI_SANA_WM_DISABLE_PER_FRAME_TIMESTEP=1  # back to scalar t
VLLM_OMNI_SANA_WM_ENABLE_COND_MASK=1            # opt-in mask restore
VLLM_OMNI_SANA_WM_DISABLE_CAM_BRANCH=1          # ablation
VLLM_OMNI_SANA_WM_DISABLE_PLUCKER_PROJ=1        # ablation
```

### 6.13a Why Stage-1 Still Misaligns: Three Contract Breaks

The remaining Stage-1 misalignment is not a generic "GDN precision"
issue. It is a denoising-contract mismatch across three layers of the
stack.

#### Break 1 — model input timestep rank

NVlabs `LTXFlowEuler.sample` builds a full latent-shaped timestep
field, then passes the model a frame-indexed tensor:

```text
timestep: (B, C, F, H, W)
model timestep argument: timestep[:, :1, :, 0, 0] -> (B, 1, F)
```

For conditioning frame 0, that tensor is forced to `0`; for generated
frames it carries the current sampling timestep. This means one forward
call contains both "clean/conditioned" and "currently denoising" frames.

Native `_run_native_smoke_backend` always calls the transformer with a
batch-scalar timestep:

```text
timestep.expand(1) -> (B,)
```

Code references:
- native scalar call: `pipeline_sana_wm.py:710-731`
- native timestep embedding: `sana_wm_transformer.py:2005-2016`
- NVlabs frame-aware model call: `flow_euler_sampler.py:153-160`

The 2026-05-28 direct Stage-1 probe confirms the consequence:
NVlabs per-frame `(B,1,F)` forward succeeds, while native per-frame
forward fails at block modulation (`2240` vs `4480`). Native scalar
forward succeeds only because it is running a different contract.

#### Break 2 — block/final modulation must be per frame

NVlabs blocks dispatch to `forward_frame_aware()` whenever
`len(t.shape) > 2`. That path reshapes timestep modulation to:

```text
t0: (B, 1, F, 6*D)
block t: (B, F, 6, D)
hidden: (B, F, spatial_tokens, D)
```

So `shift/scale/gate` are applied per frame and then broadcast over
that frame's spatial tokens. The final layer is also frame-aware: it
applies `shift/scale` over `(B, F, spatial_tokens, D)`.

Native currently flattens timestep input through
`SanaWmTimestepEmbedder.sinusoidal_embedding(timestep.reshape(-1, 1))`,
then reshapes block modulation as:

```text
timestep_modulation.reshape(batch_size, 6, -1)
```

This works for `(B,)`; for `(B,1,F)` it expands the hidden dimension
by `F`, producing the observed `2240` vs `4480` mismatch when `F=2`.
The native final layer likewise assumes `timestep_embed[:, None]`,
i.e. batch scalar timestep only.

Code references:
- native embedder flattening: `sana_wm_transformer.py:330-345`
- native block scalar reshape: `sana_wm_transformer.py:1547-1560`
- native final scalar modulation: `sana_wm_transformer.py:1609-1612`
- NVlabs frame-aware block: `sana_multi_scale_video_camctrl.py:315-424`
- NVlabs frame-aware final layer: `sana_blocks.py:907-925`

#### Break 3 — scheduler update is per token and mask-preserving

NVlabs does not update the whole latent with a scalar scheduler step.
It flattens video tokens to `(B, FHW, C)`, passes
`per_token_timesteps=(B, FHW)`, and only writes back generated tokens:

```text
condition frame tokens: timestep = 0, preserved after step
generated frame tokens: timestep = current t, updated after step
```

Native `SanaWmFlowMatchScheduler` wraps `DPMSolverMultistepScheduler`
and exposes only:

```text
step(noise_pred, timestep, latents) -> prev_sample
```

There is no `per_token_timesteps`, no token flattening, no sign flip for
the per-token FlowMatchEuler path, and no `tokens_to_denoise_mask`.

Code references:
- native scheduler wrapper: `scheduling_sana_wm.py:40-66`
- native full-latent update: `pipeline_sana_wm.py:731`
- NVlabs per-token update: `flow_euler_sampler.py:178-188`

#### Secondary mismatch — first-frame noise schedule

Native currently noises the encoded first frame once at the highest
timestep before the denoising loop (`pipeline_sana_wm.py:673-676`).
NVlabs keeps the original `init_latents`, then can re-inject
timestep-dependent motion-continuity noise inside every step through
`add_noise_to_image_conditioning_latents`. In the public config path
`condition_frame_info={0: 0.0}`, so this term is disabled by default,
but the native "noise once to highest t" behavior is still not the
same as NVlabs' "place first latent in z and preserve it via mask"
contract.

#### Fix implication

The fix is not to tune thresholds or change camera weights. The native
Stage-1 path needs a new contract:

1. build `condition_mask` over latent video tokens;
2. pass `(B, 1, F)` timestep into the transformer;
3. make `SanaWmTimestepEmbedder`, `SanaWmBlock`, and `SanaWmFinalLayer`
   frame-aware;
4. implement an LTX-style `FlowMatchEulerDiscreteScheduler` step using
   `per_token_timesteps`;
5. preserve conditioning-frame tokens after each step.

---

## 7. Outstanding Work — No GPU Required

1. **Create L2 offline expansion test** `tests/e2e/offline_inference/test_sana_wm_expansion.py`. Follow `test_wan22_t2v.py` pattern: `@pytest.mark.diffusion`, `@hardware_test`, parametrize `omni_runner`, call `send_diffusion_request` with a 2-step smoke.

2. **Create L2 online serving expansion test** `tests/e2e/online_serving/test_sana_wm_expansion.py`. Follow an existing video model expansion test pattern.

3. **Add `vllm_omni/deploy/sana_wm.yaml`** deploy config (follow `ming_flash_omni.yaml` or `glm_image.yaml`).

4. **Evaluate `model_index.json` on HF.** If the SANA-WM HF repo can accept a PR adding `{"_class_name": "SanaWmTwoStagesPipeline"}`, then the three framework changes (§6.2) can be reverted. This is the cleanest fix.

5. **Simplify transformer `load_weights()`** to ≤ 30 lines (see §6.3). Move the `SanaWmStage1LoadReport` detail to a helper or drop it.

6. **Narrow TP helper try/except** (see §6.4).

7. **Decide Cache-DiT policy** for the GDN recurrence (see §6.6). Even a documented decision ("Cache-DiT incompatible with GDN recurrence because…") closes the open item.

8. **Tighten reference-alignment threshold** — lower `SANA_WM_E2E_REFERENCE_MAX_MAE` and add PSNR/SSIM assertions (can be done before the GPU run, just set a tighter target value).

9. **Architecture diagram** for PR description. Include a diagram of the Stage-1 DiT structure (input packing → camera branch → GDN block → cross-attn → FFN). This is a strong reviewer-attention signal.

10. ~~**Port UCPE per-block attention (§6.10).** Add `vllm_omni/diffusion/models/sana_wm/ucpe.py` with `prepare_prope_fns` ported from NVlabs `sana_camctrl_blocks.py`. Rewrite `_forward_ucpe` to use bidirectional GDN recurrence (reuse `reference_bidirectional_gated_delta_net`) with `q_cam_trans`/`k_cam_trans` as the rotary inputs. Add `_prepare_cam_qkv` (project → mask → conv → norm → ReLU → scale → permute → UCPE). Add inflation_sq → β discount. Add `apply_output_gate=False` mode to `_forward_gdn`. Rewire `SanaWmSelfAttention.forward` to compute `main_raw + out_proj_cam(cam_raw)` then apply shared `output_gate` + `proj` once. Unit test against NVlabs `_GDNUCPEBase` block (atol=1e-4 fp32 single-block).~~ ✅ **Done 2026-05-27.** Module-level energy-preservation tests pass at 1e-7 precision; direct GPU parity against NVlabs `prepare_prope_fns` + `BidirectionalGDNUCPESinglePathLiteLA._forward_cam_branch` is verified in §6.12a.

11. ~~**Fix vLLM parallel linear weight loading (§6.11).**~~ ✅ **Done 2026-05-28** — fixed by tightening `use_official_backend` gating to require `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`. `qkv.weight.norm()` now matches the checkpoint (7.21e+02 ≡ 7.20e+02), `out_proj_cam.weight.norm()` is now 10.78 (was 0). The MAE drop from this fix alone is small (98.31 → 95.25) because it exposed §6.12.

12. ~~**Fix Stage-1 latent magnitude (§6.12).**~~ ✅ **Done 2026-05-28** — root cause localised to the cam branch (main-only latent was always in normal range). Cam branch rewritten to match NVlabs `BidirectionalGDNUCPESinglePathLiteLA`: single-path delta-rule recurrence (no Z denominator), `apply_fn_o` inverse output transform, and `_downscale_to_reference_rms` PostUCPERenorm. Latent at STAGE1_STEPS=1 dropped from `[-59, 61]` to `[-12.3, 11.8]`.

13. ⚠️ **Per-frame timestep contract (§6.13b + §6.13d) — Breaks 1+2+3 landed 2026-05-28.** Model side per-frame `(B, 1, F)` fp32 timestep + native `step_flow_euler_per_token` on `SanaWmFlowMatchScheduler` + `condition_mask` re-enabled by default. Current default 9-frame harness: **MAE=69.06, PSNR=9.68, SSIM-Y=-0.059** (was 102.48 / 6.57 / -0.017 at session start). Next investigations (in §6.13d): diagnose negative SSIM (single-block parity vs NVlabs; visual decode side-by-side), then move to softmax-UCPE port and camera embedder structural alignment.

---

## 8. Outstanding Work — GPU Required

1. **Re-run reference-alignment harness** (`SANA_WM_E2E_REFERENCE_ALIGNMENT=1`). 2026-05-27 morning run produced MAE=95.82 / PSNR=7.37 / SSIM=0.0047 with the SDPA cam branch. UCPE port landed later same day (§7 item 10 ✅), and direct NVlabs camera-module parity is now verified (§6.12a). Need a fresh e2e run after the §6.13 per-frame timestep refactor to confirm the full Stage-1 + refiner path drops into the comparison-meaningful range.

2. **Wire PSNR ≥ 30 / SSIM-Y ≥ 0.93** once harness produces a qualifying result.

3. **GDN multi-step parity** at full 704×1280 / 321 frames vs official NVlabs path.

4. **CUDA Graph capture smoke** — confirm no regression vs fused-GDN e2e when bucket capture fires.

5. **Real USP run** to validate `_sp_plan` under multi-GPU sequence parallelism.

6. **HSDP + CFG-parallel sweeps** from `dfx/perf` config.

7. **Live server smoke** — `vllm serve … --omni` + `POST /v1/videos/generations` with SANA-WM camera payload.

---

## 9. Recommended Next-Step Ordering

### 9.1 Immediate (no GPU, before PR open)

1. Add `tests/e2e/offline_inference/test_sana_wm_expansion.py` + `tests/e2e/online_serving/test_sana_wm_expansion.py` (§6.1).
2. Add `vllm_omni/deploy/sana_wm.yaml` (§6.5).
3. Evaluate HF `model_index.json` to remove framework changes (§6.2).
4. Simplify `load_weights()` and narrow TP helpers (§6.3, §6.4).
5. Add architecture diagram draft (§7 item 9).

### 9.2 Single-GPU correctness (GPU)

6. Run reference-alignment harness; expect MAE < 30 on 24f 256×448 smoke.
7. Wire PSNR/SSIM assertions.
8. GDN full-shape parity at 704×1280 (commit `tests/e2e/accuracy/` result as markdown).

### 9.3 Throughput (GPU after correctness closes)

9. CUDA Graph capture + latency comparison.
10. Cache-DiT decision (register or document why not).
11. Paste L2 expansion test outputs in PR description.

### 9.4 HA multi-card (GPU, deferred)

12. Real USP validation under `_sp_plan`.
13. HSDP + CFG-parallel sweeps.
14. Live server smoke.

---

## 10. vLLM Infrastructure Integration — Summary

> Full deep-dive in revision 9 (Kimi's analysis, retained in git history at `b57e96b5`).
> Headline summary only here.

All DiT attention/FFN/projection linear layers now use vLLM parallel layer primitives (`QKVParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear`, `_maybe_make_vllm_rms_norm`). `quant_config` is threaded. `_sp_plan` is declared. `vllm_omni.diffusion.attention.layer.Attention` is used for softmax-attention blocks; the UCPE camera branch now uses the ported GDN+UCPE recurrence path. Quantized and TP checkpoint loading inherit automatically from the layer primitives.

What remains model-specific by design:
- **GDN recurrence core** (`fused_gdn_chunkwise.py`): bidirectional frame-wise delta-rule, not standard attention. Cannot reuse vLLM Qwen3-Next's AR GDN backend.
- **Camera-control geometry** (`camera_control.py`): model-specific Plücker/raymap math.
- **LTX-2 refiner coupling**: architecturally correct (`pipeline_sana_wm_two_stages.py` imports `LTX2TextConnectors` because the refiner *is* LTX-2 19B).

---

## 11. References

- Spec: [`sana_wm_integration.md`](sana_wm_integration.md)
- Contributor checklist: [`.claude/skills/add-diffusion-model/references/intergration.md`](../../.claude/skills/add-diffusion-model/references/intergration.md)
- Tracking issue: <https://github.com/vllm-project/vllm-omni/issues/3656>
- HF release: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
- NVlabs reference: <https://github.com/NVlabs/Sana>
- In-tree reference: `vllm_omni/diffusion/models/ltx2/` and `tests/diffusion/models/ltx2/`
- Branch base: `4ebdcc2e feat: add Sana-WM integration scaffold`
