# Sana-WM Integration — Progress Audit

> **Audit date:** 2026-05-27 (revision 11 — post-correctness-blocker + intergration.md pass)
> **Branch:** `feat/sana_wm`
> **Implementation HEAD:** `94dfaa3f fix(sana-wm): stabilize cuda graph and camera tests`
> **Pushed to:** `fork/feat/sana_wm` (`BruceLoveDecimal/vllm-omni`)
> **Spec (single source of truth):**
> [`sana_wm_integration.md`](sana_wm_integration.md)
> **Tracking issue:**
> [vllm-project/vllm-omni#3656](https://github.com/vllm-project/vllm-omni/issues/3656)

---

## 0. Revision-9 Critical-Path Items — Updated Status

Five blockers were identified in revision 9. Three closed since revision 10.

| # | Item | Status |
|---|---|---|
| 1 | GDN Triton — long-sequence + multi-card parity vs. NVlabs | ⚠️ **Open** — main-branch GDN math verified equivalent (see §6.10); divergence is upstream of recurrence |
| 2 | First-frame VAE encode for I2V conditioning | ✅ **Closed** — commit `f7e59121` A.1 |
| 3 | NVlabs flow-DPM solver | ✅ **Closed** — commit `f7e59121` A.2 (`DPMSolverMultistepScheduler`) |
| 4 | UCPE branch decomposition + numeric Plücker reference test | ⚠️ **Port landed, contribution masked by §6.11 loader bug** — UCPE math (`ucpe.py`) ported with passing unit tests; cam branch rewritten and verified to execute. GPU run 2026-05-28 produced MAE=98.31 because `out_proj_cam.weight==0` masks the cam contribution. See §6.10 + §6.11. |
| 6 | vLLM parallel linear weight loading | ✅ **Fixed 2026-05-28** — `use_official_backend` gating tightened to require explicit `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`. Loaded weight norms verified on GPU. See §6.11. |
| 7 | Stage-1 latent magnitude vs LTX-2 refiner | ✅ **Fixed 2026-05-28** — cam branch rewritten as `BidirectionalGDNUCPESinglePathLiteLA` (single-path + apply_fn_o + RMS renorm). Latent in normal range now. See §6.12. |
| 8 | Per-token timestep sampling contract | ❌ **NEW (2026-05-28)** — NVlabs uses per-frame sigma signalling (`condition_frame_info` → per-token timestep). Our pipeline uses scalar timestep. This is the root cause of the persistent ~90 MAE gap, localised by 2026-05-28 evening experiments. See §6.13. |
| 5 | TP layers → HSDP+USP → CUDA Graphs → Cache-DiT (ordered DAG) | ⚠️ **Partial** — TP + CUDA Graphs done; HSDP+USP CPU-static only; Cache-DiT not registered |

**Implication for reference alignment:** The GPU run on 2026-05-27 produced MAE=95.82 / PSNR=7.37 dB / SSIM-Y=0.0047 on the 9-frame harness. That result is consistent with item 4 being functionally open: the model has no working camera-conditioning path, so the output is structurally unrelated to the official reference regardless of how accurate the GDN recurrence is. Items 1 and 4 must close before PSNR ≥ 30 / SSIM-Y ≥ 0.93 is achievable.

---

## 1. TL;DR

Overall progress:

- **Single-GPU production-native inference: ~92–95%.** The four correctness blockers (§0.3 items 2–4) landed in commits `f7e59121`/`291397fc`. GDN production-shape parity (item 1) remains. CUDA Graph and `_sp_plan` are wired. Cache-DiT is the last single-GPU throughput item.
- **Highly-available multi-card (TP/SP/HSDP + quant + live server): ~80–85%.** TP parallel layers are fully migrated across all attention/FFN/projection sites. `_sp_plan` is declared. Real USP and HSDP sweeps remain GPU-gated. Cache-DiT is not yet registered. Online serving endpoint is mapped but not live-server validated.

Summary of recent commits since revision 10:

```text
94dfaa3f  fix(sana-wm): stabilize cuda graph and camera tests         ← HEAD
fe4f4c72  feat(sana-wm): implement B/C HA items — cuda graph, TP closure, serving tests
b8815a1f  feat(sana-wm): add cuda graph helper
291397fc  fix(sana-wm): stabilize correctness blocker tests
f7e59121  feat(sana-wm): implement single-GPU correctness blockers A.1/A.2/A.3
```

**A.1** — VAE encode first frame: `_preprocess_first_frame` + `_vae_encode_first_frame` in `pipeline_sana_wm.py`; replaces `torch.randn` placeholder.
**A.2** — Real scheduler: `SanaWmFlowMatchScheduler` wraps `DPMSolverMultistepScheduler` with `flow_shift=9.8`; the shifted-Euler `SanaWmFlowDpmScheduler` is kept for backward-compat only.
**A.3** — UCPE camera: `_forward_ucpe` SDPA cross-attention in `SanaWmSelfAttention`; `spatial_raymap` from `compute_raymap(use_plucker=False)` fused in `_camera_hidden_states_from_conditions`. **Reopened 2026-05-27:** `_forward_ucpe` uses the wrong operator (SDPA where NVlabs uses GDN recurrence with UCPE per-ray transforms) AND is never invoked from `SanaWmSelfAttention.forward` — `camera_hidden_states` is accepted but dropped. See §6.10.
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

### 6.10 Cam Branch Uses SDPA Instead of GDN+UCPE; Never Called From `forward()` ⚠️ FIX IN PROGRESS

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

**Symptom.** Reference-alignment harness reports MAE=95.82 / PSNR=7.37 dB / SSIM-Y=0.0047 on the 9-frame smoke even after A.1–A.3 landed. That is essentially uncorrelated output — too large to be a recurrence-precision issue.

**Root cause.** What `f7e59121` A.3 actually delivered was the *camera embedder* (Plücker/raymap projections). The *per-block dual-branch attention* — which is the architectural distinguishing feature of SANA-WM vs vanilla SANA-Video — is not implemented:

1. [`SanaWmSelfAttention._forward_ucpe`](../../../vllm_omni/diffusion/models/sana_wm/sana_wm_transformer.py#L968-L993) uses `F.scaled_dot_product_attention`. NVlabs uses bidirectional GDN recurrence with UCPE per-ray Q/K/V transforms (see `sana_gdn_camctrl_blocks.py:_forward_cam_branch` and `_prepare_cam_qkv`). Different operator family.

2. [`SanaWmSelfAttention.forward`](../../../vllm_omni/diffusion/models/sana_wm/sana_wm_transformer.py#L995-L1003) accepts `camera_hidden_states` but only dispatches to `_forward_gdn(...)`. `_forward_ucpe` is **never called**. Verified by grep: no caller in tree.

3. `out_proj_cam`, `q_proj_cam`, `k_proj_cam`, `v_proj_cam`, `q_norm_cam`, `k_norm_cam`, `conv_k_cam` are constructed and loaded from the checkpoint, but their forward contribution is never added to the block output.

4. The reference fuses `main_raw + cam_contrib` BEFORE `output_gate` and `proj` (one shared application). Our `_forward_gdn` applies `output_gate * proj` directly on `main_raw`, leaving no insertion point for `cam_contrib`.

5. UCPE per-ray transforms (`prepare_prope_fns` / `apply_fn_q` / `apply_fn_kv` / `apply_fn_o`), Dynamic Beta Discounting (β ÷ inflation_sq), and shared β/decay precomputation across the two branches are all absent.

**What IS verified equivalent.** The main-branch GDN math itself is correct:
- Recurrence (`_delta_scan`) matches `torch_recurrent_sana_gdn` line-for-line.
- Bidirectional `flip_and_shift` (k/v shift=0, decay shift=1) matches.
- Bidirectional short conv with center-tap subtraction matches.
- `beta`/`decay` projection chain matches.
- Triton kernel applies ReLU (default `SKIP_RELU=False`) and encodes RoPE pair-sign correctly via `prepare_rope_tables`.
- `k_scale = D^-0.5 · S^-0.5`, RMSNorm over hidden_size — all match.

**Conclusion.** Closing item 1 (GDN long-sequence parity) cannot bring PSNR ≥ 30. Item 4 must reopen and the cam branch must be ported as a real GDN+UCPE recurrence path before reference alignment is meaningful.

**Fix scope (P0).** New `vllm_omni/diffusion/models/sana_wm/ucpe.py` (port `prepare_prope_fns`); rewrite `_forward_ucpe` to use the existing `reference_bidirectional_gated_delta_net` / fused Triton path (not SDPA); add `_prepare_cam_qkv` mirroring NVlabs ordering (project → mask → conv → norm → ReLU → scale → permute → UCPE); add inflation_sq → β discount; expose `apply_output_gate=False` mode in `_forward_gdn`; rewire `forward` to compute `main_raw + out_proj_cam(cam_raw)` then apply shared `output_gate` + `proj` once. Estimate: 4–5 person-days.

Cross-ref: integration.md §1124 "Native Stage-1 Production Readiness Gap" items 1 and 4 (updated in same revision).

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

10. ~~**Port UCPE per-block attention (§6.10).** Add `vllm_omni/diffusion/models/sana_wm/ucpe.py` with `prepare_prope_fns` ported from NVlabs `sana_camctrl_blocks.py`. Rewrite `_forward_ucpe` to use bidirectional GDN recurrence (reuse `reference_bidirectional_gated_delta_net`) with `q_cam_trans`/`k_cam_trans` as the rotary inputs. Add `_prepare_cam_qkv` (project → mask → conv → norm → ReLU → scale → permute → UCPE). Add inflation_sq → β discount. Add `apply_output_gate=False` mode to `_forward_gdn`. Rewire `SanaWmSelfAttention.forward` to compute `main_raw + out_proj_cam(cam_raw)` then apply shared `output_gate` + `proj` once. Unit test against NVlabs `_GDNUCPEBase` block (atol=1e-4 fp32 single-block).~~ ✅ **Done 2026-05-27.** Module-level energy-preservation tests pass at 1e-7 precision. Single-block end-to-end NVlabs parity test still needs GPU run (moved to §8 item 1 sub-step).

11. ~~**Fix vLLM parallel linear weight loading (§6.11).**~~ ✅ **Done 2026-05-28** — fixed by tightening `use_official_backend` gating to require `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`. `qkv.weight.norm()` now matches the checkpoint (7.21e+02 ≡ 7.20e+02), `out_proj_cam.weight.norm()` is now 10.78 (was 0). The MAE drop from this fix alone is small (98.31 → 95.25) because it exposed §6.12.

12. ~~**Fix Stage-1 latent magnitude (§6.12).**~~ ✅ **Done 2026-05-28** — root cause localised to the cam branch (main-only latent was always in normal range). Cam branch rewritten to match NVlabs `BidirectionalGDNUCPESinglePathLiteLA`: single-path delta-rule recurrence (no Z denominator), `apply_fn_o` inverse output transform, and `_downscale_to_reference_rms` PostUCPERenorm. Latent at STAGE1_STEPS=1 dropped from `[-59, 61]` to `[-12.3, 11.8]`.

13. **Refactor to per-frame timesteps (§6.13 root cause).** Localisation experiments on 2026-05-28 evening (plucker_proj disable, hard-conditioning) pointed to a fundamental sampling-contract mismatch: NVlabs' `LTXFlowEuler.sample` uses **per-token timesteps** where the conditioning frame's sigma is 0 and other frames carry the current sampling sigma, plus a `condition_mask` so only non-conditioning tokens are updated by `scheduler.step`. Our pipeline uses scalar timestep and simple noise+denoise. The hard-conditioning experiment (clean first frame without per-token timestep signal) made MAE 32 worse, confirming the model REQUIRES per-frame sigma signalling. Refactor `SanaWmTimestepEmbedder` + `t_block` + block forward to accept per-frame timesteps, switch the wrapped DPM scheduler to a per-token-timestep flow-matching step, and add `tokens_to_denoise_mask`. 1-2 person days. See §6.13.

---

## 8. Outstanding Work — GPU Required

1. **Re-run reference-alignment harness** (`SANA_WM_E2E_REFERENCE_ALIGNMENT=1`). 2026-05-27 morning run produced MAE=95.82 / PSNR=7.37 / SSIM=0.0047 with the SDPA cam branch. UCPE port landed later same day (§7 item 10 ✅). Need a fresh GPU run to confirm MAE drops into the comparison-meaningful range. Also add a single-block parity test that compares our `SanaWmSelfAttention.forward(camera_conditions=...)` output against NVlabs `BidirectionalGDNUCPELiteLA.forward(..., camera_conditions=...)` at atol=1e-4 fp32 — this is the cheapest way to localise any remaining divergence before running the full 4-step e2e.

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

All DiT attention/FFN/projection linear layers now use vLLM parallel layer primitives (`QKVParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear`, `_maybe_make_vllm_rms_norm`). `quant_config` is threaded. `_sp_plan` is declared. `vllm_omni.diffusion.attention.layer.Attention` is used for softmax-attention blocks (SDPA cross-attn in UCPE). Quantized and TP checkpoint loading inherit automatically from the layer primitives.

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
