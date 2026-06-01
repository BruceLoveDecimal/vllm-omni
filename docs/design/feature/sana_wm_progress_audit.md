# Sana-WM Integration — Progress Audit

> **Audit date:** 2026-06-01 (revision 27 — late-step block/attention + MLP stride probe)
> **Branch:** `feat/sana_wm`
> **Implementation snapshot:** fork worktree at `d3ed5411` plus native BothTriton/VAE/e2e-harness/prompt/activation diagnostic patches, native `cam_prep_func` port, MLP memory-layout parity patch, late-step probes, frame-aligned e2e gates, engine-entered input parity probes, prompt-fixed step-0 activation splits, post-prompt-mask 4-row full e2e gate, and current-workdir 321f/20 Stage-1 trajectory reruns on RTX PRO 6000 98GB
> **Pushed to:** `fork/feat/sana_wm` (`BruceLoveDecimal/vllm-omni`)
> **Spec (single source of truth):**
> [`sana_wm_integration.md`](sana_wm_integration.md)
> **Tracking issue:**
> [vllm-project/vllm-omni#3656](https://github.com/vllm-project/vllm-omni/issues/3656)

---

## 0. Revision-9 Critical-Path Items — Updated Status

Revision-9 blockers are mostly closed. Short-sequence Stage-1 alignment is now
strong; the open Stage-1 correctness gap is 321-frame trajectory-level drift in
the controlled denoising loop (§6.13p-§6.14o). Scheduler update semantics,
isolated block-local GDN/cam math, full-transformer fp32 precision, and the
engine-entered transformer call site are now ruled out as primary causes. §6.13v
also fixed a real Stage-1 prompt-tokenization mismatch. §6.13w localised the
first material step-0 divergence to block-0 self-attention; §6.13x then found
the dominant downstream error was native cross-attn ignoring the prompt padding
mask. Passing the NVlabs prompt mask through native cross-attn collapses
321-frame step-0 `noise_pred` MAE from `0.12383` to `0.01211`.
Native `cam_prep_func` has now replaced the Python Q/K/V UCPE prep path
(§6.14l); this is e2e-neutral on 9f/20 (`generated MAE 61.17 -> 61.70`,
SSIM-Y unchanged). The corrected current-workdir 321f/20 Stage-1 trajectory
rerun improves final generated latent MAE from the stale-probe `0.4085` to
`0.1177` with cosine `0.9751`; masked teacher-forced one-step replay shows
the scheduler update formula is not the first-order issue. Long-sequence full
e2e no longer shows an obvious 321-vs-9 degradation after the VAE memory-mode
and frame-index fixes (§6.14g-§6.14l). The post-prompt-mask rerun improves
321f/20 to generated MAE `39.32` / PSNR `14.30` / SSIM-Y `0.244`, but strict
NVlabs RGB parity is still far from the PSNR/SSIM target.
The follow-up late-step block/attention split (§6.14o) rules out isolated
self-attn and cross-attn as dominant amplifiers, and closes a real native MLP
bf16 Conv2d stride/layout mismatch. That semantic fix makes isolated block18/19
MLP same-input output byte-identical to NVlabs, but the full 321f/20 free-run
is essentially unchanged (`MAE 0.1177 -> 0.1210`), so the remaining Stage-1
gap is still trajectory-feedback sensitivity rather than a single late block
op bug.

| # | Item | Status |
|---|---|---|
| 1 | GDN Triton — long-sequence + multi-card parity vs. NVlabs | ⚠️ **Mostly numeric-only now** — §6.13x shows raw Q/K/V, K conv, Q/K inv-RMS, beta/decay are exact. Native `cam_prep_func` now ports the NVlabs fused camera-prep contract (§6.14l); direct kernel parity passes and 9f/20 e2e is essentially neutral. The larger step-0 gap was native cross-attn missing prompt padding mask; fixing that drops baseline 321f step-0 `noise_pred` MAE `0.12383 -> 0.01211` (cos `0.999887`). §6.14o rules out isolated late-step self-attn/cross-attn as the main amplifier and closes a real MLP bf16 Conv2d memory-layout semantic gap. Remaining deltas are trajectory-feedback numeric residuals rather than scheduler/weight/prompt/call-site semantic bugs. Frame-aligned full e2e 321f/20 did **not** degrade vs 9f/20 and improves again after the prompt-mask fix (§6.14k). |
| 2 | First-frame VAE encode for I2V conditioning | ✅ **Closed** — commit `f7e59121` A.1 |
| 3 | NVlabs flow-DPM solver | ✅ **Closed** — commit `f7e59121` A.2 (`DPMSolverMultistepScheduler`) |
| 4 | UCPE branch decomposition + numeric Plücker reference test | ✅ **Verified 2026-05-28** — UCPE math (`ucpe.py`) and native raw camera branch match NVlabs `prepare_prope_fns` + `BidirectionalGDNUCPESinglePathLiteLA._forward_cam_branch` at fp32 `~1e-7` max abs. See §6.12a. |
| 6 | vLLM parallel linear weight loading | ✅ **Fixed 2026-05-28** — `use_official_backend` gating tightened to require explicit `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`. Loaded weight norms verified on GPU. See §6.11. |
| 7 | Stage-1 latent magnitude vs LTX-2 refiner | ✅ **Fixed 2026-05-28** — cam branch rewritten as `BidirectionalGDNUCPESinglePathLiteLA` (single-path + apply_fn_o + RMS renorm). Latent in normal range now. See §6.12. |
| 8 | Per-token timestep sampling contract | ✅ **Stage-1 step-0 semantic path mostly closed** — per-frame `(B, 1, F)` timestep, native per-token FlowMatch Euler, condition-mask restore, VAE norm, softmax-UCPE, prompt tokenization, and prompt padding mask are landed. §6.13x collapses 321f step-0 baseline `noise_pred` MAE from `0.12383` to `0.01211`; with official main+cam attention ablation it is `0.01280`, showing the remaining gap is no longer dominated by scheduler or cross-attn mask semantics. §6.14l reruns controlled 321f/20 with the corrected current workdir and native cam prep: final generated latent MAE is `0.1177` / cos `0.9751`, and masked teacher-forced replay confirms the scheduler update formula is aligned. §6.14o MLP stride parity rerun is `0.1210` / cos `0.9739`, so the residual remains late-step trajectory feedback. 321f/60 remains pending. |
| 5 | TP layers → HSDP+USP → CUDA Graphs → Cache-DiT (ordered DAG) | ⚠️ **Partial** — TP + CUDA Graphs done; HSDP+USP CPU-static only; Cache-DiT not registered |

**Implication for reference alignment:** The UCPE / camera-control module is now
numerically aligned with NVlabs (§6.12a), and the Stage-1 sampling contract has
been tightened through native scheduler + softmax-UCPE fixes (§6.13m). However,
§6.13n-§6.14o show the 321-frame Stage-1 gap before Stage-2 was mostly a
cross-attn prompt-mask semantic miss, followed by trajectory-feedback
amplification of small per-step residuals. The engine-entered call site, raw
inputs, prompt path, projection weights, projection implementation, prompt
mask, native camera-prep contract, and scheduler update replay now clear.
Full-chain 321f/20 no longer degrades relative to 9f/20
under the frame-aligned e2e gate (§6.14j-§6.14l), and the post-mask
321f/20 rerun improves generated MAE to `39.32`. The absolute RGB metrics are
still well below the acceptance target, so the next work remains numeric parity
rather than photometric tuning.

---

## 1. TL;DR

Overall progress:

- **Single-GPU production-native inference: ~92–95%.** The four correctness blockers (§0.3 items 2–4) landed in commits `f7e59121`/`291397fc`. GDN production-shape parity (item 1) remains. CUDA Graph and `_sp_plan` are wired. Cache-DiT is the last single-GPU throughput item.
- **Highly-available multi-card (TP/SP/HSDP + quant + live server): ~80–85%.** TP parallel layers are fully migrated across all attention/FFN/projection sites. `_sp_plan` is declared. Real USP and HSDP sweeps remain GPU-gated. Cache-DiT is not yet registered. Online serving endpoint is mapped but not live-server validated.

Summary of recent commits since revision 10:

```text
189ea0d3  docs(sana-wm): record stage1 path ablation probes  ← HEAD (fork/feat/sana_wm)
250433e0  docs(sana-wm): add stage1 teacher-forced drift probe
9ec93bb5  docs(sana-wm): record long-sequence late-step drift
8951e6a4  perf(sana-wm): env-gated cam-branch dispatch to NVlabs triton kernel
8d31144a  fix(sana-wm): align softmax ucpe scheduler path
27ea326c  feat(sana-wm): native FlowMatchEuler scheduler + block-0 parity probes
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

### 6.12b NVlabs Stage-1 Full Forward Probe ⚠️ BLOCKED 2026-05-28 (SUPERSEDED — see §6.13/§6.14)

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

### 6.13 Persistent ~90 MAE Baseline — Cam Hurts Slightly ❌ NEW (SUPERSEDED — 9f e2e is MAE 22.55 / PSNR 17.72 after §6.14e; long-sequence gap moved to §6.13n–q)

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

### 6.13b Per-Frame Timestep Contract — Partial Fix Landed 2026-05-28 ✅ CLOSED (per-token Euler in §6.13d; full contract in §6.13m)

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

### 6.13f Stage-1 Latent Parity Probe ⚠️ 2026-05-28 late evening — frame 1 (generated) is anti-correlated (RESOLVED — §6.13m brings 3-step generated cos to +0.987)

After §6.13e brought e2e MAE to 47.35, ran the long-deferred direct
Stage-1 latent parity probe to find the *next* dominant error
source before guessing between softmax-UCPE / camera embedder /
native scheduler refactor.

**Setup.**

- New env-gated hook in `native_backend.py`:
  `SANA_WM_DUMP_STAGE1_LATENT=/tmp/nvlabs_stage1.pt` makes the
  in-process NVlabs path dump `output["latent"]` after
  `pipeline.generate(...)` — the Stage-1 sample tensor returned by
  `_sample_stage1`.
- Probe script `tools/scripts/probe_stage1.py` runs both paths in
  the same process, compares the (1, 128, 2, 22, 40) latents
  channel- and frame-wise.

**Headline.**

| Metric | Global | Frame 0 (conditioning) | Frame 1 (generated) |
|---|---|---|---|
| MAE | 0.6313 | 0.3554 | 0.9072 |
| RMSE | 0.8504 | 0.4576 | 1.1122 |
| **Cosine** | **0.6292** | **+0.9669** | **-0.1653** |
| nv std | 0.7109 | 0.874 | 0.493 |
| nt std | 1.0931 | 1.244 | 0.910 |

* **Frame 0 ≈ NVlabs (cosine 0.97).** The VAE encode + per-token
  step + condition_mask preservation work correctly — the
  conditioning frame's latent is essentially identical.
* **Frame 1 is anti-correlated (cosine -0.17).** The denoising
  of the generated frame is producing a latent whose spatial
  structure is INVERTED relative to NVlabs.
* Native magnitude is ~50% larger across frames (`nt_std` vs
  `nv_std`). Generation is over-shooting.

**Per-channel.**

* 70 / 128 channels with cosine > 0.5 (mostly OK)
* 26 / 128 channels with cosine < 0 (anti-correlated)
* Median cosine across channels: +0.60
* Worst channels show native means roughly 2× nvlabs in
  magnitude (e.g. ch 25 nv=-1.06 nt=-2.60), pointing again at
  generation step over-shoot.

**Implications for the remaining work.**

* **NOT camera embedder.** Frame 0 cosine 0.97 means camera info
  propagates correctly. The simplified `SanaWmCameraEmbedder`
  isn't the dominant gap.
* **NOT softmax-UCPE port (in isolation).** If only 5 / 20
  softmax blocks were skipping UCPE, we wouldn't see
  -0.17 cosine on generated tokens.
* **NOT loader / weights.** Frame 0 alignment confirms the
  Stage-1 weights are correctly populated.
* **IS the per-token Euler step interaction with our sigma table
  / model output.** Two specific suspects:
  - **Sigma schedule mismatch.** We reuse
    `DPMSolverMultistepScheduler(use_flow_sigmas=True, flow_shift=9.8)`
    for the sigma table; NVlabs' `LTXFlowEuler` uses
    `FlowMatchEulerDiscreteScheduler`. Both *should* produce
    matching sigmas under the same flow-shift, but this is not
    yet verified.
  - **Sign convention interacting with model output.** Default
    `-noise_pred` produced better aggregate MAE/PSNR but
    generated-frame structure is now wrong. Likely the sign that
    looks right at decoded-image level is wrong at latent level
    and the refiner masks it via downstream re-normalisation.

**Step-0 probe results (`tools/scripts/probe_step0.py`, 2026-05-28
late evening, same 9-frame harness):**

Both pipelines saved their step-0 `(latent_in, timestep, noise_pred)`
via env-gated dump hooks (`SANA_WM_DUMP_STEP0_LTX` in NVlabs
`LTXFlowEuler.sample`, `SANA_WM_DUMP_STEP0` in our
`_run_native_smoke_backend`).

| Item | NVlabs | Native | Compare |
|---|---|---|---|
| latent_in shape | (1, 128, 2, 22, 40) | same | shape match |
| latent_in stats | std=0.94, min=-4.4, max=+4.8 | std=0.94, min=-4.2, max=+4.3 | **cosine=+0.43** |
| timestep_per_frame | [0.0, 1000.0] | [0.0, 999.0] | off by one |
| noise_pred shape | (1, 128, 2, 22, 40) | same | shape match |
| noise_pred stats | std=1.02, min=-5.3, max=+5.6 | std=0.77, min=-2.9, max=+3.1 | **cosine=-0.06** |
| prompt_embeds_norm | (not probed) | **0.17** | **expected ~450 at this shape** |

**Three independent issues localised:**

1. **Prompt embedding is essentially zero (norm=0.17).** Our
   `_native_smoke_prompt_embeds` did
   `hidden_states = F.normalize(...) * y_norm_scale_factor`
   (with `y_norm_scale_factor=0.01`). NVlabs passes RAW Gemma
   hidden states directly; the model's internal
   `attention_y_norm = RMSNorm(hidden_size,
   scale_factor=y_norm_scale_factor)` handles normalisation.
   Our external pre-normalisation collapses the prompt signal,
   leaving the model effectively unconditioned on the prompt.
2. **Input latent differs (cosine=+0.43).** Even with seed=0 on
   both sides, the random noise sequence diverges (likely
   different RNG consumption order in setup code paths). This is
   a CONFOUND for the noise_pred comparison — we cannot cleanly
   say whether noise_pred would match given identical input.
3. **First-step timestep differs by 1** (1000 vs 999). The
   scheduler's terminal timestep is `num_train_timesteps`; one
   side uses `num_train_timesteps` exactly and the other uses
   `num_train_timesteps - 1`. Minor schedule alignment issue.

**Verdict on the three suspects from §6.13f:**

* "Bug in model forward" — possible, but obscured by issue #2
  (input mismatch). Need to inject identical input to test
  cleanly.
* "Bug in `step_flow_euler_per_token`" — possible, but not the
  dominant one once #1 is fixed.
* "Sigma table mismatch" — possible, but #3 hints at a smaller
  endpoint issue rather than a wholesale schedule difference.

**Prompt-embed fix landed.** Removed the `F.normalize + scale`
step from `_native_smoke_prompt_embeds` (commit `<pending>`).
GPU re-run shows:

| Metric | Before fix | **After fix** | Δ |
|---|---|---|---|
| MAE | 47.35 | 51.06 | +3.71 (worse) |
| PSNR | 13.14 | 12.43 | -0.71 (worse) |
| SSIM-Y | +0.025 | **+0.046** | **+0.021 (better)** |

SSIM-Y nearly doubled (which is the structural alignment
metric) — that is the direction we want. MAE/PSNR getting worse
suggests the prompt-conditioned output now exposes other
mismatches (noise init, timestep, etc) that were previously
masked by an unconditioned generation.

**Next investigation chunk.** Force identical noise initialisation
on both sides to control issue #2, then re-run step-0 probe. If
noise_pred matches under controlled input, scheduler is the
remaining gap. If not, we need a single-block parity test to
find the diverging model layer.

### 6.13l Native Scheduler Landed — Timestep Parity Achieved, Exposes Compounding noise_pred Drift ✅ 2026-05-29 (drift later traced to §6.13m softmax-UCPE gap, then §6.13q length-dependent gap)

Rewrote `SanaWmFlowMatchScheduler` to reproduce
`diffusers.FlowMatchEulerDiscreteScheduler(shift=9.8)` natively
(no DPMSolver wrapper). The load-bearing quirk: `sigma_min` is
established at `__init__` as the shift-transformed `1/num_train`
(= `9.8 · 0.001 / (1 + 8.8 · 0.001) ≈ 0.00971`), then
`set_timesteps` linspaces from that **already-shifted** value and
applies the shift formula a **second time**. The double-shift
produces the front-loaded "tiny early steps + huge final jump"
schedule NVlabs uses.

**Step-by-step parity (`/tmp/probe_steps.py`, 3 stage-1 steps, controlled input):**

| Step | NVlabs t_scalar | Native t_scalar | latent_in cos | noise_pred cos | noise_pred ratio |
|---|---:|---:|---:|---:|---:|
| 0 | 1000.000 | **1000.000** ✓ | +1.0000 | +0.9506 | 1.033 |
| 1 | 909.027 | **909.027** ✓ | +0.9978 | +0.8792 | 1.282 |
| 2 | 87.7046 | **87.7046** ✓ | +0.4209 | +0.1999 | 2.030 |

Timestep tables now match to fp32 precision. But latent and
noise_pred cosines **degrade more aggressively than before** —
because the now-correct giant step-2 dt (= −0.0877) propagates
the noise_pred error fully into the stage-1 output (previously
the wrong near-linear schedule had tiny dt ≈ −0.05 per step, so
errors stayed bounded).

**E2E benchmark (9-frame harness, 3+3 steps):**

| Configuration | MAE | PSNR | SSIM-Y |
|---|---:|---:|---:|
| Prior (DPMSolver wrapper, wrong schedule) | 50.53 | 12.11 | -0.002 |
| **Native FlowMatchEuler (correct schedule)** | **54.21** | **11.46** | **-0.002** |

Numbers regressed slightly on MAE/PSNR. SSIM-Y unchanged. This
is expected and informative — the scheduler was hiding the
downstream model-forward drift; fixing it exposed the real
remaining bug.

**The real remaining bug.** At step-0 with controlled-input
(identical latent + identical prompt + identical timestep),
noise_pred has **cos 0.95 and norm ratio 1.03**. Block-0 sub-
stages were verified clean (attn cos +0.9999, MLP cos +0.998 in
§6.13j), so the 5% cosine drift comes from one of:

1. **20-block compounding** of the 1.2% per-block norm-ratio
   drift seen at MLP-final (§6.13j). Theoretical 1.012²⁰ ≈ 1.27×
   in norm, but observed is 1.03× — so compounding alone
   over-predicts, suggesting it's a real but partial contributor.
2. **Final layer (`SanaWmFinalLayer`) + unpatchify** —
   completely untested at the parity level.
3. **Cross-attention / camera-fusion** at later blocks —
   block-0's cross-attn was implicitly verified by the block-0
   post-cross-attn cos +1.000 in §6.13h, but only at block-0.
4. **Refiner stage** — entirely untested at sub-step level.

**Next probe.** Multi-block compounding sweep:
dump `block_output` at blocks 0/5/10/15/19 + `final_layer` +
`noise_pred` (unpatchify out) on both sides at step-0 with
controlled inputs. Cosine and norm-ratio trajectory will
isolate which stage contributes most.

**Acceptance for native scheduler work.** Timestep table parity
achieved ✅. E2E parity unlocked but blocked on §6.13m
(noise_pred drift). Removing diffusers `DPMSolverMultistepScheduler`
dependency: ✅ done — `scheduling_sana_wm.py` no longer imports
diffusers at runtime.

**Repro.** `/tmp/probe_steps.py` (step parity),
`/tmp/run_metrics.py` (e2e) on remote
`/root/autodl-tmp/vllm-omni-feat-sana-wm-23ff624b` after syncing
`vllm_omni/diffusion/models/sana_wm/scheduling_sana_wm.py`.

### 6.13m Softmax-UCPE Native Path Landed — Stage-1 Latent Parity Tightened ✅ 2026-05-29

The multi-block probe after §6.13l found a structural omission in
the every-4th hybrid blocks (`3/7/11/15/19`): native softmax blocks
were running plain RoPE SDPA and `proj`, while NVlabs runs
`main_raw + out_proj_cam(cam_raw)` followed by the shared
`output_gate + proj`. That means two pieces were missing:

1. the softmax UCPE camera branch (`_forward_cam_branch_softmax`),
   including post-UCPE RMS downscale and inverse `apply_fn_o`;
2. the shared GDN `output_gate` path for softmax blocks.

Native now mirrors NVlabs' `_SoftmaxUCPESinglePathLiteLA` structure:
`_forward_softmax_raw(...)` returns raw SDPA output, optional
`_forward_softmax_cam_branch(...)` contributes UCPE camera output,
and both GDN and softmax paths use the same
`_apply_output_gate_and_proj(...)` helper. The helper also matches
NVlabs' dtype order more closely: output-gate linear is promoted to
fp32 before `silu` and multiply, then cast to projection weight dtype.

**Controlled DiT+scheduler verification on 5090** (identical
initial latent, prompt embedding, and per-frame timestep; `PYTHONPATH`
explicitly points at the synced worktree):

| Probe | Before §6.13m | After §6.13m |
|---|---:|---:|
| 2-step final latent MAE | 0.013053 | **0.002798** |
| 2-step final latent cosine | 0.999674 | **0.999972** |
| 3-step final latent MAE | 0.131467 | **0.063287** |
| 3-step final latent cosine | 0.968252 | **0.992491** |
| Generated-frame channels with cosine < 0 at step-0 | 3 | **0** |

Frame-level 3-step result:

| Frame | MAE | Cosine | Note |
|---|---:|---:|---|
| frame 0 (conditioning) | 0.000000 | 0.999991 | condition mask + per-token step preserve it exactly |
| frame 1 (generated) | 0.126575 | 0.986724 | remaining drift is generated-frame only |

**Block-output probe after softmax-UCPE port** (step-0 controlled
input, selected blocks):

| Block | Before output MAE | After output MAE | After cosine |
|---:|---:|---:|---:|
| 3 | 5.22 | **3.89** | 1.0003 |
| 7 | 11.84 | **9.98** | 0.9997 |
| 15 | 114.17 | **71.86** | 0.9963 |
| 19 | 455.80 | **321.67** | 0.9042 |

This closes the earlier "softmax-UCPE port" item as a real native
implementation, not just a probe. The remaining gap is now smaller
and more specific: step-0 `noise_pred` is still only cosine ~0.90
globally, and the 3-step generated-frame latent has MAE ~0.127. For Stage-1-only work, the next useful probe is no longer
scheduler/softmax structure; it is late-block sub-stage parity
(cross-attn vs MBConv/temporal conv vs final layer) under controlled
inputs. Full-chain alignment moves downstream to the refiner contract
probe in §6.14.

#### 6.13n Stage-1 Denoise Scaling Probe — Step Count OK, Long Sequence Breaks ⚠️ 2026-05-29

Ran the planned Stage-1-only denoise scaling sweep on the RTX PRO 6000
Blackwell target. The probe compares NVlabs Stage-1 `_sample_stage1`
against vLLM-Omni native Stage-1 on the same synthetic first frame,
prompt, straight camera trajectory, seed `0`, CFG `1.0`, and
`flow_euler_ltx`. It records final Stage-1 latent parity before
Stage-2 refiner or VAE decode.

| Config | Latent shape | Global MAE | Global cos | Generated MAE | Generated cos | Runtime |
|---|---|---:|---:|---:|---:|---:|
| 9 frames / 20 steps | `(1,128,2,22,40)` | 0.0804 | 0.9918 | 0.1338 | 0.9878 | 82.7s |
| 9 frames / 60 steps | `(1,128,2,22,40)` | 0.0919 | 0.9902 | 0.1568 | 0.9861 | 81.5s |
| 321 frames / 20 steps | `(1,128,41,22,40)` | 0.3930 | 0.7647 | 0.4022 | 0.7575 | 156.4s |
| 321 frames / 60 steps | — | — | — | — | — | aborted |

Frame-0 stays aligned across all completed configs (`MAE=0.0271`,
`cos=0.9978`), so the first-frame VAE encode / condition preservation
path is not the cause of the long-sequence collapse.

**Interpretation.**

> **Headline takeaway (negative result).** **Step count is not the
> bottleneck.** Increasing 9-frame Stage-1 denoise steps from `20` to
> `60` changes generated cosine `0.9878 → 0.9861` (a 0.0017 *drop*).
> Do **not** spend GPU time trying to raise step count to improve
> parity or recover Stage-1 PSNR.

> **Headline takeaway (positive signal).** Going from 2 latent frames
> to 41 latent frames at the same 20 steps collapses generated cosine
> `0.9878 → 0.7575` — a 23-point drop. The single biggest unresolved
> contract gap is **length-dependent**, not step-count-dependent.
> Implication: every parity number reported in §§6.13a-m / 6.14 was
> measured on a 9-frame (2 latent-frame) harness; those numbers do not
> certify production long-video behavior.

That points away from the scheduler step count and toward a
long-sequence contract difference: `chunk_index`, temporal
GDN/softmax-UCPE sequence handling, camera trajectory conditioning, or
chunk/stride layout.

Per the "jump out on large deviation" rule, the 321-frame / 60-step
run was stopped after `321f/20step` exposed the large latent-parity
drop. Running 60 steps at the same long shape would be expensive and
hard to interpret until the 321-frame / 20-step gap is localized.

Artifacts:

- `/root/autodl-tmp/stage1_denoise_probe/results.jsonl`
- `/root/autodl-tmp/stage1_denoise_probe/official_stage1_9f_20step.pt`
- `/root/autodl-tmp/stage1_denoise_probe/native_stage1_9f_20step.pt`
- `/root/autodl-tmp/stage1_denoise_probe/official_stage1_9f_60step.pt`
- `/root/autodl-tmp/stage1_denoise_probe/native_stage1_9f_60step.pt`
- `/root/autodl-tmp/stage1_denoise_probe/official_stage1_321f_20step.pt`
- `/root/autodl-tmp/stage1_denoise_probe/native_stage1_321f_20step.pt`


#### 6.13o Long-Sequence Probe — Inputs Match; Collapse Is Late-Step Denoise Drift ⚠️ 2026-05-29

Follow-up probes on the RTX PRO 6000 target localized the §6.13n
321-frame failure further. The goal was to separate three suspects:
long-sequence input contract, first model forward, and multi-step
denoise accumulation.

**Input-side parity (`/root/autodl-tmp/stage1_longseq_probe/input_probe.json`).**

The native camera preparation is not the cause:

| Config | Latent T | `chunk_index` | `raymap` MAE / cos | `chunk_plucker` MAE / cos |
|---|---:|---|---:|---:|
| 9 frames | 2 | `None` | `0.0 / 1.0000` | `0.0 / 1.0000` |
| 321 frames | 41 | `None` | `0.0 / 1.0000` | `0.0 / 1.0000` |

So the earlier `chunk_index` suspicion is ruled out for this config:
NVlabs does not pass a chunk index at either latent length, and the
camera tensors are byte-identical up to dtype/cosine noise.

**Controlled step-0 model-forward parity
(`/root/autodl-tmp/stage1_longseq_probe/step0_controlled_probe.json`).**

Both native runs loaded NVlabs' `latent_in` and `prompt_embeds` before
the first denoise step. Timesteps matched exactly.

| Config | Latent cos | `noise_pred` global cos | `noise_pred` generated cos | Worst generated frame |
|---|---:|---:|---:|---:|
| 9 frames / 1 step | `0.99999` | `0.98218` | `0.98203` | frame 1: `0.98203` |
| 321 frames / 1 step | `0.99991` | `0.98113` | `0.98112` | frame 38: `0.97772` |

This rules out "long sequence breaks immediately at the first forward."
The 321-frame first forward is no worse than 9-frame in aggregate, and
late latent frames are still high-cosine at step 0.

**Controlled 321-frame / 20-step multi-step probe
(`/root/autodl-tmp/stage1_longseq_probe/multistep_controlled_probe_321f_20step.json`).**

The native run again loaded NVlabs' initial latent/prompt and dumped
every pre-step latent and `noise_pred`. The final generated latent still
collapses (`MAE=0.4085`, `cos=0.7491`), matching the uncontrolled
§6.13n result. Frame 0 remains exact (`MAE=0`, `cos=0.99999`), so the
condition-frame path is not implicated.

| Step | `t` | Pre-step generated latent cos | Generated latent MAE | Generated `noise_pred` cos | Generated `noise_pred` MAE |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000.0 | `0.99990` | `0.0000` | `0.98112` | `0.1605` |
| 1 | 994.4 | `0.99990` | `0.0000` | `0.98828` | `0.1226` |
| 2 | 988.3 | `0.99989` | `0.0007` | `0.98471` | `0.1383` |
| 5 | 965.3 | `0.99983` | `0.0040` | `0.97144` | `0.1836` |
| 10 | 900.0 | `0.99949` | `0.0182` | `0.92909` | `0.2965` |
| 15 | 732.3 | `0.99150` | `0.0742` | `0.86276` | `0.4172` |
| 16 | 661.2 | `0.98191` | `0.1024` | `0.84028` | `0.4469` |
| 17 | 557.6 | `0.95725` | `0.1466` | `0.81186` | `0.4726` |
| 18 | 392.4 | `0.89055` | `0.2213` | `0.76167` | `0.5014` |
| 19 | 87.7 | `0.75868` | `0.3686` | `0.67783` | `0.4725` |

**Interpretation.**

The long-sequence failure is not caused by camera packing, `chunk_index`,
RNG, prompt embedding, timestep table, or a step-0 model-forward shape
bug. It is a late-step accumulation problem: small-but-real
`noise_pred` residuals at high timesteps are tolerated for many steps,
then the low/mid-sigma section rapidly amplifies them. The largest
collapse happens after the step-18 update (`t≈392 → 87.7`), where
pre-step generated latent cosine falls from `0.8906` to `0.7587`.

The next Stage-1 probe should therefore compare the native and NVlabs
forward internals at late denoise states, not at clean step-0 inputs.
Recommended targets:

* Capture block-level outputs at controlled 321-frame step 15/18
  (`latent_in` loaded from the NVlabs/native step dumps) to identify
  whether the growing residual is in temporal GDN, softmax-UCPE camera
  branch, cross-attn, MBConv/temporal conv, or final layer.
* Compare the scheduler update numerically on the same
  `(latent_in, noise_pred, per_token_timesteps)` triplet around steps
  17–18. The scalar timestep sequence already matches, but this will
  rule out a subtle low-sigma update/mask semantic mismatch.
* If the update formula is exact, focus implementation effort on
  length-41 model-forward parity under late-step latents rather than
  more scheduler tuning or 321-frame / 60-step runs.

#### 6.13p Late-Step Scheduler/Forward Decomposition — Scheduler Exact; Residual Accumulates ⚠️ 2026-05-29

Ran the recommended late-step probes on the RTX PRO 6000 target, reusing the
321-frame / 20-step dumps from §6.13o.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/scheduler_update_replay_probe_321f_20step.json`
- `/root/autodl-tmp/stage1_longseq_probe/late_block_forward_probe_321f_steps15_18.json`
- `/root/autodl-tmp/stage1_longseq_probe/teacher_forced_one_step_probe_321f_steps15_18.json`
- `/root/autodl-tmp/stage1_longseq_probe/teacher_forced_native_on_official_trace_321f_20step.json`
- `/root/autodl-tmp/stage1_longseq_probe/native_path_ablation_teacher_forced_321f_full.json`
- `/root/autodl-tmp/stage1_longseq_probe/native_path_ablation_teacher_forced_321f_steps_15_18_main_only.json`
- `/root/autodl-tmp/stage1_longseq_probe/oracle_correction_sampler_321f_20step.json`

**Scheduler replay.** Using the actual dumped per-frame timestep table, the
native per-token FlowMatch Euler update exactly reproduces both the NVlabs and
native next latents from `(latent_in, noise_pred, timestep_per_frame)`.

| Source | Step | Generated MAE vs dumped next latent | Generated cos |
|---|---:|---:|---:|
| NVlabs | 15 | `6.5e-09` | `0.99992` |
| NVlabs | 18 | `0.0` | `0.99994` |
| native | 15 | `0.0` | `0.99992` |
| native | 18 | `0.0` | `0.99995` |

So the late-step collapse is not a scheduler formula, sign convention,
condition-mask, or timestep-table bug.

**Same-latent model-forward probe.** Feeding the same late latent into both
model implementations keeps `noise_pred` close even at steps 15/18:

| Latent fed to both models | Step | Native-vs-NVlabs `noise_pred` generated MAE | Generated cos | Cross one-step latent MAE |
|---|---:|---:|---:|---:|
| NVlabs latent | 15 | `0.04047` | `0.99874` | `0.00288` |
| NVlabs latent | 18 | `0.03347` | `0.99897` | `0.01020` |
| native latent | 15 | `0.03992` | `0.99866` | `0.00284` |
| native latent | 18 | `0.03387` | `0.99881` | `0.01032` |

The "cross one-step" column applies the other implementation's direct
`noise_pred` to the same latent with the exact scheduler update, then compares
to that source trajectory's next latent. The local perturbation is small:
roughly `|dt| * noise_pred_residual`, with the larger step-18 jump
(`t≈392 → 87.7`) producing a `~0.010` generated-latent MAE.

**Full teacher-forced native-on-official trace.** Running native forward on the
NVlabs latent at every step confirms the local residual does not blow up late;
it generally shrinks as `t` decreases. The per-step latent perturbation grows
only where the schedule takes larger sigma jumps.

| Step | `t` | Native-vs-NVlabs `noise_pred` generated MAE | Generated cos | One-step generated MAE |
|---:|---:|---:|---:|---:|
| 0 | `1000.0` | `0.16054` | `0.98110` | `0.00482` |
| 5 | `965.3` | `0.07796` | `0.99443` | `0.00077` |
| 10 | `900.0` | `0.05718` | `0.99745` | `0.00118` |
| 15 | `732.3` | `0.04047` | `0.99874` | `0.00288` |
| 16 | `661.2` | `0.03936` | `0.99880` | `0.00408` |
| 17 | `557.6` | `0.03482` | `0.99901` | `0.00575` |
| 18 | `392.4` | `0.03347` | `0.99897` | `0.01020` |
| 19 | `87.7` | `0.02988` | `0.99880` | `0.00262` |

This separates local model parity from self-feedback stability: if every step
is reset to the NVlabs latent, native remains close; when native consumes its
own previous latent, those small perturbations push it onto a different
trajectory and the later `noise_pred` comparison degrades sharply (§6.13o).

**Path ablation.** Re-ran the same teacher-forced 321-frame trace on a clean
fork worktree at `250433e0`, comparing three native execution paths:

| Variant | Step15 noise MAE / one-step MAE | Step18 noise MAE / one-step MAE | Interpretation |
|---|---:|---:|---|
| torch bf16 default | `0.04047 / 0.00288` | `0.03347 / 0.01020` | baseline |
| cam Triton bf16 | `0.04049 / 0.00288` | `0.03347 / 0.01020` | Triton loaded successfully; no material accuracy change |
| full transformer fp32 | `0.04092 / 0.00291` | `0.03285 / 0.01001` | mixed, very small improvement only at step18 |
| main-only bf16 (cam branch disabled) | `0.05397 / 0.00384` | `0.04013 / 0.01223` | worse; native cam branch helps rather than hurts |

This rules out three tempting explanations for the remaining `~0.03–0.04`
same-latent `noise_pred` residual: the NVlabs cam Triton kernel, ordinary bf16
rounding, and the camera branch being directionally harmful. The residual is
more likely a structural per-block arithmetic difference in the native DiT
path, not a single precision or dispatch switch.

**Oracle-correction sampler.** Ran native free-running denoise from the official
step-0 latent, but periodically snapped the current latent back to the NVlabs
trajectory before selected steps:

| Snap schedule | Final generated MAE | Final generated cos | Notes |
|---|---:|---:|---|
| none | `0.53488` | `0.63222` | direct fallback self-feedback collapse; same trend as §6.13o native engine |
| every 8 steps (`8,16`) | `0.02385` | `0.99911` | late-window drift mostly contained |
| every 4 steps (`4,8,12,16`) | `0.02385` | `0.99911` | same final as every-8 because last snap is still step16 |
| every 2 steps | `0.01179` | `0.99973` | step18 snap helps, but step19 still propagates residual |
| step18 only | `0.01179` | `0.99973` | nearly same as every-2; step18 is the key correction point |
| tail steps `15–19` | `0.00262` | `0.99991` | reaches teacher-forced lower bound |
| every step | `0.00262` | `0.99991` | same lower bound as tail-only correction |

The important conclusion is that early accumulated error is reversible: even
after free-run reaches generated MAE `0.10675` at step14, snapping only steps
15–19 recovers the teacher-forced lower bound. Conversely, if step18 is not
corrected, the large sigma jump (`t≈392 → 87.7`) converts a small local
`noise_pred` residual into a much larger trajectory displacement. The next
localization target is therefore not the scheduler, but the source of the small
same-latent DiT residual before the late-window self-feedback loop.

Block-level hidden tensors diverge in magnitude through the DiT stack
(`block0` MAE `~0.31–0.37`, `block15` MAE `~11–18`, final-layer input MAE
`~33–47`), but the final projection/unpatchify compresses this to a small
`noise_pred` residual (`~0.033–0.040`, cos `>0.9986`) when the latent is held
fixed. That makes a single block-18/conv implementation bug unlikely.

**Interpretation.** The poor late-step `noise_pred` cosine from §6.13o
(`0.7617` at step 18, `0.6778` at step 19) mostly compares different
trajectories, not two model implementations on the same latent. The remaining
Stage-1 problem is trajectory-level sensitivity: small local residuals are
recursively fed back into the denoise loop and amplified over length-41 video
tokens. The next high-leverage checks are:

1. split the same-latent residual inside the DiT block path (`main_raw`,
   `cam_raw`, `cam_contrib`, `combined`, output gate/proj, final layer) at
   steps 15/18;
2. compare native fallback layers against NVlabs on the same block inputs,
   especially softmax-UCPE blocks and final-layer modulation, because
   scheduler, cam scan dispatch, and broad fp32 precision are now ruled out;
3. avoid more scheduler work unless a future probe breaks exact replay.


#### 6.13q Root-cause hypothesis: missing `chunk_size` / chunk-causal mask ❌ REFUTED by source-reading 2026-05-29 night (see §6.13r)

Looking at the §6.13n/o/p evidence end-to-end, one specific
NVlabs contract knob explains every observation simultaneously and is
verifiably absent from our native code:

```bash
$ grep -rn 'chunk_size\|chunk_causal\|chunk_split' \
    vllm_omni/diffusion/models/sana_wm/
# (no matches)
```

**NVlabs side** has `chunk_size: int = 10` (default in
`SanaMSVideoCamCtrl.__init__`) and `chunk_split_strategy:
"first_chunk_plus_one"` (default from the 1600M release config).
`_SoftmaxUCPESinglePathLiteLA.forward` documents the behavior:

> Automatically selects the correct masking mode based on `chunk_size`:
> - `chunk_size is None` or `chunk_size >= T`: full bidirectional (no mask)
> - `chunk_size < T`: chunk-causal (full within chunks, causal across)

GDN blocks accept the same kwarg via `**kwargs` and route it through the
recurrent / chunkwise scan paths. The native cam-branch (`_forward_cam_branch`,
`_forward_softmax_cam_branch`) and main forward all ignore `chunk_size`.

**Why this matches every observed data point:**

| Observation | Predicted by missing chunk-causal mask |
|---|---|
| 9-frame (`T=2`) Stage-1 latent cos ≈ 0.99 | ✅ `T=2 < chunk_size=10` → NVlabs also runs full bidirectional → both match |
| 321-frame (`T=41`) Stage-1 latent cos ≈ 0.76 | ✅ `T=41 > chunk_size=10` → NVlabs switches to chunk-causal, native stays full → different attention pattern |
| Frame 0 always matches | ✅ first chunk always contains frame 0; condition mask + first-chunk full attention preserves it |
| `noise_pred` same-latent residual ~0.04 at length 41 | ✅ aggregated effect of running unmasked attention vs chunk-causal — not a precision bug |
| fp32 / bf16 / cam-triton ablation all give ~0.04 | ✅ chunk-causal mask is a structural change, not a numerical one — precision cannot close it |
| Block-level magnitudes grow then unpatchify compresses | ✅ wrong attention pattern produces structurally different intermediate features that mostly cancel at the projection level |
| Step-18 sigma jump amplifies residual into trajectory split | ✅ low-sigma steps are where attention pattern errors propagate most into the data manifold |
| Adding more steps (20 → 60) doesn't help on 9f | ✅ at `T=2` no mask is ever active either way — step count is orthogonal |

**Lower-probability alternatives we considered:**

- *Temporal `t_conv` / bidirectional short conv boundary handling.*
  Kernel size and padding are length-independent; would also break 9-frame
  controlled-input probes, which are clean (§6.13j).
- *RoPE table generation for the temporal axis.*
  Cheap to verify but our `_apply_rotary_emb` matches NVlabs
  bit-exactly at 9 frames (§6.13j); a length-conditional bug in RoPE
  would have shown up there too.
- *GDN recurrent state buildup at length 41.*
  Possible — but `cam_scan_bidi_chunkwise` does honour the F dimension
  natively and our reference scan is shape-agnostic. The structural-mask
  hypothesis above is a strict superset of "GDN runs different code at
  long T".
- *Cross-attention with prompt at long image-token count.*
  Prompt length is fixed (300 tokens). Cross-attention is `O(N_img *
  N_text)` and bf16 saturation would degrade gradually with `T`, not
  produce a cliff at `T > chunk_size`.

**Acceptance check before code change** (cheap, no implementation
work required):

1. patch NVlabs locally to call `forward(..., chunk_size=None)` (or
   `chunk_size=9999`) at every block, re-run §6.13n's 321-frame /
   20-step config, and dump the new latent.
2. compare it to our native 321-frame Stage-1 latent. If
   `cos(NVlabs_no_chunk, native) >> cos(NVlabs_default_chunk_10,
   native)`, the chunk-causal mask is the cause and porting it closes
   the long-sequence gap.

Only if this acceptance check confirms the hypothesis should we spend
implementation effort on:

3. plumbing `chunk_size` and `chunk_split_strategy` through
   `SanaWmConfig`, `SanaWmBlock`, `SanaWmSelfAttention`
   (`_forward_softmax_raw`, `_forward_softmax_cam_branch`,
   `_forward_gdn_raw`, `_forward_cam_branch`);
4. building the chunk-causal `block_mask` and threading it into both
   SDPA paths (and into the GDN scan's per-frame decay / `beta` slicing
   so the recurrence respects chunk boundaries the same way NVlabs does).

If the acceptance check refutes the hypothesis, fall back to the §6.13p
sub-stage residual probe inside the DiT block at steps 15/18.


#### 6.13r §6.13q REFUTED — source-reading shows `chunk_size` is no-op in production forward 2026-05-29 night

Direct read of the NVlabs `main` branch (pulled via
`raw.githubusercontent.com` since the SeetaCloud SSH was unstable) shows
`chunk_size` does not affect the SANA-WM 1600M production forward. The
acceptance check from §6.13q is therefore not needed — there is no
chunk-causal path to enable on the NVlabs side.

**Evidence (file:line on NVlabs `main`):**

1. **Outer wiring exists** (`sana_multi_scale_video_camctrl.py:359-361`):
   ```python
   chunk_size = kwargs.get("chunk_size", getattr(self, "chunk_size", 10))
   if chunk_size is not None:
       self_attn_kwargs["chunk_size"] = chunk_size
   ```
   So the value reaches every block's `self.attn(...)` call.

2. **Softmax main attention swallows it**
   (`sana_gdn_blocks.py:1135-1208`, `_forward_softmax_attn`):
   ```python
   def _forward_softmax_attn(..., frame_causal: bool, **kwargs):
       ...
       attn_mask = _get_frame_causal_mask(T, S, x.device) if frame_causal else None
       out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
   ```
   The caller (`_SoftmaxUCPESinglePathLiteLA.forward`) passes
   `frame_causal=False`, so `attn_mask=None`. The `**kwargs` swallows
   `chunk_size`. Result: full bidirectional SDPA always.

3. **Softmax cam-branch also swallows it**
   (`sana_gdn_camctrl_blocks.py:1424-1505`,
   `_forward_cam_branch_softmax`): only consumes
   `invalid_kv_logit_bias` from `frame_valid_mask`; `chunk_size`
   never used.

4. **GDN main forward documents it as unused**
   (`sana_gdn_blocks.py:733-757`, `BidirectionalGDN.forward`):
   ```python
   def forward(self, x, mask=None, HW=None, rotary_emb=None,
               block_mask=None, apply_output_gate=True, **kwargs):
       """
       Args:
           ...
           **kwargs: Unused extra arguments.
       """
   ```

5. **Only one path reads `chunk_size`** — `torch_chunk_sana_gdn`
   (`sana_gdn_blocks.py:208`), wired through `__init__` as
   `partial(torch_chunk_sana_gdn, chunk_size=chunk_gdn_chunk_size=21)`
   at line 460. This is a *parallel-friendly chunked scan*: state
   accumulator `S_kv = torch.zeros(...)` is initialised once and flows
   across chunks (`sana_gdn_camctrl_blocks.py:288-303`). Algebraically
   identical to the sequential recurrence; not chunk-causal.

6. **The misleading docstring**
   (`sana_gdn_camctrl_blocks.py:1514-1517`):
   ```
   - chunk_size < T: chunk-causal (full within chunks, causal across)
   ```
   refers to a planned mode that the current implementation does not
   activate. Caller never reads back `is_chunk_causal_request(...)`
   (defined in `chunk_utils.py:26`) in the main forward path.

**Implications.**

- NVlabs at `T=41` runs the same full-bidirectional SDPA / GDN scans
  as at `T=2`. The 321-frame collapse cannot be the missing
  `chunk_size` path because that path is also dormant on NVlabs.
- No native implementation work is owed here. Adding `chunk_size`
  plumbing would not move §6.13n's 321-frame cos `0.7575` because
  the NVlabs reference number it is compared against also comes from
  a full-bidirectional run.
- §6.13q is retracted. §0 status table entries that cited §6.13q
  must be updated to point at the §6.13p sub-stage residual probe
  instead.

**Updated next probe** — pick up §6.13p exactly: same-latent
controlled-input dump of `main_raw`, `cam_raw`, `cam_contrib`,
`combined`, output gate/proj, and final layer at step 15/18 of the
321-frame trace. The `~0.04` `noise_pred` residual must be coming
from a specific sub-stage; that sub-stage is the real lead, not
`chunk_size`.

#### 6.13s Block-18 cam split — true NVlabs BothTriton does **not** use PostUCPERenorm ✅ 2026-05-30

The §6.13r follow-up probe was run on the RTX PRO 6000 target against
the 321-frame / 20-step controlled dumps at denoise steps 15 and 18.
Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/cam_internal_probe_321f_steps15_18_block18_current_v2.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps15_18_rerun_block18_after_internal.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps15_18_block18_no_post_ucpe_renorm.json`

The first internal split initially appeared to clear the block-18 cam
branch: native-on-official-attn-input `cam_raw` MAE was only `0.0023`
at step 15 and `0.0017` at step 18. That was a measurement trap. The
probe had overwritten NVlabs' real `_forward_cam_branch` with a Python
copy of `sana_gdn_camctrl_blocks.py::BidirectionalGDNUCPESinglePathLiteLA`,
so it compared native against the **Python baseline**, not against the
actual production class.

The real SANA-WM config maps
`BidirectionalGDNUCPESinglePathLiteLABothTriton` to
`diffusion/model/nets/sana_gdn_blocks_triton.py`. That path calls
`cam_prep_func(...)`, which emits raw UCPE-transformed `q/k/v` and
`inflation_sq`, then feeds them directly to
`cam_scan_bidi_chunkwise(...)`. It does **not** call
`_downscale_to_reference_rms`. Native still had the §6.12 Python
`PostUCPERenorm` shrink in the GDN cam path, so for long sequences it
was systematically under-driving the cam branch relative to NVlabs
BothTriton.

Code change applied in native:

- GDN cam branch: remove PostUCPERenorm before `_bidi_single_path`; compute
  beta discounting from the raw post-UCPE K inflation, matching
  `cam_prep_func`.
- Softmax UCPE cam branch: unchanged; it still follows the Python softmax
  path where `_stabilize_cam_transforms` is part of the reference.
- UCPE `patch_size` default restored to `(1, 1, 1)`; the loaded NVlabs
  P1 model reports `(1, 1, 1)` for `model.patch_size`,
  `x_embedder.patch_size`, and `block18.attn.patch_size`.

**Same-attn-input block-18 result (`SANA_WM_CAM_TRITON=1`):**

| Step | Stage | Before no-PostUCPERenorm fix | After fix |
|---:|---|---:|---:|
| 15 | `cam_raw` MAE | 0.182811 | **0.004031** |
| 15 | `cam_contrib` MAE | 0.078081 | **0.002465** |
| 15 | `pre_proj` MAE | 0.151536 | **0.013675** |
| 15 | `attn_out` MAE | 2.251650 | **0.268524** |
| 18 | `cam_raw` MAE | 0.133699 | **0.003118** |
| 18 | `cam_contrib` MAE | 0.055732 | **0.002120** |
| 18 | `pre_proj` MAE | 0.099670 | **0.014466** |
| 18 | `attn_out` MAE | 1.452389 | **0.283987** |

The same probe's whole-model `noise_pred` residual also improved:

| Step | `noise_pred` MAE before | `noise_pred` MAE after |
|---:|---:|---:|
| 15 | 0.04052 | **0.03184** |
| 18 | 0.03394 | **0.02989** |

**Full 321-frame / 20-step controlled trajectory rerun after the fix:**

| Metric | Before §6.13s (§6.13n) | After §6.13s |
|---|---:|---:|
| final generated latent MAE | 0.4022 | **0.2994** |
| final generated latent cos | 0.7575 | **0.8637** |
| final generated latent RMSE | n/a | 0.4060 |
| final native/ref norm | n/a | 1642.54 / 1658.74 |

The trajectory no longer collapses as badly, but it is not closed.
Generated latent cosine now stays high through step 15 (`0.9954`), then
falls through the low-sigma tail: step 17 `0.9767`, step 18 `0.9403`,
step 19 `0.8689`, final `0.8637`. The late-step `noise_pred` residual
is still material (`noise_gen_cos=0.8673`, MAE `0.3699` at step 18;
`noise_gen_cos=0.8230`, MAE `0.3491` at step 19).

**Interpretation.**

- The previous block-18 `cam_raw` gap was not camera geometry, softmax
  UCPE, chunk size, or the cam scan kernel itself. It was a wrong
  native contract: matching the Python camera branch instead of the
  loaded NVlabs BothTriton branch.
- This reinforces §6.13r: `chunk_size` is not the root cause. The
  production path is full bidirectional; the relevant difference was
  whether raw UCPE-transformed Q/K/V enter the scan.
- The remaining full-forward block-18 error is now dominated by upstream
  `attn_in` drift rather than block-18's isolated cam branch. Since the
  full 321f rerun improves but still misses, the next probe should split
  earlier GDN blocks and the final layer under the same "true BothTriton,
  no Python wrapper" rule, focusing on steps 17-19.

#### 6.13t True-production late-step probe — block math mostly clears, engine path remains open ⚠️ 2026-05-30

The follow-up probe re-ran the §6.13s split at the low-sigma tail
(`321f / 20-step`, denoise steps 17-19) and found a second probe
artifact: the official-side hook had been calling
`BidirectionalGDN.forward(...)`, which bypasses NVlabs'
production `BidirectionalGDNTriton.forward(...)` main branch. That
inflated apparent `main_raw` mismatches. The corrected probe lets
NVlabs use its production Triton main path and only captures the
sub-stage tensors.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/lite_attn_probe_321f_steps17_18_19_block0_official_triton_main.json`
- `/root/autodl-tmp/stage1_longseq_probe/lite_attn_probe_321f_steps17_18_19_block18_official_triton_main.json`

**Same-attn-input isolated block results.**

| Block | Step window | `main_raw` MAE | `cam_raw` MAE | `attn_out` MAE |
|---:|---|---:|---:|---:|
| 0 | 17-19 | `1.3e-7`-`2.3e-7` | `3.4e-6`-`3.1e-5` | `0.0248`-`0.0281` |
| 18 | 17-19 | `1.0e-6`-`2.0e-6` | `0.0035`-`0.0055` | `0.0946`-`0.1367` |

Under the same official `attn_in`, the block-local implementation is
therefore much closer than the earlier instrumentation suggested.
The direct same-latent transformer output at the same late steps is
also close: generated-frame `noise_pred` MAE is `0.0283`-`0.0291`,
with sampled cosine `0.99898`-`0.99939`.

**Full native path through those same blocks still drifts.**

| Block | Signal | Step 17 | Step 18 | Step 19 |
|---:|---|---:|---:|---:|
| 0 | full-path `attn_in` MAE | 0.00025 | 0.00144 | 0.00193 |
| 0 | full-path `attn_out` MAE | 0.126 | 0.266 | 0.394 |
| 18 | full-path `attn_in` MAE | 0.0233 | 0.0213 | 0.0220 |
| 18 | full-path `attn_out` MAE | 4.15 | 3.60 | 4.94 |

That means the late block-18 explosion is mostly the accumulated
upstream trajectory/input difference arriving at block 18, not a
fresh local cam/GDN bug in block 18.

**Engine-path controlled rerun.**

Re-running the controlled 321-frame / 20-step denoising script through
the native Omni engine still produced a large final generated-latent
gap: MAE `0.40845`, RMSE `0.54426`, cosine `0.74914`. The per-step
generated `noise_pred` MAE starts at `0.1605` even at step 0 and grows
to `~0.47`-`0.50` near steps 17-19. Toggling the diagnostic flags
`SANA_WM_CAM_TRITON=1`, `VLLM_OMNI_SANA_WM_DISABLE_VLLM_OPS=1`, and
shell-level `PYTHONPATH` did not change that trajectory.

This is the most important new localization:

- same-latent standalone transformer probing at late steps is close
  (`noise_pred` MAE `~0.029`, cosine `>0.9989`);
- native engine controlled denoising is already off at step 0
  (`noise_pred` MAE `0.1605`) and ends at cosine `0.749`;
- therefore the remaining lead is **engine pipeline inputs/dispatch vs
  standalone transformer inputs**, not scheduler math or isolated
  block-0/block-18 GDN math.

Next probe: compare engine step-0 tensors against the standalone
direct-transformer call on exactly the same `latent_in`, prompt
embeds, per-frame timestep, `camera_conditions`, `data_info`, raymap
/ chunk-plucker, and attention masks. Do this with a small metadata
dump rather than `torch.load` of the full 321-frame artifact, because
the full dump is too large for interactive inspection.


#### 6.13u Engine-vs-standalone diff probe — dump landed, recall blocked on weight loading ⚠️ 2026-05-30 night

Goal: implement the §6.13t follow-up — capture every transformer
input on the engine path at step 0, re-call `transformer.forward(...)`
standalone with byte-identical inputs, diff the resulting `noise_pred`.
A "match" implicates the inputs upstream; a "miss" implicates the
engine dispatch (vLLM custom ops, autocast, async, CUDA graph).

**What landed.**

- New env-gated hook `SANA_WM_DUMP_ENGINE_STEP0_INPUTS` in
  `_run_native_smoke_backend` (probe worktree
  `probe-vllm-omni-feat-sana-wm`'s `pipeline_sana_wm.py`) saves the
  full set of `transformer.forward(...)` arguments + the computed
  `noise_pred` to a single `.pt` file at Stage-1 step 0.
- Standalone probe `/tmp/probe_engine_dump_and_recall.py` (two phases:
  `PHASE=dump` runs the engine, `PHASE=recall` loads the dump and
  calls `transformer.forward(...)` directly).
- Artifact: `/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs.pt`
  (23 MB).

**Engine-side dump succeeded:**

| Key | Shape | Dtype | Norm |
|---|---|---|---:|
| `latents` | `(1, 128, 41, 22, 40)` | bf16 | `2141.12` |
| `model_timestep` | `(1, 1, 41)` | fp32 | `6324.55` |
| `prompt_embeds` | `(1, 300, 2304)` | bf16 | `2224.50` |
| `plucker` | `(48, 41, 22, 40)` | bf16 | `2865.78` |
| `raymap` | `(41, 20)` | bf16 | `242.09` |
| `spatial_raymap` | `(3, 41, 22, 40)` | bf16 | `189.93` |
| `condition_mask` | `(1, 1, 41, 1, 1)` | bf16 | `1.00` |
| `noise_pred` (engine) | `(1, 128, 41, 22, 40)` | bf16 | `2259.45` |

**Standalone recall failed cleanly — weights not loaded.**

| Field | Engine | Standalone | Notes |
|---|---:|---:|---|
| `noise_pred` cos | — | `+0.006` | near-orthogonal |
| `noise_pred` MAE | — | `0.948` | ~ full magnitude |
| `noise_pred` norm | `2259` | `1208` | 1.87× smaller |
| frame 0 (cond) cos | — | `+0.047` | also near zero |
| generated only cos | — | `+0.005` | broadly uncorrelated |

A near-zero cosine combined with the 1.87× norm mismatch is the
signature of "model weights are not the same instance" rather than
"dispatch noise". Inspection of the standalone instantiation confirms
the cause:

- `SanaWmPipeline(od_config=...)` constructor does **not** load
  checkpoint weights — Omni's `generate(...)` chain drives loading
  via `DiffusersPipelineLoader` separately. Re-instantiating the
  pipeline outside that chain leaves the transformer with
  default-init / lazily-materialised weights.
- Once that limitation surfaced, the standalone call also had to be
  wrapped in `set_current_vllm_config(VllmConfig())` so that
  `VllmRMSNorm` and other CustomOps could dispatch a `forward` at
  materialisation time.

**Probe iteration log (failures and what was learned).**

| Iteration | Approach | Outcome | Fix learned |
|---|---|---|---|
| v1 | `__init__` monkey-patch to register `register_forward_pre_hook` | Hook never fired | Transformer instance lives in the Omni worker process; main-process patches do not reach it |
| v2 | Class-level `forward` wrap (instead of instance hook) | Still didn't fire | Same worker-process issue + when `OFFICIAL_REPO` was set, `_run_native_smoke_backend` was bypassed entirely |
| v3 | Edit `pipeline_sana_wm.py` source in probe worktree to add the dump hook | Hook still didn't fire | With `OFFICIAL_REPO` set, the dispatch chooses `native_backend.py` (NVlabs in-process) rather than `_run_native_smoke_backend` |
| v4 | Pop `OFFICIAL_REPO`, accept cam-triton Python fallback (§6.13s fix lives in Python path) | Dump worked. Recall crashed on `VllmRMSNorm` CustomOp instantiation outside vLLM config context | Wrap standalone in `set_current_vllm_config(VllmConfig())` |
| v5 | Added vLLM-config context | Recall completes but cos ≈ 0 — standalone has wrong weights | Need explicit `DiffusersPipelineLoader.load_weights(pipe)` (or equivalent) after instantiation |

**Open work needed to close §6.13t.**

1. Load weights into the standalone pipeline using the same loader the
   engine uses. Options:
   * `DiffusersPipelineLoader(od_config).load_weights(pipe)` inside the
     `set_current_vllm_config` context (this was the v5 attempt; needs
     to actually verify the post-load weight norms match the checkpoint
     before re-running the forward).
   * Or hand-load via `safetensors.torch.load_file(...) +
     transformer.load_weights(...)`, which avoids the loader plumbing
     but needs to deal with the lazy `materialize()` step (materialise
     first by running one dummy forward, then load).
   * Or adapt one of the working probes already on remote that did
     successfully instantiate a standalone vLLM-native transformer —
     `run_attn_split_probe.py` / `run_teacher_forced_native_on_official_trace.py`
     in `/root/autodl-tmp/stage1_longseq_probe/`. These have already
     solved the weight-loading problem; the cheapest path is to lift
     their setup verbatim.
2. Sanity check the standalone transformer's block-0 weight norms match
   the checkpoint (`block 0 qkv.weight ≈ 720`,
   `block 0 mlp.point_conv.conv.weight ≈ 892`) before re-running the
   forward and comparing.
3. Once weights match, re-run the diff. The expected per-side cosines:
   * cos ≈ 1.0 → engine call site is fine, the source of §6.13t's
     `0.16` step-0 residual is one of the input tensors. Diff each
     against the same-source NVlabs reference to find which.
   * cos < 0.999 → engine dispatch (vLLM CustomOps / autocast /
     async / CUDA graph) is the source.

**Why we paused.** Five probe iterations in one session burned ~12
GPU runs without producing a usable engine-vs-standalone number, and
SeetaCloud SSH dropped twice mid-run. The remaining work is mechanical
(weight loading) but each iteration costs another ~80 s GPU run plus
the SSH instability premium. Documenting the partial state here so the
next session can either lift the existing remote probe weight-loader
or invest in the loader plumbing first.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs.pt`
  (23 MB; engine step-0 transformer inputs + `noise_pred`)
- `/tmp/probe_engine_dump_and_recall.py` (PHASE=dump|recall|all)
- Probe-worktree patch in
  `/root/autodl-tmp/probe-vllm-omni-feat-sana-wm/vllm_omni/diffusion/models/sana_wm/pipeline_sana_wm.py`
  (env-gated; default off)

#### 6.13v Engine-entered SANA-WM input parity — call site cleared, prompt bug fixed ⚠️ 2026-06-01

Follow-up to §6.13u using a corrected standalone recall harness:

- load standalone weights with the same `DiffusersPipelineLoader` path as
  the worker;
- wrap recall in both `set_forward_context(...)` and
  `set_current_vllm_config(...)`;
- verify stored and materialized block-0 weight norms before comparing;
- compare the dumped engine-entered `transformer.forward(...)` inputs
  against a standalone direct call.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs_recall_9f.pt`
- `/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs.pt`
- `/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs_promptfix_321f.pt`
- `/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs_promptfix_camtriton_321f.pt`
- `/root/autodl-tmp/stage1_longseq_probe/engine_step0_inputs_promptfix_camtriton_skipabs_321f.pt`
- local/remote probe script:
  `tools/sana_wm_engine_input_parity_probe.py`

**Engine-entered call-site result.**

| Dump | Scope | Cosine | MAE | Notes |
|---|---|---:|---:|---|
| 9f / 20-step | generated | 0.999948 | 0.007844 | fixed loader + context |
| 321f / 20-step | generated | 0.999911 | 0.006345 | old prompt dump, recall only |
| 321f / 20-step, prompt fixed + cam-triton import | generated | 0.999894 | 0.006989 | `PYTHONPATH` exposes NVlabs Triton without setting `OFFICIAL_REPO` |

This clears the generic engine/worker call site. Given byte-identical
inputs and loaded weights, standalone direct `transformer.forward(...)`
reproduces the engine-entered output closely. The remaining Stage-1
gap is therefore not a reusable engine bug; it is either native
SANA-WM transformer math or model-side input semantics before/inside
the transformer.

**Real input bug found and fixed: Stage-1 prompt encoding.**

Before this section, native `_native_smoke_prompt_embeds(...)` used
Gemma's default left padding and encoded the full chi-prompt+user text
directly to `model_max_length=300`. NVlabs does two extra things:

1. `tokenizer.padding_side = "right"`;
2. when `chi_prompt` is present, encode with
   `len(tokenizer.encode(chi_prompt)) + model_max_length - 2`, then
   keep `[BOS] + last 299 tokens`.

Measured before the fix, native engine-entered prompt embeddings vs
NVlabs controlled step-0 prompt were badly misaligned:

| Signal | Cosine | MAE | Native norm | NVlabs norm |
|---|---:|---:|---:|---:|
| `prompt_embeds` before fix | 0.1503 | 2.4107 | 2224.50 | 2280.95 |

After porting the NVlabs right-padding + selection contract, the same
prompt embedding comparison is bit-exact on the RTX PRO 6000 target:

| Signal | Cosine | MAE | Native norm | NVlabs norm |
|---|---:|---:|---:|---:|
| `prompt_embeds` after fix | 1.0000 | 0.0000 | 2280.90 | 2280.90 |

The patch also gates `spatial_raymap` / absmap injection so it is
skipped when `use_chunk_plucker_input` or `use_chunk_plucker_post_attn`
is enabled, matching NVlabs' `_skip_absmap` branch. In the current
checkpoint this did not move step-0 `noise_pred`, which is consistent
with the raymap embedder being inactive/zero-weighted under the
chunk-plucker post-attention configuration.

**Input parity after the prompt fix.**

Comparing the prompt-fixed engine-entered dump to the existing NVlabs
controlled step-0 dump:

| Input / output | Cosine | MAE | Notes |
|---|---:|---:|---|
| `latents` | ~0.99991 | 0.0000 | byte-identical; cosine print limited by fp32 accumulation |
| `model_timestep` | 1.0000 | 0.0000 | `(1, 1, 41)`, frame 0 at 0 |
| `prompt_embeds` | ~0.99996 | 0.0000 | fixed |
| `condition_mask` | 1.0000 | 0.0000 | broadcasted to official token mask |
| `chunk_plucker` | ~1.0000 | 0.0000 | checked via `build_official_camera` vs `build_native_camera` |
| `raymap` | 1.0000 | 0.0000 | checked via same camera builders |
| `noise_pred` | 0.98874 | 0.12390 | still open |

So §6.13u's "engine path" wording should be read narrowly now:
engine-entered call-site dispatch is not the root cause. The residual
is native transformer-body parity under aligned inputs. Since the
residual is already visible at step 0, the next probe should split
early-block activations under this prompt-fixed, cam-triton-importable
setup rather than continuing to diff generic engine layers.

#### 6.13w Prompt-fixed step-0 transformer activation split — first material divergence is block-0 attention projection output ⚠️ 2026-06-01

Follow-up to §6.13v. The probe reuses aligned step-0 latents,
per-frame timestep, prompt embeddings, and camera tensors, then compares
NVlabs vs native activations inside the Stage-1 transformer.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_blocks0_5_tritonmain.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_block0_projcheck.json`
- local/remote probe script:
  `tools/sana_wm_attn_split_probe.py`

Important instrumentation correction: the first block0-5 split used the
older official-side hook that calls `BidirectionalGDN.forward(...)`,
bypassing NVlabs' production Triton main path. That reproduces the
§6.13t artifact and inflates block-0 `main_raw` to MAE `0.0075`. The
accepted run below calls `BidirectionalGDNTriton.forward(...)` for
NVlabs non-softmax blocks, matching the production path.

**Step-0 block-0 isolated attention split, production Triton main path.**

| Stage | Cosine | MAE | Max abs | Notes |
|---|---:|---:|---:|---|
| `attn_in` | 1.000000 | 0.000000 | 0.0000 | block input is exact |
| `main_raw` | 1.000000 | 0.0000005 | 0.0625 | production Triton main path clears |
| `cam_raw` | 0.999822 | 0.0000126 | 0.4063 | residual exists but tiny |
| `cam_contrib` | 0.999931 | 0.0000197 | 0.0146 | after `out_proj_cam` |
| `combined` | 1.000000 | 0.0000190 | 0.0625 | `main_raw + cam_contrib` |
| `output_gate` | 1.000000 | 0.000000 | 0.0000 | exact |
| `pre_proj` | 1.000000 | 0.0000599 | 0.2500 | tiny drift before output projection |
| `attn_out` | 0.999998 | 0.0227748 | 4.0000 | first material amplification |

The full forward shows the same first jump and then accumulation:

| Signal | Cosine | MAE | Notes |
|---|---:|---:|---|
| block-0 `attn_out` | 0.999998 | 0.02277 | isolated and full are identical |
| block-1 `attn_in` | 0.99221 | 0.01208 | block-0 residual has entered the next block |
| step-0 `noise_pred` | 0.98881 | 0.12383 | same residual scale as §6.13v |

**Projection check.**

The block-0 output projection itself is not the cause:

| Check | MAE | Result |
|---|---:|---|
| native `proj.weight` vs NVlabs `proj.weight` | 0.0 | exact |
| native `proj.bias` vs NVlabs `proj.bias` | 0.0 | exact |
| official `F.linear(official_pre_proj)` vs official module output | 0.0 | exact |
| native `F.linear(official_pre_proj)` vs official `F.linear(official_pre_proj)` | 0.0 | exact |
| native module vs native `F.linear(official_pre_proj)` | 0.0 | exact |

Setting `VLLM_OMNI_SANA_WM_DISABLE_VLLM_OPS=1` does not change the
block-0 numbers, so this is not a vLLM `RowParallelLinear` dispatch
issue. With identical `pre_proj`, both sides produce identical
`attn_out`. The observed `0.0228` output delta is therefore the
projection of the tiny upstream `pre_proj` delta, not a projection
weight or implementation mismatch.

**Current localisation.**

The first non-zero difference is already inside block-0 attention raw
math at very small magnitude:

- GDN main branch: `main_raw` MAE `5.1e-7`;
- camera branch: `cam_raw` MAE `1.26e-5`, `cam_contrib` MAE `1.97e-5`;
- shared gate is exact;
- output projection is exact for identical input.

This points at accumulated bf16/fused-kernel numeric drift between the
native and NVlabs GDN/camera kernels, not a remaining scheduler, prompt,
camera tensor, engine dispatch, or linear weight-loading bug. The next
useful probe is either:

1. run a block-0 raw-branch micro-split (`qkv`, `conv_k`, `q_norm/k_norm`,
   `beta/decay`, Triton scan output, cam prep/scan) to identify which
   small raw delta appears first; or
2. quantify whether these tiny raw deltas alone explain final
   `noise_pred` MAE by teacher-forcing official block-0 `pre_proj` or
   `attn_out` into the native trajectory.

#### 6.13x Deep step-0 probe — dominant error was missing prompt padding mask in native cross-attn ✅ 2026-06-01

Follow-up to §6.13w. The initial self-attention split was correct but
incomplete: block0 attention was the first non-zero tensor delta, yet
teacher-forcing official block0 `attn_out` into native only improved
step-0 `noise_pred` MAE `0.12383 -> 0.12320`. That meant the large
trajectory error had to be regenerated downstream.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_block0_rawmicro.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_block0_cammicro.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_block0_official_cam_prep_ablation.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_block0_stage_official_main_cam.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_masked_block0_stage_official_main_cam.json`
- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps0_promptfix_step0_masked_baseline_native.json`

**Raw main GDN micro-split.** Under identical block0 input:

| Stage | MAE | Result |
|---|---:|---|
| `q_pre/k_pre/v_pre` | 0.0 | exact |
| `k_conv` | 0.0 | exact |
| `q_inv_rms/k_inv_rms` | 0.0 | exact |
| `beta/decay` | 0.0 | exact |
| `main_raw` | `5.14e-7` | first non-zero main-GDN fused-scan residual |

**Camera branch micro-split.** Camera Q/K/V projection and K-conv are exact,
but native Python UCPE prep diverges from NVlabs `cam_prep_func`:

| Stage | MAE | Notes |
|---|---:|---|
| `cam_q_raw/cam_k_raw/cam_v_raw` | 0.0 | exact |
| `cam_v_trans` | `0.00368` | first large camera-prep delta |
| `cam_inflation` | `0.00663` | beta-discount input differs |
| `cam_scan_out` | `8.58e-6` | after shared NVlabs scan |
| `cam_raw` | `1.26e-5` | after inverse UCPE |

A native-side ablation that keeps native weights/inputs but uses NVlabs
`cam_prep_func + cam_scan_bidi_chunkwise + apply_fn_o` closes the camera
branch: `cam_raw` MAE `1.31e-9`, `cam_contrib` MAE `5.82e-9`.
This is a real implementation mismatch, but not the dominant step-0 error:
the full `noise_pred` MAE stayed around `0.124`.

**Block-stage split found the dominant root.** With both main and camera
attention ablated to NVlabs functions, block0 attention was nearly closed,
but the residual reappeared after cross-attn:

| Stage | MAE before mask fix | MAE after mask fix | Notes |
|---|---:|---:|---|
| `post_attn_residual` | `0.000317` | `0.000317` | attention almost closed |
| `post_cross_attn` | `0.37017` | `0.01542` | prompt mask was missing in native |
| `x_mlp_in` | `0.00138` | `5.57e-5` | layernorm after cross-attn recovers |
| `block_output` | `0.74487` | `0.03181` | block0 no longer explodes |

Root cause: NVlabs passes the text padding mask into
`MultiHeadCrossAttention.forward(x, y, mask=mask)`. Native
`SanaWmCrossAttention.forward(...)` ignored the prompt attention mask and
therefore attended to all padded tokens in the fixed 300-token Gemma context.

Fix landed in the native implementation:

- `SanaWmCrossAttention.forward(..., attention_mask=...)` now applies the
  NVlabs SDPA mask semantics.
- `SanaWmBlock` and `SanaWmTransformer3DModel.forward` thread
  `encoder_attention_mask`.
- `SanaWmPipeline._native_smoke_prompt_embeds` records the tokenizer
  `attention_mask`, including the chi-prompt window selection.
- CUDA graph helper keys/captures now include the prompt mask.

**Step-0 result after the mask fix.**

| Configuration | `noise_pred` MAE | RMSE | Cosine |
|---|---:|---:|---:|
| pre-§6.13x baseline | `0.12383` | `0.15627` | `0.988805` |
| official main+cam ablation, no prompt mask | `0.12309` | `0.15536` | `0.988934` |
| official main+cam ablation + prompt mask | `0.01280` | `0.01663` | `0.999874` |
| real native path + prompt mask | `0.01211` | `0.01576` | `0.999887` |

So the remaining 321-frame Stage-1 gap is no longer a scheduler,
engine-dispatch, weight-loading, prompt-tokenization, or cross-attn mask
semantic bug. The next target is the smaller numeric tail:

- port/use NVlabs `cam_prep_func` semantics natively instead of the Python
  UCPE prep path;
- continue reducing the main GDN fused-scan residual (`main_raw`
  `5.1e-7`, amplified by the output projection);
- rerun 321f 20/60-step controlled denoise after the prompt-mask fix
  (full e2e gate rerun is recorded in §6.14k).

### 6.14 Refiner/VAE/RGB Photometric Chain Probe — Native Refiner Contract Dominates 🔴 2026-05-29

After §6.13m tightened Stage-1 latent parity, the next question was
whether the remaining e2e MAE/PSNR/SSIM behavior was caused by VAE
decode, simple RGB photometric bias, official-refiner amplification of
small Stage-1 drift, or the native in-process refiner algorithm.

**Hardware note.** The 5090 32GB instance is sufficient for Stage-1
probes, but not for LTX-2 refiner: even loading the refiner transformer
alone OOMs at ~31GB. The chain probes below were run on the restarted
RTX PRO 6000 98GB instance using the same 5090-synced worktree. Probe
artifacts are under `/root/autodl-tmp/refiner_photometric_probe/`:

- `official_refiner_vae_photometric_metrics.json`
- `prompt_connector_compare_metrics.json`
- `native_refiner_algorithm_metrics.json`
- `native_refiner_algorithm_rgb_metrics.json`

#### 6.14a VAE/RGB decode is not the dominant error source

Using the same official VAE decode on controlled Stage-1 latents:

| Comparison | Scope | MAE | PSNR | SSIM-Y (global) |
|---|---|---:|---:|---:|
| native Stage-1 RGB vs official Stage-1 RGB | all frames | 2.90 | 35.96 | 0.9946 |
| native Stage-1 RGB vs official Stage-1 RGB | generated only | 3.20 | 35.46 | 0.9777 |
| after per-channel affine fit | generated only | 2.60 | 37.12 | 0.9779 |

This makes pure VAE decode and simple brightness/contrast drift a
secondary effect. Stage-1 direct RGB is visually/photometrically close
once decoded by the same VAE.

#### 6.14b Official refiner amplifies the remaining Stage-1 drift

Feeding both Stage-1 latents into the same NVlabs official refiner and
then the same official VAE decode gives:

| Comparison | Scope | MAE | PSNR | SSIM-Y (global) |
|---|---|---:|---:|---:|
| official_refiner(native Stage-1) vs official_refiner(official Stage-1) | generated only | 25.86 | 16.29 | 0.4426 |
| after per-channel affine fit | generated only | 23.70 | 17.49 | 0.3569 |

Latent-side, the generated-frame comparison is MAE `0.7266`, cosine
`0.6049`. The same Stage-1 residual that is only RGB MAE ~3 before
refiner becomes RGB MAE ~26 after the official refiner. A simple
photometric correction does not recover structure; it actually lowers
SSIM-Y. This is refiner sensitivity, not just RGB calibration.

> **Reconciliation with §6.13m.** §6.13m reported generated-frame
> Stage-1 latent MAE `0.1266` / cosine `0.987` — that figure was measured
> under a **controlled-input probe** where NVlabs' step-0 latent and
> prompt embeddings were injected into the native pipeline. The
> `0.7266` / `0.6049` here is from the **full e2e path** (native
> generates its own initial noise and runs its own prompt-encoder
> chain), which adds upstream divergence that the controlled probe
> deliberately removes. Both numbers are correct for their respective
> setups; the ~5× gap between them is the answer to "what does the
> §6.13m improvement look like once upstream uncontrolled steps are
> included".

#### 6.14c Refiner prompt connector is effectively aligned

The native prompt-connector path was compared against NVlabs packed
Gemma hidden-state path before transformer denoising:

| Connector output | MAE | RMSE | Max | Cosine | Mask |
|---|---:|---:|---:|---:|---|
| video prompt embeds | 0.000607 | 0.00245 | 0.25 | ~1.000 | equal |
| audio prompt embeds | 0.001004 | 0.00281 | 0.375 | 0.9999 | equal |

This rules out the text/refiner connector as the source of the large
refiner-chain gap.

#### 6.14d Current native refiner algorithm is structurally non-equivalent

A probe-only native-like refiner run used the current in-process native
algorithm on the same controlled Stage-1 latents, with TP mocked to
single-card for offline comparison. Against NVlabs official refiner on
the **same official Stage-1 input**:

| Comparison | Scope | Latent MAE | Latent cos | RGB MAE | PSNR | SSIM-Y |
|---|---|---:|---:|---:|---:|---:|
| native-like refiner vs official refiner | generated only | 1.2622 | 0.4207 | 56.25 | 11.51 | 0.0087 |

The native-like output has much larger generated-frame magnitude
(`norm 565.35` vs `372.47`, `std 1.684` vs `1.110`). In contrast,
native-like refiner sensitivity to native-vs-official Stage-1 input is
small by comparison: latent generated-frame MAE `0.1924`, cosine
`0.9906`; RGB generated-frame MAE `3.01`, PSNR `31.81`, SSIM-Y
`0.8631`.

> **Direction-flip clue.** Stage-1 block-19 controlled-input probes
> in §6.13m showed native generated-frame norm was *below* NVlabs
> (`block-19 output ratio ≈ 0.908`). After the native refiner runs,
> the same generated frames come out *above* NVlabs
> (`norm 565 / 372 ≈ 1.52×`, `std 1.684 / 1.110 ≈ 1.52×`). The sign
> flips and the magnitude is amplified ~1.5×. That is consistent with
> a missing per-token / sink-current update — without it, the native
> refiner integrates over the wrong sigma table for the wrong tokens
> and overshoots in the opposite direction.

So the dominant refiner-side gap is not that Stage-1 is still slightly
off; it is that the native refiner denoising contract is currently not
NVlabs' contract.

The specific contract breaks in `pipeline_sana_wm_two_stages.py` are:

1. native refiner updates the packed full latent stream directly;
2. NVlabs splits `sink = z[:, :, :sink_size]` and `current = z[:, :, sink_size:]`;
3. NVlabs initializes current frames as `(1 - start_sigma) * current + start_sigma * eps` with seed 42;
4. NVlabs uses per-token timestep: sink tokens at `0`, current tokens at `sigma`;
5. NVlabs runs a video-only transformer path with sink/current streaming self-attention mask;
6. NVlabs predicts current `x0` and updates via velocity `(noisy_tokens - denoised) / sigma`;
7. native currently uses the public transformer forward with an audio stream and scalar timestep.

#### 6.14e Native refiner contract port — structure fixed, parity now limited by accumulated bf16 drift ⚠️ 2026-05-29

The native in-process refiner was updated to follow the NVlabs
`DiffusersLTX2Refiner` denoising contract:

1. split `sink` and `current` frames;
2. seed-42 current-frame noise initialization;
3. per-token refiner timestep with sink/context tokens at `0`;
4. video-only transformer path with sink/current streaming self-attention;
5. x0/velocity update loop;
6. terminal `0.0` sigma appended when the installed diffusers
   `STAGE_2_DISTILLED_SIGMA_VALUES` omits it.

Initial contract-port probe against the old `official_refiner_*`
artifact improved the old native-like baseline but exposed a probe
consistency issue:

| Probe | Reference | Generated latent MAE | Generated cos | Generated norm |
|---|---|---:|---:|---:|
| pre-port native-like | old official artifact | 1.2622 | 0.4207 | 565.35 vs 372.47 |
| contract port, 2 effective steps | old official artifact | 0.5473 | 0.7759 | 263.95 vs 372.47 |
| contract port, 3 steps + terminal zero | old official artifact | 0.7683 | 0.5630 | 370.60 vs 372.47 |

The 3-step run fixes the generated-frame magnitude, but comparison
against the old artifact gets worse because the saved prompt-connector
artifact and the old official-refiner artifact are not same-source.
Re-running the NVlabs official refiner manually with
`prompt_connector_compare.pt` gives a different "official" result:

| Comparison | Generated latent MAE | Generated cos | Note |
|---|---:|---:|---|
| official manual from `prompt_connector_compare.pt` vs old official artifact | 0.7754 | 0.5596 | baseline artifact mismatch |
| native contract port vs same-source official manual | 0.3589 | 0.8912 | true current native-vs-official gap |

Step-level same-source comparison:

| Stage | MAE | Cosine |
|---|---:|---:|
| initial noisy current | 0.0000 | ~1.0000 |
| step-0 predicted x0 | 0.1825 | 0.9703 |
| after step-0 update | 0.0370 | 0.9976 |
| step-1 predicted x0 | 0.3723 | 0.8888 |
| after step-1 update | 0.1636 | 0.9538 |
| final / after step-2 | 0.3589 | 0.8912 |

Block-level isolation rules out gross structural or weight-load errors.
All 3126 native refiner parameters load. Feeding the same official block
input into native and NVlabs blocks gives close local agreement:

| Same official input | Sub-stage | MAE | Cosine |
|---|---|---:|---:|
| block 25 | after self-attn | 0.0023 | ~1.000 |
| block 25 | after ff | 0.0044 | ~1.000 |
| block 37 | after self-attn | 0.0042 | ~1.000 |
| block 37 | after ff | 0.0074 | ~1.000 |

Full-chain e2e was re-run after the refiner contract port with the
current 9-frame reference-alignment harness. The harness currently
hard-codes Stage-1 `num_inference_steps=1`; Stage-2 was run with
`REFINER_STEPS=3` (native in-process refiner vs NVlabs bridge):

| Configuration | MAE | PSNR | SSIM-Y |
|---|---:|---:|---:|
| §6.13e full-chain baseline (`3+3` probe config) | 47.35 | 13.14 | +0.025 |
| §6.13l native scheduler exposed drift (`3+3` probe config) | 54.21 | 11.46 | -0.002 |
| **§6.14e native refiner contract port** | **22.55** | **17.72** | **+0.587** |

Because the earlier table rows were collected under a `3+3` probe
configuration, treat this as a strong smoke/alignment result rather
than a strict apples-to-apples delta. It still confirms the refiner
contract port moved the full decoded output into a much more correlated
regime; a follow-up should add a Stage-1 step env override to the e2e
harness and re-run exact `3+3` and normal-step comparisons.

So the remaining gap is not scheduler, sink/current masking, prompt
connector, or missing weights. It is accumulated bf16 numerical drift
between the vLLM-native LTX-2 layers and diffusers' reference layers
across 48 refiner blocks, then amplified by the 3-step distilled
refiner loop.

Artifacts:

- `/root/autodl-tmp/refiner_photometric_probe/native_refiner_contract_fix_steps3_metrics.json`
- `/root/autodl-tmp/refiner_photometric_probe/official_refiner_step_dump.pt`
- `/root/autodl-tmp/refiner_photometric_probe/native_refiner_step_dump_official_prompt_sdpa.pt`
- `/root/autodl-tmp/refiner_photometric_probe/native_vs_official_step_dump_sdpa_metrics.json`
- `/root/autodl-tmp/refiner_photometric_probe/native_vs_official_block_detail_25_37_metrics.json`

#### 6.14f Updated fix order

Do **not** chase RGB photometric tuning first. The next correctness work
is now narrower:

1. regenerate the official refiner baseline and prompt-connector dump in
   one run so acceptance compares same-source artifacts;
2. decide whether strict `cos >= 0.98` parity requires a
   diffusers-exact refiner fallback, or a refiner-specific torch-linear
   layer stack instead of vLLM parallel layers;
3. only after same-source latent parity clears should VAE/RGB
   photometric checks become actionable again.

Updated acceptance target: compare against a same-source official
refiner run, not the stale `official_refiner_official_stage1.pt`
artifact. Current same-source generated-frame latent is MAE `0.3589` /
cos `0.8912`; the target remains cos `>=0.98`, MAE `<=0.20`.

#### 6.14g 321-frame full e2e gate — no 321-vs-9 degradation after VAE memory fix ✅ 2026-05-30

The requested e2e gate was run on the RTX PRO 6000 target using the
controllable full-chain path:

- reference: NVlabs in-process backend (`SanaWmTwoStagesPipeline`,
  official repo path set, CLI not forced);
- candidate: vLLM-Omni native Stage-1 + native in-process refiner/VAE;
- Stage-2 refiner steps: `3`;
- prompt/image/intrinsics fixed to the existing e2e harness;
- long trajectory action fixed to `w-(num_frames-1)` so the 321-frame
  request actually has 321 camera poses.

Two harness issues were found before the valid run:

1. The first 321-frame attempt reused `action="w-16"`. That action only
   expands to 17 camera poses. NVlabs therefore returned a 16-frame
   video for the nominal 321-frame request, while native built 41 latent
   frames and failed with `camera_conditions frames 3 != latent frames 41`.
   The gate runner now scales the action duration with `num_frames`.
2. Native VAE decode initially OOMed on the 321-frame path. NVlabs enables
   LTX-2 VAE tiling and framewise decoding during VAE construction; native
   `_ensure_vae()` did not. Native now calls `enable_tiling()` and sets
   `use_framewise_encoding/use_framewise_decoding`, matching NVlabs'
   memory-mode contract. This is not an OOM-only benchmark trick; it is
   part of the upstream LTX-2 VAE runtime contract that native had failed
   to reproduce. After this change, 321-frame native decode completes on
   the 98GB card.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_321_vs_9_wmatched/e2e_gate_summary.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_321_vs_9_wmatched/metrics_9f_20s.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_321_vs_9_wmatched/metrics_321f_20s.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_321_vs_9_wmatched/run_321f_60s.log`

Full-chain e2e metrics, comparing native output to NVlabs common-prefix
frames. These rows were collected before the explicit frame-index fix
in §6.14i, so treat the absolute MAE/PSNR values as provisional until
the gate is rerun:

| Config | Common frames | MAE | PSNR | SSIM-Y | Generated MAE | Generated PSNR | Generated SSIM-Y |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9f / 20-step | 8 | 84.8630 | 8.71 | 0.0768 | 86.1813 | 8.51 | 0.0080 |
| 321f / 20-step | 320 | **73.7393** | **9.45** | **0.1299** | **73.8153** | **9.44** | **0.1295** |
| 321f / 60-step | 320 | 76.1839 | 9.12 | **0.1890** | 76.2746 | 9.11 | **0.1888** |

Gate decision: `321f/20` has no obvious degradation against `9f/20`
under the runner's generated-frame thresholds (`MAE <= max(1.2x,
+5)`, `PSNR drop <= 1 dB`, `SSIM-Y drop <= 0.05`), so `321f/60`
was run. The 60-step result improves structural similarity
(`SSIM-Y 0.1295 -> 0.1888`) but slightly worsens absolute error
(`MAE 73.8153 -> 76.2746`, `PSNR 9.44 -> 9.11`).

Interpretation:

- The earlier "321 degrades badly" conclusion does **not** hold for the
  current full e2e RGB gate after the BothTriton cam fix and VAE memory
  fix. Production-length e2e is at least as stable as the 9-frame gate.
- The absolute RGB metrics remain far from the target (`PSNR >= 30`,
  `SSIM-Y >= 0.93`), so this is not acceptance parity. It is a gating
  result: long sequence length is no longer the first-order e2e failure.
- The controlled Stage-1 latent residual in §6.13t still matters because
  it is a cleaner numeric signal than RGB MAE after the refiner/VAE stack.
  Keep probing the late-step Stage-1 residual and same-source refiner
  latent parity; do not tune RGB photometric offsets first.

#### 6.14h E2E harness controls — frame/action/step metrics fixed ✅ 2026-05-30

The local e2e reference-alignment harness was tightened so future RGB
benchmark rows can be reproduced without one-off runner scripts:

- `SANA_WM_E2E_STAGE1_STEPS` now controls Stage-1 denoise steps, with
  fallback to the old `SANA_WM_E2E_NUM_INFERENCE_STEPS` env and default
  `1`;
- default camera action now scales with the requested sequence length:
  `w-(num_frames-1)`, while `SANA_WM_E2E_ACTION` can still override it;
- `_assert_video_reference_alignment(...)` now returns the measured
  frame counts, common-prefix length, MAE, PSNR, and SSIM-Y;
- `SANA_WM_E2E_METRICS_JSON` writes those metrics plus
  `num_frames`, `stage1_steps`, `refiner_steps`, and `action`.
- `SANA_WM_E2E_NATIVE_SMOKE_MAX_TOKENS` now forwards to
  `sana_wm_native_smoke_max_tokens` so 321-frame native candidate runs
  can intentionally lift the default 4096-token smoke cap.

This closes the earlier harness ambiguity where a nominal `321f` run
could silently use a short `w-16` camera path and where the Stage-1
step count was hard-coded inside the test. The 321-frame e2e gate uses
`SANA_WM_E2E_NATIVE_SMOKE_MAX_TOKENS >= 36080`.

#### 6.14i E2E frame-index alignment — condition frame is now explicit ✅ 2026-05-30

The §6.14g RGB gate still compared common-prefix frames only. That is
unsafe for SANA-WM because the native decoded video can include the I2V
conditioning frame while the NVlabs mp4 bridge commonly starts at the
generated frames. In that case a common-prefix compare shifts the whole
video by one frame: native frame 0 (conditioning image) is compared to
NVlabs generated frame 1, and every generated frame is compared against
the wrong neighbor.

The local e2e harness now makes the frame-index contract explicit:

- default `SANA_WM_E2E_FRAME_ALIGNMENT=auto`;
- if native has one more frame than NVlabs, prepend the known first
  frame to the reference (`auto_prepend_reference_frame0`);
- if NVlabs has one more frame than native, prepend the known first
  frame to the prediction (`auto_prepend_prediction_frame0`);
- override modes exist for `none`, `prepend_reference_frame0`,
  `prepend_prediction_frame0`, `drop_prediction_frame0`, and
  `drop_reference_frame0`;
- metrics now include `frame0_mae`, `frame0_psnr`, `generated_mae`,
  `generated_psnr`, `generated_ssim_y`, and `frame_alignment`;
- `SANA_WM_E2E_FRAME0_MIN_PSNR` defaults to `30.0`.

This makes frame 0 a hard sanity check. If the alignment is correct,
frame-0 PSNR should be comfortably above 30 dB because it is the
conditioning frame / its VAE round-trip. If frame-0 PSNR falls below
20 dB, the e2e gate is almost certainly comparing different semantic
frame indices and the aggregate RGB MAE/PSNR rows are not actionable.

#### 6.14j Frame-aligned full e2e rerun — 321f/20 beats 9f/20, 60-step trades MAE for SSIM ✅ 2026-05-30

After the §6.14i frame-index fix, reran the controllable full-chain
gate on the RTX PRO 6000 target with the same backend split:

- reference: NVlabs in-process backend (`SanaWmTwoStagesPipeline`);
- candidate: vLLM-Omni native Stage-1 + native in-process refiner/VAE;
- Stage-2 refiner steps: `3`;
- default frame alignment: `SANA_WM_E2E_FRAME_ALIGNMENT=auto`;
- native token cap lifted with `SANA_WM_E2E_NATIVE_SMOKE_MAX_TOKENS=40000`
  for 321-frame runs.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_framealign_20260530/metrics_9f_20s_framealign_twostage.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_framealign_20260530/metrics_321f_20s_framealign_twostage.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_framealign_20260530/metrics_321f_60s_framealign_twostage.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_framealign_20260530/run_9f_20s_framealign_twostage.log`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_framealign_20260530/run_321f_20s_framealign_twostage.log`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_framealign_20260530/run_321f_60s_framealign_twostage.log`

Frame-aligned RGB metrics:

| Config | Pred frames | Ref frames | Alignment | Frame0 PSNR | Generated frames | Generated MAE | Generated PSNR | Generated SSIM-Y | Global MAE | Global PSNR | Global SSIM-Y |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9f / 20-step | 9 | 8 | `auto_prepend_reference_frame0` | 35.63 | 8 | 69.5170 | 8.72 | 0.1076 | 62.1609 | 9.23 | 0.1924 |
| 321f / 20-step | 321 | 320 | `auto_prepend_reference_frame0` | 35.31 | 320 | **41.7494** | **13.82** | 0.1947 | **41.6305** | **13.84** | 0.1970 |
| 321f / 60-step | 321 | 320 | `auto_prepend_reference_frame0` | 33.75 | 320 | 53.0073 | 11.69 | **0.2313** | 52.8565 | 11.70 | **0.2335** |

Interpretation:

- The earlier frame-index suspicion is confirmed. Native produced the
  conditioning frame plus generated frames, while the NVlabs mp4 bridge
  produced generated frames only. All three valid reruns needed
  `auto_prepend_reference_frame0`.
- Frame-0 sanity passes for all rows (`33.75-35.63 dB`), so the new
  aggregate rows are comparing the intended semantic frames. The old
  §6.14g common-prefix values are superseded for absolute MAE/PSNR.
- `321f/20` is materially better than `9f/20` on generated-frame MAE,
  PSNR, and SSIM-Y. The long-sequence degradation hypothesis is not
  supported by the frame-aligned e2e gate.
- `321f/60` improves structure (`generated SSIM-Y 0.1947 -> 0.2313`)
  but worsens absolute error (`generated MAE 41.7494 -> 53.0073`,
  `generated PSNR 13.82 -> 11.69`). More denoise steps are not a free
  RGB win yet; after §6.13v cleared engine-entered call-site parity,
  the next numeric probes should target prompt-fixed Stage-1 early-block
  transformer parity and same-source refiner latent parity before
  photometric tuning.

#### 6.14k Post-prompt-mask full e2e rerun — 321f/20 improves, 60-step still trades MAE for SSIM ✅ 2026-06-01

After the §6.13x prompt-mask fix, reran the full-chain gate on the RTX
PRO 6000 target for all four requested rows:

- reference: NVlabs in-process backend through `SanaWmTwoStagesPipeline`;
- candidate: vLLM-Omni native Stage-1 + native in-process refiner/VAE;
- Stage-2 refiner steps: `3`;
- `SANA_WM_CAM_TRITON=1`;
- `SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline`;
- native token cap lifted with `SANA_WM_E2E_NATIVE_SMOKE_MAX_TOKENS=40000`;
- default frame alignment: `SANA_WM_E2E_FRAME_ALIGNMENT=auto`.

One harness pitfall was caught before accepting the row: leaving
`SANA_WM_E2E_MODEL_CLASS` at its default `SanaWmPipeline` made both
the reference and candidate calls use the same NVlabs in-process backend,
producing a false `0.0` MAE. The valid rows below all use
`SanaWmTwoStagesPipeline`.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/metrics_9f_20s_promptmask_twostage.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/metrics_9f_60s_promptmask_twostage.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/metrics_321f_20s_promptmask_twostage.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/metrics_321f_60s_promptmask_twostage.json`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/run_9f_20s_promptmask_twostage.log`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/run_9f_60s_promptmask_twostage.log`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/run_321f_20s_promptmask_twostage.log`
- `/root/autodl-tmp/stage1_longseq_probe/e2e_gate_promptmask_20260601_twostage/run_321f_60s_promptmask_twostage.log`

Frame-aligned RGB metrics:

| Config | Pred frames | Ref frames | Alignment | Frame0 PSNR | Generated frames | Generated MAE | Generated PSNR | Generated SSIM-Y | Global MAE | Global PSNR | Global SSIM-Y |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9f / 20-step | 9 | 8 | `auto_prepend_reference_frame0` | 35.78 | 8 | 61.1686 | 9.75 | 0.1050 | 54.7352 | 10.26 | 0.1882 |
| 9f / 60-step | 9 | 8 | `auto_prepend_reference_frame0` | 35.98 | 8 | 68.1117 | 9.02 | 0.0346 | 60.8908 | 9.53 | 0.1271 |
| 321f / 20-step | 321 | 320 | `auto_prepend_reference_frame0` | 32.85 | 320 | **39.3226** | **14.30** | 0.2438 | **39.2151** | **14.31** | 0.2458 |
| 321f / 60-step | 321 | 320 | `auto_prepend_reference_frame0` | 33.25 | 320 | 49.3557 | 12.26 | **0.2968** | 49.2166 | 12.27 | **0.2987** |

Interpretation:

- Frame-0 sanity passes on every valid row (`32.85-35.98 dB`), so the
  frame-index contract is still correct.
- The prompt-mask fix improves the production-length 20-step full chain:
  `321f/20` generated MAE improves from `41.7494` (§6.14j) to
  `39.3226`, generated PSNR from `13.82` to `14.30`, and generated
  SSIM-Y from `0.1947` to `0.2438`.
- `321f/60` also improves versus §6.14j (`53.0073 -> 49.3557` generated
  MAE, `0.2313 -> 0.2968` generated SSIM-Y), but it still worsens MAE/PSNR
  versus `321f/20`. More Stage-1 steps remain a structure-vs-photometric
  tradeoff, not a strict RGB-parity win.
- `9f/60` is worse than `9f/20` across generated MAE/PSNR/SSIM-Y, so the
  60-step issue is not only a long-sequence phenomenon.
- The long-sequence degradation hypothesis remains unsupported: `321f/20`
  is substantially better than `9f/20` on generated MAE, PSNR, and SSIM-Y.
  Strict RGB acceptance is still far away; next numeric work should focus on
  controlled post-mask Stage-1 20/60-step latent parity and same-source
  Stage-2 refiner latent parity.

#### 6.14l Native cam-prep + post-mask Stage-1 trajectory replay — scheduler update ruled out ⚠️ 2026-06-01

Implemented a native PyTorch `cam_prep_func` port matching NVlabs'
`diffusion.model.ops.fused_cam_gdn.cam_prep_func` contract:

- full-channel Q/K RMSNorm, ReLU, and K scale;
- UCPE 4x4 ray projection on the first half of each head;
- interleaved-pair RoPE on the second half;
- V transform without Q/K RMSNorm/ReLU;
- K norm inflation reporting for dynamic beta discounting;
- output layout `(B, H, D, N)`, matching the fused kernel.

Validation:

- local compile / diff checks passed;
- remote RTX PRO 6000 unit parity against NVlabs CUDA `cam_prep_func`
  passed (`tests/diffusion/models/sana_wm/test_sana_wm_cam_prep.py`);
- camera-branch smoke and UCPE/blocker subset passed on GPU.

**9f / 20-step full-chain e2e after native cam-prep.**

Same harness contract as §6.14k:

- reference: NVlabs in-process backend through `SanaWmTwoStagesPipeline`;
- candidate: vLLM-Omni native Stage-1 + native in-process refiner/VAE;
- Stage-1 steps: `20`;
- Stage-2 refiner steps: `3`;
- `SANA_WM_CAM_TRITON=1`;
- `SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline`;
- frame alignment: `auto_prepend_reference_frame0`.

Artifact:

- `/root/autodl-tmp/stage1_longseq_probe/e2e_cam_prep_20260601_031344/metrics_9f_20s_cam_prep.json`

| Config | Global MAE | Global PSNR | Global SSIM-Y | Frame0 PSNR | Generated MAE | Generated PSNR | Generated SSIM-Y |
|---|---:|---:|---:|---:|---:|---:|---:|
| §6.14k 9f/20 pre-cam-prep | 54.7352 | 10.26 | 0.1882 | 35.78 | 61.1686 | 9.75 | 0.1050 |
| native `cam_prep_func` | 55.2057 | 10.22 | 0.1889 | 35.91 | 61.7005 | 9.71 | 0.1051 |

Interpretation: the native cam-prep port is e2e-neutral on the short
full-chain gate (`+0.53` generated MAE, SSIM unchanged). This is not a new
RGB-parity lever; it closes the camera-prep semantic contract and keeps the
remaining error in Stage-1 trajectory/refiner sensitivity.

**Probe harness caveat fixed.**

The existing remote `run_multistep_current.py` wrapper only changed the
outer probe workdir. The imported
`/root/autodl-tmp/stage1_denoise_probe/run_stage1_denoise_probe.py` still had
a module-level `WORKDIR` pointing to the stale
`/root/autodl-tmp/synced-from-5090/vllm-omni-5090-stage1-align` checkout.
That stale path reproduced old pre-prompt-mask behavior and produced an
incorrect post-mask free-run final generated MAE `0.4085` / cosine `0.7491`.

Added a small wrapper:

- local: `tools/sana_wm_multistep_current_fixed.py`;
- remote synced copy:
  `/root/autodl-tmp/probe-vllm-omni-feat-sana-wm/tools/sana_wm_multistep_current_fixed.py`.

The wrapper fixes both the outer multistep probe workdir and the inner
stage-1 probe module's `WORKDIR` / `sys.path` before running the existing
controlled probe.

**Corrected current-workdir 321f / 20-step Stage-1 free-run.**

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/multistep_controlled_probe_321f_20step.json`
- `/root/autodl-tmp/stage1_longseq_probe/run_multistep_current_fixed_file_cam_prep_*.log`

| Step | `t` | Pre-step generated latent MAE | Generated `noise_pred` MAE | Generated `noise_pred` cos |
|---:|---:|---:|---:|---:|
| 0 | 1000.0000 | 0.000000 | 0.011681 | 0.999835 |
| 1 | 994.4205 | 0.000000 | 0.017176 | 0.999705 |
| 5 | 965.2842 | 0.000714 | 0.038419 | 0.998628 |
| 10 | 900.0266 | 0.004103 | 0.076625 | 0.994394 |
| 15 | 732.2700 | 0.018916 | 0.122004 | 0.986119 |
| 18 | 392.4376 | 0.061878 | 0.152962 | 0.973940 |
| 19 | 87.7046 | 0.106024 | 0.143141 | 0.965639 |

Final generated latent:

| Metric | Value |
|---|---:|
| MAE | 0.117717 |
| RMSE | 0.174175 |
| Cosine | 0.975069 |

This materially improves on the stale-probe row (`MAE 0.4085`, cosine
`0.7491`) and shows the post-prompt-mask Stage-1 trajectory is much closer,
but not closed.

**Masked teacher-forced per-step replay.**

To separate local transformer residuals from trajectory feedback, reran the
teacher-forced probe with two fixes:

1. use the current workdir;
2. explicitly pass the prompt `encoder_attention_mask` reconstructed from
   the same Gemma tokenizer contract. The mask has shape `(1, 300)` and
   `mask_sum=12.0`.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/teacher_forced_native_on_official_trace_321f_20step_masked_cam_prep.json`
- `/root/autodl-tmp/stage1_longseq_probe/run_teacher_forced_masked_cam_prep_*.log`

| Step | `t` | Native-vs-NVlabs `noise_pred` MAE | `noise_pred` cos | Native one-step MAE vs official next | Official source replay MAE |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000.0000 | 0.011657 | 0.999836 | 0.004797 | 0.0048 |
| 1 | 994.4205 | 0.017074 | 0.999707 | 0.000093 | 0 |
| 5 | 965.2842 | 0.013703 | 0.999800 | 0.000134 | 7.69e-11 |
| 10 | 900.0266 | 0.012166 | 0.999846 | 0.000250 | 1.44e-08 |
| 15 | 732.2700 | 0.011668 | 0.999865 | 0.000829 | 6.54e-09 |
| 18 | 392.4376 | 0.016033 | 0.999690 | 0.004886 | 0 |
| 19 | 87.7046 | 0.018768 | 0.999478 | 0.001647 | 0 |

Interpretation:

- The scheduler update formula is effectively aligned. Official
  `noise_pred` replay reaches the official next latent at zero or near-zero
  error for all non-step-0 rows; step 0's `0.0048` is the known conditioning
  first-step edge and matches native one-step scale.
- Native local `noise_pred` residual on the official trajectory is small
  (`~0.012-0.019` generated MAE, cos `>0.99947`), and one-step latent
  perturbation is usually `<=0.001`, rising to `0.0049` at the large
  low-sigma step 18.
- The free-run gap (`final generated MAE 0.1177`) is therefore a
  trajectory-feedback / manifold-sensitivity issue: small per-step residuals
  compound once the native latent leaves the official trajectory, and the
  feedback becomes visible in later denoise steps (`step15+`), rather than a
  scheduler-sign or timestep-update formula bug.

Next useful Stage-1 probe: run block/attention split on the corrected
current-workdir free-run latents at steps `15`, `18`, and `19`, where the
feedback-amplified `noise_pred` MAE reaches `0.122-0.153`, and compare it
against masked teacher-forced residuals. The target is to identify which
block/path is most sensitive to off-manifold latent drift, not to change the
scheduler formula.

#### 6.14m Refiner same-source diff + 48-block cliff localisation ⚠️ 2026-06-01

Reconfirmed §6.14e same-source refiner gap from the existing May-29
artifacts (refiner code unchanged across commits `985e3a07..d3ed5411`):

| Stage | cos | MAE | max |
|---|---:|---:|---:|
| `initial_noisy` | **+1.0000** | 0.0000 | 0.0 |
| `denoised_step0` | +0.9703 | 0.183 | 3.57 |
| `velocity_step0` | +0.9781 | 0.201 | 3.93 |
| `noisy_after_step0` | +0.9976 | 0.037 | 0.72 |
| `denoised_step1` | +0.8888 | 0.372 | 5.55 |
| `velocity_step1` | +0.8908 | 0.501 | 7.36 |
| `noisy_after_step1` | +0.9538 | 0.164 | 2.44 |
| `denoised_step2` | +0.8912 | 0.359 | 5.20 |
| `final` (full) | +0.9326 | 0.179 | 5.20 |
| `final` frame 0 (sink) | **+1.0000** | 0.0000 | 0.0 |
| `final` frame 1 (generated) | **+0.8912** | 0.359 | 5.20 |

Source dumps:
`/root/autodl-tmp/refiner_photometric_probe/official_refiner_step_dump.pt`,
`native_refiner_step_dump_official_prompt_sdpa.pt`.

Confirms: (a) sink/current mask is bit-exact, (b) refiner-forward divergence
appears already at the first denoise call (`denoised_step0` cos `0.97`), then
feedback compounds to `0.89` over the remaining two steps. Same structural
pattern as the Stage-1 §6.13t engine-vs-standalone forward residual.

**Step-0 48-block diff** (from `native_vs_official_step0_block_metrics.json`,
all blocks fed real cumulative native state):

| Phase | Blocks | `cos` | `MAE` | `max` |
|---|---|---:|---:|---:|
| Smooth precision floor | `0-22` | `~1.0` | `0.001 → 0.022` | `4-8` |
| **First sparse spike** | **`23`** | `+1.002` | `0.028` | **`104`** (25× jump) |
| Mid drift | `24-25` | `+0.999` | `0.04 → 0.06` | `120 → 788` |
| Sustained mid | `26-36` | `0.994-1.000` | `0.08 → 0.18` | `~1k` |
| **Cliff (amplifier)** | **`37`** | **`+0.853`** | `0.205` | **`16380`** |
| Cliff persistent | `38-45` | `0.84-0.89` | `0.22 → 0.76` | `~19k` |
| Direction self-correct | `46-47` | `+0.99` | `1.45` (mag only) | `~2k` |

Two distinct events:

1. **Block 23 — first sparse-element divergence.** Up through block 22 the
   residual stream has uniform max=4 (bf16 ULP / fp16 boundary floor). At
   block 23 a single residual element diverges to max=104. Cos stays ~1.0
   because the diff is sparse. Block_25 detail dump shows all sub-stages
   (`self_attn / cross / ff / after_*`) cos `>0.9999` when fed identical
   NVlabs input, so the divergence at block 23+ is from cumulative input
   drift hitting a threshold, not a fresh op bug at that block.
2. **Block 37 — cliff amplifier.** First block where `after_self` and
   `after_ff` show large modulated residual contributions (in detail-dump:
   `self_attn max=0.5 → after_self max=8`, `ff max=8 → after_ff max=128`,
   consistent with `scale_msa`/`scale_mlp ≈ 16`). Under the real cumulative
   drift this `~16×` modulation amplifies sparse single-element drift to
   max=16380, dropping cos to `0.853`. Late blocks 46-47 partially
   re-project direction (`cos → 0.99`) while magnitude still differs.

**Implication.** Refiner per-block ops are correct in isolation. The
`final generated MAE 0.359 / cos 0.8912` is dominated by:

- accumulated bf16 / numeric drift across blocks 0-22 reaching a critical
  threshold at block 23;
- nonlinear scale-modulation amplification at block 37+;
- 3-step feedback compounding `denoised_step0` cos `0.97 → 0.89`.

This is the refiner analog of Stage-1 §6.13o → §6.13s: the first scalar
sub-op responsible is not yet localised; the cleanest next experiment is a
**block-22/23 sub-stage drill with NATIVE accumulated input** (the
existing block_25/37 detail used official input and therefore cannot show
the drift-onset op). High-value because the entire downstream cliff is
downstream of block 23.

#### 6.14n Refiner RMSNorm replacement — REFUTED (byte-identical output) ❌ 2026-06-01

Source-diffed `vllm_omni.diffusion.models.ltx2.ltx2_transformer` vs upstream
`diffusers.models.transformers.transformer_ltx2`. The two implementations of
`LTX2VideoTransformerBlock` are textually almost identical; the only
documented divergence is the per-block weightless RMSNorms (`norm1/norm2/norm3`):
vllm_omni uses `vllm.model_executor.layers.layernorm.RMSNorm`, diffusers uses
its own `diffusers.models.normalization.RMSNorm`. Hypothesis: vllm fused
kernel reduction order differs from diffusers' pure-PyTorch fp32-promote,
ULP drift compounds across `3 × 48 = 144` calls per refiner step.

Replaced `_make_rms_norm` with `_DiffusersStyleRMSNorm` (verbatim port of the
diffusers forward) and reran the same-source probe with `TORCHDYNAMO_DISABLE=1`,
identical `initial_noisy` + `official_video` prompt connector outputs + sigmas.

Result: **byte-identical** to the prior baseline.

| Stage | Pre-fix (§6.14m) | Post-RMSNorm-swap |
|---|---:|---:|
| `denoised_step0` cos / MAE | 0.9703 / 0.183 | 0.9703 / 0.183 |
| `denoised_step1` cos / MAE | 0.8888 / 0.372 | 0.8888 / 0.372 |
| `denoised_step2` cos / MAE | 0.8912 / 0.359 | 0.8912 / 0.359 |
| `final` frame 1 (generated) | **0.8912 / 0.359** | **0.8912 / 0.359** |

Root cause: pipeline log shows
`IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])`.
The vllm `RMSNorm.forward_cuda` short-circuits to `forward_native`, which
dispatches to `ir.ops.rms_norm` — a pure PyTorch path equivalent to
diffusers' `RMSNorm.forward` to the bit. The kernel-divergence hypothesis is
therefore wrong under this priority config; the 0.0089 cos / 0.359 MAE
generated-frame gap is **not** RMSNorm precision drift.

Remaining `LTX2VideoTransformerBlock` ↔ diffusers divergence surface narrows to:

1. `LTX2Attention` — the local class uses `QKVParallelLinear` (with explicit
   TP fallback), a custom rotary apply (`apply_interleaved_rotary_emb` /
   `apply_split_rotary_emb` with explicit fp32 promote), and a SDPA call
   wrapper. Diffusers uses `nn.Linear` + processor pattern.
2. `LTX2FeedForward` — `ColumnParallelApproxGELU` + `RowParallelLinear` vs
   diffusers' standard `FeedForward(nn.Linear→GELU→nn.Linear)`.
3. Attention backend selection: probe log shows
   `Resolved diffusion attention backend 'SDPA' for role='self'`; whether
   diffusers' processor uses the same SDPA call or another path is unverified.

Next high-ROI A/B: swap `LTX2FeedForward` and/or `LTX2Attention` with
diffusers' equivalents and rerun the same probe.

Artifact: `/root/autodl-tmp/refiner_photometric_probe/native_refiner_step_dump_post_rmsnorm_fix.pt`.

#### 6.14p Refiner = diffusers swap — same-source bit-exact, e2e gain minimal ⚠️ 2026-06-01

After §6.14n refuted RMSNorm, ran a stronger A/B: swap the entire vllm_omni
local LTX-2 refiner transformer for upstream
`diffusers.models.transformers.transformer_ltx2.LTX2VideoTransformer3DModel`,
same weights, same surrounding sana-wm pipeline (`_run_inprocess_refiner`,
`_predict_refiner_current_x0`, streaming attention helpers, sink/current
split, sigmas). Same official prompt connector outputs + official
`initial_noisy` as §6.14m.

**Refiner same-source diff (3-step loop):**

| Stage | vllm_omni LTX-2 (§6.14m) | **diffusers LTX-2 swap** |
|---|---:|---:|
| `denoised_step0` cos / MAE | 0.9703 / 0.183 | **1.0000 / 0.0000** |
| `denoised_step1` cos / MAE | 0.8888 / 0.372 | **1.0000 / 0.0000** |
| `denoised_step2` cos / MAE | 0.8912 / 0.359 | **1.0000 / 0.0000** |
| `final` frame 1 (generated) | **0.8912 / 0.359** | **1.0000 / 0.0000** |

Diffusers refiner is **byte-identical** to NVlabs reference. This
conclusively proves the entire Stage-2 refiner same-source `11%` cos gap
is in `vllm_omni.diffusion.models.ltx2.LTX2VideoTransformer3DModel`
(transformer_blocks, proj_in, time_embed, rope, scale_shift_table,
proj_out, or norm_out — any subset). It is **not** in the surrounding
sana-wm pipeline (sink/current split, scheduler, packing, prompt
connector, streaming attention helpers).

Probe artifact:
`/root/autodl-tmp/refiner_photometric_probe/native_refiner_step_dump_diffusers_swap.pt`.

**Wired into production:** added `SANA_WM_USE_DIFFUSERS_REFINER=1`
env-gated branch in
`pipeline_sana_wm_two_stages._ensure_refiner_transformer` (instantiates
`diffusers.LTX2VideoTransformer3DModel(**filtered)`, loads the same
safetensors via standard `load_state_dict(..., strict=False)`, sets
`eval`, `requires_grad_(False)`, attempts `_disable_caching()` from
`CacheMixin`). Refiner sampling loop already runs under
`@torch.inference_mode()` on `_predict_refiner_current_x0`.

**E2E RGB metrics with the swap enabled:**

| Config | MAE | PSNR | SSIM-Y | Reference baseline (§6.14k) |
|---|---:|---:|---:|---|
| 9f / 20-step (common-prefix 8, no frame-align) | 63.67 | 9.00 | 0.176 | Global 54.74 / 10.26 / 0.188; Generated 61.17 / 9.75 / 0.105 |
| 321f / 20-step | — | — | — | OOM (see below); §6.14k vllm_omni-refiner baseline: Generated 39.32 / 14.30 / 0.244 |
| 321f / 60-step | not attempted | | | §6.14k baseline: Generated 49.36 / 12.26 / 0.297 |

The 9f number is in the **same ballpark** as the vllm_omni-refiner baseline
(slightly worse on MAE, comparable on PSNR/SSIM-Y). The strict frame
alignment is different between this run and §6.14k so the comparison is
not exact, but the refiner swap clearly does **not** dramatically move
9f e2e RGB.

321f/20 attempts under SANA_WM_USE_DIFFUSERS_REFINER=1 CUDA-OOM on the
native side (tries to allocate `4.41 GiB` with `~92 GiB` already
resident; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did not
help). The same 321f/20 runs cleanly with the local LTX-2 port at
§6.14k, so the diffusers transformer consumes roughly **~30 GiB more
activation memory** than the local port at this video length. Root
cause not yet localised (likely attention kernel selection inside
diffusers' `LTX2AudioVideoAttnProcessor` or remaining `CacheMixin`
state); did not chase further since the bit-exact same-source result
already proves the port hypothesis without needing 321f e2e
confirmation.

**Implication.**

1. The vllm_omni LTX-2 port has a real numeric divergence vs upstream
   diffusers, but it is **not** the dominant 9f-e2e pixel-space
   blocker. Stage-1 latent (cos `0.9751`, §6.14l) and VAE / frame
   alignment dominate.
2. Whether the port is a 321f e2e blocker is unverified due to OOM; the
   §6.14k 321f/20 generated MAE `39.32` is already much closer to a
   useful number than 9f's `61.17`, so refiner port cleanup could in
   principle close a meaningful fraction at long sequences — but only
   after solving the diffusers-refiner memory delta.
3. Production decision pending: (a) cut over refiner to diffusers and
   chase the extra memory budget, or (b) leave refiner on the local
   port and focus on Stage-1 latent parity first since refiner is not
   the 9f bottleneck.

#### 6.14q Apples-to-apples refiner A/B at 9f/20 — refiner port effectively 0 MAE on e2e RGB ✅ 2026-06-01

§6.14p left "is refiner the e2e bottleneck?" as an open question (the
9f/20 number after the diffusers swap was MAE `63.67` but couldn't be
directly compared to §6.14k's `54.74` because of differing
frame-alignment). Ran a clean same-harness A/B (same probe script
`run_one_side.py`, no frame alignment, common-prefix 8 frames):

| Pred path | MAE | PSNR | SSIM-Y |
|---|---:|---:|---:|
| vllm_omni LTX-2 refiner (`SANA_WM_USE_DIFFUSERS_REFINER=0`) vs reference | **63.43** | 8.92 | 0.173 |
| diffusers LTX-2 refiner (`SANA_WM_USE_DIFFUSERS_REFINER=1`) vs reference | **63.67** | 9.00 | 0.176 |
| vllm_omni refiner vs diffusers refiner (both native) | 18.66 | 17.38 | 0.865 |

Artifacts:
`/tmp/ref_9_20.npy`, `/tmp/nat_9_20_omni.npy`, `/tmp/nat_9_20_diffusers.npy`.

**Conclusion.** Refiner port choice contributes **`0.24` MAE / `0.08` PSNR
/ `0.003` SSIM-Y** to 9f/20 e2e RGB — effectively zero. Both refiners
land at `~63` MAE from reference. The two native outputs do differ from
each other by MAE `18.66` (so the refiner *does* produce a structurally
different latent), but both are equidistant from the NVlabs reference.

Interpretation: even a byte-exact refiner (diffusers, §6.14p) fed a
slightly-wrong Stage-1 latent (cos `0.975`, §6.14l) produces a wrong
final RGB by the same magnitude as the imperfect refiner fed the same
wrong latent. The 63-MAE 9f/20 gap is dominated by **Stage-1 (+ VAE +
pre-refiner path)**, not refiner port.

**ROI implication.** Stage-1 latent parity (cos `0.975 -> 0.99+`) is the
highest-leverage Stage-2 reliant work for e2e RGB at this video length.
Refiner port (§6.14p) is a real numeric port bug but is **not** the e2e
bottleneck at 9f. 321f e2e impact still unverified due to diffusers
refiner OOM.

#### 6.14o Stage-1 late-step block/attention split — MLP stride parity fixed, trajectory drift remains ⚠️ 2026-06-01

Ran the §6.14l follow-up on the corrected current workdir at 321 frames /
20 Stage-1 steps, focusing on denoise steps `15`, `18`, and `19`.

Artifacts:

- `/root/autodl-tmp/stage1_longseq_probe/attn_split_probe_321f_steps15_18_19_late_steps15_18_19_blocks0_15_18_19_current_cam_prep.json`
- `/root/autodl-tmp/stage1_longseq_probe/block_stage_late_20260601_052115_gated_summary.json`
- `/root/autodl-tmp/stage1_longseq_probe/mlp_substage_same_input_321f_step19_blocks18_19.json`
- `/root/autodl-tmp/stage1_longseq_probe/run_multistep_stridefix_20260601_054105.log`

**Attention split result.** Same-latent `noise_pred` is close on the official
trajectory:

| Step | Generated `noise_pred` MAE | Cosine |
|---:|---:|---:|
| 15 | 0.011823 | 0.999899 |
| 18 | 0.016113 | 0.999784 |
| 19 | 0.018826 | 0.999555 |

Full-path attention tensors look large in absolute value at late blocks
because upstream hidden-state drift has already accumulated:

| Step / block | Full-path `attn_out` MAE | Isolated-on-official-`attn_in` `attn_out` MAE |
|---|---:|---:|
| 19 / block18 | 5.1405 | 0.1312 |
| 19 / block19 | 6.4452 | 0.2918 |

Single-block `attn_out` teacher forcing changes final same-step
`noise_pred` MAE by only `~1e-4` to `8e-4`, sometimes worsening it. This
rules out an isolated late self-attention or camera-attention block as the
main remaining error source.

**Block-stage split.** Block18/19 stage dumps show LayerNorm/adaptive
modulation makes the attention/MLP inputs very close even when the raw hidden
state differs substantially:

| Step / block | Input MAE | `x_msa_in` MAE | `x_mlp_in` MAE | Gated MLP-out MAE | Block-output MAE |
|---|---:|---:|---:|---:|---:|
| 15 / block18 | 8.791 | 0.01018 | 0.00363 | 9.802 | 12.470 |
| 18 / block18 | 9.260 | 0.01096 | 0.00359 | 10.126 | 13.101 |
| 19 / block18 | 12.468 | 0.01516 | 0.00451 | 14.651 | 19.109 |
| 19 / block19 | 19.109 | 0.02676 | 0.00413 | 17.589 | 25.066 |

Cross-attn barely changes the residual (`post_cross_attn` tracks
`post_attn_residual`). The biggest block-local amplifier is the GLUMBConvTemp
FFN/MLP path, but the probe then found this is largely a **memory-layout
numeric contract**, not a weight or formula bug.

**MLP stride/layout root cause.** NVlabs `GLUMBConvTemp.forward()` does:

```python
x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
x = self._apply_spatial_autochunked(x)
```

It intentionally leaves the NHWC-derived NCHW view non-contiguous
(`stride=(1971200, 1, 89600, 2240)` in the 321f probe). Native had inserted
`.contiguous()` before the 2D conv stack. On bf16 CUDA Conv2d this switches
kernel/layout and produces a deterministic same-input output difference:

| Block | Official MLP forward vs native contiguous | Official vs native non-contiguous |
|---:|---:|---:|
| 18 | MAE 1.5411 / max 48 | **0.0 / 0.0** |
| 19 | MAE 1.4483 / max 64 | **0.0 / 0.0** |

After removing the native `.contiguous()`, direct same-input MLP parity for
step19 block18/19 is byte-identical (`off_vs_native = 0.0 / 0.0`). This is a
real semantic parity fix: native now follows NVlabs' bf16 Conv2d layout.

**Trajectory rerun after the stride fix.** Re-ran corrected 321f/20 controlled
Stage-1 free-run with the MLP stride fix:

| Config | Final generated MAE | RMSE | Cosine |
|---|---:|---:|---:|
| §6.14l pre-stride-fix | 0.117717 | 0.174175 | 0.975069 |
| MLP stride/layout fix | 0.120997 | 0.178146 | 0.973891 |

The fix slightly worsens the full free-run. Interpretation: the previous
contiguous MLP path was semantically wrong but happened to compensate a small
part of the trajectory feedback. Keeping the stride fix is still correct for
NVlabs parity, but it does **not** solve the remaining 321f free-run drift.

**Current Stage-1 conclusion.** Scheduler update, prompt mask, camera prep,
isolated self-attn/cross-attn, and MLP formula/weights are all substantially
cleared. The residual is distributed trajectory sensitivity: small local
`noise_pred` residuals push the native latent off the official manifold, and
normalisation hides that drift at each block input until large FFN/MLP
projections re-expand it. The next useful Stage-1 test is therefore a 321f/60
post-stride rerun and/or an oracle-correction/sensitivity sweep, not another
single-block formula port.

### 6.13k Stage-1 Multi-Step Timestep Schedule Diverges — Native Scheduler Now Urgent 🔴 2026-05-28 night (RESOLVED — see §6.13l)

With §6.13j confirming model forward is clean at block-0 (attn
cos +0.9999, MLP cos +0.998), the natural next probe is whether
the e2e MAE-50 gap comes from scheduler-side timestep drift or
multi-block model-forward compounding. Extended the step-0 dump
hook to dump steps 0/1/2 on both pipelines, controlled-input
inject step-0 latent + prompt on the native side, then compare
`latent_in`, `noise_pred`, and `timestep_per_frame` per step.

**Result (`/tmp/probe_steps.py`, 3 stage-1 steps):**

| Step | NVlabs t_scalar | Native t_scalar | latent_in cos | noise_pred cos | noise_pred ratio |
|---|---:|---:|---:|---:|---:|
| 0 | 1000.0 | 999.0 | **+1.0000** | +0.9435 | 1.030 |
| 1 | 909.0 | 951.0 | +0.9993 | +0.9056 | 1.330 |
| 2 | **87.7** | **830.0** | **+0.6450** | **+0.3173** | **2.049** |

**Diagnosis.** Our `DPMSolverMultistep(use_flow_sigmas=True)`
produces a near-linear schedule (999 → 951 → 830) over 3 steps.
NVlabs `LTXFlowEuler` produces a flow-shifted schedule that
front-loads tiny early steps then takes one huge final jump
(1000 → 909 → 87.7). By step-2 the two pipelines are at
completely different points on the flow trajectory, the latent
cosine collapses to 0.645, and noise_pred cosine to 0.32.

The model forward is **not** the bug. The scheduler timestep
schedule **is**. Specifically:

* Step-0 (controlled input): noise_pred cos 0.9435 — already
  slightly off purely because `t_scalar` differs by 1 (999 vs
  1000).
* Step-1 latent_in cos 0.9993 — small Euler step took us
  somewhere reasonable, but the divergence is growing.
* Step-2 latent_in cos 0.6450 — catastrophic; we're at t=830
  while NVlabs is at t=87.7 (~last step of the denoising
  trajectory). The two pipelines are doing different things at
  this point.

**Implication for native scheduler work.** Native scheduler is
no longer a code-cleanup / dependency-removal task — it is the
**primary remaining correctness bug** for stage-1 alignment.
Replacing the `DPMSolverMultistep` wrapper with a faithful port
of `LTXFlowEuler`'s schedule (init sigmas + flow shift +
per-step `t = sigma * num_train_timesteps`) should drop e2e MAE
significantly in one shot, given that all upstream stages
(model forward, controlled-input noise_pred, per-token Euler
step formula) are already verified clean.

**Acceptance.** After native scheduler lands, step-1 and step-2
`t_scalar` should match NVlabs to within 1.0 (matching step-0's
tolerance). `latent_in` cosine should hold ≥ 0.99 through all
stage-1 steps. E2E MAE should drop from ~50 to the
comparison-meaningful range (target ≥ 25 dB PSNR).

**Repro.** `/tmp/probe_steps.py` on remote
`/root/autodl-tmp/vllm-omni-feat-sana-wm-23ff624b`. Dump hooks
gated by `SANA_WM_DUMP_STEPS_PREFIX` + `SANA_WM_DUMP_STEP_COUNT`.

### 6.13j Block-0 Sub-stage Parity Re-verified — §6.13h Findings Were a Measurement Artifact ✅ 2026-05-28 night

After §6.13h flagged `attn_out` (3× smaller, cos +0.26) and `mlp_out`
(6× larger then 6× smaller, cos < 0) as the two block-0 layer bugs,
a deeper drill-down with stage-by-stage dump hooks **retracts both
findings**. Both modules are essentially correct; the §6.13h
divergences were caused by an apples-to-oranges comparison where
NVlabs's `forward_frame_aware` dumped the **gated** output
(`gate_msa * attn_out`, `gate_mlp * mlp_out`) while our hooks dumped
the **raw** sub-module output.

**Attn sub-stage parity (controlled inputs, after fixing the
gated/raw mismatch):**

| Stage | NVlabs norm | Native norm | Cosine |
|---|---:|---:|---:|
| main_raw (post-`out_proj`, pre-gate) | match | match | **+0.9999** |
| cam_raw | match | match | +0.9999 |
| cam_contrib (after camera fusion) | match | match | +0.9999 |
| combined | match | match | +0.9999 |
| attn_out (final) | — | — | **+0.9999** |

**MLP sub-stage parity (controlled inputs, same fix):**

| Stage | NVlabs norm | Native norm | Cosine | Ratio |
|---|---:|---:|---:|---:|
| mlp_in (4D `(B*F,C,H,W)`) | match | match | +1.0000 | 1.000 |
| after_inverted_silu | match | match | +1.0000 | 1.000 |
| after_depth | match | match | +1.0000 | 1.000 |
| after_glu | match | match | +1.0000 | 1.000 |
| after_point | match | match | +1.0000 | 1.000 |
| t_conv_in (5D `(B,C,F,H*W)`) | 1210.31 | 1220.75 | +0.9980 | 1.009 |
| t_conv_out | 9093.37 | 9201.14 | +0.9981 | 1.012 |
| after_tconv_add | 9491.44 | 9608.66 | +0.9982 | 1.012 |
| **final_mlp_out** | **9491.44** | **9608.65** | **+0.9982** | **1.012** |

`t_conv.weight.norm` is byte-identical on both sides (180.5021).

**What this means.**

* `SanaWmMbConvFfn` rewrite from §6.13h (per-frame `(B*F,C,H,W)`
  reshape + post-`inverted_conv` SiLU) was correct. The spatial
  MBConv pipeline is now numerically equivalent to NVlabs's
  `GLUMBConvTemp`.
* `SanaWmSelfAttention._forward_cam_branch` single-path delta rule
  + `apply_fn_o` + RMS renorm is also correct. The §6.13h "3× too
  small" was the gated-vs-raw artifact — NVlabs's
  `forward_frame_aware` multiplies by `gate_msa` (~0.3 in block 0)
  before the dump point we anchored on.
* Block-0 has **no remaining layer bug**. The residual e2e
  divergence (still ~50 MAE, 12 dB PSNR) must come from one of:
  1. **Compounding** — the 1.2% per-block norm-ratio drift at
     `final_mlp_out` (1.012²⁰ ≈ 1.27× by block 19). At bf16
     precision this is plausibly within numerical tolerance, but
     it compounds.
  2. **Final layer / unpatchify** (untested).
  3. **Multi-step scheduler** beyond step 0 (only step-0 noise_pred
     was verified controlled-input parity).
  4. **Stage-2 refiner** path (also untested at sub-stage level).

**Action.** §6.13h's "two layer bugs" narrative is retracted. The
MbConv reshape + SiLU fix from §6.13h still stands as a real
correctness fix; the leftover negative cosine on the gated output
seen in §6.13h was just gate dynamics. Next probe will be one of
the four candidates above — see §9.

**Repro.** `/tmp/probe_attn0.py`, `/tmp/probe_mlp0.py`,
`/tmp/probe_mlp_final.py` on remote
`/root/autodl-tmp/vllm-omni-feat-sana-wm-23ff624b`. Dump hooks
gated by `SANA_WM_DUMP_ATTN0`, `SANA_WM_DUMP_MLP0`,
`SANA_WM_DUMP_MLP_FINAL`; injection by `SANA_WM_LOAD_LATENT_FROM`
+ `SANA_WM_LOAD_PROMPT_FROM`.

### 6.13h Block-0 Staged Parity Probe — Located 2 Layer Bugs ⚠️ 2026-05-28 night (SUPERSEDED — see §6.13j)

> **2026-05-28 update:** Both "bugs" identified below are retracted
> by §6.13j. The 3×-smaller `attn_out` and 6×-larger `mlp_out`
> cosines were a measurement artifact (NVlabs dumped the gated
> output; we dumped raw). The MbConv reshape + SiLU fix described
> below is still a real correctness improvement, but the "remaining
> single-block work" and "open questions" bullets at the end of
> this section no longer apply.


Added env-gated dump hooks at four points inside our
`SanaWmBlock._forward_frame_aware` and at matching points in
NVlabs `SanaVideoMSCamCtrlBlock.forward_frame_aware`. Ran the
probe with `SANA_WM_LOAD_LATENT_FROM` + `SANA_WM_LOAD_PROMPT_FROM`
injecting NVlabs's step-0 state so both pipelines see byte-
identical inputs.

**Block-0 staged comparison (controlled inputs):**

| Stage | NVlabs norm | Native norm | Cosine | Status |
|---|---:|---:|---:|---|
| block_input | 1165.6 | 1165.6 | **+1.000** | ✅ control works |
| shift/scale/gate (msa+mlp, ×6) | match | match | **+1.000** | ✅ modulation OK |
| scale_shift_table (param) | 73.52 | 73.52 | **+1.000** | ✅ loaded correctly |
| x_msa_in (post-modulation, pre-attn) | 2174 | 2199 | +0.9999 | ✅ |
| **attn_out** (self.attn output) | **191574** | **71134** | **+0.2574** | ❌ **3× smaller** |
| post_attn_residual | 191600 | 192481 | +1.000 | OK (residual dominates) |
| post_cross_attn | 192860 | 193457 | +1.000 | OK |
| x_mlp_in | 117.1 | 117.5 | +1.000 | ✅ |
| **mlp_out** (FFN output) | **55989** | **343716** | **+0.0373** | ❌ **6× larger, uncorrelated** |
| block_output | 147975 | 654294 | +0.6241 | ❌ MLP-dominated |

**Two layer bugs localised.** With control inputs identical and
modulation correct, two specific sub-modules diverge:

1. **`self.attn` (the GDN+UCPE main attention) output is 3×
   smaller than NVlabs and only partially correlated** (cosine
   +0.26). Could be wrong scale somewhere in the GDN
   recurrence, a missing factor in `out_proj`, an over-eager
   `_downscale_to_reference_rms`, or some sign issue. Hidden
   by the post-attention residual at block 0 (the latent
   itself dominates) but matters when compounded across 20
   blocks.

2. **`mlp_out` (SanaWmMbConvFfn) is 6× larger and essentially
   uncorrelated** (cosine +0.04). This dominates the block
   output divergence. Two root causes identified:

   * **Wrong reshape layout.** Our code reshaped to
     `(B, C, F, H*W)` and ran the 3×3 depthwise conv with its
     kernel spanning `(F, H*W)` — treating F as height and the
     flattened H*W as 1D width. NVlabs reshapes to
     `(B*T, C, H, W)`, applying the 3×3 conv as a proper 2D
     spatial convolution per frame.
   * **Missing post-`inverted_conv` SiLU.** NVlabs `ConvLayer`
     applies `act=act[0]="silu"` after the 1×1 conv inside
     `inverted_conv`. We stored only the raw `nn.Conv2d` in
     `_ConvWrapper.conv` (to match the checkpoint key
     `mlp.inverted_conv.conv.weight`) and called `.conv(x)`
     directly, skipping the SiLU.

**Fix attempt (commit `<pending>`):** rewrote
`SanaWmMbConvFfn.forward` to reshape per-frame as
`(B*F, C, H, W)` and apply SiLU after `inverted_conv`. Result:

| Configuration | block_0 mlp_out cos | block_0 output cos | e2e MAE | PSNR | SSIM-Y |
|---|---|---|---|---|---|
| Before fix | +0.04 | +0.6241 | 51.06 | 12.43 | +0.046 |
| **After fix** | -0.29 | **+1.0000** | **50.53** | 12.11 | -0.002 |

Block-0 OUTPUT now matches perfectly (cos +1.0000) but
`mlp_out` flipped from 6× too large to 6× too small with a
negative cosine. The block-output match comes from the
post-cross-attn residual (norm 193k) dominating the small mlp
contribution (residual-dominance argument).

E2E MAE barely changed (-0.53) and SSIM regressed slightly to
near zero. So the fix moves block-0 in the right direction but
the deeper architectural alignment of the MLP is still off
(likely a 3rd subtle difference — activation choice, norm
position, or some scale factor). Compounding across 20 blocks
prevents real e2e gains until both `mlp_out` and `attn_out`
match more tightly.

**Remaining single-block work:**

* Find the third MLP discrepancy (compare per-step intermediate
  inside MLP; possibly NVlabs uses different `norm` in
  `ConvLayer`, or our SiLU position is wrong).
* Diagnose `attn_out` — break down inside
  `SanaWmSelfAttention._forward_gdn_raw + _forward_cam_branch`
  to find which stage diverges. The 3×-smaller output suggests
  a missing scale factor or an over-eager
  `_downscale_to_reference_rms`.

**Open questions for next session:**

* Compare our and NVlabs's `attn_out` BEFORE the
  `out_proj` Linear — isolate whether the bug is in the
  attention compute or the output projection.
* Print our and NVlabs's `mlp_out` standard deviation at
  several intermediate points inside the MLP (post-inverted,
  post-depth, post-glu, post-point) to localise the 6× scale
  divergence.

### 6.13g Step-0 Controlled-Input Probe — Bug Is In Model Forward ⚠️ 2026-05-28 late (CLOSED — drilled into §6.13h→j, full model fwd resolved by §6.13m)

Added env-gated input-injection hooks to `_run_native_smoke_backend`:

* `SANA_WM_LOAD_LATENT_FROM=path` — overrides the initial latent
  with `latent_in` loaded from a saved dump (typically the
  NVlabs LTXFlowEuler step-0 dump).
* `SANA_WM_LOAD_PROMPT_FROM=path` — overrides `prompt_embeds`
  with the `prompt_embeds` field of a saved dump; handles
  NVlabs' `(B, 1, N, D)` shape and CFG-doubled batch.

Re-ran the step-0 probe with BOTH `latent_in` and `prompt_embeds`
injected from NVlabs into our pipeline. Verified the injection
worked: native latent_in cosine vs NVlabs = `+0.999987`, native
prompt_embeds norm = NVlabs norm (`2280.903 vs 2280.949`).

**Result — model forward is anti-correlated even with identical
inputs:**

```
NVlabs noise_pred: shape (1, 128, 2, 22, 40)  std=1.02  norm=487.6
native noise_pred: shape (1, 128, 2, 22, 40)  std=0.81  norm=384.0
MAE(noise_pred)     = 1.062
cosine(noise_pred)  = -0.057
  frame 0 cosine: -0.231
  frame 1 cosine: +0.104
```

**Isolation: cam disabled gives WORSE cosine (-0.110)**, ruling
out cam branch as the dominant bug. The shared
model-forward components are the source. Most likely suspects:

* main GDN attention path (`_forward_gdn_raw` + recurrence)
* cross-attention with prompt
* timestep modulation (`scale_shift_table` + `_modulate`)
* patch embedding (`x_embedder.Conv3d`)
* `attention_y_norm`
* MLP / FFN (`SanaWmMbConvFfn`)
* final layer modulation

**Decisive rule-outs from this round:**

1. ❌ Noise initialisation differences — controlled, cosine
   +0.999987 on latent_in.
2. ❌ Prompt encoding — controlled, NVlabs prompt injected with
   matching shape and norm.
3. ❌ Scheduler / sigma — model forward divergence happens
   BEFORE any scheduler.step on step 0.
4. ❌ Cam branch — disabling it makes cosine worse.

**Next high-leverage diagnostic.** Single-block parity test:
- Capture both pipelines' input to and output from `blocks[0]`
  on the same control inputs.
- If block-0 output matches → bug is in patch embedding /
  `attention_y_norm` / final layer.
- If block-0 output differs → bug is in shared block components
  (GDN, cross-attn, MLP, modulation).

This is the next 0.5–1 day work chunk and is the natural
continuation of the §6.13f probe approach.

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

12. ~~**Fix Stage-1 latent magnitude (§6.12).**~~ ✅ **Done 2026-05-28** — root cause localised to the cam branch (main-only latent was always in normal range). The first fix matched NVlabs' Python `BidirectionalGDNUCPESinglePathLiteLA` path with single-path recurrence, `apply_fn_o`, and PostUCPERenorm, dropping STAGE1_STEPS=1 from `[-59, 61]` to `[-12.3, 11.8]`. §6.13s later corrected the production GDN path to match the loaded NVlabs BothTriton class, where GDN cam Q/K/V are **not** PostUCPERenorm-shrunk before the scan.

13. ⚠️ **Stage-1 native scheduler/softmax-UCPE/BothTriton/prompt-mask/cam-prep/MLP layout alignment (§6.13m-§6.14o).** Controlled 3-step and 9-frame 20/60-step parity are strong (generated cos `0.986-0.988`). Production-length 321-frame / 20-step parity previously dropped to generated cos `~0.75`; §6.13s found a real long-sequence GDN cam-branch bug, §6.13v cleared engine-entered input parity and fixed prompt tokenization, and §6.13w localised the first non-zero attention residual. §6.13x then found the dominant downstream gap: native cross-attn ignored the prompt padding mask. Passing `encoder_attention_mask` drops 321f step-0 baseline `noise_pred` MAE `0.12383 -> 0.01211` (cos `0.999887`). §6.14l ports native `cam_prep_func`, confirms 9f/20 e2e neutrality, fixes a stale-workdir trajectory-probe pitfall, and reruns corrected 321f/20 Stage-1: final generated latent MAE is `0.1177` / cos `0.9751`. §6.14o closes a real MLP bf16 Conv2d memory-layout semantic gap, but the post-fix 321f/20 free-run is essentially unchanged (`0.1210` / cos `0.9739`). Masked teacher-forced replay rules out scheduler update formula as the first-order issue; remaining Stage-1 work is feedback-amplified late-step trajectory sensitivity and 321f/60 rerun.

14. ⚠️ **Stage-2 refiner/VAE contract alignment (§6.14).** The structural NVlabs refiner contract is now ported: sink/current split, seed-42 current noise, per-token timestep, video-only streaming mask, x0/velocity loop, and terminal-zero sigma. Same-source native vs official-manual refiner is generated-frame latent MAE=0.3589 / cos=0.8912. §6.14g also aligns the LTX-2 VAE memory-mode contract (`enable_tiling` + framewise decoding), allowing native 321-frame decode on 98GB; this is upstream-native memory behavior, not an OOM workaround. §6.14h/§6.14i fix the local e2e harness controls for action length, Stage-1 steps, metrics JSON, frame-index alignment, and frame-0 sanity; §6.14j/§6.14k record frame-aligned 9f/321f e2e reruns before and after the prompt-mask fix. Strict refiner latent parity still needs a same-source rerun before attributing the remaining RGB error to photometric tuning.

---

## 8. Outstanding Work — GPU Required

1. **Rerun 321-frame 60-step Stage-1 after §6.14o.** The corrected current-workdir 321f/20 run is now final generated latent MAE `0.1210` / cos `0.9739` after the MLP stride parity fix (`0.1177` / `0.9751` before it). 321f/60 still needs the same corrected-workdir treatment. Record generated-frame latent MAE/RMSE/cos and compare against the 20-step row before drawing conclusions about step count.

2. **Regenerate same-source Stage-2 refiner acceptance artifacts (§6.14).** The 321-frame e2e gate now runs through native refiner+VAE after the VAE memory-mode, frame-index, and prompt-mask fixes (§6.14g-§6.14k), but strict refiner latent parity is still not closed. Regenerate the official refiner baseline and prompt-connector dump in one run, then decide whether strict cos ≥0.98 requires a diffusers-exact fallback or a refiner-specific torch-linear layer stack.

3. **Wire PSNR ≥ 30 / SSIM-Y ≥ 0.93** once harness produces a qualifying result.

4. **Late-step trajectory sensitivity** at full 704×1280 / 321 frames vs official NVlabs path. §6.14l shows the scheduler update formula and cam-prep semantics are no longer the first-order issue; §6.14o rules out isolated self-attn/cross-attn and closes MLP stride parity. The remaining task is an oracle-correction / perturbation-sensitivity sweep to quantify why small local residuals still compound in free-run.

5. **CUDA Graph capture smoke** — confirm no regression vs fused-GDN e2e when bucket capture fires.

6. **Real USP run** to validate `_sp_plan` under multi-GPU sequence parallelism.

7. **HSDP + CFG-parallel sweeps** from `dfx/perf` config.

8. **Live server smoke** — `vllm serve … --omni` + `POST /v1/videos/generations` with SANA-WM camera payload.

---

## 9. Recommended Next-Step Ordering

### 9.1 Immediate (no GPU, before PR open)

1. Add `tests/e2e/offline_inference/test_sana_wm_expansion.py` + `tests/e2e/online_serving/test_sana_wm_expansion.py` (§6.1).
2. Add `vllm_omni/deploy/sana_wm.yaml` (§6.5).
3. Evaluate HF `model_index.json` to remove framework changes (§6.2).
4. Simplify `load_weights()` and narrow TP helpers (§6.3, §6.4).
5. Add architecture diagram draft (§7 item 9).

### 9.2 Single-GPU correctness (GPU)

6. Rerun corrected-workdir 321-frame Stage-1 60-step controlled denoise after §6.14o.
7. Run an oracle-correction / perturbation-sensitivity sweep on the post-stride 321f trajectory.
8. Wire PSNR/SSIM assertions once the short harness qualifies.
9. GDN full-shape parity at 704×1280 (commit `tests/e2e/accuracy/` result as markdown).

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
