# Sana-WM Integration — Progress Audit

> **Audit date:** 2026-05-29 (revision 15 — Stage-1 long-sequence late-step drift probe)
> **Branch:** `feat/sana_wm`
> **Implementation snapshot:** remote 5090-synced worktree containing §6.13m softmax-UCPE + native scheduler fixes; refiner probes run on RTX PRO 6000 98GB
> **Pushed to:** `fork/feat/sana_wm` (`BruceLoveDecimal/vllm-omni`)
> **Spec (single source of truth):**
> [`sana_wm_integration.md`](sana_wm_integration.md)
> **Tracking issue:**
> [vllm-project/vllm-omni#3656](https://github.com/vllm-project/vllm-omni/issues/3656)

---

## 0. Revision-9 Critical-Path Items — Updated Status

Revision-9 blockers are mostly closed. Short-sequence Stage-1 alignment is now
strong; the open Stage-1 correctness gap is 321-frame late-step drift in the
denoising loop (§6.13o). The active short-sequence full-chain blocker remains
the Stage-2 LTX-2 refiner contract (§6.14).

| # | Item | Status |
|---|---|---|
| 1 | GDN Triton — long-sequence + multi-card parity vs. NVlabs | ⚠️ **Open** — main-branch GDN math verified equivalent (see §6.10); divergence is upstream of recurrence |
| 2 | First-frame VAE encode for I2V conditioning | ✅ **Closed** — commit `f7e59121` A.1 |
| 3 | NVlabs flow-DPM solver | ✅ **Closed** — commit `f7e59121` A.2 (`DPMSolverMultistepScheduler`) |
| 4 | UCPE branch decomposition + numeric Plücker reference test | ✅ **Verified 2026-05-28** — UCPE math (`ucpe.py`) and native raw camera branch match NVlabs `prepare_prope_fns` + `BidirectionalGDNUCPESinglePathLiteLA._forward_cam_branch` at fp32 `~1e-7` max abs. See §6.12a. |
| 6 | vLLM parallel linear weight loading | ✅ **Fixed 2026-05-28** — `use_official_backend` gating tightened to require explicit `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`. Loaded weight norms verified on GPU. See §6.11. |
| 7 | Stage-1 latent magnitude vs LTX-2 refiner | ✅ **Fixed 2026-05-28** — cam branch rewritten as `BidirectionalGDNUCPESinglePathLiteLA` (single-path + apply_fn_o + RMS renorm). Latent in normal range now. See §6.12. |
| 8 | Per-token timestep sampling contract | ⚠️ **Short-sequence Stage-1 mostly closed; long-sequence still open** — per-frame `(B, 1, F)` timestep, native per-token FlowMatch Euler, condition-mask restore, VAE norm, and softmax-UCPE camera branch are landed. Short 9-frame Stage-1 parity stays strong at 20/60 steps (generated cos `0.9878` / `0.9861`). The 321-frame / 20-step drop is now localized to late-step drift after input camera tensors, `chunk_index`, step-0 latent/prompt/timestep, and initial model forward all check out; see §6.13n–§6.13o. |
| 5 | TP layers → HSDP+USP → CUDA Graphs → Cache-DiT (ordered DAG) | ⚠️ **Partial** — TP + CUDA Graphs done; HSDP+USP CPU-static only; Cache-DiT not registered |

**Implication for reference alignment:** The UCPE / camera-control module is now
numerically aligned with NVlabs (§6.12a), and the Stage-1 sampling contract has
been tightened through native scheduler + softmax-UCPE fixes (§6.13m). However,
§6.13n–§6.13o show a separate long-sequence Stage-1 gap at 321 frames before
Stage-2. For short 9-frame alignment the dominant full-chain blocker remains
the LTX-2 refiner contract (§6.14); for production-length video, fix the
321-frame late-step Stage-1 drift first.

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

### 6.13f Stage-1 Latent Parity Probe ⚠️ 2026-05-28 late evening — frame 1 (generated) is anti-correlated

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

### 6.13l Native Scheduler Landed — Timestep Parity Achieved, Exposes Compounding noise_pred Drift ⚠️ 2026-05-29

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

**Interpretation.** Increasing Stage-1 denoise steps from `20` to `60`
at 9 frames barely changes generated-frame parity (`cos 0.9878 →
0.9861`). The dominant new failure appears when moving from 2 latent
frames to 41 latent frames: generated cosine falls to `0.7575` even at
20 steps. That points away from the scheduler step count and toward a
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

### 6.13g Step-0 Controlled-Input Probe — Bug Is In Model Forward ⚠️ 2026-05-28 late

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

12. ~~**Fix Stage-1 latent magnitude (§6.12).**~~ ✅ **Done 2026-05-28** — root cause localised to the cam branch (main-only latent was always in normal range). Cam branch rewritten to match NVlabs `BidirectionalGDNUCPESinglePathLiteLA`: single-path delta-rule recurrence (no Z denominator), `apply_fn_o` inverse output transform, and `_downscale_to_reference_rms` PostUCPERenorm. Latent at STAGE1_STEPS=1 dropped from `[-59, 61]` to `[-12.3, 11.8]`.

13. ⚠️ **Stage-1 native scheduler/softmax-UCPE alignment (§6.13m–§6.13o).** Controlled 3-step and 9-frame 20/60-step parity are strong (generated cos `0.986–0.988`). Production-length 321-frame / 20-step parity still drops to generated cos `~0.75`, but the gap is now localized: camera tensors, `chunk_index`, prompt/latent injection, timestep table, and step-0 model forward are aligned; divergence accumulates in late denoise steps, especially the step-18 update around `t≈392 → 87.7`.

14. ⚠️ **Stage-2 refiner contract alignment (§6.14).** The structural NVlabs contract is now ported: sink/current split, seed-42 current noise, per-token timestep, video-only streaming mask, x0/velocity loop, and terminal-zero sigma. Same-source native vs official-manual refiner is generated-frame latent MAE=0.3589 / cos=0.8912. Remaining gap is accumulated bf16 drift across vLLM-native vs diffusers LTX-2 layers, not scheduler, prompt connector, or missing weights.

---

## 8. Outstanding Work — GPU Required

1. **Fix 321-frame Stage-1 late-step drift (§6.13o).** Input camera tensors and `chunk_index` are ruled out, and step-0 controlled forward parity is high (`noise_pred` generated cos `0.9811`). The collapse appears late in 321-frame / 20-step denoise (`generated latent cos 0.9915 at step 15 → 0.7587 at step 19`). Next compare scheduler update semantics and block-level model internals at controlled late-step latents (steps 15/18) before spending GPU time on 321-frame / 60-step.

2. **Regenerate same-source Stage-2 refiner acceptance artifacts (§6.14).** For short 9-frame alignment, the next correctness target is still native LTX-2 refiner parity against NVlabs on controlled Stage-1 latents. Regenerate the official refiner baseline and prompt-connector dump in one run, then decide whether strict cos ≥0.98 requires a diffusers-exact fallback or a refiner-specific torch-linear layer stack.

3. **Wire PSNR ≥ 30 / SSIM-Y ≥ 0.93** once harness produces a qualifying result.

4. **GDN multi-step parity** at full 704×1280 / 321 frames vs official NVlabs path.

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

6. Localize the 321-frame Stage-1 parity drop from §6.13n.
7. Re-run short 9-frame full-chain reference alignment after any Stage-1 fix.
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
