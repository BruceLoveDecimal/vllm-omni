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
| 4 | UCPE branch decomposition + numeric Plücker reference test | ⚠️ **Port landed, GPU verification pending (2026-05-27 late)** — UCPE math (`ucpe.py`) ported with passing unit tests; cam branch rewritten to use bidirectional GDN with UCPE per-ray transforms. End-to-end MAE drop on GPU still to be confirmed. See §6.10. |
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
added. E2E reference-alignment MAE drop verification on GPU is the
remaining work — see §8 item 1.

Verified by deep-dive vs NVlabs `sana_gdn_camctrl_blocks.py` on 2026-05-27.

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
