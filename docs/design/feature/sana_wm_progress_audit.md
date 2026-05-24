# Sana-WM Integration — Progress Audit

> **Audit date:** 2026-05-25 (revision 2)
> **Branch:** `feat/sana_wm`
> **Branch HEAD:** `76b8138e fix(sana-wm): validate refiner e2e path`
> **Pushed to:** `fork/feat/sana_wm` (`BruceLoveDecimal/vllm-omni`)
> **Worktree state at audit:** clean (sync'd with fork HEAD; only `.agents/` untracked)
> **Spec (single source of truth):**
> [`sana_wm_integration.md`](sana_wm_integration.md) (1120 lines).
> **Tracking issue:**
> [vllm-project/vllm-omni#3656](https://github.com/vllm-project/vllm-omni/issues/3656).
>
> This is a snapshot. The previous audit (2026-05-24) is preserved
> via git; this revision tracks the substantial GPU-validation push
> that landed overnight. Re-run by reading the spec end-to-end, then
> walking the file inventory in §3 and the P0–P5 table in §4.

## 1. TL;DR

Overall progress against the post-release implementation plan:
**roughly 60–65%** (up from 40–45% in the 2026-05-24 snapshot).

- **P0 + P0.5 (scaffold + official CLI bridge): 100% done.** Unchanged.
- **P1 (Stage-1 reference forward path): ~95% done.** Real-GPU 1-step
  shape verified during the overnight GPU run; all 50 unit tests pass,
  2 skipped (gated by hardware not present in this environment).
- **P2 (real fused Triton GDN + offline native e2e): ~35% done.**
  Plücker rasterization remains complete; `gated_deltanet_triton.py`
  is **still a PyTorch reference fallback**, not the actual Triton
  kernel. Cindy's msg `234705a3` decomposes the blocker into four
  concrete sub-items — see §6.
- **P3 (LTX-2 refiner attach + dual text encoder): ~70% done.**
  `pipeline_sana_wm_two_stages.py` has grown from 45 → 163 lines.
  `Gemma3ForConditionalGeneration` (refiner text encoder),
  `LTX2TextConnectors` (refiner connectors), and
  `LTX2VideoTransformer3DModel` (refiner transformer) are now **all
  real-loadable** from `refiner/text_encoder/`, `refiner/connectors/`,
  and `refiner/transformer/`. The 9-minute e2e test (RTX PRO 6000
  Blackwell 96 GB) produced a `(8, 704, 1280, 3)` video tensor.
  **What's still missing:** the in-process Stage-1 → refiner denoising
  loop. Today the refiner components are *loaded* in vLLM-Omni but the
  actual Stage-2 forward pass is delegated to NVlabs' CLI subprocess
  through the bridge; a true native two-stage forward is still pending.
- **P4 (online serving + recipe + accuracy): ~50% done.**
  `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md` landed
  (194 lines); offline example committed; OpenAI-style online serving
  hookup and `accuracy ≥ PSNR 30 / SSIM 0.93` reference-alignment tests
  still pending.
- **P5 (SP/USP/CFG-parallel/HSDP + dfx perf): ~10% done.**
  `test_sana_wm_hsdp.py`, `test_sana_wm_cfg_parallel_adaptation.py`,
  `test_sana_wm_cfg_parallel_parity.py`, and the dfx-perf JSON now
  exist as `pytest.skip(...)` / "pending_gpu_validation" stubs that
  document the contract. No actual SP/USP wiring yet.

The system today can produce real videos end-to-end via the **official
CLI bridge** (Stage 1 + LTX-2 refiner inside the NVlabs subprocess).
The vLLM-Omni native path can now (a) load every refiner component,
(b) run Stage-1 reference math under a PyTorch GDN fallback, and (c)
decode through the real `AutoencoderKLLTX2Video`. It still relies on
the CLI bridge for actual Stage-2 denoising and for production-quality
Stage-1 generation.

## 1.1. Changes since 2026-05-24 snapshot

Five commits landed on `feat/sana_wm` and were pushed to `fork`:

```text
76b8138e  fix(sana-wm): validate refiner e2e path
361c5065  feat(sana-wm): support external official refiner root
40381be0  fix(sana-wm): set PYTHONPATH for official CLI
10922de5  fix(sana-wm): use official refiner root flag
e21b582e  feat(sana-wm): add native smoke and refiner loaders
4ebdcc2e  feat: add Sana-WM integration scaffold       ← previous HEAD
```

Concrete deltas:

- **`pipeline_sana_wm.py`:** 554 → 696 lines. `AutoencoderKLLTX2Video.from_pretrained(..., subfolder="vae", local_files_only=True)` is now wired; `AutoModelForCausalLM.from_pretrained(text_encoder_model_id, ...)` is wired for Stage-1 Gemma-2-2B-IT (env-overridable via `VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER`); `_native_smoke_prompt_embeds` falls back to deterministic `hash_smoke` only when the real encoder cannot be obtained.
- **`pipeline_sana_wm_two_stages.py`:** 45 → 163 lines. Adds `_ensure_refiner_text_encoder` (real `Gemma3ForConditionalGeneration` load), `_ensure_refiner_connectors` (real `LTX2TextConnectors` load), `_ensure_refiner_transformer` (real `LTX2VideoTransformer3DModel` via `create_transformer_from_config` + `safetensors.load_file`), and `ensure_refiner_components` orchestrator. Forward path triggers loading when `sampling_params.extra_args["sana_wm_load_refiner_components"]` is set; the actual Stage-2 denoising still routes through Stage-1's official-CLI path.
- **`official_backend.py`:** 344 → 401 lines. Supports external official refiner root (`SANA_WM_REFINER_ROOT_ENV`), CLI flag, and `PYTHONPATH` propagation so the NVlabs subprocess can find the locally cached weights.
- **`gated_deltanet_triton.py`:** unchanged file name; +236 line edits but still a PyTorch reference, no Triton kernel yet.
- **`recipes/Efficient-Large-Model/SANA-WM-bidirectional.md`:** new, 194 lines. Documents checkpoint layout, the three execution paths (official / native smoke / two-stage), and the GPU requirement bands.
- **New test files (7 of 8 spec-listed):**
  - `test_sana_wm_camera_control.py` — real test (Plücker condition shape).
  - `test_sana_wm_pipeline.py` — real test (component-slot declarations + preprocess).
  - `test_sana_wm_two_stages.py` — real test (isolated refiner slots + loader pre-conditions).
  - `test_sana_wm_cfg_parallel_adaptation.py` — `pytest.skip` stub.
  - `test_sana_wm_hsdp.py` — `pytest.skip` stub.
  - `tests/examples/offline_inference/test_sana_wm_cfg_parallel_parity.py` — `pytest.skip` stub.
  - `tests/dfx/perf/tests/test_sana_wm_vllm_omni.json` — `{"status": "pending_gpu_validation"}`.
- **`test_sana_wm_scaffold.py`:** 664 → 766 lines (+ many new cases including refiner-loader contract).
- **`test_sana_wm_video_e2e.py`:** 81 → 91 lines. Now reads `SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline`, `SANA_WM_E2E_NUM_FRAMES`, and accepts deterministic intrinsics to bypass the optional `pi3` dependency in NVlabs' code.
- **GPU validation result (Cindy's msg `1dd5a346`):** `seeta-gpu` RTX PRO 6000 Blackwell 96 GB; `50 passed, 2 skipped` on the SANA-WM test suite; LTX-2 refiner component load uses ~60.6 GiB VRAM; end-to-end `SanaWmTwoStagesPipeline` + official CLI bridge + local refiner ran for ~9 minutes and produced a `(8, 704, 1280, 3)` video tensor.
- **Three concrete fixes shipped during GPU validation:** default `TORCHDYNAMO_DISABLE=1` for the CLI subprocess to dodge a TorchInductor/FLA-import quirk on Blackwell; two-stage refiner loader now creates a temporary `VllmConfig` only when one is not already active (so real worker contexts are not overwritten); e2e test exposes the env knobs and intrinsics path so optional NVlabs dependencies are not required.

## 2. Method

For each spec line item below, the audit:

1. Confirms whether the file exists on `feat/sana_wm` worktree
   (including uncommitted changes).
2. Reads the file's intent (docstring, exported surface, key
   class/function signatures).
3. Cross-checks against the relevant subsection of
   [`sana_wm_integration.md`](sana_wm_integration.md):
   §§ "Confirmed Release Inventory", "Concrete Architecture",
   "Updated Implementation Plan", "Interface declarations",
   "Request schema", "Recommended e2e test layout", "Phased rollout".

The status icons mean:

- ✅ Implementation matches the spec.
- ⚠️ Implementation is present but partial, or relies on a non-native
  backend (e.g. NVlabs CLI bridge) where the spec also requires a
  native equivalent.
- ❌ Spec item is missing from the worktree.

## 3. File inventory (as of `feat/sana_wm` worktree)

```text
vllm_omni/diffusion/models/sana_wm/
  __init__.py                       157 LOC   ✅ public surface; re-exports camera/transformer/two-stages
  config.py                         165 LOC   ✅ SanaWmConfig + from_yaml
  sana_wm_transformer.py           1084 LOC   ✅ reference DiT — GDN PyTorch fallback, Wan RoPE, camera branch
  pipeline_sana_wm.py               696 LOC   ✅ Stage-1 pipeline w/ HF download, validation, 3-backend dispatch,
                                              real Gemma-2-2B-IT + LTX-2 VAE loaders
  pipeline_sana_wm_two_stages.py    163 LOC   ⚠️ real refiner-component loaders shipped; in-process Stage-2 forward
                                              still delegates to Stage-1 / official CLI path
  camera_control.py                 332 LOC   ✅ Plücker / raymap + all camera schemas
  weight_mapping.py                  49 LOC   ✅ Stage-1 prefix remap helper (slightly extended)
  gated_deltanet_triton.py          244 LOC   ⚠️ PyTorch reference fallback; real Triton kernel NOT written
  official_backend.py               401 LOC   ✅ NVlabs CLI bridge w/ external refiner root + PYTHONPATH wiring
  native_backend.py                 489 LOC   ✅ direct-import NVlabs Python modules (no subprocess)
  scheduling_sana_wm.py              52 LOC   ✅ SanaWmFlowDpmScheduler w/ inference_flow_shift=9.8
vllm_omni/model_executor/stage_input_processors/sana_wm.py  309 LOC   ✅ request payload schema validator
recipes/Efficient-Large-Model/SANA-WM-bidirectional.md      194 LOC   ✅ checkpoint layout + 3 backends + GPU bands
tests/diffusion/models/sana_wm/test_sana_wm_scaffold.py     766 LOC   ✅ ~30+ test functions (incl. refiner-loader contract)
tests/diffusion/models/sana_wm/test_sana_wm_pipeline.py      37 LOC   ✅ real test (component declarations + preprocess)
tests/diffusion/models/sana_wm/test_sana_wm_two_stages.py    28 LOC   ✅ real test (isolated refiner slots + loader gate)
tests/diffusion/models/sana_wm/test_sana_wm_camera_control.py  16 LOC ✅ real test (Plücker shape)
tests/diffusion/models/sana_wm/test_sana_wm_hsdp.py          10 LOC   ⚠️ pytest.skip stub
tests/diffusion/models/sana_wm/test_sana_wm_cfg_parallel_adaptation.py
                                                             10 LOC   ⚠️ pytest.skip stub
tests/model_executor/stage_input_processors/test_sana_wm.py 236 LOC   ✅ request schema tests
tests/e2e/accuracy/test_sana_wm_video_e2e.py                 91 LOC   ✅ env-driven gates; supports two-stage class
tests/examples/offline_inference/test_sana_wm_cfg_parallel_parity.py
                                                             10 LOC   ⚠️ pytest.skip stub
tests/dfx/perf/tests/test_sana_wm_vllm_omni.json              7 LOC   ⚠️ {"status": "pending_gpu_validation"}
examples/offline_inference/sana_wm/sana_wm.py               130 LOC   ✅ offline example w/ camera + action support
```

All 8 spec-listed files now exist on disk (4 are real tests, 4 are
documented stubs gated on GPU validation).

## 4. Phase-by-phase status

| Phase | Spec scope | Concrete status | Coverage |
| --- | --- | --- | --- |
| **P0** — scaffold + registry + pipeline classes + processor stub | All declared in spec §"Phased rollout" | 4 pipeline registry entries, 4 metadata entries, pre/post-process registered, `SanaWmConfig` dataclass, `normalize_sana_wm_payload`, all tests collect, all compileall passes | ✅ **100%** |
| **P0.5** — official CLI backend bridge for GPU smoke | `official_backend.py` ships, supports external refiner root (`SANA_WM_REFINER_ROOT_ENV`), `PYTHONPATH` propagation, and `TORCHDYNAMO_DISABLE=1` workaround for Blackwell | `official_backend.py` 401 LOC; `test_sana_wm_video_e2e.py` 91 LOC gated by `SANA_WM_E2E=1` and class env knob | ✅ **100%** |
| **P1** — Stage-1 weight load (softmax fallback only) + image/camera input packing | `weight_mapping.remap` + `load_weights` + Wan-RoPE positional path + GDN PyTorch fallback + image/camera packing through `normalize_sana_wm_payload` | All present statically; real-GPU validation reported as passing `50 passed, 2 skipped` on `seeta-gpu` (RTX PRO 6000 Blackwell 96 GB). | ✅ **~95%** |
| **P2** — `GatedDeltaNetTriton` kernel + Plücker camera injection → offline native e2e | `camera_control.py` (Plücker, raymap, action DSL, all camera schemas) is done; `gated_deltanet_triton.py` remains **only** a PyTorch reference recurrence — no Triton kernel | Plücker done (✅), Triton GDN missing (❌); offline native e2e for the Stage-1 reference path is exercised inside the CLI bridge (production-grade native is still pending) | ⚠️ **~35%** |
| **P3** — LTX-2 refiner attach via `refiner/transformer/`, `refiner/connectors/`, and dual-text-encoder loading | `pipeline_sana_wm_two_stages.py` (163 LOC) **now loads** `Gemma3ForConditionalGeneration` from `refiner/text_encoder/`, `LTX2TextConnectors` from `refiner/connectors/`, and `LTX2VideoTransformer3DModel` (via `create_transformer_from_config` + `safetensors.load_file`) from `refiner/transformer/`. The `ensure_refiner_components` orchestrator is wired into `forward()` behind `extra_args["sana_wm_load_refiner_components"]`. Cindy's GPU run loaded all three components into ~60.6 GiB VRAM. | **Pending:** the in-process Stage-1 → refiner denoising loop. Current `forward()` defers Stage-2 inference to NVlabs' CLI subprocess after loading the components in-process. | ⚠️ **~70%** |
| **P4** — Online serving + recipes + accuracy thresholds | `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md` (194 LOC) committed; offline example present; the `(8, 704, 1280, 3)` e2e tensor proves the full pipeline runs end-to-end on a real Blackwell GPU; online OpenAI-style serving hookup and explicit PSNR/SSIM thresholds against NVlabs reference are still pending | Recipe ✅; e2e via CLI bridge ✅; online serving ❌; accuracy reference-alignment ❌ | ⚠️ **~50%** |
| **P5** — SP/USP/CFG-parallel/HSDP + dfx perf; cache-DiT if useful | `test_sana_wm_hsdp.py`, `test_sana_wm_cfg_parallel_adaptation.py`, `test_sana_wm_cfg_parallel_parity.py`, and `test_sana_wm_vllm_omni.json` exist as documented stubs. No actual SP/USP wiring; no DiT cache integration. | All four stub files in place ✅; actual perf and parallelism wiring ❌ | ⚠️ **~10%** |

## 5. What's already strong

- **Spec discipline.** `SanaWmConfig.from_yaml` reads the root
  `config.yaml` and never hardcodes architecture constants outside
  `SanaWmConfig` defaults. Matches spec §"`config.yaml` is the single
  source of truth".
- **Interface declarations** match spec §"Interface declarations"
  exactly:
  `support_image_input = True`, `color_format = "RGB"`,
  `_dit_modules = ["transformer"]`,
  `_encoder_modules = ["text_encoder", "camera_encoder"]`
  (Stage 1) or `+ ["refiner_text_encoder", "refiner_connectors"]`
  (two-stages), `_vae_modules = ["vae"]`,
  `_resident_modules = ["refiner_transformer"]` on two-stages.
- **Backend dispatch is explicit.** `forward()` chooses between
  `run_sana_wm_official_backend`, `run_sana_wm_native_backend`, and
  the in-process `_run_native_smoke_backend` based on env vars
  (`VLLM_OMNI_SANA_WM_OFFICIAL_REPO`,
  `VLLM_OMNI_SANA_WM_NATIVE_SMOKE`,
  `VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI`) and request fields. Errors
  are loud and explicit.
- **Plücker / raymap rasterization** supports `extrinsics_4x4`,
  `c2w_4x4`, `w2c_4x4`, `relative_6dof`, and the WASD/IJKL action
  DSL (`action_string_to_c2w`). Matches the request schema in spec
  §"Request schema (concrete recommendation)".
- **Test coverage** is broad for the scope that exists. 29 tests in
  `test_sana_wm_scaffold.py` cover registry, exports, config parse,
  scheduler shift, transformer shape, hybrid softmax routing, Plücker
  post-attention blocks, GDN reference forward, download patterns,
  local path validation (incl. missing-refiner case), weight mapping,
  fail-fast surface, native smoke opt-in, deterministic prompt
  embeddings, official-backend env handling, action rollout, native
  backend script resolution, Stage-1 remap, audit-materialization,
  reject of unconsumed/unmapped keys, preprocess normalization,
  registry lazy load, checkpoint resolution from `od_config`, default
  stage config layout, and weight-source declaration.

## 6. What's weak

(Items 1, 4, 5, 6, 7 from the 2026-05-24 snapshot are now resolved
and recorded in §1.1.)

1. **In-process Stage-1 → refiner forward is not implemented.**
   `pipeline_sana_wm_two_stages.py::forward` loads the refiner
   components and then calls `super().forward(req, ...)`, which is
   the Stage-1 pipeline path. When the official-backend env vars
   are set, the actual two-stage denoising executes inside the
   NVlabs CLI subprocess via the bridge. A native two-stage
   denoising loop (Stage-1 latent → `LTX2TextConnectors` →
   `LTX2VideoTransformer3DModel` refinement → VAE decode) is still
   pending.
2. **`gated_deltanet_triton.py` is still PyTorch-only.** Cindy
   (msg `234705a3`) decomposed the blocker into four sub-items:
   - GDN ≠ standard attention — uses `A_log`, `beta_proj`, `conv_k`
     state-space / recurrent parameters; cannot reuse the generic
     vLLM attention backend interface.
   - Triton kernel must reproduce the bidirectional scan +
     gating + depthwise conv + chunking + dtype boundaries of the
     official FLA kernel; shape-only parity is insufficient — video
     quality will drift if the math is off.
   - The native Stage-1 path must integrate with vLLM-Omni's token
     layout, camera Plücker injection, scheduler step, VAE/refiner
     hand-off, device-offload, and (eventually) SP/CFG-parallel.
     `native smoke` only proves "weights load + forward shape OK",
     not production parity.
   - Reference-alignment tests against the official NVlabs path at
     identical `(seed, prompt, camera, steps)` are not yet wired.
3. **Stage-1 reference-alignment vs official CLI not yet measured.**
   The full e2e through the CLI gave a `(8, 704, 1280, 3)` tensor,
   but there is no automated comparison that holds the native path
   to PSNR ≥ 30 / SSIM ≥ 0.93 against that reference. This is the
   gate the spec sets for P4 closure.
4. **Online serving entry point still missing.** No OpenAI-style
   `/v1/videos/generations` wiring nor request schema mapping to
   `additional_information["sana_wm"]` outside of offline tests.
5. **P5 stubs document but do not exercise SP/USP/CFG-parallel/HSDP
   or dfx perf.** The four stub files mark the intent; the wiring
   has not started.

## 7. Outstanding work that does NOT require GPU access

(The 2026-05-24 list of six items is now mostly resolved — see §1.1.
Remaining non-GPU work.)

1. **Implement the in-process Stage-1 → refiner denoising loop.**
   Even before tackling the Triton GDN kernel, `SanaWmTwoStagesPipeline.forward`
   can be extended to actually invoke the loaded
   `LTX2VideoTransformer3DModel` for the Stage-2 refinement step
   instead of deferring to the official-CLI subprocess. The
   loaders, the text-encoder, the connectors, and the VAE are all
   in place.
2. **Online serving wiring.** Add the `/v1/videos/generations`
   request-side mapping (image + camera trajectory + action) into
   `additional_information["sana_wm"]`. Currently only the offline
   path is exercised.
3. **Reference-alignment test harness.** Write the comparison
   between vLLM-Omni native Stage-1 output and the CLI bridge
   output at the latent level (PSNR-on-latents is cheaper than
   video PSNR and runs without ffmpeg); land it as a skip-by-default
   test that flips on once the native path returns a valid latent.
4. **Recipe extensions.** The existing recipe documents the three
   execution paths well; extending it with the
   `SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline` invocation and
   the `60.6 GiB` measured GPU footprint would help reviewers
   understand the actual cost.
5. **Backfill SP/USP/CFG-parallel/HSDP stubs into real CPU-static
   smoke** where possible (registry presence, attribute declarations,
   etc.), keeping the `pytest.skip` for the GPU-dependent assertions.

## 8. Outstanding work that DOES require GPU access

1. **Triton GDN kernel + reference-alignment math.** See Cindy's
   four-point blocker breakdown captured in §6 item 2.
2. **vLLM-Omni native two-stage forward.** The end-state has Stage-1
   reference latent → in-process `LTX2VideoTransformer3DModel`
   refinement → `AutoencoderKLLTX2Video` decode, all inside vLLM-Omni
   without the CLI subprocess.
3. **Accuracy thresholds.** PSNR ≥ 30 / SSIM-Y All ≥ 0.93 vs the
   CLI-bridge reference once the native path produces a tensor.
4. **`SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline` smoke
   on more GPUs.** Validated on RTX PRO 6000 Blackwell 96 GB; not
   yet on H100/H800/A100 — useful before claiming Tier 2/3 support.
5. **dfx / HSDP / SP / CFG-parallel sweeps** per spec §"GPU tier
   policy" Tier 3.

## 9. Suggested next-step ordering

The biggest single open item is **the in-process two-stage forward
loop** (§7 item 1). It does not require new GPU work — the loaders
and the CLI-bridge reference output already give us everything
needed to write the loop and then validate it against the bridge
output once the GPU instance is open. After that:

1. Write the in-process Stage-1 → refiner forward (§7 item 1).
2. Plumb the latent-level reference-alignment test (§7 item 3) and
   run it against the CLI bridge output to confirm parity.
3. Online serving wiring (§7 item 2).
4. Triton GDN kernel (§8 item 1).
5. SP/USP/CFG-parallel/HSDP wiring (§8 item 5).
6. Accuracy thresholds (§8 item 3).

## 10. References

- Spec: [`sana_wm_integration.md`](sana_wm_integration.md)
- Tracking issue: <https://github.com/vllm-project/vllm-omni/issues/3656>
- HF release: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
- NVlabs reference: <https://github.com/NVlabs/Sana>
- In-tree LTX-2 precedent:
  `vllm_omni/diffusion/models/ltx2/{pipeline_ltx2,pipeline_ltx2_latent_upsample}.py`
  and `tests/diffusion/models/ltx2/`.
- Branch base: `4ebdcc2e feat: add Sana-WM integration scaffold`
  (single commit on `feat/sana_wm` at audit time).
