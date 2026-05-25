# Sana-WM Integration — Progress Audit

> **Audit date:** 2026-05-26 (revision 7)
> **Branch:** `feat/sana_wm`
> **Implementation HEAD:** `0ec9a7af feat(sana-wm): wire online and alignment smokes`
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
**roughly 80%** (up from 40–45% in the 2026-05-24 snapshot, and from
~75% in revision 6).

- **P0 + P0.5 (scaffold + official CLI bridge): 100% done.** Unchanged.
- **P1 (Stage-1 reference forward path): ~95% done.** Real-GPU 1-step
  shape verified during the overnight GPU run; SANA-WM units now report
  `54 passed, 2 skipped` on the GPU instance (up from 50 passed in
  revision 2).
- **P2 (real fused Triton GDN + offline native e2e): ~70% done.**
  The NVlabs Apache-2.0 fused GDN/chunkwise Triton kernels are now
  vendored under `sana_wm/`, Stage-1 GDN prefers the fused path on CUDA,
  and `VLLM_OMNI_SANA_WM_REQUIRE_TRITON_GDN=1` forces fail-fast if the
  pipeline would fall back. Small fused-vs-reference parity and the
  in-process e2e both pass on RTX PRO 6000 Blackwell.
- **P3 (LTX-2 refiner attach + dual text encoder): ~92% done.**
  `pipeline_sana_wm_two_stages.py` has grown from 45 → 476 lines.
  `Gemma3ForConditionalGeneration` (refiner text encoder),
  `LTX2TextConnectors` (refiner connectors), and
  `LTX2VideoTransformer3DModel` (refiner transformer) are now **all
  real-loadable** from `refiner/text_encoder/`, `refiner/connectors/`,
  and `refiner/transformer/`. The official-bridge e2e produced a
  `(8, 704, 1280, 3)` video tensor. The in-process refiner now runs
  end-to-end on RTX PRO 6000 Blackwell 96 GB **for both
  latent-output and decoded-video output** (latent: 200.92 s;
  decoded `(1, 9, 704, 1280, 3)`: 212.22 s).
  **What's still missing:** numerical reference-alignment results against
  the NVlabs bridge and multi-step quality validation.
- **P4 (online serving + recipe + accuracy): ~70% done.**
  `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md` landed
  (217 lines); offline example committed; OpenAI-style online serving
  hookup now accepts the SANA-WM camera payload through
  `/v1/videos/generations`; the e2e test now includes a reference-alignment
  harness. GPU numeric thresholds are still pending.
- **P5 (SP/USP/CFG-parallel/HSDP + dfx perf): ~30% done.**
  SANA-WM now inherits `CFGParallelMixin`, Stage-1 exposes HSDP shard
  conditions for `blocks.*`, the skip stubs were replaced with CPU-static
  contract tests, and the dfx perf JSON now contains concrete official,
  in-process, and cfg-parallel benchmark cases. Actual multi-GPU sweeps
  are still pending.

The system today can produce real videos end-to-end via the **official
CLI bridge** (Stage 1 + LTX-2 refiner inside the NVlabs subprocess).
The vLLM-Omni native path can now (a) load every refiner component,
(b) run Stage-1 GDN through the fused Triton kernel on CUDA with PyTorch fallback, (c) run one
in-process LTX-2 refiner denoising step on the Stage-1 latent, and (d)
decode through the real `AutoencoderKLLTX2Video`. It still relies on the
CLI bridge as the decoded-video reference until multi-step numeric alignment is measured.

## 1.4. Changes since revision 6

Commit `0ec9a7af feat(sana-wm): wire online and alignment smokes` landed after
the fused-GDN revision:

- Added an online serving path for SANA-WM camera control:
  `VideoGenerationRequest.sana_wm`, multipart form parsing for `sana_wm`,
  `/v1/videos/generations` and `/v1/videos/generations/sync` aliases, and
  request mapping from `sana_wm` / `extra_params.sana_wm` into
  `prompt["sana_wm"]`.
- Added a reference-alignment harness in
  `tests/e2e/accuracy/test_sana_wm_video_e2e.py`. The opt-in
  `SANA_WM_E2E_REFERENCE_ALIGNMENT=1` path runs the official bridge and
  in-process refiner path and compares decoded frames with an env-controlled
  MAE threshold.
- Multi-step in-process refiner smoke is now an explicit test contract:
  the fake-component unit test exercises `sana_wm_inprocess_refiner_steps=2`,
  the e2e path reads `SANA_WM_E2E_REFINER_STEPS`, and the recipe documents a
  2-step smoke command.
- Replaced P5 skip stubs with CPU-static contract tests:
  `test_sana_wm_hsdp.py` verifies Stage-1 HSDP block matching,
  `test_sana_wm_cfg_parallel_adaptation.py` verifies `CFGParallelMixin`
  inheritance and combine math, and the offline cfg-parallel parity test
  checks the example/perf config contract.
- Replaced the placeholder dfx perf JSON with concrete benchmark entries for
  official-bridge, in-process fused-GDN, and cfg-parallel eager cases.
- Local static validation: `compileall` passed for the changed packages/tests,
  `git diff --check` passed, and the SANA-WM perf JSON validates with
  `python3 -m json.tool`. Local pytest is still blocked on this macOS Python by
  missing `torch`. The current `sana-wm-seeta` SSH endpoint returns
  `Connection closed by 198.18.4.22 port 34732`, so GPU long-run validation was
  not rerun for this revision.

## 1.2. Changes since revision 2

Five commits landed on `feat/sana_wm` and were pushed to `fork` between
the revision-2 audit commit (`2d8253cc`) and this revision-5 snapshot:

```text
b69a24a3  test(sana-wm): accept decoded in-process refiner output  ← HEAD
41b776ef  docs(sana-wm): record in-process refiner validation
f9c09825  fix(sana-wm): keep refiner tensors on runtime device
08c8c37d  test(sana-wm): expose in-process refiner e2e switch
abfe3d4a  feat(sana-wm): add in-process refiner step
2d8253cc  docs(sana_wm): refresh progress audit (2026-05-25)        ← rev 2 HEAD
```

Concrete deltas:

- **In-process Stage-1 → LTX-2 refiner forward landed** (`abfe3d4a`).
  `SanaWmTwoStagesPipeline` exposes an opt-in native refiner path
  behind `sampling_params.extra_args["sana_wm_inprocess_refiner"]=True`.
  When enabled, Stage 1 produces a latent, the bundled Gemma3 refiner
  text encoder + `LTX2TextConnectors` + `LTX2VideoTransformer3DModel`
  run a minimal sigma loop in-process, and optional VAE decode follows.
  Default behaviour is unchanged — the previously validated official
  CLI bridge remains the reference for now.
- **E2E switch + fake-component coverage** (`08c8c37d`). New
  `SANA_WM_E2E_INPROCESS_REFINER=1` env gate; fake-component unit tests
  cover the refiner prompt encoder + in-process step call contract.
- **Refiner device-discipline fix** (`f9c09825`). Root cause: vLLM
  `RMSNorm(has_weight=False)` stores `weight` as a plain tensor
  attribute, not as a parameter or buffer. `Module.to()` and the
  first migration helper missed it, causing CUDA activations to meet
  CPU RMSNorm weights inside the LTX-2 refiner.
  `_force_module_tensors_to` now migrates parameters, buffers, **and**
  plain tensor attributes. Regression test
  `test_sana_wm_two_stage_force_module_tensors_handles_plain_tensor_attrs`
  pins the behaviour.
- **In-process refiner validation now passes on GPU.** Command for
  the latent path:
  `SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline`,
  `SANA_WM_E2E_INPROCESS_REFINER=1`,
  `SANA_WM_E2E_OUTPUT_TYPE=latent`,
  `SANA_WM_E2E_REFINER_STEPS=1`.
  Result on `sana-wm-seeta` RTX PRO 6000 Blackwell 96 GB:
  `1 passed, 4 warnings in 200.92s`.
- **Decoded-output validation also passes** (`b69a24a3`). Setting
  `SANA_WM_E2E_OUTPUT_TYPE=np` runs the same e2e, producing
  `1 passed, 4 warnings in 212.22s` with output shape
  `(1, 9, 704, 1280, 3)`. The e2e test now collapses the leading
  batch dimension when the in-process refiner returns a 5-D
  `(B, F, H, W, C)` tensor before asserting against the 4-D
  `(F, H, W, C)` contract.
- **Recipe updated for both paths** (`41b776ef` + `b69a24a3`).
  `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md` now
  documents the latent-only smoke (`SANA_WM_E2E_OUTPUT_TYPE=latent`)
  and the heavier decoded-output smoke (`SANA_WM_E2E_OUTPUT_TYPE=np`),
  records the 60.6 GiB measured VRAM footprint when the refiner trio
  is resident, and pins the `(1, 9, 704, 1280, 3)` shape from the
  Blackwell 96 GB run.
- **Unit validation:** `tests/diffusion/models/sana_wm`
  + `tests/model_executor/stage_input_processors/test_sana_wm.py`
  now pass as `54 passed, 2 skipped` on the GPU instance (up from
  50 passed in revision 2).

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
- **`pipeline_sana_wm_two_stages.py`:** 45 → 476 lines. Adds `_ensure_refiner_text_encoder` (real `Gemma3ForConditionalGeneration` load), `_ensure_refiner_connectors` (real `LTX2TextConnectors` load), `_ensure_refiner_transformer` (real `LTX2VideoTransformer3DModel` via `create_transformer_from_config` + `safetensors.load_file`), `ensure_refiner_components`, and the opt-in in-process latent refiner step. The path is validated for latent output; decoded-output and accuracy alignment remain open.
- **`official_backend.py`:** 344 → 401 lines. Supports external official refiner root (`SANA_WM_REFINER_ROOT_ENV`), CLI flag, and `PYTHONPATH` propagation so the NVlabs subprocess can find the locally cached weights.
- **`gated_deltanet_triton.py` + fused kernels:** the wrapper now dispatches to vendored NVlabs Triton kernels on CUDA and keeps the PyTorch recurrence as fallback; `VLLM_OMNI_SANA_WM_REQUIRE_TRITON_GDN=1` validates fail-fast fused execution.
- **`recipes/Efficient-Large-Model/SANA-WM-bidirectional.md`:** new, 217 lines. Documents checkpoint layout, the three execution paths (official / native smoke / two-stage), fused-GDN controls, and the GPU requirement bands.
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


## 1.3. Changes since revision 5

Triton GDN work landed after `b69a24a3`:

- Vendored the Apache-2.0 NVlabs fused GDN implementation into
  `vllm_omni/diffusion/models/sana_wm/fused_gdn.py` and
  `fused_gdn_chunkwise.py`, with imports rewritten to stay inside the
  vLLM-Omni package.
- Added `triton_bidirectional_gated_delta_net_from_qkv`, which wraps
  the fused kernel from raw `[B, N, 3, H, D]` QKV and returns the same
  `[B, H, D, N]` layout as the PyTorch reference recurrence.
- Wired `SanaWmSelfAttention._forward_gdn` to prefer the fused kernel on
  CUDA, while retaining the PyTorch recurrence as a correctness fallback.
  `VLLM_OMNI_SANA_WM_DISABLE_TRITON_GDN=1` disables the fused path;
  `VLLM_OMNI_SANA_WM_REQUIRE_TRITON_GDN=1` makes fallback fail-fast for
  validation.
- GPU validation on RTX PRO 6000 Blackwell 96 GB:
  - fused-vs-reference small parity: `1 passed` (`atol=rtol=1e-2`);
  - SANA-WM unit/stage-input suite: `54 passed, 2 skipped`;
  - in-process e2e with `VLLM_OMNI_SANA_WM_REQUIRE_TRITON_GDN=1`,
    latent output, one refiner step: `1 passed, 4 warnings in 172.39s`.

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
  sana_wm_transformer.py           1120 LOC   ✅ Stage-1 DiT — fused GDN on CUDA + fallback, Wan RoPE, camera branch
  pipeline_sana_wm.py               698 LOC   ✅ Stage-1 pipeline w/ HF download, validation, 3-backend dispatch,
                                              CFGParallelMixin, real Gemma-2-2B-IT + LTX-2 VAE loaders
  pipeline_sana_wm_two_stages.py    476 LOC   ⚠️ real refiner-component loaders + in-process latent refiner
                                              and decoded-output smokes; reference alignment still open
  camera_control.py                 332 LOC   ✅ Plücker / raymap + all camera schemas
  weight_mapping.py                  49 LOC   ✅ Stage-1 prefix remap helper (slightly extended)
  gated_deltanet_triton.py          341 LOC   ✅ fused GDN wrapper + PyTorch fallback
  fused_gdn.py                      271 LOC   ✅ vendored NVlabs fused QK/RMS + BiGDN entry point
  fused_gdn_chunkwise.py           2269 LOC   ✅ vendored NVlabs chunkwise Triton kernels
  official_backend.py               401 LOC   ✅ NVlabs CLI bridge w/ external refiner root + PYTHONPATH wiring
  native_backend.py                 489 LOC   ✅ direct-import NVlabs Python modules (no subprocess)
  scheduling_sana_wm.py              52 LOC   ✅ SanaWmFlowDpmScheduler w/ inference_flow_shift=9.8
vllm_omni/model_executor/stage_input_processors/sana_wm.py  309 LOC   ✅ request payload schema validator
recipes/Efficient-Large-Model/SANA-WM-bidirectional.md      265 LOC   ✅ checkpoint layout + 3 backends + GPU bands
tests/diffusion/models/sana_wm/test_sana_wm_scaffold.py     774 LOC   ✅ ~30+ test functions (incl. refiner-loader contract)
tests/diffusion/models/sana_wm/test_sana_wm_pipeline.py      37 LOC   ✅ real test (component declarations + preprocess)
tests/diffusion/models/sana_wm/test_sana_wm_two_stages.py   165 LOC   ✅ real test (refiner slots, prompt encode, in-process step)
tests/diffusion/models/sana_wm/test_sana_wm_camera_control.py  16 LOC ✅ real test (Plücker shape)
tests/diffusion/models/sana_wm/test_sana_wm_gdn_triton.py     74 LOC   ✅ GPU fused-vs-reference parity
tests/diffusion/models/sana_wm/test_sana_wm_hsdp.py          29 LOC   ✅ HSDP block-matching contract
tests/diffusion/models/sana_wm/test_sana_wm_cfg_parallel_adaptation.py
                                                             30 LOC   ✅ CFG mixin + combine contract
tests/model_executor/stage_input_processors/test_sana_wm.py 236 LOC   ✅ request schema tests
tests/e2e/accuracy/test_sana_wm_video_e2e.py                167 LOC   ✅ e2e gates + reference-alignment harness
tests/examples/offline_inference/test_sana_wm_cfg_parallel_parity.py
                                                             25 LOC   ✅ offline cfg/perf contract
tests/dfx/perf/tests/test_sana_wm_vllm_omni.json            149 LOC   ✅ concrete official/in-process/cfg2 cases
examples/offline_inference/sana_wm/sana_wm.py               144 LOC   ✅ offline example w/ camera + action + in-process refiner support
```

All 8 spec-listed files now exist on disk. The previous skip stubs were
converted into CPU-static contract tests where possible; only the actual
multi-GPU/perf execution remains GPU-gated.

## 4. Phase-by-phase status

| Phase | Spec scope | Concrete status | Coverage |
| --- | --- | --- | --- |
| **P0** — scaffold + registry + pipeline classes + processor stub | All declared in spec §"Phased rollout" | 4 pipeline registry entries, 4 metadata entries, pre/post-process registered, `SanaWmConfig` dataclass, `normalize_sana_wm_payload`, all tests collect, all compileall passes | ✅ **100%** |
| **P0.5** — official CLI backend bridge for GPU smoke | `official_backend.py` ships, supports external refiner root (`SANA_WM_REFINER_ROOT_ENV`), `PYTHONPATH` propagation, and `TORCHDYNAMO_DISABLE=1` workaround for Blackwell | `official_backend.py` 401 LOC; `test_sana_wm_video_e2e.py` 167 LOC gated by `SANA_WM_E2E=1` and class env knob | ✅ **100%** |
| **P1** — Stage-1 weight load (softmax fallback only) + image/camera input packing | `weight_mapping.remap` + `load_weights` + Wan-RoPE positional path + GDN PyTorch fallback + image/camera packing through `normalize_sana_wm_payload` | All present statically; real-GPU validation reported as passing `54 passed, 2 skipped` on `seeta-gpu` (RTX PRO 6000 Blackwell 96 GB). | ✅ **~95%** |
| **P2** — `GatedDeltaNetTriton` kernel + Plücker camera injection → offline native e2e | `camera_control.py` (Plücker, raymap, action DSL, all camera schemas) is done; NVlabs fused GDN/chunkwise Triton kernels are vendored and wired into Stage-1 GDN on CUDA with PyTorch fallback. | Plücker ✅; fused-vs-reference small parity ✅; require-fused in-process e2e ✅; production multi-step numeric alignment still pending | ⚠️ **~70%** |
| **P3** — LTX-2 refiner attach via `refiner/transformer/`, `refiner/connectors/`, and dual-text-encoder loading | `pipeline_sana_wm_two_stages.py` now loads `Gemma3ForConditionalGeneration` from `refiner/text_encoder/`, `LTX2TextConnectors` from `refiner/connectors/`, and `LTX2VideoTransformer3DModel` (via `create_transformer_from_config` + `safetensors.load_file`) from `refiner/transformer/`. The in-process path now runs Stage-1 latent through the loaded LTX-2 refiner transformer and can return either latent output or decoded video output; unit tests exercise a 2-step sigma loop. | **Pending:** real-GPU multi-step quality validation; production-quality Stage-1 still depends on reference alignment. | ⚠️ **~92%** |
| **P4** — Online serving + recipes + accuracy thresholds | `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md` committed; offline example present; `/v1/videos/generations` and `/v1/videos/generations/sync` aliases now accept `sana_wm` form payloads; the e2e test includes an opt-in official-vs-in-process reference-alignment harness. | Recipe ✅; e2e via CLI bridge ✅; in-process latent e2e ✅; in-process decoded e2e ✅; online serving mapping ✅; reference-alignment harness ✅; numeric threshold results ❌ | ⚠️ **~70%** |
| **P5** — SP/USP/CFG-parallel/HSDP + dfx perf; cache-DiT if useful | `SanaWmPipeline` inherits `CFGParallelMixin`; Stage-1 exposes HSDP shard conditions for `blocks.*`; cfg/HSDP/offline parity tests are CPU-static contract tests; dfx perf JSON has concrete official, in-process, and cfg2 cases. | Static wiring ✅; real multi-GPU/perf sweeps ❌ | ⚠️ **~30%** |

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

1. **In-process refiner is still an integration smoke, not a quality
   replacement.** It now executes Stage-1 latent →
   `LTX2TextConnectors` → `LTX2VideoTransformer3DModel` → VAE decode,
   and has a 2-step unit contract, but real-GPU multi-step quality
   validation against the official bridge is still pending.
2. **Triton GDN is integrated but not fully quality-closed.** The
   fused path now runs in the real pipeline; remaining sub-items are:
   - GDN ≠ standard attention — uses `A_log`, `beta_proj`, `conv_k`
     state-space / recurrent parameters; cannot reuse the generic
     vLLM attention backend interface.
   - Multi-step numeric parity against the official NVlabs path is
     still unmeasured; small fused-vs-reference parity is covered.
   - The native Stage-1 path must integrate with vLLM-Omni's token
     layout, camera Plücker injection, scheduler step, VAE/refiner
     hand-off, device-offload, and (eventually) SP/CFG-parallel.
     `native smoke` plus the require-fused e2e proves the fused kernel
     executes, but not production video quality parity.
   - Reference-alignment tests against the official NVlabs path are now
     wired, but have not been run after this revision because the GPU SSH
     endpoint is currently closing connections.
3. **Stage-1 reference-alignment vs official CLI not yet measured.**
   The full e2e through the CLI gave a `(8, 704, 1280, 3)` tensor,
   and there is now an automated comparison harness, but no GPU run has
   established PSNR ≥ 30 / SSIM ≥ 0.93 or a stricter MAE threshold.
4. **Online serving is mapped but not live-server validated.** The
   `/v1/videos/generations` aliases and `sana_wm` payload mapping are
   covered by API unit tests; a real OpenAI-compatible server smoke on
   GPU is still pending.
5. **P5 static wiring exists, but distributed execution is not validated.**
   HSDP block conditions, CFG mixin inheritance, and dfx perf cases are
   concrete now; SP/USP/CFG-parallel/HSDP still need actual multi-GPU
   runs.

## 7. Outstanding work that does NOT require GPU access

(The four revision-6 non-GPU items are now implemented at code/test-contract
level. Remaining non-GPU work is mostly cleanup around thresholds and docs.)

1. **Tighten reference-alignment thresholds after the first GPU run.**
   The harness defaults to a permissive `SANA_WM_E2E_REFERENCE_MAX_MAE=255.0`
   so it can be enabled before native quality parity is known. After a GPU
   reference run, lower this and/or add PSNR/SSIM assertions.
2. **Add server-doc examples for async `/v1/videos/generations`.** The sync
   alias is documented in the recipe; async lifecycle examples can still be
   added.
3. **Decide whether SP/USP should be supported for Stage-1 native smoke or
   only for the production-quality path.** Current static wiring is ready,
   but actual SP/USP integration needs a design decision.

## 8. Outstanding work that DOES require GPU access

1. **Run the new reference-alignment harness on GPU.** Current SSH to
   `sana-wm-seeta` returns `Connection closed by 198.18.4.22 port 34732`,
   so revision-7 GPU checks were not run.
2. **Triton GDN multi-step reference-alignment math.** The fused kernel
   executes; parity must still be checked against the official NVlabs path
   over realistic shapes and steps.
3. **vLLM-Omni native two-stage quality validation.** Latent and decoded
   output are validated structurally; numeric parity against the CLI
   bridge is still pending.
4. **Accuracy thresholds.** PSNR ≥ 30 / SSIM-Y All ≥ 0.93 vs the
   CLI-bridge reference once the native path produces a tensor.
5. **`SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline` smoke
   on more GPUs.** Validated on RTX PRO 6000 Blackwell 96 GB; not
   yet on H100/H800/A100 — useful before claiming Tier 2/3 support.
6. **dfx / HSDP / SP / CFG-parallel sweeps** per spec §"GPU tier
   policy" Tier 3.

## 9. Suggested next-step ordering

The biggest single open item is now **running** the reference-alignment harness
against the official bridge once the GPU endpoint is reachable again:

1. Run `SANA_WM_E2E_REFERENCE_ALIGNMENT=1` on GPU and record MAE/shape results.
2. Tighten the reference threshold and add PSNR/SSIM if MAE is stable.
3. Run the 2-step in-process refiner e2e (`SANA_WM_E2E_REFINER_STEPS=2`).
4. Run a live `/v1/videos/generations/sync` SANA-WM request against the OpenAI server.
5. Run SP/USP/CFG-parallel/HSDP sweeps from the dfx perf config.

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
