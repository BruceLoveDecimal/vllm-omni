# Sana-WM Integration — Progress Audit

> **Audit date:** 2026-05-24
> **Branch:** `feat/sana_wm`
> **Branch HEAD:** `4ebdcc2e feat: add Sana-WM integration scaffold`
> **Worktree state at audit:** 13 modified files + 2 untracked
> (`vllm_omni/diffusion/models/sana_wm/native_backend.py`,
>  `vllm_omni/diffusion/models/sana_wm/scheduling_sana_wm.py`).
> **Spec (single source of truth):**
> [`sana_wm_integration.md`](sana_wm_integration.md) (1120 lines).
> **Tracking issue:**
> [vllm-project/vllm-omni#3656](https://github.com/vllm-project/vllm-omni/issues/3656).
>
> This is a snapshot. Re-run by reading the spec end-to-end, then walking
> the file inventory in §3 and the P0–P5 table in §4.

## 1. TL;DR

Overall progress against the post-release implementation plan:
**roughly 40–45%**.

- **P0 + P0.5 (scaffold + official CLI bridge): 100% done.**
- **P1 (Stage-1 reference forward path): ~90% done** statically;
  real-GPU 1-step shape verification still pending.
- **P2 (real fused Triton GDN + offline native e2e): ~30% done.**
  Plücker rasterization is complete; `gated_deltanet_triton.py` is a
  PyTorch reference fallback, not the actual Triton kernel.
- **P3 (LTX-2 refiner attach + dual text encoder): ~5% done.**
  `pipeline_sana_wm_two_stages.py` is a 45-line shell; refiner
  modules are declared as `None` and never loaded or invoked.
- **P4 (online serving + recipe + accuracy): ~5% done.** Offline
  example exists; recipe and online wiring missing.
- **P5 (SP/USP/CFG-parallel/HSDP + dfx perf): 0%.**

The only path that today can produce a real video end-to-end is the
**official CLI bridge** (`VLLM_OMNI_SANA_WM_OFFICIAL_REPO` →
NVlabs/Sana). The vLLM-Omni native path runs the Stage-1 reference
math through a PyTorch GDN fallback and a size-capped "native smoke"
sampler; production-quality native generation is still pending.

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
  __init__.py                       151 LOC   ✅ public surface exports the 6 modules
  config.py                         165 LOC   ✅ SanaWmConfig + from_yaml (parses model/scheduler/text_encoder blocks)
  sana_wm_transformer.py           1084 LOC   ✅ reference DiT — GDN PyTorch fallback, Wan RoPE, camera branch
  pipeline_sana_wm.py               554 LOC   ✅ Stage-1 pipeline w/ HF download, validation, 3-backend dispatch
  pipeline_sana_wm_two_stages.py     45 LOC   ❌ shell only — refiner attach untouched
  camera_control.py                 332 LOC   ✅ Plücker / raymap + all camera schemas
  weight_mapping.py                  47 LOC   ✅ Stage-1 prefix remap helper
  gated_deltanet_triton.py          244 LOC   ⚠️ PyTorch reference fallback; real Triton kernel NOT written
  official_backend.py               344 LOC   ✅ NVlabs CLI bridge (P0.5)
  native_backend.py                 488 LOC   ✅ direct-import NVlabs Python modules (no subprocess)
  scheduling_sana_wm.py              52 LOC   ✅ SanaWmFlowDpmScheduler w/ inference_flow_shift=9.8
vllm_omni/model_executor/stage_input_processors/sana_wm.py  309 LOC   ✅ request payload schema validator
tests/diffusion/models/sana_wm/test_sana_wm_scaffold.py     664 LOC   ✅ 29 test functions
tests/model_executor/stage_input_processors/test_sana_wm.py 236 LOC   ✅ request schema tests
tests/e2e/accuracy/test_sana_wm_video_e2e.py                 81 LOC   ✅ SANA_WM_E2E=1 gated
examples/offline_inference/sana_wm/sana_wm.py               130 LOC   ✅ offline example w/ camera + action support
```

Spec-listed but **missing from disk**:

- `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md`
- `tests/diffusion/models/sana_wm/test_sana_wm_pipeline.py`
- `tests/diffusion/models/sana_wm/test_sana_wm_two_stages.py`
- `tests/diffusion/models/sana_wm/test_sana_wm_camera_control.py`
  (partial coverage folded into `test_sana_wm_scaffold.py`)
- `tests/diffusion/models/sana_wm/test_sana_wm_hsdp.py`
- `tests/diffusion/models/sana_wm/test_sana_wm_cfg_parallel_adaptation.py`
- `tests/examples/offline_inference/test_sana_wm_cfg_parallel_parity.py`
- `tests/dfx/perf/tests/test_sana_wm_vllm_omni.json`

## 4. Phase-by-phase status

| Phase | Spec scope | Concrete status | Coverage |
| --- | --- | --- | --- |
| **P0** — scaffold + registry + pipeline classes + processor stub | All declared in spec §"Phased rollout" | 4 pipeline registry entries, 4 metadata entries, pre/post-process registered, `SanaWmConfig` dataclass, `normalize_sana_wm_payload`, all tests collect, all compileall passes | ✅ **100%** |
| **P0.5** — official CLI backend bridge for GPU smoke | `official_backend.py` ships and `SANA_WM_E2E=1` test exercises the bridge | `official_backend.py` 344 LOC; `test_sana_wm_video_e2e.py` 81 LOC gated by env vars | ✅ **100%** |
| **P1** — Stage-1 weight load (softmax fallback only) + image/camera input packing | `weight_mapping.remap` + `load_weights` + Wan-RoPE positional path + GDN PyTorch fallback + image/camera packing through `normalize_sana_wm_payload` | All present statically; `test_sana_wm_stage1_weight_loads_with_remap`, `test_sana_wm_stage1_weight_audit_materializes_cached_weights` pass at the unit level. **Real-GPU 1-step shape check pending.** | ⚠️ **~90%** |
| **P2** — `GatedDeltaNetTriton` kernel + Plücker camera injection → offline native e2e | `camera_control.py` (Plücker, raymap, action DSL, all camera schemas) is done; `gated_deltanet_triton.py` is **only** a PyTorch reference recurrence, no Triton kernel | Plücker done (✅), Triton GDN missing (❌), offline native e2e green path missing | ⚠️ **~30%** |
| **P3** — LTX-2 refiner attach via `refiner/transformer/`, `refiner/connectors/`, and dual-text-encoder loading | `pipeline_sana_wm_two_stages.py` is a 45-line subclass that declares `refiner_transformer = refiner_text_encoder = refiner_connectors = None` and forwards to `super().forward()` (i.e. Stage-1 only). No `LTX2VideoTransformer3DModel`, no `LTX2TextConnectors`, no `Gemma3ForConditionalGeneration`, no `AutoencoderKLLTX2Video.from_pretrained`. | The class is registered and declares the right `_resident_modules`/`_encoder_modules` slots, but the entire refiner pass is missing. | ❌ **~5%** |
| **P4** — Online serving + recipes + accuracy thresholds | Offline example present; recipe missing; OpenAI-style serving not wired; accuracy tests gated on completion of P3 | `examples/offline_inference/sana_wm/sana_wm.py` ✅; `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md` ❌; no online entry; no PSNR/SSIM thresholds wired | ❌ **~5%** |
| **P5** — SP/USP/CFG-parallel/HSDP + dfx perf; cache-DiT if useful | Not started | No SP plan, no CFG-parallel adaptation, no HSDP wiring, no `tests/dfx/perf/tests/test_sana_wm_vllm_omni.json` | ❌ **0%** |

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

1. **`pipeline_sana_wm_two_stages.py` is a 45-line shell.**
   Declares the right slots but never loads
   `refiner/transformer/diffusion_pytorch_model.safetensors`,
   `refiner/connectors/diffusion_pytorch_model.safetensors`, or
   `refiner/text_encoder/`, and never runs a refiner forward.
2. **`gated_deltanet_triton.py` is misnamed.** Module docstring is
   explicit: "fused Bidirectional Gated DeltaNet Triton kernel is
   not implemented yet"; the implementation is a pure-PyTorch
   reference recurrence. The real Triton kernel is the actual P2
   exit gate.
3. **Stage-1 text encoder is not real.** The native smoke path uses
   a deterministic SHA-256-seeded random tensor (`hash_smoke`) as
   prompt embeddings. `SanaWmConfig.chi_prompt` is read from
   `config.yaml` but never fed through a real Gemma-2-2B-IT
   instance. Spec §"Stage 1 text encoder — Gemma-2-2B-IT (separate
   download)" requires the real model.
4. **VAE slot is unattached.** `self.vae: nn.Module | None = None`
   is set in `__init__` but no code path constructs an
   `AutoencoderKLLTX2Video` from `vae/`. The current pipeline
   returns latents only.
5. **Recipe missing.** Spec §"Updated Implementation Plan / Recipes"
   names `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md`;
   no such file on disk.
6. **Branch hygiene.** `feat/sana_wm` HEAD is still the single
   scaffold commit; 13 modified + 2 untracked files sit in the
   worktree. Pre-PR cleanup is overdue.
7. **Spec-listed test files split.** Five of the eight test files
   the spec recommends are absent; their coverage is folded into
   one large `test_sana_wm_scaffold.py`. Acceptable interim, but
   the split should happen before P3 lands to prevent that file
   from becoming unreviewable.

## 7. Outstanding work that does NOT require GPU access

(These are the items Cindy can keep pushing on a CPU box.)

1. **P3 — LTX-2 refiner attach.** Expand
   `pipeline_sana_wm_two_stages.py` from ~45 LOC to ~400+ LOC:
   - Load `LTX2VideoTransformer3DModel` from
     `refiner/transformer/`.
   - Load `LTX2TextConnectors` from `refiner/connectors/`.
   - Load `Gemma3ForConditionalGeneration` from
     `refiner/text_encoder/`.
   - Implement `forward()` as Stage-1 latent → refiner pass →
     VAE decode.
   - Keep the two text-encoder instances **strictly isolated**
     per spec §"Text-encoder split".
2. **Stage-1 real Gemma-2-2B-IT + `chi_prompt` verbatim.**
   Replace the `hash_smoke` random-tensor path with a
   `transformers.AutoModel.from_pretrained("google/gemma-2-2b-it")`
   load, prepend `chi_prompt` lines, multiply by
   `y_norm_scale_factor=0.01`. The chi_prompt strings are already
   parsed into `SanaWmConfig.chi_prompt`.
3. **VAE real load.**
   `AutoencoderKLLTX2Video.from_pretrained(release_paths.root, subfolder="vae")`
   wired into the Stage-1 pipeline so non-`latent` `output_type`
   actually returns video.
4. **Recipe.** Write
   `recipes/Efficient-Large-Model/SANA-WM-bidirectional.md` using
   the structure from `recipes/Wan-AI/Wan2.2-I2V.md` or
   `recipes/Wan-AI/Wan2.2-S2V.md` as a template; cover the GPU tier
   policy from spec §"GPU tier policy".
5. **Commit + push.** Stage the 13 modified + 2 untracked files,
   produce well-scoped commits on `feat/sana_wm`, and push to the
   `fork` remote so a PR draft can be opened.
6. **Test-file split** per spec §"Recommended e2e test layout":
   - `test_sana_wm_pipeline.py`
   - `test_sana_wm_two_stages.py`
   - `test_sana_wm_camera_control.py`
   - `test_sana_wm_hsdp.py`
   - `test_sana_wm_cfg_parallel_adaptation.py`
   - `tests/examples/offline_inference/test_sana_wm_cfg_parallel_parity.py`
   - `tests/dfx/perf/tests/test_sana_wm_vllm_omni.json`
   Many of these can start as `pytest.skip()` stubs that document
   the contract until GPU access lands.

## 8. Outstanding work that DOES require GPU access

1. **P1 exit gate.** Load
   `dit/sana_wm_1600m_720p.safetensors` for real, do a 1-step
   forward at `256×448, 24f`, and confirm the latent shape comes
   out correct.
2. **Official CLI smoke.** Run `VLLM_OMNI_SANA_WM_OFFICIAL_REPO=…
   SANA_WM_E2E=1 pytest tests/e2e/accuracy/test_sana_wm_video_e2e.py`
   against a checkout of NVlabs/Sana that contains
   `inference_video_scripts/inference_sana_wm.py` (not yet on
   upstream `main` as of audit date — track upstream).
3. **vLLM-Omni native path end-to-end.** Requires items 1+2 above
   plus the real Triton GDN kernel and the real Gemma-2-2B-IT load.
4. **Accuracy.** PSNR ≥ 30 / SSIM-Y All ≥ 0.93 vs official runner
   output, after P3 lands.
5. **dfx / HSDP / SP / CFG-parallel** parity sweeps per spec §"GPU
   tier policy" Tier 3.

## 9. Suggested next-step ordering

A reasonable local-only sequence that maximizes review value before
GPU access lands:

1. **Commit + push the existing worktree.** Otherwise everything
   below sits on an untracked working copy and is hard to review.
2. **Recipe + spec-test-split (low risk, high docs leverage).**
3. **Stage-1 real Gemma-2-2B-IT + chi_prompt verbatim.**
4. **VAE real load.**
5. **P3 refiner attach in `pipeline_sana_wm_two_stages.py`.**
6. **(Optional) Triton GDN kernel** — only attempt after a CUDA
   environment is available; the PyTorch fallback can stand in for
   CPU correctness checks meanwhile.

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
