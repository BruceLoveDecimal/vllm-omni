# Sana-WM Integration Spec

> **Status as of 2026-05-21 (consolidated single source of truth).**
> This document is the **canonical** Sana-WM integration spec for
> vllm-omni. It supersedes and absorbs the earlier
> `sana_wm_spec_v0.1.md` pre-release planning draft (now removed): all
> remaining-valid content from v0.1 (scope statement, non-goals,
> milestone breakdown, test plan) has been merged in below, and the
> speculative parts (T2V-first framing, dual-conv-stem camera
> conditioning, GDN-as-generic-attention-backend) have been dropped
> because the HF release contradicts them.
>
> NVIDIA released the SANA-WM weights to Hugging Face on 2026-05-18
> (Beijing time evening). All "Work Allowed Before Weight Release"
> guidance below has been superseded by concrete checkpoint
> inspection — see §§ "Confirmed Release Inventory",
> "Concrete Architecture (from `config.yaml` + HF component configs)",
> and "Updated Implementation Plan" further down. The earlier
> pre-release sections are kept for historical context only.
>
> **Tracking:** GitHub issue
> [vllm-project/vllm-omni#3656](https://github.com/vllm-project/vllm-omni/issues/3656),
> local branch `feat/sana_wm`.

## Status

Sana-WM is tracked by https://github.com/vllm-project/vllm-omni/issues/3656.
**As of 2026-05-19, the public Hugging Face state is:**

- Model repo: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
  (private=False, gated=False, apache-2.0, last updated 2026-05-19 14:57 BJT).
- Initial commit and core weights pushed 2026-05-18 19:14–19:48 BJT.
- Reference code: <https://github.com/NVlabs/Sana>.
- The model card says the bundled LTX-2 refiner and VAE inherit the LTX-2
  upstream license. The repository-level license tag is `apache-2.0`.

**Original draft (kept for context):** before 2026-05-18, no public
Sana-WM repo existed under the obvious names, and this doc recommended
triage-only work. That guidance is now historical.

## Goal

Add production inference support for Sana-WM once official weights and configs
are released.

The integration should support:

- First-frame image-to-video generation with text prompting.
- 704×1280 output video through the released SANA-WM serving contract.
- Stage-1 latent decode with the LTX2 video VAE, plus the optional LTX-2
  refiner path.
- 6-DoF camera trajectory control.
- The official WASD/IJKL action DSL as a convenience input, if feasible.
- The Sana-WM transformer backbone, including the camera-control dual branch.
- The attention or linear-attention path required by the official architecture,
  including Gated DeltaNet if it is part of the released transformer.
- Offline inference and online serving examples.
- Correctness tests against the official reference implementation.

## Non-goals

- Do not register a fake Sana-WM pipeline that cannot load official weights.
- Do not infer exact tensor shapes, camera branch layouts, or weight mappings
  from the paper alone.
- Do not treat SANA-Video as Sana-WM. It is only a public baseline for shared
  components such as Gemma text encoding, flow scheduling, and LTX2 VAE usage.
- Do not merge performance features before baseline output parity is established.

## Expected Architecture

The expected high-level pipeline is:

```text
request
  -> first-frame image resize / center-crop to 704×1280
  -> text prompt normalization
  -> camera trajectory or action-DSL normalization
  -> Stage-1 Gemma-2-2B-IT prompt enhancement / text encoding
  -> camera control encoder / dual branch conditioning
  -> Sana-WM DiT denoising loop
  -> LTX2 video VAE decode or LTX-2 refiner pass
  -> video post-process
```

The exact text encoder, scheduler, transformer shape, and camera branch layout
must come from the released `config.yaml`, HF component configs, safetensors
headers, or official reference code. The current HF repo does **not** include a
root `model_index.json`, so vLLM-Omni discovery needs explicit registry support
or a download/setup script that generates a minimal local `model_index.json`.

## Work Allowed Before Weight Release

These tasks are useful and safe before official weights are available.

### 1. Release tracking

Track the expected public artifacts:

- Hugging Face model ID.
- Whether the release is diffusers-style, custom `.pth`, or hybrid.
- `model_index.json` if diffusers-style.
- `transformer/config.json`, `vae/config.json`, scheduler config, tokenizer,
  text encoder config.
- Camera-control metadata or example input format.
- Official inference script and sample prompts.

No code should hardcode speculative artifact names.

### 2. Camera-control request schema

Define the request-side schema independently from model internals. The schema
should be model-facing but not tied to weight shapes.

Historical pre-release payload:

```json
{
  "camera_trajectory": {
    "format": "extrinsics_4x4",
    "coordinate_system": "opencv",
    "fps": 16,
    "frames": 257,
    "poses": [[[...]]]
  }
}
```

This was intentionally generic before release. It is superseded by the
post-release "Request schema" section below, which adds the required first-frame
image input, explicit camera-to-world matrices, action DSL, and official
intrinsics shapes.

The processor should also reserve room for:

- `format`: `extrinsics_4x4`, `c2w_4x4`, `w2c_4x4`, `relative_6dof`,
  or an official release-specific name.
- `intrinsics`: optional camera intrinsics when the model needs them.
- `frame_rate` / `fps`.
- `num_frames`.
- `height` and `width`.

Pre-release implementation may include validation and normalization utilities
only. It must not project the trajectory into model embeddings unless the
official camera branch contract is known.

### 3. Stage input processor skeleton

Create a processor only if it can be tested without model weights.

Proposed file:

```text
vllm_omni/model_executor/stage_input_processors/sana_wm.py
```

Responsibilities:

- Extract camera fields from `additional_information`, `extra_args`, or an
  online serving `extra_params` equivalent.
- Validate frame count, matrix dimensions, dtype-convertible values, and
  consistent `height` / `width` / `fps`.
- Normalize the payload into a single dictionary under
  `additional_information["camera_trajectory"]`.
- Preserve original prompt text and sampling params.

Pre-release tests:

```text
tests/model_executor/stage_input_processors/test_sana_wm.py
```

Test only validation and normalization behavior with synthetic inputs.

### 4. SANA-Video baseline study

Use `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` as a separate baseline
to understand shared mechanics:

- `SanaVideoPipeline`.
- `SanaVideoTransformer3DModel`.
- `Gemma2Model` text encoder.
- `AutoencoderKLLTX2Video` VAE.
- Flow / DPM scheduler configuration.

This can inform file layout and LTX2 VAE handling, but it should not be
committed as Sana-WM support.

### 5. Gated DeltaNet research spike

If the paper or official code gives enough algorithmic detail, add a standalone
experimental backend behind tests and explicit opt-in. Otherwise wait.

Safe pre-release scope:

- Review whether Gated DeltaNet is used as a full attention replacement or as
  only one branch in a hybrid block.
- Decide whether it fits the existing `AttentionBackend` interface or needs a
  model-local module because the operation is not Q/K/V softmax attention.
- Prototype a small PyTorch reference function using synthetic tensors.

Do not wire it into Sana-WM until the released transformer confirms call sites,
state layout, recurrence semantics, and weight names.

## Work Required After Weight Release

These tasks require official weights, config, or reference code.

### 1. Classify release format

Inspect the released repository:

- If it has a standard diffusers `model_index.json`, follow the diffusers
  migration path.
- If it has only `.pth` checkpoints and custom code, follow the custom model
  path.
- If only some components are diffusers-style, use a hybrid path.

Record the exact model ID and file list in the PR description.

### 2. Implement model files

Proposed files:

```text
vllm_omni/diffusion/models/sana_wm/
  __init__.py
  camera_control.py
  pipeline_sana_wm.py
  sana_wm_transformer.py
  weight_mapping.py
```

File responsibilities:

- `pipeline_sana_wm.py`: component loading, prompt encoding, camera condition
  preparation, denoising loop, VAE decode, post-process output.
- `sana_wm_transformer.py`: ported transformer blocks, attention/linear
  attention modules, camera dual branch fusion, TP/SP/HSDP hooks.
- `camera_control.py`: model-internal camera embedding and packing logic,
  once the official branch contract is known.
- `weight_mapping.py`: checkpoint key remapping and expected missing/unexpected
  key policy.
- `__init__.py`: exports the public pipeline and helper factories.

Register the model in:

```text
vllm_omni/diffusion/registry.py
```

Only register after the pipeline can load official weights.

### 3. Implement attention or linear-attention support

If Sana-WM uses Gated DeltaNet in a way that fits the backend abstraction, add:

```text
vllm_omni/diffusion/attention/backends/gated_deltanet.py
```

and update:

```text
vllm_omni/diffusion/attention/backends/registry.py
```

If the operation requires recurrent state, block state, camera-specific state,
or a non-QKV signature, keep it as a model-local module in
`sana_wm_transformer.py` or a helper file instead of forcing it into
`AttentionBackend`.

Required after release:

- Match official output numerically for synthetic block-level tests.
- Validate dtype support.
- Validate CUDA and CPU/eager fallback behavior.
- Confirm behavior under sequence parallelism before enabling SP.

### 4. Weight loading

Implement weight loading only after official files are available.

Requirements:

- Support the official release layout exactly.
- Fail loudly on unexpected shape mismatches.
- Maintain a small allowlist for intentionally missing or ignored keys.
- Add a smoke test that instantiates the pipeline and verifies no unapproved
  missing/unexpected keys.

### 5. Correctness validation

Required parity tests:

- Component load smoke test.
- Text encoder output shape parity.
- Camera processor to camera embedding shape parity.
- Transformer forward parity on a short synthetic or official fixture.
- End-to-end deterministic sample comparison against official reference code.

End-to-end quality validation should use official sample prompts, camera paths,
seed, resolution, frame count, scheduler settings, and dtype.

### 6. Examples and serving

Add examples only after the model runs:

```text
examples/offline_inference/sana_wm/
  README.md
  sana_wm.py

examples/online_serving/sana_wm/
  README.md
  run_server_sana_wm.sh
  run_curl_sana_wm.sh
```

The examples should include one official first-frame image-to-video request,
one explicit camera-file request, and one action-DSL request if the action
rollout utility is ported.

### 7. Performance support

Add performance features after correctness:

- CPU offload via `SupportsComponentDiscovery`.
- HSDP when transformer blocks are identified.
- TP after linear layers and weight mappings are sharded correctly.
- SP/USP only after attention/linear-attention call sites support sharded
  sequence dimensions.
- CFG parallel only if the released sampler uses classifier-free guidance.
- Cache-DiT only after baseline quality and deterministic behavior are known.

Each feature needs a separate smoke or e2e test.

### 8. Documentation updates

After full integration, update:

```text
docs/models/supported_models.md
docs/user_guide/diffusion_features.md
docs/user_guide/examples/offline_inference/text_to_video.md
docs/user_guide/examples/online_serving/text_to_video.md
```

If new attention backend controls are user-visible, update:

```text
docs/user_guide/diffusion/attention_backends.md
```

## Proposed File Design

### Model package

```text
vllm_omni/diffusion/models/sana_wm/
  __init__.py
  pipeline_sana_wm.py
  sana_wm_transformer.py
  camera_control.py
  weight_mapping.py
```

The package should keep camera-control model logic inside the diffusion model
package. The stage input processor should only normalize request payloads.

### Stage input processor

```text
vllm_omni/model_executor/stage_input_processors/sana_wm.py
```

This file should not import the diffusion pipeline. It should stay lightweight
and usable in multi-stage request routing.

### Attention backend

```text
vllm_omni/diffusion/attention/backends/gated_deltanet.py
```

Only add this if the released architecture confirms the operation is a reusable
backend. Otherwise, keep Gated DeltaNet implementation local to the Sana-WM
transformer.

### Tests

```text
tests/model_executor/stage_input_processors/test_sana_wm.py
tests/diffusion/models/sana_wm/test_sana_wm_components.py
tests/diffusion/attention/test_gated_deltanet.py
tests/e2e/accuracy/test_sana_wm_video_similarity.py
```

Pre-release tests should be limited to the stage input processor and pure
synthetic utilities. Weight-loading and e2e tests must wait for official
artifacts.

## Release Checklist

Do not mark Sana-WM as supported until all of these pass:

- Official model ID is public and accessible.
- Pipeline loads official weights without unapproved missing/unexpected keys.
- At least one official first-frame image + prompt + camera/action request
  produces a valid video.
- Output matches the official implementation within an agreed visual/numerical
  tolerance.
- CPU offload or documented memory requirements are provided.
- Offline and online examples are documented.
- Supported model tables are updated.

## Open Questions

- What is the official HF model ID? — **Resolved**: `Efficient-Large-Model/SANA-WM_bidirectional`.
- Is the release diffusers-style, custom `.pth`, or hybrid? — **Hybrid**: see "Confirmed Release Inventory" below. Stage 1 DiT is a custom flat safetensors checkpoint plus root `config.yaml`; VAE and refiner components are diffusers-style subfolders; there is no root `model_index.json`.
- What exact camera trajectory schema does the official implementation accept? — Partially resolved: the model card documents `--camera` as NumPy `(F, 4, 4)` camera-to-world matrices and `--action` as a WASD/IJKL DSL rolled out to `(F+1, 4, 4)`. NVlabs uses Plücker coordinates injected at every block (`plucker_embedder.*` + `raymap_embedder.*` in Stage 1 checkpoint). Trajectory → raymap → plücker pipeline details still need reference-code parity.
- Is Gated DeltaNet exposed as a reusable attention backend or embedded inside custom transformer blocks? — **Embedded**, see below. NVlabs ships its own Triton kernel via `BidirectionalGDNTriton`.
- Does Sana-WM require two-stage generation or an LTX2 refiner path, and is that required for the advertised quality? — **Yes for advertised quality.** The current HF repo exposes the refiner as `refiner/transformer/`, `refiner/connectors/`, and `refiner/text_encoder/` diffusers-style folders, not as a single `refiner/refiner.safetensors`.
- What frame count and fps should be the default serving contract for minute-scale generation? — Partially resolved: the model-card example uses `--num_frames 321`; exact fps/default scheduler settings should be mirrored from upstream code.

---

## NVlabs-Recommended Production Inference Config

Single source of truth for "production-mode" inference defaults, mirrored from
upstream NVlabs Sana-WM
([`inference_video_scripts/inference_sana_wm.py`](https://github.com/NVlabs/Sana)
`GenerationParams` + `argparse` defaults + `configs/sana_wm/sana_wm_1600m_720p.yaml`).
These are the values to use when benchmarking realistic end-user generation
quality; do not deviate from these in published metrics unless explicitly
noting the deviation. Internal latent-parity probes may use cheaper configs
(e.g. `num_frames=9`, `step=20`, `cfg_scale=1.0`) for iteration speed; those
absolute RGB numbers are not production-representative.

### Stage-1 (DiT) generation knobs

| Knob | Production value | Notes |
|---|---|---|
| `num_frames` | **161** | 10 s @ 16 fps; the default production length. `321` is a ~20 s stress configuration that NVlabs's model card example demonstrates but is not the recommended default. |
| `fps` | **16** | |
| `step` (DiT sampling steps) | **60** | NVlabs `GenerationParams.step` default. Smaller step counts (e.g. 20) trade quality for speed and **must not** be reported as production results. |
| `cfg_scale` | **5.0** | Classifier-free guidance scale. `1.0` disables guidance and is *not* production. |
| `sampling_algo` | `flow_euler_ltx` | NVlabs default; also supports `flow_euler` and `flow_dpm-solver` but `flow_euler_ltx` is the production path. |
| `flow_shift` (inference) | **9.8** | From `configs/sana_wm/sana_wm_1600m_720p.yaml::scheduler.inference_flow_shift`. Training-time `flow_shift` is `9.95`; inference uses the shifted value. Setting `None` falls back to the scheduler default which already encodes `9.8`. |
| `seed` | **42** | NVlabs `GenerationParams.seed` default. |
| `negative_prompt` | `""` (empty) | |

### Stage-2 (LTX-2 refiner) knobs

| Knob | Production value | Notes |
|---|---|---|
| `STAGE_2_DISTILLED_SIGMA_VALUES` | `[0.909375, 0.725, 0.421875, 0.0]` | 3-step Euler refiner; sigmas hard-coded as a distilled schedule. Imported from `diffusers.pipelines.ltx2.utils.STAGE_2_DISTILLED_SIGMA_VALUES`. |
| `sink_size` | **1** | Number of leading frames preserved bit-exact through the refiner (the conditioning image). |
| `seed` | **42** | Refiner noise seed (separate from Stage-1 seed; happens to be the same value). |

### VAE knobs

| Knob | Production value | Notes |
|---|---|---|
| `vae_type` | `LTX2VAE_diffusers` | |
| `use_framewise_encoding` / `use_framewise_decoding` | `true` / `true` | |
| `tile_sample_stride_num_frames` | **64** | Required for tiled long-sequence decoding (321 f path). |
| `tile_sample_min_num_frames` | **96** | |

### vLLM-Omni wiring

In `OmniDiffusionSamplingParams`:

```python
OmniDiffusionSamplingParams(
    height=704,
    width=1280,
    num_frames=161,
    seed=42,
    fps=16,
    num_inference_steps=60,
    guidance_scale=5.0,
    guidance_scale_provided=True,
    extra_args={
        "sana_wm_sampling_algo": "flow_euler_ltx",
        "sana_wm_inprocess_refiner": True,
        "sana_wm_inprocess_refiner_steps": 3,
        "sana_wm_refiner_sink_size": 1,
        "sana_wm_refiner_seed": 42,
        # Long-sequence VAE memory contract:
        "sana_wm_offload_vae": True,
    },
)
```

E2E harness env vars to match (for `tests/e2e/accuracy/test_sana_wm_video_e2e.py`
and the `sana_wm_321_metrics_*.py` family):

```bash
SANA_WM_E2E_NUM_FRAMES=161
SANA_WM_E2E_STAGE1_STEPS=60
SANA_WM_E2E_REFINER_STEPS=3
SANA_WM_E2E_ACTION=w-160         # 161 frames - 1 conditioning frame
SANA_WM_E2E_PREDICTION_OUTPUT_TYPE=np
SANA_WM_E2E_NATIVE_MAX_TOKENS=50000
```

If a probe deviates from any of these knobs (e.g. for fast iteration or for
memory budget), it must:

1. State the deviation explicitly in the experiment writeup.
2. Treat any absolute MAE / PSNR / SSIM result as *non-production*; only
   relative deltas between paired runs at the same deviation are admissible
   as evidence for design decisions.
3. Re-validate the design decision under production config before landing
   the change.

---

## Confirmed Release Inventory (2026-05-19)

Repo: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>

API snapshot:

```text
sha:          90e0ff3b8f1f9b54a92b4b707edeaa27073aec84
lastModified: 2026-05-19T06:57:12Z
private:      false
gated:        false
tags:         diffusers, safetensors, text-to-video, image-to-video,
              camera-control, world-model, diffusion, arxiv:2605.15178,
              license:apache-2.0
```

```text
README.md
config.yaml                                  # full Stage-1 / VAE / scheduler wiring
dit/sana_wm_1600m_720p.safetensors           # Stage 1 custom Sana-WM DiT
vae/config.json
vae/diffusion_pytorch_model.safetensors      # LTX2 video VAE, diffusers folder
refiner/transformer/config.json
refiner/transformer/diffusion_pytorch_model.safetensors
                                             # LTX2VideoTransformer3DModel refiner
refiner/connectors/config.json
refiner/connectors/diffusion_pytorch_model.safetensors
                                             # LTX2TextConnectors
refiner/text_encoder/
    model-00001-of-00011.safetensors
    model-00002..00011-of-00011.safetensors
    model.safetensors.index.json
    config.json                              # Gemma3ForConditionalGeneration
    tokenizer.json / tokenizer.model / tokenizer_config.json
    processor_config.json / preprocessor_config.json
    chat_template.jinja / special_tokens_map.json / added_tokens.json
```

There is no root `model_index.json` in the current release. Note that the HF
README's repository-layout table still mentions `refiner/refiner.safetensors`;
the actual repository file list is split into diffusers-style refiner subfolders
as shown above.

Plus the runtime-fetched **`google/gemma-2-2b-it`** for Stage 1 (not bundled,
per the HF model card; loader must hit a public mirror separately).

Note: this asymmetric two-text-encoder layout is intentional — Stage 1 uses
the small Gemma-2-2B-IT for fast prompt enhancement; Stage 2 (the LTX-2
refiner) uses its own bundled Gemma-3-12B multimodal encoder plus
`LTX2TextConnectors`.

---

## Concrete Architecture (from `config.yaml` + HF component configs)

### Stage 1 — Sana camera-controlled DiT (1.6B)

From `config.yaml` `model:` block:

```yaml
model:        SanaMSVideoCamCtrl_1600M_P1_D20    # 20 transformer blocks (D20)
image_size:   720                                  # 720p latent target
aspect_ratio_type: ASPECT_RATIO_VIDEO_720_MS_DIV32 # multi-scale, /32 alignment
mixed_precision: bf16
fp32_attention: true
multi_scale: true
camctrl_type: BidirectionalGDNUCPESinglePathLiteLABothTriton
attn_type:    BidirectionalGDNTriton
softmax_every_n: 4                                 # every 4th block uses softmax; others GDN
linear_head_dim: 112
conv_kernel_size: 4                                # depthwise conv inside GDN (K only)
k_conv_only:  true
ffn_type:     GLUMBConvTemp                        # GLU + MBConv + temporal — non-standard
t_kernel_size: 3
mlp_acts:     [silu, silu, null]
mlp_ratio:    3
use_pe:       true
pos_embed_type: wan_rope                           # Wan-style RoPE
qk_norm:      true
cross_norm:   true
chunk_split_strategy: first_chunk_plus_one
cam_attn_compress: 1
init_cam_from_base: true                           # camera branch warm-started from base
use_chunk_plucker_post_attn: true
chunk_plucker_channels: 48
chunk_plucker_post_attn_blocks: 20                 # all 20 blocks inject plücker
```

From the safetensors header (`dit/sana_wm_1600m_720p.safetensors`):

```text
total tensors: 872 (F32)
top-level prefixes:
  blocks           — 850   (20 blocks × ~42 tensors/block)
  y_embedder       — 5     (text-condition projection)
  t_embedder       — 4     (timestep MLP)
  t_block          — 2     (timestep ada-ln modulation)
  x_embedder       — 2     (latent input proj)
  final_layer      — 3     (output head)
  plucker_embedder — 2     ← camera control
  raymap_embedder  — 2     ← camera control
  pos_embed        — 1
  attention_y_norm — 1
sample per-block keys (block 0):
  blocks.0.attn.A_log              [20]              ← GDN state-space param
  blocks.0.attn.beta_proj.{w,b}    [20, 2240]/[20]   ← GDN beta gate
  blocks.0.attn.conv_k.weight      [2240, 1, 4]      ← K depthwise conv
  ... (more GDN-specific tensors per block)
```

Hidden dim ≈ 2240 (from `attention_y_norm.weight: [2240]`).

### Stage 2 — refiner = LTX-2 video transformer + text connectors

The current HF repo exposes the refiner as diffusers-style component folders,
not as a single merged `refiner/refiner.safetensors` file:

```text
refiner/transformer/config.json
refiner/transformer/diffusion_pytorch_model.safetensors
refiner/connectors/config.json
refiner/connectors/diffusion_pytorch_model.safetensors
refiner/text_encoder/...
```

`refiner/transformer/config.json`:

```json
{
  "_class_name": "LTX2VideoTransformer3DModel",
  "_diffusers_version": "0.37.0.dev0",
  "attention_head_dim": 128,
  "num_attention_heads": 32,
  "num_layers": 48,
  "in_channels": 128,
  "out_channels": 128,
  "caption_channels": 3840,
  "cross_attention_dim": 4096,
  "qk_norm": "rms_norm_across_heads",
  "rope_type": "split",
  "rope_double_precision": true,
  "patch_size": 1,
  "patch_size_t": 1,
  "vae_scale_factors": [8, 32, 32],
  "audio_in_channels": 128,
  "audio_out_channels": 128,
  "audio_cross_attention_dim": 2048
}
```

`refiner/connectors/config.json`:

```json
{
  "_class_name": "LTX2TextConnectors",
  "caption_channels": 3840,
  "text_proj_in_factor": 49,
  "video_connector_num_layers": 2,
  "video_connector_num_attention_heads": 30,
  "video_connector_attention_head_dim": 128,
  "audio_connector_num_layers": 2,
  "audio_connector_num_attention_heads": 30,
  "audio_connector_attention_head_dim": 128,
  "rope_type": "split",
  "rope_double_precision": true
}
```

**Implication:** refiner support should reuse vLLM-Omni's existing LTX2
transformer, connector, Gemma3 text-encoder, scheduler, and VAE patterns, but
the existing `LTX2TwoStagesPipeline` cannot be used by only swapping the Stage-1
transformer. It currently assumes LTX2 release naming/layout and checks for a
`distilled` model path. `SanaWmTwoStagesPipeline` needs its own component
discovery and loader paths:

```text
Stage 1 transformer weights:  dit/sana_wm_1600m_720p.safetensors
Stage 2 transformer weights:  refiner/transformer/diffusion_pytorch_model.safetensors
Stage 2 connectors weights:   refiner/connectors/diffusion_pytorch_model.safetensors
Stage 2 text encoder:         refiner/text_encoder/
Shared video VAE:             vae/
```

The current release does **not** include `refiner/audio_vae/` or
`refiner/vocoder/` folders. Do not design the first video-only integration
around filtering those components unless upstream adds them later.

### VAE — already the LTX-2 video VAE (bf16 on disk)

```text
vae/diffusion_pytorch_model.safetensors
  total tensors: 184 (BF16)
  encoder.*  — 92
  decoder.*  — 90
  latents_mean — 1
  latents_std  — 1
decoder.conv_out.conv.weight: [48, 128, 3, 3, 3]    # latent_dim 128, patch 4×4
```

This matches the released `vae/config.json` class
`AutoencoderKLLTX2Video` and the `vae_type: LTX2VAE_diffusers` line in
`config.yaml`. Map to `diffusers.AutoencoderKLLTX2Video` in vLLM-Omni —
zero new VAE code.

### Refiner text encoder — Gemma-3-12B-IT (multimodal)

`refiner/text_encoder/config.json`:

```json
{
  "architectures": ["Gemma3ForConditionalGeneration"],
  "model_type": "gemma3",
  "text_config": {
    "hidden_size": 3840,         "num_hidden_layers": 48,
    "num_attention_heads": 16,   "num_key_value_heads": 8,
    "vocab_size": 262208,        "max_position_embeddings": 131072,
    "sliding_window": 1024,      "sliding_window_pattern": 6,
    "rope_scaling": {"factor": 8.0, "rope_type": "linear"}
  },
  "vision_config": {
    "hidden_size": 1152, "num_hidden_layers": 27,
    "patch_size": 14, "image_size": 896,
    "model_type": "siglip_vision_model"
  },
  "boi_token_index": 255999, "eoi_token_index": 256000,
  "image_token_index": 262144, "mm_tokens_per_image": 256
}
```

Shard 1 carries the `vision_tower.*` (437 tensors) + `multi_modal_projector.*`
(2 tensors). Shards 2–11 hold the LM tower. Total ≈ 50 GB.

### Stage 1 text encoder — Gemma-2-2B-IT (separate download)

From `config.yaml`:

```yaml
text_encoder:
  text_encoder_name: gemma-2-2b-it
  y_norm: true
  y_norm_scale_factor: 0.01
  model_max_length: 300
  chi_prompt:                  # prepended to user prompts
    - 'Given a user prompt, generate an "Enhanced prompt" ...'
    - ...
```

NVlabs prepends a multi-line "Enhanced prompt" instruction so Gemma-2-2B-IT
rewrites the user prompt before encoding. **Our pipeline must replicate this
template verbatim** or the conditioning distribution will shift.

### Scheduler

```yaml
scheduler:
  predict_flow_v: true
  noise_schedule: linear_flow
  pred_sigma: false
  flow_shift: 9.95              # training
  inference_flow_shift: 9.8     # inference
  vis_sampler: flow_dpm-solver  # DPM-Solver at inference
```

So Stage 1 uses **flow-DPM-Solver**, not the Euler scheduler we used for
Wan/Lance. We will need either the diffusers `FlowMatchDPMSolverMultistepScheduler`
or NVlabs' custom DPM-Solver port.

---

## Updated Implementation Plan (post-release)

These items replace the original "Work Required After Weight Release" plan
where the older plan made guesses that the safetensors-header inspection
has since invalidated.

### Components — new code we need

1. `SanaWmTransformer3DModel` — 1.6B DiT with:
   - 20 blocks, hidden 2240, `mlp_ratio=3`, `GLUMBConvTemp` FFN.
   - Hybrid attention: **Bidirectional Gated DeltaNet** with depthwise
     1D `conv_k` (kernel=4), softmax fallback every 4th block.
   - Plucker camera ray injection post-attention in **every** block.
   - Wan-style RoPE positional encoding.
   - QK + cross RMSNorm.
2. `gated_deltanet_triton.py` — model-local GDN kernel (Triton). Do **not**
   force it into the generic `AttentionBackend` interface yet; it's a
   model-internal operator with state-space `A_log` / `beta_proj` / `conv_k`
   parameters that don't fit Q/K/V softmax assumptions.
3. `camera_control.py` — Plucker raymap encoder. Takes the validated
   trajectory from the stage input processor and produces the embedding
   that gets injected post-attention.
4. `stage_input_processors/sana_wm.py` — trajectory schema validator
   (kept lightweight, no model imports).
5. `pipeline_sana_wm.py` — Stage 1 only (no refiner) image-to-video pipeline:
   first-frame preprocessing, prompt enhancement / encoding, camera condition
   packing, denoising, and VAE decode.
6. `pipeline_sana_wm_two_stages.py` — Stage 1 + LTX-2 refiner. Reuse LTX2
   helper code where possible, but own the Sana-WM release layout and
   dual-text-encoder loading explicitly.
7. `weight_mapping.py` — flat (no `transformer.*` wrapping) → vLLM-Omni
   module names. Prefix table:

```text
blocks.{0..19}.attn.{...}        →  transformer.blocks.{i}.attention.{...}
blocks.{0..19}.cross_attn.{...}  →  transformer.blocks.{i}.cross_attention.{...}
blocks.{0..19}.mlp.{...}         →  transformer.blocks.{i}.mlp.{...}
blocks.{0..19}.plucker_post.{...}→  transformer.blocks.{i}.camera_post.{...}
y_embedder.{...}                  →  transformer.text_embedder.{...}
t_embedder.{...}                  →  transformer.time_embedder.{...}
t_block.{...}                     →  transformer.time_modulation.{...}
x_embedder.{...}                  →  transformer.latent_in.{...}
final_layer.{...}                 →  transformer.out_head.{...}
plucker_embedder.{...}            →  camera_encoder.plucker.{...}
raymap_embedder.{...}             →  camera_encoder.raymap.{...}
pos_embed                         →  transformer.pos_embed
attention_y_norm.weight           →  transformer.cross_attn_y_norm.weight
```

Exact mapping will be finalised once the per-block tensor list is dumped
in full (the 8 MB probe captured every block-0 tensor; the rest follow
the same pattern by symmetry).

### Components — reuse from `vllm_omni/diffusion/models/ltx2/`

- **`LTX2VideoTransformer3DModel`** — reuse for the Stage 2 refiner transformer
  loaded from `refiner/transformer/`.
- **`LTX2TextConnectors`** — reuse for `refiner/connectors/`.
- **`AutoencoderKLLTX2Video`** — use as-is for VAE encode/decode from `vae/`.
- **`Gemma3ForConditionalGeneration`** — use as-is for the refiner's bundled
  text encoder under `refiner/text_encoder/`.
- **LTX2 scheduler / postprocess helpers** — reuse carefully, but do not depend
  on `LTX2TwoStagesPipeline` constructor assumptions because the Sana-WM HF
  layout is not an LTX2 model-root layout.
- For Stage 1 add `FlowMatchDPMSolverMultistepScheduler` or port NVlabs'
  custom flow-DPM solver configured by root `config.yaml`.

### Text-encoder split (critical detail)

```text
Stage 1 text encoder: load from `google/gemma-2-2b-it` (small, ~5 GB)
                       prepend `config.yaml::text_encoder.chi_prompt` to user prompt
                       wrap in `y_norm * y_norm_scale_factor (0.01)`
Stage 2 text encoder: load from local `refiner/text_encoder/` (~50 GB,
                       multimodal Gemma3-12B-IT)
                       feed LTX2TextConnectors from `refiner/connectors/`
                       no chi_prompt; LTX-2 conventions
```

Two separate text-encoder instances; never share state.

---

### Interface declarations (concrete)

All diffusion pipeline interface protocols live in
`vllm_omni/diffusion/models/interface.py`. The correct interfaces for Sana-WM are:

**`pipeline_sana_wm.py`** (Stage 1 only):

```python
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

class SanaWmPipeline(nn.Module, SupportImageInput, ProgressBarMixin):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"
```

`SupportImageInput` is the vllm-omni equivalent of vLLM's `SupportsMultiModal` for
diffusion pipelines. It signals to the serving layer that the pipeline accepts image
inputs. **Without it, first-frame image payloads will be silently dropped by the
request router.** `color_format = "RGB"` matches the official preprocessing order
expected by the model card.

**`pipeline_sana_wm_two_stages.py`** (Stage 1 + LTX-2 refiner):

```python
from vllm_omni.diffusion.models.interface import (
    SupportImageInput,
    SupportsComponentDiscovery,
)

class SanaWmTwoStagesPipeline(
    nn.Module,
    SupportImageInput,
    SupportsComponentDiscovery,
    ProgressBarMixin,
):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = [
        "text_encoder",          # google/gemma-2-2b-it (Stage 1, ~5 GB)
        "refiner_text_encoder",  # refiner/text_encoder, Gemma-3-12B-IT (Stage 2, ~50 GB)
        "refiner_connectors",    # refiner/connectors, LTX2TextConnectors
    ]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["refiner_transformer"]
```

`_dit_modules` covers Stage 1 only. The LTX-2 refiner transformer stays in
`_resident_modules` rather than `_dit_modules` to prevent it from being offloaded
between the two denoising stages — mirroring `LTX2TwoStagesPipeline`'s component
layout. Both text encoders and the connectors go in `_encoder_modules` so they can
be CPU-offloaded after encoding is complete.

---

### `config.yaml` loading strategy (concrete decision)

Stage 1's full architecture spec lives in the root `config.yaml`; there is no
`transformer/config.json`. **Decision: implement a `SanaWmConfig` dataclass** that
reads from `config.yaml` directly. Do not generate a synthetic `model_index.json`
at runtime — it would be a non-reproducible artifact that silently becomes stale
when the HF repo layout changes.

Proposed file:

```text
vllm_omni/diffusion/models/sana_wm/config.py
```

Minimal surface:

```python
@dataclass
class SanaWmConfig:
    # --- transformer architecture ---
    num_blocks: int             # 20
    hidden_size: int            # 2240 (from attention_y_norm.weight shape)
    mlp_ratio: float            # 3.0
    attn_type: str              # "BidirectionalGDNTriton"
    softmax_every_n: int        # 4  (every 4th block uses softmax fallback)
    linear_head_dim: int        # 112
    conv_kernel_size: int       # 4  (depthwise K-only conv inside GDN)
    ffn_type: str               # "GLUMBConvTemp"
    pos_embed_type: str         # "wan_rope"
    mixed_precision: str        # "bf16"
    fp32_attention: bool        # True
    image_size: int             # 720
    # --- camera control ---
    chunk_plucker_channels: int         # 48
    chunk_plucker_post_attn_blocks: int # 20 (all blocks)
    # --- scheduler ---
    inference_flow_shift: float # 9.8
    scheduler_type: str         # "flow_dpm-solver"
    # --- text encoder ---
    chi_prompt: list[str]       # multi-line system prompt prepended to user prompt
    y_norm_scale_factor: float  # 0.01
    model_max_length: int       # 300

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SanaWmConfig":
        ...  # parse model:, scheduler:, and text_encoder: blocks from config.yaml
```

The pipeline constructor calls `SanaWmConfig.from_yaml(model_path / "config.yaml")`
and passes the result to `SanaWmTransformer3DModel.__init__`. This makes the model
directly instantiable from the HF checkout without any intermediate generated file.

Registry entry in `vllm_omni/diffusion/registry.py`:

```python
_DIFFUSION_MODELS["SanaWmTwoStagesPipeline"] = (
    "sana_wm",
    "vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm_two_stages",
    "SanaWmTwoStagesPipeline",
)
```

Only add this entry after the pipeline can successfully load official weights (P1
exit gate), not during scaffold work in P0.

---

### Weight loading integration

Standard pattern in vllm-omni is to delegate `load_weights` to `AutoWeightsLoader`:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights)
```

Stage 1's flat safetensors keys (e.g. `blocks.0.attn.A_log`) do not include a
`transformer.` prefix, so they will not match the vllm-omni module tree without
remapping. `weight_mapping.py` must expose a `remap(name: str) -> str | None`
function used inside `load_weights`:

```python
# pipeline_sana_wm.py
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    from vllm_omni.diffusion.models.sana_wm import weight_mapping
    remapped = (
        (weight_mapping.remap(name), tensor)
        for name, tensor in weights
        if weight_mapping.remap(name) is not None
    )
    loader = AutoWeightsLoader(self)
    return loader.load_weights(remapped)
```

`remap` returns `None` only for keys on the explicit ignore allowlist (e.g.
training-only buffers confirmed absent at inference). **Do not silently drop
unexpected keys** — let `AutoWeightsLoader` raise on shape mismatches so breakage
is visible immediately.

Stage 2 components (refiner transformer, connectors, text encoder, VAE) load from
their diffusers-style subfolders and use unmodified key paths — no remapping needed.
Each component's `load_weights` is called in isolation; do not attempt to load all
stages from a single call. Follow `LTX2TwoStagesPipeline`'s precedent.

### Request schema (concrete recommendation)

Official usage is first-frame image-to-video plus either an explicit camera
trajectory or the WASD/IJKL action DSL. Normalize user-facing payloads into one
canonical internal dictionary while preserving the original input for debugging:

```json
{
  "image": "<first frame image or URL>",
  "prompt": "A camera-controlled driving scene ...",
  "sana_wm": {
    "num_frames": 321,
    "camera": {
      "format": "c2w_4x4",
      "coordinate_system": "official",
      "poses": [[[/* F x 4 x 4 */]]]
    },
    "action": null,
    "intrinsics": {
      "fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0,
      "width": 1280, "height": 704
    }
  }
}
```

Validation rules:
- `image` is required for the official bidirectional checkpoint path. Text-only
  should remain unsupported unless upstream exposes and validates it.
- Exactly one of explicit `camera.poses` or `action` should be supplied.
- Explicit `camera.poses` follows the model card: NumPy-like `(F, 4, 4)`
  camera-to-world matrices.
- `action` follows the model card's WASD/IJKL DSL, e.g.
  `"w-80,jw-40,w-40,lw-60,w-100"`, and rolls out to `(F+1, 4, 4)` before
  downstream conversion. Do not force `len(poses) == num_frames` on this path.
- `intrinsics` may be `(3, 3)`, `(F, 3, 3)`, or `(4,)`; if omitted, upstream
  estimates intrinsics with Pi3X and rejects FOV outside `[25°, 120°]`. We can
  either port that estimator or require explicit intrinsics in vLLM-Omni.
- The processor converts camera-to-world poses to raymap / Plücker coordinates
  using NVlabs' formula before injecting into the DiT.
- `fps` and `num_frames` must round-trip with the latent temporal
  compression (LTX-2 VAE strides 8 in time; SANA-WM uses
  `chunk_plucker_channels=48`).
- The official output size is fixed at 704×1280; inputs are aspect-preserving
  resized and center-cropped to that resolution.

### Recommended e2e test layout

Mirror `tests/diffusion/models/ltx2/` and `tests/e2e/accuracy/test_ltx2_3_video_similarity.py`:

```text
tests/diffusion/models/sana_wm/
  test_sana_wm_pipeline.py             # smoke: Stage 1 only I2V, 256×448, 24f, 2 steps
  test_sana_wm_two_stages.py           # smoke: + refiner
  test_sana_wm_camera_control.py       # camera/action → plücker shape parity
  test_sana_wm_hsdp.py                 # mirror test_ltx2_hsdp.py
  test_sana_wm_cfg_parallel_adaptation.py  # mirror existing LTX2 cfg-parallel test

tests/model_executor/stage_input_processors/
  test_sana_wm.py                      # trajectory validation/normalisation

tests/e2e/accuracy/
  test_sana_wm_video_similarity.py     # vs NVlabs reference, PSNR ≥ 30, SSIM-Y All ≥ 0.93

tests/examples/offline_inference/
  test_sana_wm_cfg_parallel_parity.py  # mirror test_ltx2_cfg_parallel_parity.py

tests/dfx/perf/tests/
  test_sana_wm_vllm_omni.json
```

### GPU tier policy (same shape as Lance §9)

- **Tier 1 PR CI smoke** (single 40-48 GB): Stage 1 only I2V, 256×448, 24 frames, 2 steps.
- **Tier 2 nightly** (single 80 GB): Stage 1 only at 704×1280, full inference shift `9.8`.
- **Tier 3 expansion + refiner** (4× 80 GB): full Stage 1 + LTX-2 refiner;
  Stage 1 alone is ~10 GB, refiner transformer/connectors + Gemma-3-12B +
  VAE require substantially more than a single 80 GB card if kept resident
  → single-card 80 GB cannot host all of it concurrently. Stage CPU
  offload (per LTX-2's existing pattern) or multi-GPU.

### Phased rollout (post-release-update)

| Phase | Scope | Exit |
| --- | --- | --- |
| P0 | scaffold + registry + Stage-1/Stage-2 pipeline classes + processor stub | compileall + import smoke green |
| P0.5 | Official CLI backend bridge for GPU smoke while native DiT is still being ported | `SANA_WM_E2E=1` e2e test can call NVlabs runner |
| P1 | Stage 1 DiT weight load (no GDN forward yet, softmax fallback only) + image/camera input packing | single 1-step forward returns correct shape |
| P2 | Add `GatedDeltaNetTriton` kernel + plücker camera injection → end-to-end Stage 1 I2V 256×448 24f | offline e2e green |
| P3 | LTX-2 refiner attach via `refiner/transformer/`, `refiner/connectors/`, and dual-text-encoder loading | Stage 1 + refiner e2e green at 704×1280 |
| P4 | Online serving, recipes, accuracy ≥ thresholds | CI online green; accuracy passes |
| P5 | SP/USP/CFG-parallel/HSDP + dfx perf; cache-DiT if useful | dfx green |

Each phase = one PR; PR description should link issue #3656 and this doc.

### Current code status (2026-05-20)

Implemented in this branch:

- `SanaWmPipeline` / `SanaWmTwoStagesPipeline` are registered and validate the
  released HF layout.
- Request preprocessing normalizes first-frame image + action/camera +
  intrinsics, and preserves official CLI motion options:
  `translation_speed`, `rotation_speed_deg`, and `official_repo_path`.
- `SanaWmConfig` reads root `config.yaml`; `weight_mapping.py` remaps flat
  Stage-1 checkpoint names into the internal module namespace.
- `vllm_omni.diffusion.models.sana_wm.official_backend` provides a real GPU
  e2e bridge to NVlabs' official runner. Set:

```bash
export VLLM_OMNI_SANA_WM_OFFICIAL_REPO=/path/to/NVlabs/Sana
python examples/offline_inference/sana_wm/sana_wm.py \
  --official-repo "$VLLM_OMNI_SANA_WM_OFFICIAL_REPO" \
  --model Efficient-Large-Model/SANA-WM_bidirectional \
  --image asset/sana_wm/demo_0.png \
  --prompt asset/sana_wm/demo_0.txt \
  --action "w-80,jw-40,w-40,lw-60,w-100" \
  --num-frames 321 \
  --output sana_wm_output.mp4
```

GPU e2e test entry:

```bash
SANA_WM_E2E=1 \
VLLM_OMNI_SANA_WM_OFFICIAL_REPO=/path/to/NVlabs/Sana \
pytest tests/e2e/accuracy/test_sana_wm_video_e2e.py -q
```

Important limitation: as of the checked public NVlabs/Sana `main` branch, the
repository has `inference_video_scripts/inference_sana_video.py` but does not
yet contain the model-card referenced `inference_sana_wm.py`. The bridge is
therefore the first executable path once that official runner is present; the
native vLLM-Omni transformer path still requires P1/P2.

### Pitfalls (post-release-update)

1. **HF README and file list disagree on the refiner layout.** The README still
   mentions `refiner/refiner.safetensors`; the actual repository file list
   currently uses `refiner/transformer/` and `refiner/connectors/`. Implement
   against the file list, and keep a compatibility note if upstream changes it
   again.
2. **`config.yaml::text_encoder.chi_prompt` must be applied verbatim.** Skipping
   it shifts the prompt distribution Stage 1 was trained on and degrades quality
   silently.
3. **`flow_shift` differs between train/inference.** Training used 9.95; inference
   uses 9.8. The pipeline must read `inference_flow_shift` at request time.
4. **GDN is hybrid, not pure linear attention.** `softmax_every_n: 4` means every
   4th block is plain softmax. Don't replace ALL attention with GDN.
5. **Plucker is injected post-attention in every block** (`chunk_plucker_post_attn_blocks: 20`).
   Skipping it on any block produces a model that runs but loses camera control.
6. **`google/gemma-2-2b-it` is not in the SANA-WM repo.** Loader must download
   it explicitly; SANA-WM repo only ships Gemma-3-12B for the refiner.
7. **The official checkpoint is I2V-first.** Do not build a text-only serving
   path unless upstream validates one; first-frame image preprocessing is part
   of the contract.
8. **F32 on disk for Stage 1.** Cast non-attention paths to bf16 at load time
   per `config.yaml::mixed_precision: bf16`; keep attention math aligned with
   `fp32_attention: true`.
9. **Refiner/VAE license inheritance is not captured by a local metadata blob
   in the current split layout.** The HF model card says the bundled LTX-2
   refiner and VAE inherit the LTX-2 upstream license; document this in recipes
   and release notes.
10. **Stage 1's `fp32_attention: true` + `mixed_precision: bf16` requires explicit
    dtype discipline in the Triton GDN kernel.** Recommended policy:
    - Linear projections entering the block (Q/K/V, gate): bf16.
    - `A_log` accumulator and state-space recurrence path: f32 (precision-sensitive;
      the log-domain state diverges in bf16 on long sequences).
    - `conv_k` depthwise (K-only, kernel=4): bf16.
    - Output projection back to residual stream: bf16.
    The Triton kernel must accept an explicit `compute_dtype` argument rather than
    inferring from input tensors to prevent accidental mixed-mode promotion. Do not
    cast the entire block to f32 — it doubles memory bandwidth on non-attention paths.
11. **20-block × 42-tensors/block** budget on the loader: the weight map
    has to enumerate per-block keys; can't just regex-skip prefixes.
12. **`config.yaml` is the single source of truth for Stage 1 architecture.**
    If the HF repo is later updated (e.g. a new checkpoint adds `num_blocks: 28`),
    `SanaWmConfig.from_yaml` will pick it up automatically. Do not hardcode
    architecture constants (num_blocks=20, hidden_size=2240) anywhere except as
    defaults in `SanaWmConfig` with a comment pointing to `config.yaml`.

---

## Native Stage-1 Production Readiness Gap (post-revision-8 review)

> Added 2026-05-27 after an independent peer review of the
> revision-8 `feat/sana_wm` branch HEAD (`7148ecdf`). The progress
> audit puts the branch at ~82–85% against the post-release plan,
> but the remaining gap is concentrated in **five mutually-coupled
> items** that together gate any production claim on the native
> (non-CLI-bridge) Stage-1 path. Each item below ships with a
> concrete acceptance criterion and a cross-reference to the audit
> doc so the two files stay in sync.

### Why this section exists separately

The reference-alignment harness landed in `0ec9a7af` and produced
`MAE = 69.64 / 255.0` on the GPU instance. That number proves the
*harness* is functional. It does **not** prove the native path
matches the official path, because — as the next subsection makes
explicit — `_run_native_smoke_backend` still seeds latents from
`torch.randn` instead of encoding the first frame. Until that
single line changes, native-vs-official MAE measures "noise +
conditioning distribution similarity," not model behaviour, and
the audit's loose `MAE ≤ 255` threshold cannot meaningfully tighten.

The five items below are the critical path to closing that gap.
They must land in roughly this order: items 2 and 4 unblock the
"can a number actually mean something" question; item 3 unblocks
multi-step quality; item 1 unblocks quality at scale; item 5 is
deliberately last because every perf optimisation is a quality risk
that cannot be evaluated until 1–4 land.

### 1. GDN Triton kernel — production correctness pass

**Status:** vendored kernel in `fused_gdn_chunkwise.py` (2,269 LOC,
Apache-2.0 from NVlabs) is wired into `SanaWmSelfAttention._forward_gdn`
on CUDA with PyTorch fallback. Small fused-vs-reference parity passes
(`atol=rtol=1e-2`) and the multi-shape / fp32+bf16 / disable-env /
warmup matrix added in `7148ecdf` passes on RTX PRO 6000 Blackwell.

**Main-branch math is verified equivalent to NVlabs.** Deep-dive on
2026-05-27 (audit §6.10) confirmed that `_delta_scan`, bidirectional
`flip_and_shift`, bidirectional short conv, β/decay gates, ReLU
kernel placement, `prepare_rope_tables` pair-sign encoding, and
RMSNorm over hidden_size all match `torch_recurrent_sana_gdn` and
`BidirectionalGDN.forward` line-for-line. The catastrophic
reference-alignment MAE (95.82 on 2026-05-27) is **not** caused by
the GDN recurrence itself — it is caused by item 4 (cam branch
missing, see §6.10 / item 4 of this section).

**What's still open:**

- **Frame-level parity vs. the official NVlabs path** at realistic
  Stage-1 video shape (`T=11, H=22, W=40` at 704×1280, 321 frames),
  not just the small test shapes. The chunkwise scan has subtle
  numerical sensitivity in the log-domain state; small-shape parity
  does not imply long-sequence parity.
- **Multi-card correctness.** The current parity tests are
  single-GPU. Once Stage-1 routes through vLLM TP layers (item 5),
  the fused kernel needs to run correctly under shard splits along
  the head dim and produce the same output as the single-card path.
- **Dtype discipline audit.** Spec §"Pitfalls" #10 specifies the
  policy (bf16 projections, f32 `A_log` state, bf16 `conv_k`,
  bf16 output). The vendored kernel needs an explicit audit that
  it doesn't silently up- or down-cast the state-space path; the
  `compute_dtype` argument should be threaded through.
- **Inference-only scope.** No backward pass needed. State this
  explicitly to prevent scope creep into training-style kernels.

**Acceptance criterion:**

```text
[ ] tests/diffusion/models/sana_wm/test_sana_wm_gdn_triton.py
    extended with a "long-sequence parity" case at T=11, with
    rtol/atol ≤ 1e-2 bf16 / 1e-4 fp32 against the PyTorch reference.
[ ] Block-0 isolation harness: dump one block's output from the
    official NVlabs path on a fixed input + state; vllm-omni's
    fused path matches to rtol=1e-2 bf16 / 1e-4 fp32.
[ ] tp_size=2 fused vs. tp_size=1 fused produces identical decoded
    output for a 24-frame 256×448 smoke (after item 5 lands).
[ ] `compute_dtype` is an explicit argument to the kernel entry
    points, not inferred from input dtypes.
```

Cross-ref: audit §6 item 2, §8 item 2.

### 2. First-frame VAE encode — root non-comparability cause

**Status (verified in code):** `pipeline_sana_wm.py::_run_native_smoke_backend`
currently does:

```python
latents = torch.randn(
    (1, 128, latent_frames, latent_height, latent_width),
    device=device, dtype=dtype, generator=generator,
)
```

The first frame from `prompt["image"]` is validated by the request
processor and resized inside the official-CLI-bridge path, but is
**never encoded** in the native smoke path. The VAE is initialised
(`_ensure_vae` runs to allow decoding the result), but no
`_ensure_vae_encode(image)` exists.

**Why this matters more than any other item:** the official
checkpoint is image-to-video first. Reference-alignment MAE between
official-bridge output and native-smoke output is currently
measuring "noise plus camera conditioning plus partial Gemma
prompt similarity," not "model output." The audit's
`MAE = 69.64 / 255.0` result will not move meaningfully no matter
how good the GDN kernel, scheduler, or camera embedder become,
because the model is never told what the first frame is.

**Required changes (concrete):**

```python
# pipeline_sana_wm.py
def _ensure_vae_encode(self, image: PIL.Image.Image | torch.Tensor,
                       *, device, dtype, num_frames) -> torch.Tensor:
    """Encode the first frame with the LTX-2 VAE and return the
    Stage-1 first-frame latent slot (B, C, 1, H/32, W/32)."""
    self._ensure_vae(device=device, dtype=dtype)
    frame_tensor = _resize_and_center_crop(image, (704, 1280))  # CHW, [-1, 1]
    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(2)        # (1, 3, 1, H, W)
    posterior = self.vae.encode(frame_tensor.to(self.vae.dtype)).latent_dist
    return posterior.sample().to(dtype=dtype)                    # (1, 128, 1, H/32, W/32)

# in _run_native_smoke_backend, replace the torch.randn call with:
first_frame_latent = self._ensure_vae_encode(
    payload["image"], device=device, dtype=dtype, num_frames=params.num_frames,
)
noise = torch.randn(
    (1, 128, latent_frames, latent_height, latent_width),
    device=device, dtype=dtype, generator=generator,
)
latents = noise
# I2V conditioning: at every denoising step, overwrite the first
# frame slot with first_frame_latent before calling the transformer,
# matching the official I2V pattern.
```

**Acceptance criterion:**

```text
[ ] `_ensure_vae_encode` exists and is unit-tested with a synthetic
    PIL image: returns (1, 128, 1, 22, 40) for 704×1280 input
    (modulo VAE-actual spatial compression).
[ ] `_run_native_smoke_backend` calls `_ensure_vae_encode` and
    pins the first-frame latent slot at every denoising step.
[ ] Reference-alignment MAE on the same fixed prompt + image +
    camera drops below the current 69.64 baseline. Target: MAE
    falls under 30 for a 24-frame 256×448 smoke.
```

Cross-ref: audit §6 item 1, §6 item 3.

### 3. Scheduler — vendor NVlabs flow-DPM solver (not diffusers)

**Status (verified in code):** `SanaWmFlowDpmScheduler` is a
dataclass with one Euler step:

```python
def step(self, latents, noise_pred, delta) -> torch.Tensor:
    return latents - delta * noise_pred  # explicit Euler
```

Its docstring acknowledges: "exact numerical parity still belongs
to the official backend until the upstream solver is ported."
This does not match `config.yaml::vis_sampler: flow_dpm-solver`.

**Recommendation: vendor NVlabs' solver directly, do not use diffusers'
`FlowMatchDPMSolverMultistepScheduler`.** Same Apache-2.0 vendor
pattern we used for `fused_gdn_chunkwise.py`. Rationale:

- Bit-level parity with the reference path. Sigma schedules,
  history-buffer initialisation, and final-step handling vary
  enough between solvers that "looks the same" turns into "MAE
  drifts by 10–20" over 30 steps.
- `inference_flow_shift = 9.8` is non-standard. The diffusers
  scheduler accepts a `shift` parameter but the way it composes
  with the DPM-Solver coefficient table is solver-implementation-
  dependent; we'd be debugging the diffusers internals to chase
  parity. Cheaper to vendor.
- We already vendor 2,269 LOC of Triton kernel code from the same
  upstream repo. Adding ~300 LOC of pure-Python solver is small
  incremental maintenance surface.

**Required surface:**

```python
# scheduling_sana_wm.py
class SanaWmFlowDpmSolverMultistep:
    """Vendored from NVlabs/Sana::diffusion/schedulers/scheduling_flow_dpm.py
    (Apache-2.0)."""
    def __init__(self, num_inference_steps: int,
                 flow_shift: float = 9.8,
                 solver_order: int = 2):
        ...
    def set_timesteps(self, *, device: torch.device) -> None: ...
    def step(self, model_output: Tensor, timestep: Tensor,
             sample: Tensor) -> Tensor: ...
```

History-buffer-based multi-step formulation; signature should
match `LTX2`'s scheduler so the in-process refiner can reuse the
step loop without restructuring.

**Acceptance criterion:**

```text
[ ] `SanaWmFlowDpmSolverMultistep` ships in `scheduling_sana_wm.py`
    with file-header provenance comment ("ported from NVlabs/Sana,
    Apache-2.0, NVIDIA copyright preserved").
[ ] Scheduler-call-sequence unit test (mirrors Wan2.2's
    `test_wan22_pipeline_diffuse.py`): monkeypatch
    `predict_noise_maybe_with_cfg`, assert 30-step solver calls
    `step` in the right order with the right history-buffer state.
[ ] `SanaWmFlowDpmScheduler` (the current Euler dataclass) is
    retained as a debug-only opt-in (`SANA_WM_USE_EULER_DEBUG=1`)
    or removed once the new solver lands.
```

Cross-ref: audit §7 item 2.

### 4. Camera embedder + per-block UCPE attention

> **Two distinct sub-problems, intentionally kept under one item.**
> (a) The camera *embedder* converts Plücker / raymap inputs into
>     hidden-state-shaped features. Partially closed by commit
>     `f7e59121` A.3.
> (b) The per-block *UCPE attention branch* is what makes those
>     features actually condition the recurrence. Verified missing
>     on 2026-05-27 — see sub-section 4b below.

#### 4a. Camera embedder structure

**Status (verified in code):** `SanaWmCameraEmbedder` is two trivial
projections:

```python
self.plucker.proj = nn.Conv3d(config.chunk_plucker_channels,
                              config.hidden_size, kernel_size=1)
self.raymap.proj = nn.Linear(20, config.hidden_size)
```

Its own docstring: "Small camera branch used by the native smoke path."

This does not match the official architecture. `config.yaml` declares:

```yaml
camctrl_type: BidirectionalGDNUCPESinglePathLiteLABothTriton
cam_attn_compress: 1
init_cam_from_base: true
use_chunk_plucker_post_attn: true
chunk_plucker_channels: 48
chunk_plucker_post_attn_blocks: 20   # injected into ALL blocks
```

UCPE = Unified Camera-Pose Encoding. "SinglePathLite" + "LABoth"
indicate a specific compress-then-broadcast architecture, not a
single 1×1 conv.

**Required decomposition:**

```text
plucker_embedder:
  - K=4 conv (matches `conv_kernel_size: 4` in main attention),
    multi-scale per `aspect_ratio_type: ASPECT_RATIO_VIDEO_720_MS_DIV32`.
  - Input channels: 48 (chunk_plucker_channels).
  - Output channels: 2240 (hidden_size).
  - Injected post-attention in every one of the 20 blocks.

raymap_embedder:
  - Input: 20-d ray features (already correct shape).
  - Multi-block injection (currently only injects via the plucker
    side; raymap needs its own per-block path).

UCPE branch:
  - cam_attn_compress=1: 1x compression on camera attention.
  - init_cam_from_base=True: camera-branch weights warm-started
    from the base attention block at load time (weight_mapping must
    handle the duplication explicitly).
  - LABoth: linear attention on both spatial and temporal axes.
```

**Numeric reference test (Kimi P2 item):** beyond
`test_sana_wm_camera_control.py`'s current shape-only assertions,
add a numeric check against a reference Plucker embedding produced
by a verbatim port of NVlabs' `camera_utils.py` on a fixed
trajectory (e.g. straight-line `w-80,w-40,w-40,w-100`).

**Acceptance criterion:**

```text
[ ] `SanaWmCameraEmbedder` is renamed or replaced; the docstring
    "smoke path" qualifier is removed.
[ ] Three sub-modules — `plucker_embedder`, `raymap_embedder`,
    `ucpe_branch` — exist with `weight_mapping.py` entries that
    correctly map `plucker_embedder.*` and `raymap_embedder.*`
    Stage-1 checkpoint keys (currently mapped to
    `camera_encoder.plucker.*` / `camera_encoder.raymap.*`).
[ ] init_cam_from_base warm-start is implemented in `load_weights`
    or `weight_mapping.remap` and tested.
[ ] tests/diffusion/models/sana_wm/test_sana_wm_camera_control.py
    asserts numeric equivalence (atol=1e-5 fp32) to a reference
    Plucker tensor dumped from NVlabs camera_utils on a fixed
    trajectory.
```

Cross-ref: audit §6 item 6, §7 item 3.

#### 4b. Per-block UCPE attention branch (verified missing 2026-05-27)

**Status (verified by deep-dive vs `sana_gdn_camctrl_blocks.py` on
2026-05-27):** ❌ Not implemented. The camera *embedder* in 4a is
correct shape and loads from the checkpoint, but the *per-block
camera attention branch* that consumes its output is wrong on every
axis that matters. Full evidence is in audit §6.10.

**What NVlabs does** (`_GDNUCPEBase.forward`, ~70 LOC):

```text
1. precomputed_gates = self._compute_frame_gates(x, HW)    # shared
2. main_raw = super().forward(x, ..., apply_output_gate=False,
                              precomputed_gates=precomputed_gates)
3. cam_raw  = self._forward_cam_branch(x, HW, camera_conditions,
                                       rotary_emb,
                                       precomputed_gates=precomputed_gates)
   #  └── _prepare_cam_qkv: project → mask → conv → norm → ReLU →
   #         scale → permute → UCPE per-ray transforms (q_cam_trans,
   #         k_cam_trans) + inflation_sq
   #  └── beta_cam = beta / inflation_sq.clamp_min(1.0)
   #  └── bidirectional GDN recurrence with the SAME _delta_scan as
   #         main branch, but using q_cam_trans/k_cam_trans as the
   #         rotary inputs
4. combined = main_raw + self.out_proj_cam(cam_raw)
5. output   = self.proj(self._apply_output_gate(combined, x))
```

**What we have** (`SanaWmSelfAttention`):

1. `_forward_ucpe` uses `F.scaled_dot_product_attention` — **wrong
   operator family** (SDPA vs GDN recurrence with per-ray transforms).
2. `_forward_ucpe` is **never invoked** from `forward()` — verified
   by grep. `camera_hidden_states` is accepted as a kwarg and dropped.
3. `_forward_gdn` applies `output_gate` and `proj` directly on
   `main_raw`, leaving no insertion point for `cam_contrib`.
4. `prepare_prope_fns`, `_prepare_cam_qkv`, Dynamic Beta Discounting
   (β ÷ inflation_sq) — none exist.
5. `out_proj_cam`, `q_proj_cam`, `k_proj_cam`, `v_proj_cam`,
   `q_norm_cam`, `k_norm_cam`, `conv_k_cam` are constructed and
   weight-loaded but **dead code** at inference time.

**Impact.** The 2026-05-27 reference-alignment harness reported
MAE=95.82 / PSNR=7.37 dB / SSIM-Y=0.0047 — essentially uncorrelated
to the official path. That is exactly the signature of a model
that has been stripped of its camera-conditioning mechanism: the
GDN recurrence runs, the pipeline produces a 128-channel latent of
the right shape, but the latent does not respond to the camera
trajectory at all.

**Required port (P0 — estimated 4–5 person-days):**

```text
1. New file vllm_omni/diffusion/models/sana_wm/ucpe.py
   - Port prepare_prope_fns() from NVlabs sana_camctrl_blocks.py
     (returns apply_fn_q, apply_fn_kv, apply_fn_o closures).
   - Pure-PyTorch reference first; Triton fusion can come later.

2. SanaWmSelfAttention._prepare_cam_qkv(...)
   - Order: project (fused qkv_w) → mask → temporal short-conv →
            q_norm_cam/k_norm_cam → ReLU → k_scale → permute →
            UCPE transforms → inflation_sq measurement.

3. SanaWmSelfAttention._forward_cam_branch(...)
   - Reuse reference_bidirectional_gated_delta_net.
   - Pass q_cam_trans / k_cam_trans as query_rot / key_rot.
   - Pre-compute β / decay once at the block level; share between
     main and cam branch; discount β_cam by inflation_sq.

4. Refactor _forward_gdn to accept apply_output_gate: bool.
   - When False, return raw GDN output (B, N, C) before
     output_gate/proj.

5. SanaWmSelfAttention.forward rewrite:
       precomputed_gates = self._compute_frame_gates(x, spatial_shape)
       main_raw = self._forward_gdn(x, spatial_shape, rotary_emb,
                                    apply_output_gate=False,
                                    precomputed_gates=precomputed_gates)
       if camera_hidden_states is not None:
           cam_raw = self._forward_cam_branch(
               x, spatial_shape, camera_hidden_states, rotary_emb,
               precomputed_gates=precomputed_gates)
           combined = main_raw + self.out_proj_cam(cam_raw)
       else:
           combined = main_raw
       gate = F.silu(self.output_gate(x))
       return self.proj(combined * gate)

6. Route camera_hidden_states from the top-level transformer
   down to every block's forward() (currently the kwarg exists on
   the attention layer but the block doesn't forward it).
```

**Acceptance criterion:**

```text
[ ] tests/diffusion/models/sana_wm/test_sana_wm_ucpe.py exists:
    - UCPE prope_fns closures match NVlabs reference at atol=1e-5
      fp32 on a fixed camera trajectory.
[ ] tests/diffusion/models/sana_wm/test_sana_wm_block_vs_nvlabs.py
    exists: single-block end-to-end forward, our block output vs
    NVlabs _GDNUCPEBase.forward at atol=1e-4 fp32 single-card.
[ ] _forward_ucpe is either deleted or rewritten to use GDN
    recurrence; SDPA on the cam branch is gone.
[ ] SanaWmSelfAttention.forward routes camera_hidden_states into
    a cam branch when present; verified by grep.
[ ] e2e reference-alignment MAE drops below 30 (currently 95.82),
    proving the cam branch is contributing.
```

Cross-ref: audit §6.10, audit §7 item 10.

### 5. Performance optimisations — ordered DAG, not parallel todos

**Status:** static wiring exists; no multi-GPU sweep ran.
`CFGParallelMixin` is inherited, HSDP block conditions are
declared, dfx perf JSON has three concrete entries. None of
this has been validated end-to-end on a multi-GPU instance.

**Critical constraint: these are NOT independent.** Treating them
as a parallel todo list will produce optimisations that look good
on a microbenchmark and silently regress quality. Required ordering:

```text
5a (must precede everything else): TP-aware layers.
    Replace plain nn.Linear / nn.Conv3d / SanaWmRMSNorm with
    QKVParallelLinear / ColumnParallelLinear / RowParallelLinear /
    vLLM RMSNorm. This is Kimi's ⭐⭐☆☆☆ framework-consistency item
    (audit §10.3). Without it, TP/SP/quant inherit nothing.

5b (after 5a + items 1-4): HSDP + USP/Ulysses.
    HSDP shard policy on `blocks.*` is already declared. USP/Ulysses
    requires routing attention through
    vllm_omni.diffusion.attention.layer.Attention so the SP plan can
    inject all-to-all at the right boundary. Cannot validate without
    5a because the layer-level SP boundaries don't exist yet.

5c (after 5b): CUDA Graphs.
    Static shapes are partially given (fixed 704×1280), but
    num_frames varies per request → bucket by num_frames (e.g.
    {65, 129, 193, 257, 321}) and capture per-bucket graphs.
    Skip this entirely if num_frames variation is dominant
    in production traffic.

5d (last, gated by quality): Cache-DiT.
    Spec §"Phased rollout" item 7 already says "Cache-DiT only
    after baseline quality and deterministic behavior are known."
    Spatial-feature cache (within-step across blocks) is lower-risk;
    temporal-feature cache (across denoising steps) requires the
    scheduler from item 3 to be deterministic. Do not enable
    Cache-DiT for the production default until reference-alignment
    PSNR ≥ 30 / SSIM-Y ≥ 0.93 has been demonstrated WITHOUT it.
```

**Acceptance criterion:**

```text
[ ] 5a: SanaWmTransformer3DModel uses vLLM parallel layers
    end-to-end; tp_size=2 produces identical decoded output to
    tp_size=1 for a 24-frame 256×448 smoke (deterministic seed).
[ ] 5b: dfx perf entry for HSDP + USP shows >1.6x speedup over
    single-GPU at 4×80 GB; output PSNR vs. single-GPU ≥ 40.
[ ] 5c: num_frames-bucketed CUDA graph capture works for the top-3
    request shapes in observed traffic; no quality regression vs.
    eager.
[ ] 5d: Cache-DiT temporal+spatial both opt-in (env var or
    sampling_params extra arg) and ship disabled by default until
    reference-alignment thresholds are met without it.
```

Cross-ref: audit §6 items 4 and 9, §8 items 5 and 6, §10.

### Rev-10 progress on item 5a

Commits `63c58d08`, `845247bd`, `c77360a3`, and `7317c76e` landed
after the revision-9 spec snapshot and partially close item 5a:
Stage-1 FFN / cross-attention / GDN projection paths migrated to
`ColumnParallelLinear` / `QKVParallelLinear` / `RowParallelLinear`;
`quant_config` threaded; `_sp_plan` with explicit USP split/gather
boundaries declared; GPU-gated TP-layer completeness tests added.
Remaining item 5a debt (camera attention projections, `x_embedder`
Conv3d, `t_block` modulation, `final_layer.linear`, `plucker_proj`,
`quant_config` end-to-end, real USP sweep) is tracked in audit §0.6
and §6 item 4.

CUDA Graphs (`cuda_graph.py`) and Cache-DiT (`enable_cache_for_sana_wm`)
scaffolding for items 5c–5d is on disk but uncommitted; GPU smoke
required before landing.

### Recap — critical path to "production native"

> **Updated 2026-05-27** after deep-dive vs NVlabs reference: item 4
> (camera UCPE branch) is the dominant blocker for reference
> alignment — not the GDN kernel. Main-branch GDN math is verified
> equivalent. See §6.10 of the audit and item 4b above.

```text
ITEM 2 (VAE encode)      ───┐
                            ├── ITEM 3 (scheduler) ───┐
ITEM 4 (camera UCPE)*    ───┘                          ├── reference-
   * dominant blocker —                                │   alignment
     reopened by audit §6.10                           │   tightening
                                                       │
ITEM 1 (GDN long-seq parity)  ─────────────────────────┘
   main-branch math verified;
   long-seq Triton parity still open
                                                                │
ITEM 5a (TP layers)  ───────────────────────────────────────────┤
                                                                │
ITEM 5b (HSDP+USP)  ────────────────────────────────────────────┤
                                                                │
ITEM 5c (CUDA Graphs)  ─────────────────────────────────────────┤
                                                                │
ITEM 5d (Cache-DiT)  ───────────────────────────────────────────┘
                                                                │
                                                  ┌─────────────┘
                                                  ▼
                                  Production-native Stage-1
                                  (PSNR ≥ 30, SSIM-Y ≥ 0.93
                                   vs. official CLI bridge)
```

The current `feat/sana_wm` branch can ship the **official CLI
bridge path** today as a "preview" — that path is real, GPU-validated,
and produces decoded video. The **native path** should remain
behind `VLLM_OMNI_SANA_WM_NATIVE_SMOKE=1` / `_run_native_smoke_backend`
until items 1–4 land and the reference-alignment threshold tightens.
Marking Sana-WM as production-supported requires the full DAG above.
