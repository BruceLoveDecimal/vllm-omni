# Sana-WM Integration Spec

> **Status as of 2026-05-19 (post-weight-release update).**
> NVIDIA released the SANA-WM weights to Hugging Face on 2026-05-18
> (Beijing time evening). All "Work Allowed Before Weight Release"
> guidance below has been superseded by concrete checkpoint
> inspection — see §§ "Confirmed Release Inventory",
> "Concrete Architecture (from `config.yaml` + HF component configs)",
> and "Updated Implementation Plan" further down. The earlier
> pre-release sections are kept for historical context only.

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
