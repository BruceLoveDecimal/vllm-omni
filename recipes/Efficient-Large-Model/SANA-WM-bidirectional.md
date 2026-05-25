# SANA-WM Bidirectional

> First-frame image-to-video generation with camera control

## Summary

- Vendor: Efficient-Large-Model / NVIDIA Research
- Model: `Efficient-Large-Model/SANA-WM_bidirectional`
- Task: Image-to-video generation with camera trajectory control
- Mode: Offline smoke, official-reference bridge, and vLLM-Omni diffusion serving
- Maintainer: Community

## When To Use This Recipe

Use this recipe when you want to validate SANA-WM through vLLM-Omni while the
native port is still staged. The current implementation has three execution
paths:

- **Official backend**: calls an NVlabs/Sana checkout for real output parity.
- **Native smoke backend**: exercises vLLM-Omni Stage-1 request, camera, and
  scheduler plumbing at reduced resolution.
- **Two-stage backend**: loads the SANA-WM release layout, the bundled LTX-2
  refiner slots, and an opt-in in-process refiner step for native integration
  testing.

## References

- Model: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
- Tracking issue: <https://github.com/vllm-project/vllm-omni/issues/3656>
- Reference implementation: <https://github.com/NVlabs/Sana>

## Checkpoint Layout

The release is not a single monolithic diffusers pipeline. vLLM-Omni expects:

- `config.yaml`
- `dit/sana_wm_1600m_720p.safetensors`
- `vae/config.json`
- `vae/diffusion_pytorch_model.safetensors`
- `refiner/transformer/config.json`
- `refiner/transformer/diffusion_pytorch_model.safetensors`
- `refiner/connectors/config.json`
- `refiner/connectors/diffusion_pytorch_model.safetensors`
- `refiner/text_encoder/`

Stage 1 uses `google/gemma-2-2b-it` as a separate text encoder. Stage 2 uses
the bundled `refiner/text_encoder`, which is a Gemma-3 multimodal encoder. Do
not share the two text encoder instances.

## Downloading Refiner Weights

The refiner weights come from the same Hugging Face model repository:

```text
Efficient-Large-Model/SANA-WM_bidirectional/refiner/
```

The NVlabs/Sana checkout contains the official inference code and downloader,
but it does not contain the 80GB+ refiner weights by itself. The current
official code resolves this default path:

```text
hf://Efficient-Large-Model/SANA-WM_bidirectional/refiner
```

For a GPU machine with Hugging Face access, download only the needed files:

```bash
huggingface-cli download Efficient-Large-Model/SANA-WM_bidirectional \
  --include 'refiner/**' \
  --local-dir /autodl-pub/data/models/SANA-WM_bidirectional \
  --local-dir-use-symlinks False
```

If the GPU machine has no external network, download on a networked machine and
sync the `refiner/` directory:

```bash
rsync -az /path/to/SANA-WM_bidirectional/refiner \
  seeta-gpu:/autodl-pub/data/models/SANA-WM_bidirectional/
```

After syncing, point vLLM-Omni at the parent directory:

```bash
export VLLM_OMNI_SANA_WM_OFFICIAL_REPO=/root/autodl-tmp/NVlabs-Sana
export SANA_WM_E2E_MODEL=/autodl-pub/data/models/SANA-WM_bidirectional
```

Older notes may mention a monolithic `refiner/refiner.safetensors`. If that is
the only artifact available, convert it with NVlabs/Sana before using it:

```bash
python /path/to/NVlabs/Sana/tools/convert_sana_wm_refiner_to_diffusers.py \
  --checkpoint /path/to/refiner.safetensors \
  --output_dir /autodl-pub/data/models/SANA-WM_bidirectional/refiner
```

## GPU Support

### Tier 1: 40-48GB Smoke

Use this tier for native-smoke shape checks only. Keep the request small:

```bash
VLLM_OMNI_SANA_WM_NATIVE_SMOKE=1 \
python examples/offline_inference/sana_wm/sana_wm.py \
  --model Efficient-Large-Model/SANA-WM_bidirectional \
  --height 256 \
  --width 448 \
  --num-frames 24 \
  --num-inference-steps 1
```

### Tier 2: 80GB Stage-1 Validation

Use this tier for the real Stage-1 checkpoint and VAE decode path.

```bash
python examples/offline_inference/sana_wm/sana_wm.py \
  --model Efficient-Large-Model/SANA-WM_bidirectional \
  --height 704 \
  --width 1280 \
  --num-frames 81 \
  --num-inference-steps 4
```

### Tier 3: Multi-GPU Refiner / Accuracy

Use 4x or 8x 80GB GPUs for Stage 2 refiner validation and accuracy runs. The
refiner is a repackaged LTX-2 19B distilled refiner, so memory pressure is much
higher than Stage 1.

The current single-GPU integration smoke was validated on an RTX PRO 6000
Blackwell Server Edition (96GB). Loading the refiner text encoder, connectors,
and transformer uses roughly 60.6 GiB VRAM before the denoising step.

## Official Backend

Until the native Gated DeltaNet kernel is numerically validated, use the
official backend for real-video smoke tests:

```bash
export VLLM_OMNI_SANA_WM_OFFICIAL_REPO=/path/to/NVlabs/Sana
export VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1

SANA_WM_E2E=1 \
pytest tests/e2e/accuracy/test_sana_wm_video_e2e.py -q
```

The official repo must contain:

```text
inference_video_scripts/inference_sana_wm.py
```

## In-Process Refiner Smoke

The two-stage pipeline also exposes an opt-in native refiner path. This bypasses
the official CLI subprocess and routes the Stage-1 latent through the loaded
`refiner/text_encoder`, `refiner/connectors`, and `refiner/transformer`
components:

```bash
export VLLM_OMNI_SANA_WM_OFFICIAL_REPO=/path/to/NVlabs/Sana
export SANA_WM_E2E_MODEL=/path/to/SANA-WM_bidirectional
export SANA_WM_E2E_MODEL_CLASS=SanaWmTwoStagesPipeline
export SANA_WM_E2E_INPROCESS_REFINER=1
export SANA_WM_E2E_OUTPUT_TYPE=latent
export SANA_WM_E2E_REFINER_STEPS=1

SANA_WM_E2E=1 \
pytest tests/e2e/accuracy/test_sana_wm_video_e2e.py -q
```

This path is for integration validation. With `SANA_WM_E2E_OUTPUT_TYPE=latent`,
it validates the Stage-1 latent handoff through the loaded LTX-2 refiner without
spending additional time on VAE decode. The official CLI bridge remains the
quality reference until the native Stage-1 Gated DeltaNet kernel is numerically
matched against NVlabs/Sana.

## Request Shape

SANA-WM requires a first-frame image and exactly one camera control source.
Camera can be either an action DSL string or explicit poses.

```json
{
  "prompt": "A forward driving shot through a quiet city street.",
  "multi_modal_data": {"image": "/absolute/path/to/first_frame.png"},
  "sana_wm": {
    "action": "w-16",
    "num_frames": 81,
    "height": 704,
    "width": 1280,
    "translation_speed": 0.055,
    "rotation_speed_deg": 1.2
  }
}
```

Explicit camera poses use:

```json
{
  "sana_wm": {
    "camera": {
      "format": "c2w_4x4",
      "coordinate_system": "official",
      "poses": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    },
    "num_frames": 1
  }
}
```

## Known Limitations

- Native Gated DeltaNet is still a PyTorch reference fallback, not the fused
  Triton implementation.
- The in-process refiner path is an integration smoke, not yet a quality
  replacement for the official CLI bridge.
- There is no tagged upstream release yet; pin an exact vLLM-Omni commit and
  exact SANA-WM snapshot for reproducible tests.
