# AuK Speech Generation and Editing

AuK and AuK-Flash use a single diffusion stage: Qwen2.5-Omni-3B semantic
conditioning, continuous-latent flow matching, and an audio VAE decoder. The
output is 24 kHz mono audio. The initial implementation supports one GPU,
one request per forward, and complete audio responses.

## Checkpoints

Use the upstream YAML layout without converting the weights:

```text
ckpts/
  AuK/
    config.yaml
    auk_base.safetensors
    vae.safetensors
  AuK-Flash/
    config.yaml
    auk_flash.safetensors
    vae.safetensors
  Qwen2.5-Omni-3B/
```

The encoder is resolved from `additional_config.qwen_path`, the YAML path,
a sibling `Qwen2.5-Omni-3B` directory, or `Qwen/Qwen2.5-Omni-3B` on the Hub.
AuK's advertised 1.5B size describes its diffusion backbone; the encoder and
VAE consume additional memory. No prompt-enhancer service or ASR model is required.

## Generate speech

```bash
python examples/offline_inference/text_to_speech/auk/end2end.py \
  --model ckpts/AuK-Flash \
  --text "Hello, welcome to the demonstration." \
  --gen-seconds 3 --output speech.wav
```

Add `--ref-audio reference.wav` for voice cloning. Reference transcription is
optional and is not used by the model. Use `--qwen-path /path/to/Qwen2.5-Omni-3B`
when the encoder is stored elsewhere.

## Edit audio

```bash
python examples/offline_inference/text_to_speech/auk/end2end.py \
  --model ckpts/AuK \
  --instruction "Remove the background noise and preserve the speech." \
  --ref-audio noisy.wav --output clean.wav
```

An explicit `instruction` is passed to AuK unchanged. With `input` or `text`,
the pipeline builds a speech-generation instruction. This distinction prevents
editing commands from being synthesized as spoken text.

`gen_seconds` is required for reference-free generation. When reference audio
is present, omitting it requests an output with approximately the same duration.
The initial serving limit is 30 seconds for reference plus target audio.
Output duration is quantized to 20 ms latent frames.

## Sampling

Base defaults to 32 Euler steps, `cfg_strength=2`, and
`sway_sampling_coef=-1`. Pass these model-specific controls through
`OmniDiffusionSamplingParams.extra_args`; `num_inference_steps` controls the
Base step count. AuK uses `v_cond + cfg_strength * (v_cond - v_uncond)`.
The generic image `guidance_scale` is not the AuK CFG parameter.

Flash always uses its released four-step time grid with CFG disabled,
regardless of caller-supplied Base sampling controls. Both reference latent
sampling and target noise use request-local random generators. Repeating a seed
with identical inputs and execution settings is reproducible; the upstream
CLI's seed controls only target noise, so reference-audio comparisons must also
align the VAE random state.

## Speech API

```bash
vllm-omni serve ckpts/AuK-Flash --dtype bfloat16 --enforce-eager
```

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"ckpts/AuK-Flash","input":"Hello, welcome home.",
       "instructions":"A warm, gentle voice.",
       "extra_params":{"gen_seconds":3},"seed":42,"response_format":"wav"}' \
  --output speech.wav
```

This uses the existing pure-diffusion speech route. Audio editing is exposed by
the structured offline interface; there is no new editing HTTP endpoint.
Incremental streaming, tensor/sequence parallelism, quantization, and diffusion
cache approximations are not part of the initial integration.

## Module parity

Use a separate checkout of the official implementation for numerical parity:

```bash
git clone https://github.com/Tencent-Hunyuan/AuK /tmp/auk-reference
git -C /tmp/auk-reference checkout d9f30ffe4231dbc90b48cc83a35d310fece0b060
AUK_REFERENCE_PATH=/tmp/auk-reference \
AUK_PROCESSOR_PATH=/path/to/Qwen2.5-Omni-3B \
pytest tests/diffusion/models/auk/test_auk_modules.py
```

The reference tests require `torchdiffeq` and `qwen-omni-utils`. They compare
nonzero DiT outputs, multi-layer semantic fusion, Base/Flash Euler sampling,
VAE encode/decode and normalization, audio preprocessing, and real processor
token IDs/features/masks. Missing reference resources are reported as skipped;
they must be supplied when validating a change to these modules.

Run `pytest tests/diffusion/models/auk/test_auk_integration.py` for registry,
checkpoint validation, request determinism, and Omni SDPA mask/layout checks.
The CPU attention test selects SDPA and bypasses accelerator dispatch while
running the real backend implementation. These small-model checks do not
replace loading the released weights and generating audio on a GPU.
