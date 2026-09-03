# dots.tts

> Continuous-AR TTS at 48 kHz (rednote-hilab)

## Summary

- Vendor: rednote-hilab
- Model: `dots-studio/dots.tts-soar`
- Task: Text-to-speech, zero-shot synthesis and voice cloning
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use this recipe as a known-good starting point for running
`dots-studio/dots.tts-soar` on vLLM-Omni on consumer-class GPUs.
dots.tts is a ~1.7B-parameter continuous-AR TTS model (Qwen2.5-1.5B base LM
+ 344M DiT flow-matching head + 180M AudioVAE) that emits 48 kHz mono audio.
It follows the same "vLLM-native base LM + side-path computation" pattern as
VoxCPM2 — single-stage pipeline
`Qwen2.5-1.5B base LM → DiT (10-step Euler flow matching) → patch_encoder AR
loopback → AudioVAE (streaming decode)` — with a plain Qwen2 backbone
instead of MiniCPM4, and no FSQ / residual-LM stage.

This is an early integration; review the [Known limitations](#known-limitations)
before deploying it in production.

## References

- Offline end-to-end script:
  [`examples/offline_inference/text_to_speech/dots_tts/end2end.py`](../../examples/offline_inference/text_to_speech/dots_tts/end2end.py)
- Example guide:
  [`examples/offline_inference/text_to_speech/README.md`](../../examples/offline_inference/text_to_speech/README.md#dotstts)
- Online serving guide:
  [`examples/online_serving/text_to_speech/README.md`](../../examples/online_serving/text_to_speech/README.md#dotstts)
- Default deploy config:
  [`vllm_omni/deploy/dots_tts.yaml`](../../vllm_omni/deploy/dots_tts.yaml)
- Talker / pipeline source:
  [`vllm_omni/model_executor/models/dots_tts/`](../../vllm_omni/model_executor/models/dots_tts/)
- Upstream: [rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts)

## Hardware Support

This recipe documents one tested 16 GB consumer-GPU configuration. Other
vendor sections (ROCm, NPU) and larger-VRAM configurations are welcome as
community validation lands.

## GPU

### 1 x RTX 5080 16GB (Single GPU, Minimum Recommended)

dots.tts (~1.7B params across the base LM + DiT + AudioVAE + CAM++
speaker encoder, bfloat16) fits comfortably on a single 16 GB GPU. The
bundled default config at
[`vllm_omni/deploy/dots_tts.yaml`](../../vllm_omni/deploy/dots_tts.yaml)
(`gpu_memory_utilization: 0.8`, `max_num_seqs: 4`, `enforce_eager: true`,
`enable_prefix_caching: false`) loads cleanly with ~5.1 GiB for model
weights and ~0.3 GiB peak activation; the remainder of the configured
budget is available for KV cache. Total resident footprint at idle is
roughly **7-8 GiB / 16 GB** — the only tight spot in the full CUDA-Graph
roadmap would be step 8's graph capture (not implemented yet; this
release runs `enforce_eager: true`, so it doesn't apply today).

#### Environment

- OS: Linux (WSL2)
- Python: 3.12
- Driver / runtime: NVIDIA driver 595.95
- torch: 2.11.0+cu130
- vLLM: 0.26.0
- vLLM-Omni: 0.22.1.dev (current `main`)

#### Command

```bash
python examples/offline_inference/text_to_speech/dots_tts/end2end.py \
    --model dots-studio/dots.tts-soar \
    --text "Hello, this is a test of dots TTS running on vLLM Omni."
```

The deploy config at
[`vllm_omni/deploy/dots_tts.yaml`](../../vllm_omni/deploy/dots_tts.yaml)
is loaded automatically by the model registry (HF `model_type=dots_tts`).
Pass `--deploy-config <path>` to override.

#### Verification

**T1 — offline zero-shot synthesis**:

```bash
python examples/offline_inference/text_to_speech/dots_tts/end2end.py \
    --model dots-studio/dots.tts-soar \
    --text "Hello, this is a test of dots TTS running on vLLM Omni."
```

Observed: `output_audio/output.wav`, 3.52 s @ 48 kHz mono. Single
one-shot process (init → one `generate()` → exit), so the reported
numbers include engine init and first-request warmup, not just steady-state
per-step throughput — same caveat as VoxCPM2's recipe. `Inference: 5.52s`,
`RTF: 1.569`.

Weight-loading breakdown from the same run (all tensors matched, no
missing/extra keys): 951/951 AudioVAE, 244/244 DiT, 270/270
patch_encoder, 198 Qwen2, 938/938 CAM++ speaker encoder.

Whisper transcription of the output matched the input text with no
dropped leading word (confirms the streaming-vocoder patch-boundary fix
described in [Known limitations](#known-limitations)).

**T2 — online zero-shot synthesis**:

```bash
vllm serve dots-studio/dots.tts-soar --omni --trust-remote-code --port 8091
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a test of dots TTS online serving.",
        "voice": "default",
        "response_format": "wav"
    }' --output output.wav
```

The endpoint also supports raw streaming audio with `stream=true`,
`stream_format="audio"`, and `response_format="pcm"`.

The server issues a synthetic warmup request at startup, moving the side
path's lazy initialization off the first real request. Measured on the
RTX 5090 box below, the warmup request took **3.8 s** while the same
request in steady state takes **1.24-1.40 s** (5 runs) — roughly 2.5 s
moved off the first real request.

**T3 — voice cloning**. Three conditioning modes:

| Request fields | Conditioning |
|---|---|
| `input` | zero-shot |
| `input`, `ref_audio` | CAM++ x-vector conditions the DiT (`g_cond`) |
| `input`, `ref_audio`, `ref_text` | additionally prefills the reference's audio latents into the DiT history and the patch-encoder KV cache |

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello from a cloned voice.",
        "ref_audio": "file:///path/to/reference.wav",
        "ref_text": "transcript of reference.wav",
        "response_format": "wav"
    }' --output cloned.wav
```

Verified on 1 x RTX 5090 32GB by taking a zero-shot output as the reference
and measuring CAM++ x-vector cosine similarity between the reference and
each generated clip:

| Conditioning | cosine similarity to reference |
|---|---|
| zero-shot (no reference) | 0.10 - 0.38 |
| `ref_audio` only | 0.73 - 0.81 |
| `ref_audio` + `ref_text` (prompt prefill) | 0.76 - 0.78 |

Both conditioning modes move speaker identity decisively away from the
unconditioned baseline. The two modes are not separable on this test: the
reference is itself a zero-shot output of the same model, and the
per-request DiT noise varies run to run, so the spread across repeats
exceeds the gap between them. Measured over 3 offline and 6 online runs.

Prompt prefill lengthens the DiT's flow-matching sequence by five slots per
prompt patch, and the sampler rebuilds that whole sequence on every Euler
step. It does not show up in wall clock at these sizes: with the decode-step
count pinned at 40, three runs per arm measured 6.85-7.03 s zero-shot,
6.83-6.96 s at 21 prompt patches (3.5 s reference), 6.87-7.02 s at 44, and
6.95-7.28 s at 66 (10.7 s reference) — the per-step cost is dominated by the
eager sampler's fixed work, not by sequence length.

The x-vector and the reference latent distribution are cached process-wide
by reference-audio identity, so a repeated voice re-runs neither encoder.
For a 3.7 s reference that is **63 ms** of engine-blocking work per request
(CAM++ 53 ms + AudioVAE 10 ms) elided on a cache hit — blocking for every
request in the step, not just the one that supplied the reference.

**T4 — concurrency**: 9 overlapping `/v1/audio/speech` requests mixing all
three conditioning modes across two different references completed with no
server errors, confirming per-request isolation of the prompt-prefill state.

#### Notes

- Output: 48 kHz mono WAV.
- Checkpoints: `dots-studio/dots.tts-soar` is the validated default
  used throughout this recipe. `rednote-hilab/dots.tts-base` shares the
  same architecture but is unvalidated in this repo. `rednote-hilab/dots.tts-mf`
  (MeanFlow, 2-4 step) is not supported — see below.
- `enforce_eager: true` and `enable_prefix_caching: false` in the deploy
  config are load-bearing, not just conservative defaults: with prefix
  caching enabled, vLLM-Omni's prefix-cache multimodal-output merge path
  does not preserve the `sparse_audio` marker this model relies on to
  route audio output correctly, and generation silently truncates to a
  single ~160 ms patch. Do not override `enable_prefix_caching` for this
  model until that framework-level gap is fixed.

## Known limitations

- **Precomputed speaker embeddings are not supported.** Conditioning goes
  through reference audio (`ref_audio`, optionally with `ref_text`); the
  Qwen3-TTS-style `speaker_embedding` / `x_vector_only_mode` fields are
  rejected.
- **Reference audio is capped at 30 s** by the serving adapter. Prompt
  prefill costs one prompt token per 160 ms of reference audio and shares
  the talker's 1024-patch FM workspace with the generated audio, and the
  CAM++ extractor crops to 10 s regardless.
- **Reference encoding runs on the engine's critical path.** The CAM++ and
  AudioVAE encoders run inside `preprocess()`, so a cache-missing reference
  stalls the whole engine step. The cross-request cache makes this a
  once-per-voice cost; a burst of distinct references still pays it per
  request.
- **`dots.tts-mf` (MeanFlow, 2-4 step) checkpoint is not supported.** Only
  the fixed 10-step Euler DiT sampler used by `dots.tts-soar` /
  `dots.tts-base` is implemented.
- **No CUDA graph capture.** The talker runs fully eager. voxcpm2's three
  captured graphs (base LM decode, CFM solver, VAE decode) have no
  dots.tts equivalent yet.
- **Concurrent requests do not scale.** Each request's 10-step DiT Euler
  integration runs serially in the side path (no cross-request batching,
  unlike voxcpm2's `enable_batched_cfm`). A community review of this
  integration measured no throughput gain at `c=4` concurrent requests
  versus `c=1`.
- **`SamplingParams.seed` does not control audio-generation randomness.**
  The DiT's flow-matching noise is deterministically derived per-request
  from a fixed internal seed (reproducible run-to-run, matching
  voxcpm2's `deterministic_cfm_seed` convention) rather than from the
  caller-supplied `seed` field — the same limitation voxcpm2 has today.
