# IndexTTS2 TTS Support

## Status

Draft design for supporting IndexTTS2 in vLLM-Omni.

Related request: <https://github.com/vllm-project/vllm-omni/issues/3413>

Upstream reference: <https://github.com/index-tts/index-tts>

## Goals

- Add IndexTTS2 as a text-to-speech model available through vLLM-Omni offline
  inference and the OpenAI-compatible `/v1/audio/speech` endpoint.
- Preserve the upstream zero-shot voice-cloning behavior with speaker reference
  audio.
- Preserve the upstream emotion-control inputs where practical:
  - speaker audio prompt
  - optional emotion audio prompt
  - optional explicit emotion vector
  - optional emotion text prompt
  - `emo_alpha`
- Provide a safe MVP path that can be validated before doing deeper vLLM-native
  optimization.
- Define the path from MVP to a production-grade multi-stage implementation.

## Non-Goals

- Do not support IndexTTS1 in the first implementation.
- Do not promise token-level low-latency streaming in the MVP. Upstream
  `infer_generator()` streams by completed text segment, not by codec frame.
- Do not vendor large upstream source trees or model weights into this
  repository.
- Do not bypass license review. The upstream package declares a custom
  `LicenseRef-Bilibili-IndexTTS` license, so redistribution and vendoring must
  be checked before merging implementation code.
- Do not add DeepSpeed as a required runtime dependency.

## Upstream Architecture Summary

IndexTTS2's current inference path is implemented around `IndexTTS2.infer()` and
`IndexTTS2.infer_generator()` in upstream `indextts/infer_v2.py`.

The high-level data flow is:

```text
text + speaker prompt + emotion prompt
  -> text normalization + BPE tokenizer
  -> w2v-bert semantic feature extraction for prompt audio
  -> semantic codec quantization for prompt condition
  -> CAMPPlus speaker/style embedding
  -> UnifiedVoice GPT autoregressive mel-code generation
  -> GPT latent refinement
  -> semantic-code embedding
  -> s2mel / CFM mel generation
  -> BigVGAN waveform synthesis
  -> 22.05 kHz audio
```

Important upstream configuration values from `checkpoints/config.yaml`:

| Field | Value |
|---|---:|
| `gpt.number_text_tokens` | 12000 |
| `gpt.number_mel_codes` | 8194 |
| `gpt.start_mel_token` | 8192 |
| `gpt.stop_mel_token` | 8193 |
| `gpt.max_mel_tokens` | 1815 |
| `dataset.sample_rate` | 24000 |
| `s2mel.preprocess_params.sr` | 22050 |
| `s2mel.preprocess_params.spect_params.hop_length` | 256 |
| `vocoder.name` | `nvidia/bigvgan_v2_22khz_80band_256x` |

The serving sample rate should be treated as 22050 Hz because the final
waveform is produced by the s2mel/BigVGAN branch configured at 22.05 kHz.

## Design Overview

Use a two-step delivery plan:

1. MVP: single-stage AR wrapper around the upstream IndexTTS2 runtime.
2. Production path: split the model into an AR code-generation stage and a
   code-to-waveform generation stage.

The MVP is intentionally conservative. It gives users a working model and gives
the project concrete correctness data before committing to a deep rewrite of the
GPT2-style AR module into vLLM-native decoder layers.

The production path is where the model can start benefiting from vLLM-Omni's
multi-stage execution, async scheduling, batching, and eventually chunked
streaming.

## MVP Architecture: Single-Stage Wrapper

### Pipeline

```text
Stage 0: indextts2
  execution_type: LLM_AR
  final_output: true
  final_output_type: audio
  engine_output_type: audio
  async_chunk: false
```

This mirrors the single-stage generator pattern used by MOSS-TTS-Nano:

- load the upstream IndexTTS2 runtime lazily in `load_weights()`
- create one per-request generator from `infer_generator()`
- return one audio delta per `forward()` call
- force EOS in `compute_logits()` after the final chunk
- keep all cross-step state keyed by `_omni_req_id`

### Request Contract

The MVP should support these OpenAI speech request fields:

| API field | IndexTTS2 field |
|---|---|
| `input` | `text` |
| `ref_audio` | `spk_audio_prompt` |
| `instructions` or extra arg `emo_text` | `emo_text` when `use_emo_text=true` |
| extra arg `emo_audio` | `emo_audio_prompt` |
| extra arg `emo_alpha` | `emo_alpha` |
| extra arg `emo_vector` | `emo_vector` |
| extra arg `max_text_tokens_per_segment` | `max_text_tokens_per_segment` |
| sampling params | `top_p`, `top_k`, `temperature`, `num_beams`, `repetition_penalty`, `max_mel_tokens` |

For the first online integration, `ref_audio` should be required. Upstream
IndexTTS2 is zero-shot TTS and the reference path is central to voice quality.

`ref_audio` must support:

- local path in offline examples
- data URL in online tests
- uploaded temporary file after the existing serving layer decodes a user
  supplied data URL

### Per-Request State

Upstream `IndexTTS2` caches prompt-derived state on the model object:

- `cache_spk_cond`
- `cache_s2mel_style`
- `cache_s2mel_prompt`
- `cache_spk_audio_prompt`
- `cache_emo_cond`
- `cache_emo_audio_prompt`
- `cache_mel`

This cannot remain a single shared model-level cache in vLLM-Omni. The wrapper
must either:

- disable those caches, or
- move them into a bounded cache keyed by stable prompt identity and protected
  by request-safe access, or
- store active streaming generator state by `_omni_req_id`.

The MVP should start with per-request generator state and no mutable shared
prompt cache. A later optimization can add a content-addressed prompt cache.

### Output Semantics

The wrapper must document its `forward()` output contract as delta audio:

```text
Each forward call returns only the newly produced audio samples for this
request. Offline output is reconstructed by concatenating audio chunks.
```

If the upstream generator yields a full text-segment waveform, the wrapper can
emit that entire segment as one delta. It must not re-emit previously returned
samples.

### Dependency Handling

Optional or fragile dependencies should be imported in `load_weights()`, not at
module import time:

- `indextts`
- `modelscope`
- `torchaudio`
- `librosa`
- `omegaconf`
- `safetensors`
- `sentencepiece`
- BigVGAN custom CUDA kernel

If BigVGAN fused CUDA kernels are unavailable, the implementation should fall
back to the torch path, matching upstream behavior.

The implementation should avoid requiring DeepSpeed. If users request it and the
package is unavailable, log a warning and continue without DeepSpeed.

## Production Architecture: Two-Stage Pipeline

The production path should split IndexTTS2 into AR code generation and waveform
generation:

```text
Stage 0: indextts2_gpt
  text + prompt conditions -> mel codes + GPT latent

Stage 1: indextts2_code2wav
  mel codes + prompt condition + style + emotion condition -> waveform
```

### Stage 0 Responsibilities

- Load the GPT/UnifiedVoice AR model.
- Build speaker and emotion conditioning inputs.
- Tokenize and segment text.
- Generate mel codes with stop token `8193`.
- Emit:
  - generated mel codes
  - code lengths
  - speech conditioning latent required by the second GPT forward path
  - prompt condition
  - reference mel
  - style embedding
  - emotion vector/condition metadata

### Stage 1 Responsibilities

- Load semantic codec, s2mel/CFM, CAMPPlus-dependent assets if not already
  produced by Stage 0, and BigVGAN.
- Convert mel codes to semantic embeddings.
- Apply the GPT latent branch and length regulator.
- Run CFM inference.
- Run BigVGAN.
- Return 22.05 kHz mono waveform chunks.

### Why Not Start With Async Chunk

IndexTTS2's upstream streaming boundary is text-segment based, while
vLLM-Omni's low-latency TTS async_chunk design expects codec-frame chunks with
left context. Before enabling `async_chunk: true`, the implementation needs a
validated chunking strategy for:

- mel-code accumulation
- CFM context windows
- BigVGAN boundary handling
- silence insertion between text segments
- audio delta trimming

Therefore, the first two-stage version should run with `async_chunk: false`.
After full-sequence correctness is stable, add chunked streaming as a separate
phase.

## File Plan

MVP files:

```text
vllm_omni/model_executor/models/indextts2/
  __init__.py
  configuration_indextts2.py
  modeling_indextts2.py
  pipeline.py

vllm_omni/deploy/indextts2.yaml
examples/offline_inference/text_to_speech/indextts2/end2end.py
examples/online_serving/text_to_speech/indextts2/run_server.sh
examples/online_serving/text_to_speech/indextts2/speech_client.py
```

Registry and serving updates:

```text
vllm_omni/model_executor/models/registry.py
vllm_omni/config/pipeline_registry.py
vllm_omni/entrypoints/openai/serving_speech.py
examples/offline_inference/text_to_speech/README.md
examples/online_serving/text_to_speech/README.md
docs/models/supported_models.md
```

Production split adds:

```text
vllm_omni/model_executor/models/indextts2/indextts2_gpt.py
vllm_omni/model_executor/models/indextts2/indextts2_code2wav.py
vllm_omni/model_executor/stage_input_processors/indextts2.py
```

## Configuration Sketch

MVP deploy config:

```yaml
async_chunk: false
trust_remote_code: true
dtype: float16

stages:
  - stage_id: 0
    max_num_seqs: 4
    gpu_memory_utilization: 0.8
    enforce_eager: true
    enable_prefix_caching: false
    max_num_batched_tokens: 4096
    max_model_len: 4096
    devices: "0"
    skip_mm_profiling: true
    default_sampling_params:
      temperature: 0.8
      top_p: 0.8
      top_k: 30
      max_tokens: 4096
      repetition_penalty: 10.0
```

Notes:

- `enforce_eager: true` is expected for the MVP because the wrapper calls
  upstream code with dynamic control flow and non-vLLM modules.
- `max_num_seqs: 4` should be used for concurrency tests, but real concurrency
  may be limited by upstream generator cost and memory footprint.
- The sample rate should be surfaced as `sr=22050` in multimodal output.

## Implementation Plan

### Phase 0: License and Reference Baseline

1. Confirm whether the upstream license permits the intended integration.
2. Run upstream IndexTTS2 with official weights.
3. Save a small set of reference outputs:
   - Chinese text with speaker reference
   - English text with speaker reference
   - speaker reference plus emotion audio
   - speaker reference plus explicit emotion vector
4. Record waveform properties:
   - sample rate
   - channel count
   - duration
   - dtype/range
   - RTF on target GPU

### Phase 1: MVP Single-Stage Model

1. Add config and pipeline registration.
2. Implement lazy upstream runtime loading in `load_weights()`.
3. Implement request parameter extraction from `additional_information`.
4. Implement per-request generator state keyed by `_omni_req_id`.
5. Implement delta waveform output and EOS signaling.
6. Add offline `end2end.py`.
7. Add online speech serving integration.

### Phase 2: Hardening

1. Replace shared upstream caches with request-safe state.
2. Add data-URL reference audio handling in serving tests.
3. Add defensive tensor/list multimodal output handling in examples.
4. Validate concurrent requests.
5. Add clear error messages for missing optional dependencies.

### Phase 3: Two-Stage Split

1. Extract Stage 0 AR code generation.
2. Extract Stage 1 code-to-waveform generation.
3. Define stage transition payload schema.
4. Verify full-sequence output equivalence against MVP and upstream.
5. Tune memory split across stages.

### Phase 4: Chunked Streaming Research

1. Determine whether CFM and BigVGAN can decode stable partial windows.
2. Define codec chunk size and left-context policy.
3. Implement `async_chunk_process_next_stage_input_func`.
4. Verify streaming audio boundaries by listening tests and waveform checks.
5. Enable `async_chunk: true` only after the chunked output is stable.

## Unit Test Plan

### Config and Registry Tests

Add tests that verify:

- `AutoConfig` can load `model_type="indextts2"`.
- `OmniModelRegistry` resolves the configured architecture.
- `pipeline_registry` returns the `indextts2` pipeline.
- deploy config loads with exactly one stage for MVP.

Suggested file:

```text
tests/model_executor/models/indextts2/test_indextts2_config.py
```

### Request Parameter Tests

Test the serving parameter builder without loading model weights:

- empty `input` is rejected
- missing `ref_audio` is rejected for MVP
- `emo_alpha` is range-checked
- `emo_vector` must have exactly 8 numeric values
- `max_text_tokens_per_segment` must be positive
- unsupported response/stream combinations return clear errors

Suggested file:

```text
tests/entrypoints/openai/test_indextts2_speech_params.py
```

### Per-Request State Tests

Use a fake upstream generator to avoid loading real weights. Verify:

- two request IDs get independent generators
- chunks from request A are never returned to request B
- generator state is removed after final chunk
- abort/exception cleanup removes request state

Suggested file:

```text
tests/model_executor/models/indextts2/test_indextts2_state.py
```

### Audio Output Contract Tests

Use fake generator chunks:

- `forward()` returns `multimodal_output["audio"]`
- returned audio is a 1-D float32 tensor
- `multimodal_output["sr"] == 22050`
- repeated `forward()` calls emit deltas, not cumulative audio
- offline consolidation concatenates all chunks in order

Suggested file:

```text
tests/model_executor/models/indextts2/test_indextts2_output_contract.py
```

### Optional Dependency Tests

Mock missing optional packages and verify:

- import of the model module does not fail
- missing dependencies fail in `load_weights()` with actionable errors
- missing BigVGAN CUDA extension falls back when the torch path is available

Suggested file:

```text
tests/model_executor/models/indextts2/test_indextts2_optional_deps.py
```

### Stage Input Processor Tests

For the two-stage phase, test payload conversion without full model execution:

- generated code tensor layout is preserved
- code lengths match stop-token trimming
- prompt condition, style, reference mel, and emotion condition are carried
- batched payloads are split back into per-request outputs

Suggested file:

```text
tests/model_executor/stage_input_processors/test_indextts2.py
```

## E2E Test Plan

### Offline E2E

Add a hardware-gated test that runs:

```bash
python examples/offline_inference/text_to_speech/indextts2/end2end.py \
  --model /path/to/IndexTTS-2 \
  --text "Hello, this is an IndexTTS2 test." \
  --ref-audio tests/assets/indextts2/ref_voice.wav \
  --output /tmp/indextts2.wav
```

Assertions:

- output file exists
- sample rate is 22050 Hz
- channel count is mono
- duration is within a broad expected range for the input length
- waveform is not all zeros
- waveform has no NaN or Inf
- RTF is recorded in logs, but not used as a strict pass/fail threshold for the
  initial test

Suggested file:

```text
tests/e2e/offline_inference/test_indextts2.py
```

### Online Non-Streaming E2E

Launch:

```bash
vllm serve /path/to/IndexTTS-2 --omni --port 8091
```

Send:

```json
{
  "input": "Hello, this is an IndexTTS2 online serving test.",
  "voice": "default",
  "ref_audio": "data:audio/wav;base64,...",
  "response_format": "wav"
}
```

Assertions:

- HTTP 200
- response decodes as WAV
- sample rate is 22050 Hz
- audio duration is positive and plausible
- output is not silent

Suggested file:

```text
tests/e2e/online_serving/test_indextts2_speech.py
```

### Online Streaming MVP E2E

For MVP, streaming may be segment-level. If enabled, require PCM only:

```json
{
  "input": "Hello. This has two short segments.",
  "voice": "default",
  "ref_audio": "data:audio/wav;base64,...",
  "response_format": "pcm",
  "stream": true
}
```

Assertions:

- response starts before the full request timeout
- at least one PCM chunk is received
- concatenated PCM decodes to non-silent audio at 22050 Hz
- chunks are deltas, not cumulative replays

If the MVP disables online streaming, add an E2E test that verifies
`stream=true` returns a clear validation error.

### Concurrent E2E

Run four concurrent requests with different text prompts and at least two
different reference audios.

Assertions:

- all requests complete successfully
- output durations differ according to prompt length
- no output is byte-identical across different prompts
- no request returns another request's reference-conditioned cached result
- no per-request generator state remains after completion

### Quality Regression E2E

For nightly or manual CI only:

- compare generated duration against upstream reference for the same prompt
- compute RMS energy and silence ratio
- optionally run ASR-based text similarity if an existing CI ASR helper is
  available
- keep subjective listening as a release checklist item, not an automated gate

## CI Strategy

- Unit tests should not download real IndexTTS2 weights.
- E2E tests should be hardware-gated and require a pre-downloaded model path or
  CI cache.
- Online tests should use data URLs for reference audio so the server does not
  fetch external URLs during CI.
- Use one `OmniServerParams` configuration per E2E test file to avoid
  module-scoped server restart races.
- Mark core GPU coverage with the same conventions used by other TTS models.

## Acceptance Criteria

MVP acceptance:

- Offline inference produces valid 22.05 kHz mono audio.
- `/v1/audio/speech` produces valid WAV for text plus reference audio.
- Concurrent requests do not share generator or prompt state incorrectly.
- Missing optional dependencies fail with actionable messages.
- Unit tests cover config, serving validation, state isolation, and output
  contract.
- E2E tests cover offline and online non-streaming paths.

Production acceptance:

- Two-stage full-sequence output matches MVP/upstream within expected stochastic
  variation.
- Stage payload schema is covered by unit tests.
- Multi-request batching does not corrupt outputs.
- Streaming is enabled only after chunk-boundary quality is verified.

## Risks and Open Questions

- License compatibility may block vendoring or direct dependency use.
- Upstream depends on specific torch/transformers versions that may conflict
  with vLLM-Omni's runtime.
- `modelscope` and Qwen emotion text classification add extra dependency and
  download surfaces.
- Upstream prompt caches are not concurrency-safe as-is.
- The final waveform sample rate differs from the dataset sample rate in config;
  serving must use 22050 Hz.
- Token-level async streaming may require non-trivial changes to CFM and BigVGAN
  invocation.
- Rewriting UnifiedVoice's GPT2-style AR path into vLLM-native layers may be
  substantial and should be treated as a separate optimization project.
