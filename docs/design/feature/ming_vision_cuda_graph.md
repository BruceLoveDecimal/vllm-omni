# Ming vision encoder CUDA graphs

This work integrates Ming-flash-omni's thinker with the upstream encoder graph
manager. Audio, talker and diffusion graphs are separate features.

## Upstream contract

The implementation targets the `SupportsEncoderCudaGraph` protocol with
`EncoderItemSpec`, `select_encoder_cudagraph_items`, and capture input `values`.
The source reference is vLLM commit
`7b942936276d59cc1912185a442b97dea8d2386c`. This is an API reference, not a claim
of GPU validation. The model adapter was GPU-tested with official vLLM 0.27
and 0.28 wheels. The pre-rebase Omni engine baseline required 0.27 for the full
Thinker test: 0.28 changes the engine startup signature, and 0.29 removes the
legacy input preprocessor. The current upstream baseline includes later engine migrations.

References:

- [vLLM tracker](https://github.com/vllm-project/vllm/issues/38175)
- [Omni integration](https://github.com/vllm-project/vllm-omni/issues/7452)

## Change 1: separate metadata from vision computation

`MingVisionEncoder` retains the original upstream eager forward. Its optional
`encoder_metadata` path reuses the same patch embedding, blocks and mergers,
with no new parameters or changes to checkpoint names. This small execution
adapter can be removed when Qwen3-Omni supports precomputed metadata upstream.

Both image and video grids stay on CPU via `MultiModalFieldConfig.keep_on_cpu`.
Position interpolation and attention metadata preparation happen before graph
capture/replay. The CPU `max_seqlen` scalar supplies the capture-time upper
bound; attention does not need to read a GPU scalar. Graph padding uses a separate attention sequence followed by repeated terminal
offsets for empty sequences, as described below.

### GPU validation

On a CUDA host with a compatible vLLM installation and Omni test dependencies:

```bash
python -m pytest tests/model_executor/models/ming_flash_omni/test_vision_metadata.py -v
```

No model download is required. The tests instantiate a small real BF16 ViT
with FlashAttention and compare its original eager path against the metadata
path. They cover images, differing aspect ratios, multiple items, video,
DeepStack, sequence padding, invalid grids, and actual CUDA capture/replay.
GPU grids are rejected explicitly rather than silently copied to CPU.

Pass criteria: no skipped tests on the CUDA host, no capture error, and all
embedding comparisons pass (`atol=2e-3`, `rtol=2e-2`). Full-model accuracy and
performance remain separate validation steps; these tolerances are for the
small BF16 regression model only.

## Change 2: image encoder protocol

The thinker and its registered stage wrapper implement the same model-side
protocol through `MingVisionCudaGraphMixin`. vLLM owns the manager, graph pool,
budget packing, output scattering/cloning and encoder cache. No model-specific
runner branch is needed. Capture includes the ViT, Ming's main-feature slice,
vision projector and normalization. The output width is the language model
hidden size, not the ViT width.

Budgets count merged output tokens; a merge size of two requires four input
patches per output token. Capture buffers are allocated directly from this
budget without constructing an arbitrarily wide dummy image. The replay path
keeps the captured CPU `max_seqlen` upper bound unchanged.

The initial backend is CUDA FlashAttention with encoder TP mode `weights`.
Encoder data parallelism is rejected because the wrapped Qwen3-Omni MLP still
uses tensor-parallel linear layers. Other attention backends need their own
metadata-layout validation before being enabled.

### GPU validation for image capture

```bash
python -m pytest tests/model_executor/models/ming_flash_omni/test_encoder_cudagraph.py -v
```

This exercises the real upstream manager and the production wrapper/thinker
methods with a small real ViT and projector. It checks graph hits, greedy
packing, original item order, aspect-ratio changes, eager fallback for large
items, stable cached outputs after later replays, and a fixed capture bound.
Run change 1's tests as well. All tests must pass without skips on the GPU host.

Padding patches form a separate attention sequence. The final cumulative offset
covers the whole capture buffer, ensuring that attention writes every output
row; leaving the tail uncovered can produce NaNs with real checkpoint weights.
The padding callback computes this terminal offset from buffer shape and merge
size, without reading a GPU value on the host.

## Change 3: video, deployment and Thinker parity

Images and videos share the same captured vision computation. Item selection
uses temporal grid sizes to preserve frame boundaries. Each temporal grid
contributes at least one merged output token, so a graph reserves
`token_budget + 2` cumulative offsets: real sequences, one padding sequence,
and the initial offset. This conservative allocation supports many tiny frames
without relying on newer upstream capture-axis APIs. It may reserve more
sequence metadata than a typical request needs. Token budgets still bound
patch buffers, and oversized items use the upstream eager fallback.

`vllm_omni/deploy/ming_flash_omni_thinker_only.yaml` enables the feature for
the existing four-GPU Thinker topology. The YAML selects FlashAttention, weight
tensor parallelism and the encoder graph switch. Ming supplies a model-specific
automatic range of 512--2048 merged vision tokens; the upstream manager derives
the `[512, 1024, 2048]` capture budgets and a maximum of four packed items.
Deployments can still override the upstream budget fields when workload-specific
tuning is needed.
Keep decoder CUDA graphs enabled: this upstream runner skips encoder capture
when its overall graph mode is `NONE`.

### Reproducible validation

```bash
# CUDA regression tests; checkpoint tests skip unless MING_CHECKPOINT is set.
python -m pytest tests/model_executor/models/ming_flash_omni -v

# Real vision tower and projector weights, without loading the language model.
export MING_CHECKPOINT=/path/to/Ming-flash-omni-2.0
python -m pytest tests/model_executor/models/ming_flash_omni/test_vision_checkpoint_parity.py -v

# Full Thinker: text, image, video and image+audio, eight greedy output tokens.
# Optional: override deployment for the available hardware.
export MING_CG_DEPLOY_CONFIG=/path/to/deployment.yaml
python -m pytest tests/e2e/offline_inference/test_ming_vision_cudagraph.py -v -s
```

The full Thinker test compares exact token IDs with encoder graphs disabled
and enabled while preserving decoder settings. A worker extension checks actual
capture and graph-hit counters, so silent eager fallback cannot satisfy it.
Audio retains its existing eager encoder path.

### Recorded GPU results

The results below were collected before rebasing onto upstream commit
`9fbc36ab2`. The three feature commits were subsequently rebased onto that
`origin/main` revision, retaining upstream serving imports and processor changes.
GPU parity has not been rerun on the rebased baseline; the recorded results
should not be interpreted as validation of those newer upstream changes.

Validation used one NVIDIA H20 96 GB, Torch 2.13.0+cu130, FlashAttention,
and checkpoint revision `acdd193bcadf5fbf58b24af4e0c665c1e7d7a37c` of
`Jonathan1909/Ming-flash-omni-2.0`.

| Environment | Result |
| --- | --- |
| vLLM 0.27, model/manager/Thinker routing tests | 33 passed, no skips |
| vLLM 0.28, model/manager tests before adding two routing cases | 31 passed, no skips |
| Real checkpoint vision/projector parity (five cases, both versions) | Maximum absolute error 0.0 |

Real-weight cases cover square and rectangular images, a padded multi-image
batch, video, and an oversized eager fallback. They also check finite outputs.
The GPU run caught an uncovered padding sequence that produced NaNs with real
weights; the dedicated padding sequence fixes this and is regression-tested.
These single-GPU results do not establish a throughput gain.

### Full Thinker generation parity (TP=2)

After expanding the instance to two H20 96 GB GPUs and a 300 GiB host-memory
limit, the full Thinker test passed with official vLLM 0.27.0. It loads the
complete BF16 checkpoint, including the language, vision and audio encoders;
no model layers or audio outputs are mocked in this end-to-end test.

The test deployment uses TP=2, `cpu_offload_gb: 24` per rank,
`gpu_memory_utilization: 0.95`, `max_model_len: 4096`,
`max_num_batched_tokens: 1024`, and `max_num_seqs: 1`. Both runs use compilation
mode 0, decoder graph mode `FULL_DECODE_ONLY`, decode capture size `[1]`, and
seed 17. Encoder budgets are `[64, 256, 512, 1024]`, with at most two vision
items per batch. Only `cudagraph_mm_encoder` changes between the two runs.

| Input | Generated tokens | Encoder eager vs graph |
| --- | --- | --- |
| Text | 2, including EOS | Exact token-ID match |
| Image | 2, including EOS | Exact token-ID match |
| Video | 8 | Exact token-ID match |
| Image + audio | 8 | Exact token-ID match |

Each rank captured four encoder graphs and recorded two graph hits with zero
misses. The repeated image can reuse the existing encoder cache. The worker
extension reads `get_cumulative_stats()` to verify capture and replay; token
IDs and rank statistics are included in the JUnit properties. The final run
passed in 285.71 seconds with warmed checkpoint file caches. First cold loading
was much slower, so this duration is not an inference performance benchmark.

This is a complete-model generation smoke parity test with a maximum of eight
greedy tokens per request. It does not establish long-generation accuracy,
throughput gains, or TP=4 parity. The earlier single-H20/150-GiB attempt stopped
at the host-memory limit; the TP=2 result supersedes that incomplete validation.
