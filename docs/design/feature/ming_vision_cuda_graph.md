# Ming vision encoder CUDA graphs

This work integrates Ming-flash-omni's thinker with the upstream encoder graph
manager. Audio, talker and diffusion graphs are separate features.

## Upstream contract

The implementation targets the `SupportsEncoderCudaGraph` protocol with
`EncoderItemSpec`, `select_encoder_cudagraph_items`, and capture input `values`.
The source reference is vLLM commit
`7b942936276d59cc1912185a442b97dea8d2386c`. This is an API reference, not a claim
of GPU validation. Use that revision for the initial GPU checks and record the
actual vLLM commit with results. The repository's older default CUDA image is
not the validation environment for this feature.

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
bound; attention does not need to read a GPU scalar. Padding cumulative lengths
repeats the terminal offset, representing empty attention sequences.

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
