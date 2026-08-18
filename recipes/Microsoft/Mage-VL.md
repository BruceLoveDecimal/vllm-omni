# Mage-VL

> Online serving and offline inference for image and frame-sampled video understanding
> (text / image / video → text)

## Summary

- Vendor: Microsoft
- Model: [`microsoft/Mage-VL`](https://huggingface.co/microsoft/Mage-VL)
- Task: Multimodal understanding — accepts text and images (video enters as sampled
  frames through the image tensors); emits text
- Mode: Online serving via the OpenAI-compatible `/v1/chat/completions` API, and offline
  inference via `LLM.generate`
- Architecture: Mage-ViT vision tower (~0.3B, codec-native) + a stock Qwen3-4B decoder.
  The decoder is instantiated through `init_vllm_registered_model` as `Qwen3ForCausalLM`;
  positions are plain 1-D, since the checkpoint's `text_config` carries no `mrope_section`.

## When to use this recipe

Use this as a known-good starting point for serving `microsoft/Mage-VL` on vLLM-Omni.
The port covers the checkpoint's **online path**: still images, and video that the caller
has already sampled into frames (which is what the reference implementation itself serves
online). The codec-native long-video path and the StreamMind cognition gate ship in the
checkpoint but are not wired up here yet — see [Limitations](#limitations).

## References

- Default deploy config (auto-loaded by HF `model_type=mage_vl`):
  [`vllm_omni/deploy/mage_vl.yaml`](../../vllm_omni/deploy/mage_vl.yaml)
- Model source:
  [`vllm_omni/model_executor/models/mage_vl/`](../../vllm_omni/model_executor/models/mage_vl/)
- Vendored config / processor:
  [`vllm_omni/transformers_utils/configs/mage_vl.py`](../../vllm_omni/transformers_utils/configs/mage_vl.py),
  [`vllm_omni/transformers_utils/processors/mage_vl.py`](../../vllm_omni/transformers_utils/processors/mage_vl.py)
- Parity drivers against the HuggingFace reference (GPU, real checkpoint):
  [`tests/model_executor/models/mage_vl/parity/`](../../tests/model_executor/models/mage_vl/parity/)
- Online e2e test:
  [`tests/e2e/online_serving/test_mage_vl.py`](../../tests/e2e/online_serving/test_mage_vl.py)
- Design spec:
  [`docs/design/feature/mage_vl_integration.md`](../../docs/design/feature/mage_vl_integration.md)
- Upstream model card: [`microsoft/Mage-VL`](https://huggingface.co/microsoft/Mage-VL)

## Hardware Support

| Layout | GPUs | Notes |
| --- | --- | --- |
| Single GPU (validated) | 1x RTX 5090 32GB | bf16 weights ≈8.83 GiB; validated at `--gpu-memory-utilization 0.85`, `max_model_len 32768`, `image: 8` |
| Single GPU (minimum) | 1x 24GB | Fits the weights plus a working KV cache at a smaller `max_model_len` |

Tensor parallelism beyond `tp=1` is **not validated** — see [Limitations](#limitations).

## Quickstart

### Online serving

Through the deploy config (recommended — this is the path
[`vllm_omni/deploy/mage_vl.yaml`](../../vllm_omni/deploy/mage_vl.yaml) describes):

```bash
vllm-omni serve microsoft/Mage-VL --omni --deploy-config vllm_omni/deploy/mage_vl.yaml --port 8077
```

Or as a plain vLLM server, when you want to override engine args directly:

```bash
vllm serve microsoft/Mage-VL --dtype bfloat16 --max-model-len 8192 --limit-mm-per-prompt '{"image":8}' --port 8077
```

`--trust-remote-code` is **not required**: the config and processor are vendored in
vllm-omni, and the checkpoint's own remote code is never imported.

Then send an image:

```bash
curl -s http://127.0.0.1:8077/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "microsoft/Mage-VL", "temperature": 0, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Stop_sign.jpg/240px-Stop_sign.jpg"}}, {"type": "text", "text": "What is in this image?"}]}]}'
```

Video is sent as its sampled frames — one `image_url` part per frame, in order.

### Offline inference

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="microsoft/Mage-VL", dtype="bfloat16", max_model_len=8192,
          limit_mm_per_prompt={"image": 4})
out = llm.generate(
    {
        "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is in this image?<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"image": Image.open("frame.jpg").convert("RGB")},
    },
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(out[0].outputs[0].text)
```

## Accuracy vs the reference implementation

The issue asks for SSIM/PSNR. Those do not apply — Mage-VL emits text, not images. The
criterion used instead is **token-exact generation**, which is strictly stronger than a
perceptual metric, backed by operator-level `rel_l2` in fp32.

Measured on 1x RTX 5090 against `AutoModelForCausalLM` + the checkpoint's own processor
(`transformers` 5.8.1, bf16 weights, greedy):

| Check | Result |
| --- | --- |
| Processor outputs (`input_ids`, `pixel_values`, `image_grid_thw`, `patch_positions`) | 3/3 cases elementwise identical |
| Vision tower, fp32 (same bf16 weights upcast on both sides) | `rel_l2` **4.1865e-06**, cosine **1.0000000000** |
| Vision tower, bf16 end to end | cosine 0.99980, `rel_l2` 2.0062e-02 — accumulated rounding over 24 layers, not a defect; the fp32 figure above is what rules a bug out |
| Weight loading | 297/297 vision tower parameters, none missing |
| Offline generation | **3/3 token-identical** |
| Online `/v1/chat/completions` | **3/3 token-identical**, plus a 24-chunk streaming response |
| Frame-sampled video (4 frames as images) | token-identical; 4 concurrent requests returned one distinct output |
| Codec-native video preprocessing (720-frame clip) | canvases, `image_grid_thw`, `pixel_values`, `patch_positions` and the rewritten prompt all **elementwise identical** to the reference; prompt token ids identical (6518 tokens, 4608 visual) |

Reproduce with the drivers in
[`tests/model_executor/models/mage_vl/parity/`](../../tests/model_executor/models/mage_vl/parity/)
(`run_hf_baseline.py` first, then the rest — its README explains the order).

## Performance

Measured on 1x RTX 5090 32GB with the deploy config above (bf16, one
image, `max_tokens=64`, greedy, streaming). Numbers are a smoke baseline for spotting
regressions, not a tuned benchmark:

| Scenario | TTFT (median) | Decode throughput |
| --- | --- | --- |
| Single stream | 76 ms (52-86 ms over 5 runs) | 161.7 tok/s |
| 4 concurrent streams | 124 ms | 422.3 tok/s aggregate |

## Tuning

### Pixel budget

The checkpoint's `preprocessor_config.json` allows `max_pixels = 4,000,000`, so vLLM
profiles the encoder with one image of that maximum size. Measured at
`--gpu-memory-utilization 0.85 --max-model-len 32768 --limit-mm-per-prompt '{"image":8}'`
on a 32GB card:

| `max_pixels` | Encoder cache budget | KV cache |
| --- | --- | --- |
| checkpoint default (4,000,000) | 3906 tokens | 16.67 GiB / 121,408 tokens |
| `150000` (the reference pipeline's own codec budget) | 2048 tokens | 16.97 GiB / 123,552 tokens |

The default costs ~0.3 GiB of KV cache — profiling uses a *single* max-size image, not
`limit_mm_per_prompt` copies of one — so the default is left as the checkpoint ships it.
Lower it when you want more KV cache and your inputs are small:

```bash
vllm serve microsoft/Mage-VL --mm-processor-kwargs '{"max_pixels": 150000}'
```

### Codec-native video

Long video can go through the checkpoint's codec sampler instead of uniform frames: it
groups frames by how much new information each carries (read off the codec's own bit-cost
map) and tiles the selected patches onto a few square canvases, so a long clip costs far
fewer visual tokens than one image per frame would. Measured on the checkpoint's own
720-frame `examples/soccer-broadcast.mp4`: 256 sampled frames become **32 canvases →
4608 visual tokens**, and the codec pass itself takes ~1.9 s.

The engine is an external tool. Install it alongside vllm-omni:

```bash
pip install codec-video-prep  # provides cv-preinfer; also needs ffmpeg/ffprobe on PATH
```

`codec-video-prep` pins `numpy<2.0`, which will fight a vLLM install. Keep it in its own
virtualenv and point `CV_PREINFER_BIN` at that venv's `cv-preinfer`, or pre-produce the
assets and pass the directory:

```python
processor(text=prompt, videos="clip.mp4", video_backend="codec")           # runs the engine
processor(text=prompt, video_backend="codec",
          codec_config={"asset_dir": "/path/to/assets"})                    # reuses a run
```

Results are cached per (video, sampling budget) under `ONLINE_CODEC_CACHE_DIR` (default
`$HF_HOME/online_codec`).

**Do not set `codec_config={"patch": ...}` by hand.** The checkpoint ships `codec.patch =
14` while its image processor uses `patch_size = 16`; with 14 the canvases tile as 36x20 =
720 patches per canvas and the processor reads 527, so the position table cannot line up
and preprocessing fails outright. vllm-omni derives the value from the image processor and
rejects a conflicting override.

`engine="dcvc-rt"` (the neural codec) is not driven from here: it needs the checkpoint's
bundled `neural_codec` package with compiled CUDA extensions. Asking for it raises
`NotImplementedError` naming what is missing.

## Limitations

Untested support is reported as unknown, not as supported:

- **`tp > 1` is not validated.** The vision tower is written with upstream's parallel
  linear layers so weights-TP should work, but no multi-GPU run has been made. The model
  leaves `supports_encoder_tp_data` at `False`, so vLLM downgrades encoder "data"
  parallelism to weights-TP on its own rather than running with incomplete state.
- **Codec video is preprocessing-verified, not served end to end.** Preprocessing matches
  the reference elementwise on a real video, and the `hevc` engine runs from an installed
  `cv-preinfer`. What is not wired yet is the engine path *through vLLM's multimodal
  plumbing* — the model declares image only, so codec inputs are built processor-side
  today rather than by passing a video to the server.
- **The StreamMind cognition gate is not wired up.** `streammind_gate.safetensors` is
  present in the checkpoint but proactive streaming is a later milestone.
- **Checkpoint revision:** validation ran against a local snapshot; pin an explicit
  `--revision` before treating these numbers as reproducible against the live repo.
