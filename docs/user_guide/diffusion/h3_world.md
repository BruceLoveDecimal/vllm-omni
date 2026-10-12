# H3-World

[H3-World](https://huggingface.co/DANNY621/H3-World) is an action-conditioned
rank-32 LoRA for the original MiniMax-H3 **FL2VA** checkpoint. Use the base
model as `model`, and `step-10000.safetensors` as the LoRA. The adapter alone
is not a complete pipeline.

This integration extends `MiniMaxH3Pipeline` without a new engine, deployment
topology, REST endpoint, or DiffSynth dependency. It reuses H3's text encoder,
first-frame preprocessing, tensor parallel projections, LoRA manager, packed
AV tokens, noise initialization, scheduler, codecs, and MP4 output.

## Run

Download the adapter with `huggingface_hub.hf_hub_download`:

```python
from huggingface_hub import hf_hub_download
hf_hub_download("DANNY621/H3-World", "step-10000.safetensors", local_dir="checkpoints/H3-World")
```

On a CUDA host with sufficient GPU memory for H3, its encoder and VAEs:

```bash
python examples/offline_inference/minimax_h3/h3_world.py \
  --checkpoint checkpoints/H3-World/step-10000.safetensors \
  --first-frame first_frame.png \
  --scene-prompt "A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage." \
  --action-preset forward --num-frames 124 --steps 50 --seed 2 \
  --tp-size 2 --out h3_world.mp4
```

The example center-crops the image to 832×480, generates 124 frames at 24 fps,
and uses the existing H3 MP4 path to preserve generated audio. The scene prompt
describes the static scene; actions are supplied separately. Presets match the
[official inference script](https://github.com/Danzer1xxxxChan/H3-World/blob/f0c7be2acbbde8256b2473f6e7b088f58272acae/code/abot/infer.py).

For programmatic use, select the original FL2VA base (`task_type="fl2va"`),
set `sampling_params.lora_request` to the downloaded adapter, and use
`sampling_params.quality="lossless"`, `guidance_scale=1`, and
`sampling_params.extra_args["h3_world_actions"]`:

```python
# Hold a combination throughout the clip.
{"keys": ["W", "J", "F"]}

# Or specify each video latent separately: 124 output frames have 37 latents.
{"frames": [["W"]] * 18 + [["D", "L"]] * 19}
```

Pass either structure to the example through `--actions-json actions.json`.
Keys are case-insensitive. W/S move forward/backward, A/D strafe left/right,
J/L pan left/right, **I tilts down and K tilts up**, and F makes panning fast.
Opposing pairs cancel; F alone has no effect. Frame entries refer to video
latents, whose pixel cadence is `(1,4,4,4,4)`, not individual output frames.
For `17k+5` output frames there are `5k+2` video latents, using H3's existing
shape planner. The current H3 interface requires a minimum four-second clip.

## What is added

Loading the LoRA alone cannot reproduce H3-World. The adapter requires all of:

1. The official deterministic key-to-language mapping.
2. Independent encoding of each action sentence; deduplication is request-local.
3. One Token Refiner segment for the scene head and one for each action sentence.
4. Mirrored temporal RoPE positions, maintaining a fixed offset to each latent.
5. Directed DiT attention: only action k and video latent k can exchange action
   information directly. Scene text, first-frame anchors, audio and other
   actions cannot read action k. Action k can read scene/anchors/audio and its
   own video latent. Video-to-video attention remains dense.

The CUDA mask uses FlexAttention block masks, avoiding a dense quadratic video
mask. It is constructed once per denoise branch and reused for every step.
As in the official patch, a bounded process-local cache reuses small label
buffers to keep compiled closure tensor identities stable. This relies on
serial request-mode forwards; overlapping forwards on the same worker are
unsupported. For serving with varying text lengths, H3's existing
`extra_args["pad_seq_len"]` can pin a sufficiently large aligned row bucket
to reduce FlexAttention recompilation.
The 104 LoRA pairs are validated against the published global shapes, and
QKV B matrices use the existing native H3 grouped-to-Q/K/V conversion.
DiffSynth's unit adapter scaling is preserved by `alpha=rank=32`.

## Current scope and validation

The minimal path supports one first-frame image, request execution, CUDA,
tensor parallelism with sequence parallel size 1, a packed-varlen refiner
backend (`FLASH_ATTN`), no diffusion cache, and guidance scale 1. Use the
example's eager configuration. Dynamic LoRA with DiT CPU/layerwise offload,
Ref2VA, FastH3/VDN/distilled checkpoints, step execution, latent editing,
guides, continuation, refine/upscale and a negative-action CFG branch are
not currently supported. Ordinary MiniMax-H3 requests retain their existing
execution path.

LingBot World's action/session APIs operate on a causal model with a KV cache;
Cosmos world-model adapters expose their own conditioning/state contracts.
H3-World's released code uses a full-clip bidirectional H3 denoiser, so this
integration reuses H3 rather than claiming those models' real-time session
capabilities. A session adapter should follow only after a compatible
continuation algorithm is validated.

| Existing implementation | Reuse decision |
| --- | --- |
| `minimax_h3` video pipeline | Reuse the complete AV inference path and native QKV LoRA packing. |
| `wan2_2`, `hunyuan_video`, `ltx2` | Reuse framework loading/output conventions; their text encoders, VAEs and transformer layouts are incompatible with H3 weights. |
| `lingbot_world` | Follow its per-latent key-list input convention; its causal DMD/session engine requires a different model architecture. |
| Cosmos3 / DreamZero state adapters | Keep conditioning and mask ownership within a request; defer durable session state until H3 continuation semantics exist. |

CPU tests cover keyboard semantics, action encoding isolation, RoPE binding,
refiner boundaries, directed information paths and the actual published LoRA
header shapes. CUDA tests compare sparse attention against dense SDPA while
changing action boundaries between same-size requests:

```bash
pytest tests/diffusion/models/minimax_h3/test_h3_world.py
```

An opt-in real-weight test runs forward and still controls on the same engine:

```bash
VLLM_TEST_MINIMAX_H3_FL2VA_MODEL=/models/MiniMax-H3 \
VLLM_TEST_H3_WORLD_LORA=/models/H3-World/step-10000.safetensors \
VLLM_TEST_H3_WORLD_FIRST_FRAME=/data/first_frame.png \
pytest tests/e2e/offline_inference/test_h3_world.py
```

Full vLLM adapter binding, CUDA kernels and real-weight video quality require
GPU validation. For a reference comparison, use the official test frame,
scene prompt, forward preset, seed 2, 50 steps, 124 frames and CFG 1. Compare
against the pinned upstream code rather than expecting bitwise identical
outputs across different attention kernels.

Reference implementation: [H3-World commit f0c7be2](https://github.com/Danzer1xxxxChan/H3-World/tree/f0c7be2acbbde8256b2473f6e7b088f58272acae).
Source contributions are Apache-2.0; the adapter and resulting model remain
subject to the MiniMax-H3 model license described in the model card.
