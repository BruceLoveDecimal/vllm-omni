# EasyCache Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration Parameters](#configuration-parameters)
- [How It Works](#how-it-works)
- [Best Practices](#best-practices)
- [MiniMax H3](#minimax-h3)
- [Troubleshooting](#troubleshooting)

---

## Overview

EasyCache accelerates diffusion model inference by skipping the whole transformer block stack on
denoising steps whose output is predicted to change very little. The prediction uses an online
estimate of how strongly input drift translates into output drift, so **no model-specific
coefficient fitting or calibration run is required**. It was designed for video DiTs and is the
cache policy used with SANA-Video.

Paper: [Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching](https://arxiv.org/abs/2507.02860)

EasyCache is implemented with vLLM-Omni's hook system and does not modify model code. It attaches
to every transformer whose blocks live in an `nn.ModuleList` and return `hidden_states` (or a
`(hidden_states, encoder_hidden_states)` tuple).

---

## Quick Start

### Basic Usage

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    cache_backend="easy_cache",
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### Custom Configuration

```python
omni = Omni(
    model="Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    cache_backend="easy_cache",
    cache_config={
        "easy_threshold": 0.1,  # Controls speed/quality tradeoff
        "easy_warmup_steps": 5,
        "easy_cooldown_steps": 1,
        "easy_max_skip_steps": 0,
    },
)
```

### Serving

```bash
vllm serve Efficient-Large-Model/SANA-Video_2B_480p_diffusers --omni \
  --cache-backend easy_cache \
  --cache-config '{"easy_threshold": 0.1, "easy_warmup_steps": 5, "easy_cooldown_steps": 1}'
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `easy_threshold` | `0.1` | Accumulated predicted relative output change above which the block stack is recomputed. Higher values skip more steps (faster, lower quality). |
| `easy_warmup_steps` | `5` | Number of initial steps that always compute. At least two full steps are needed before the estimator can skip. |
| `easy_cooldown_steps` | `1` | Number of final steps that always compute. |
| `easy_max_skip_steps` | `0` | Maximum consecutive skipped steps before a full computation is forced. `0` disables the cap. |

`num_inference_steps` is taken from each request; it defines the cooldown window and the
end-of-run state reset. A pipeline whose request step count is not the number of transformer
forwards reports the real count, so the schedule lines up with the run (see
[MiniMax H3](#minimax-h3)).

---

## How It Works

For every denoising step, with `x_t` the input of the first transformer block:

1. Warmup and cooldown steps always run the full block stack.
2. Otherwise, the relative output change is predicted from the input drift:
   `e_t = k * mean|x_t - x_{t-1}| / mean|y_last|`, where `y_last` is the output of the last fully
   computed step and `k = mean|y_last - y_prev| / mean|x_last - x_prev|` is the transformation rate
   measured between the last two fully computed steps. `e_t` is accumulated across skipped steps.
3. If the accumulated value stays below `easy_threshold`, the cached residual is reused:
   `y_t = x_t + (y_last - x_last)`. Otherwise the block stack runs, `k` and the residual are
   refreshed and the accumulator resets.

When classifier-free guidance runs the transformer once per branch, positive and negative branches
keep independent state. Under sequence parallelism the `mean|.|` statistics are reduced across the
SP group, so every rank makes the same skip decision.

---

## Best Practices

- Start with the default `easy_threshold=0.1` and raise it gradually while checking output quality;
  values around `0.2`–`0.3` trade noticeably more quality for speed on large video models.
- Keep `easy_warmup_steps` at least 2. Few-step distilled models (≤ 8 steps) leave little room to
  skip; lower the warmup to 2 and consider `easy_max_skip_steps=1`.
- Set `easy_max_skip_steps` (e.g. `2`–`3`) when you observe temporal artifacts with an
  aggressive threshold.
- Compare cached and uncached outputs with the same prompt, seed, scheduler, resolution and frame
  count before adopting a threshold for production.

---

## MiniMax H3

H3 is a CFG-distilled joint video/audio DiT that runs one block-stack forward
per denoising step, so EasyCache attaches to it without any model-specific
setting. Two H3 particulars are worth knowing:

- **`num_inference_steps` counts sigma points, not forwards.** H3 denoises the
  intervals between them, so a 50-step request runs 49 forwards. The pipeline
  reports the forward count to the cache backend, which is what the warmup,
  cooldown and the end-of-run summary are scheduled against.
- **`quality=high` is rejected.** That request quality installs Cache-DiT, which
  cannot be stacked on EasyCache's hooks. Omit `quality`, or restart the server
  with `--cache-backend cache_dit`.

Cache acceleration is unavailable in step mode (`--step-execution`).

Measured on 2x RTX PRO 6000 Blackwell with the two-GPU recipe configuration
(TP2, BF16, `CUDNN_ATTN`, tiled VAE), T2VA at 1344x768, 124 frames, 50 sigma
points, `flow_shift=12`, `seed=1101`, two warmup requests before the measured
one:

| Measurement | No cache | `easy_threshold=0.1` |
| --- | ---: | ---: |
| Denoise | 285.64 s | 239.68 s (-16.1%) |
| Peak HBM per GPU | 78,394 MiB | 79,818 MiB (+1,424 MiB) |
| Steps computed / skipped | 49 / 0 | 41 / 8 |

The skipped steps are the second half of the schedule (31, 33, 35, ..., 45),
where the run settles into a compute/skip alternation. Against the uncached
output, video SSIM was 0.9934 on average (0.9915 worst frame) and PSNR 50.19 dB
(48.07 dB worst frame). The **audio track is markedly more sensitive**: its
SDR against the uncached run was 16.2 dB, an error an order of magnitude larger
in relative terms than the video difference. Audio rows share the packed
sequence and the same skipped steps, so listen to the result before adopting a
threshold on a lip-sync or foley-sensitive workload.

---

## Troubleshooting

- **No speedup**: the run summary log (`EasyCache: run finished ... computed N, skipped M`) shows
  how many steps were skipped. If `M` is 0, raise `easy_threshold` or reduce warmup/cooldown; note
  that `easy_warmup_steps + easy_cooldown_steps >= num_inference_steps` disables skipping entirely.
- **Quality regression**: lower `easy_threshold`, increase `easy_warmup_steps`, or cap
  `easy_max_skip_steps`.
- **Unsupported block output**: EasyCache raises a `TypeError` on the first computed step if a block
  returns something other than a tensor or a `(hidden_states, encoder_hidden_states)` tuple.
