# Mage-VL parity drivers

Directly executable scripts that compare this port against the HuggingFace reference
implementation of `microsoft/Mage-VL`. They are **not** collected by pytest (the `run_`
prefix keeps them out of collection) because they need a GPU and the real checkpoint;
the CPU-only assertions live in `../test_mage_vl_config.py`.

Naming follows the in-tree precedent of `tests/e2e/online_serving/run_minicpmo_realtime_duplex_*.py`.

## Prerequisites

- One CUDA GPU with >= 24 GB (validated on RTX 5090, 32 GB).
- The checkpoint on disk. Override the path with `MAGE_VL_PATH` (default
  `/root/autodl-tmp/models/Mage-VL`).
- Reference tensors produced by `run_hf_baseline.py`, written to `REF_OUT`
  (default `/root/autodl-tmp/mage_ref`).

Two environment variables were required on the validation box and are unrelated to this
model — they work around a flashinfer install whose cubin package version does not match
the wheel, and whose sampler JIT fails its arch check on sm120:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## Order of use

| Script | What it establishes |
|---|---|
| `run_hf_baseline.py` | Dumps reference processor tensors, vision embeddings, logits and greedy tokens. Everything else compares against this. Run first. |
| `run_parity_vision.py` | Vision tower vs reference vision output. `MAGE_FORCE_SDPA=1` pins the attention kernel to match the reference's. |
| `run_fp32_equivalence.py` | The decisive check: same bf16 weights upcast to fp32 on both sides. Proves algorithmic equivalence independently of accumulation noise. |
| `run_offline_parity.py` | vLLM offline generation vs reference, token by token. |
| `run_online_parity.py` | `/v1/chat/completions` vs reference, plus a streaming smoke. Needs a running server. |
| `run_video_parity.py` | Frame-sampled video through the online multi-image path (the mode the reference itself supports online), plus a 4-way concurrency check. Needs a running server. |
| `run_codec_parity.py` | Vision tower on **codec** canvases -- sparse `t`, several source frames per canvas. The image drivers cannot reach that layout. |
| `run_codec_generation_parity.py` | Codec inputs end to end: our tower into vLLM's decoder, against the checkpoint's `generate`. `MAGE_PARITY_DTYPE=float32` runs the attribution mode described below. |
| `run_gate_parity.py` | The cognition gate's perception encoder against the checkpoint's own, both full-stream and one segment at a time. Needs `mamba_ssm` importable -- see below. |

Diagnostics, for when a parity check fails:

| Script | Use |
|---|---|
| `run_diag_layers.py` | Per-encoder-layer divergence. Smooth growth means accumulation; a jump localises a bug. |
| `run_diag_submodule.py` | Bisects layer 0 into patch embed / norms / attention / MLP / activation / rotary tables on identical inputs. |
| `run_rope_probe.py` | Compares candidate upstream rotary formulations against the reference op. Documents why none of them match. |

## The codec drivers need an engine

`run_codec_parity.py` and `run_codec_generation_parity.py` run the codec sampler unless
you hand them a directory it already produced:

```bash
pip install codec-video-prep          # in a SEPARATE venv: it pins numpy<2.0
export CV_PREINFER_BIN=/path/to/that/venv/bin/cv-preinfer   # plus ffmpeg/ffprobe on PATH
# or skip the engine entirely:
export MAGE_CODEC_ASSETS=/path/to/assets
```

## Interpreting the numbers

bf16 end-to-end vision `rel_l2` around 2e-2 with cosine ~0.9998 is expected: it is
rounding accumulated over 24 layers, amplified by large outlier activations. It is *not*
evidence of a bug — `run_fp32_equivalence.py` returning ~4e-6 is what rules a bug out.
The load-bearing acceptance criterion is token-exact generation, not tensor closeness.

### Why the codec path is not judged on greedy token equality

The image drivers are: short prompts, short answers, and the port reproduces the
reference token for token. Codec inputs are the opposite -- 4608 visual tokens feeding an
open-ended description -- and there the continuation is unstable under perturbations far
smaller than the port's:

| Comparison (same decoder, same decode path, 32 new tokens) | Agreement |
|---|---|
| our tower (fp32) vs reference tower (fp32), visual embeds `rel_l2` 7.39e-06 | 5/32 |
| **reference tower (fp32) vs reference tower (bf16)** -- the reference against itself | **6/32** |
| vLLM decoder vs HF decoder, both fed *identical* embeddings | **32/32 exact** |

The middle row is the point: the reference disagrees with itself when only its own dtype
changes, so token equality cannot tell a correct port from an incorrect one on this input.
What it is judged on instead: preprocessing identical elementwise (canvases, grid,
`pixel_values`, `patch_positions`, rewritten prompt, prompt token ids), tower `rel_l2`
7.39e-06 in fp32, and a decoder that is bit-identical to the reference's given the same
embeddings.

### Running the gate reference

The checkpoint's `streammind_gate.py` builds its Mamba block through `mamba_ssm`, whose
CUDA extension has no wheel for the torch versions this project targets. Two steps make it
importable anyway:

```bash
MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation --no-deps mamba-ssm einops
mkdir -p /tmp/shims && touch /tmp/shims/selective_scan_cuda.py   # mamba_ssm imports it eagerly
MAGE_SHIM_DIR=/tmp/shims python run_gate_parity.py
```

The driver then points the reference at `selective_scan_ref`, the pure-torch recurrence
`mamba_ssm` ships as its own definition. The port does not depend on `mamba_ssm` at all.
