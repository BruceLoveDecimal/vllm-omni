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

Diagnostics, for when a parity check fails:

| Script | Use |
|---|---|
| `run_diag_layers.py` | Per-encoder-layer divergence. Smooth growth means accumulation; a jump localises a bug. |
| `run_diag_submodule.py` | Bisects layer 0 into patch embed / norms / attention / MLP / activation / rotary tables on identical inputs. |
| `run_rope_probe.py` | Compares candidate upstream rotary formulations against the reference op. Documents why none of them match. |

## Interpreting the numbers

bf16 end-to-end vision `rel_l2` around 2e-2 with cosine ~0.9998 is expected: it is
rounding accumulated over 24 layers, amplified by large outlier activations. It is *not*
evidence of a bug — `run_fp32_equivalence.py` returning ~4e-6 is what rules a bug out.
The load-bearing acceptance criterion is token-exact generation, not tensor closeness.
