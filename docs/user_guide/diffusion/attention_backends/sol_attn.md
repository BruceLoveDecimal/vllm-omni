# Sol-Attn

`SOL_ATTN` runs [Sol-Attn](https://arxiv.org/abs/2607.24027), the
training-free on-the-fly sparse attention from the NVlabs/Sana
[`sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine) branch, on
CUDA GPUs. Inside one online-softmax pass the kernel scores every 64-token
key block with a lightweight proxy, evaluates the blocks above a per-query
threshold exactly, and reuses the proxy score for the rest instead of
dropping them. No routing map is materialized and no calibration pass is
needed, so the backend works on stock checkpoints.

The integration follows the MiniMax-H3 policy validated in Sol-Engine:

- The packed prefix (text, visual conditions, audio: everything before the
  target-video rows) is marked as an exact KV sink, so every query attends
  those keys exactly. The prefix's own query rows are then recomputed
  densely.
- The first `dense_steps` denoise steps and the DiT blocks in `dense_layers`
  stay dense.

Every call the kernel cannot serve — warmup steps, dense layers, attention
masks, joint or piecewise attention, non-BF16 activations, a kernel failure
when `strict` is off — delegates to a dense backend with the same metadata, so
a model can select `SOL_ATTN` globally. One selection covers every role a
model declares, so a role the kernel can never serve (a causal role, or one
whose head size is not 128) delegates for the whole run instead of failing
startup; `strict` turns that back into an error.

## Installation

The kernel ships as the external `sol-attn` package on the `sol-engine`
branch. It needs PyTorch >= 2.10, CUDA >= 12.8 and Triton >= 3.6; the CuTe DSL
kernels for RTX 4090 (SM89), H100 (SM90), B200/GB200 (SM100) and RTX 5090
(SM120) additionally need `nvidia-cutlass-dsl` and `cuda-python`. Other
architectures with compute capability >= 8.0 use the Triton reference kernel,
which is correct but slower than the published speedups.

```bash
git clone -b sol-engine https://github.com/NVlabs/Sana
pip install -e Sana/techniques/sparse_backends
```

Selecting `SOL_ATTN` without the package raises during backend resolution,
before the model is built.

## Configuration

| Key | Valid values | Meaning |
| --- | --- | --- |
| `tau` | finite float | Routing threshold coefficient; larger routes fewer key blocks exactly. Default `1.0` |
| `thresh_type` | `"diag"`, `"exact"` | Diagonal or full-covariance threshold estimate. Default `"diag"` |
| `kv_splits` | `"auto"`, `1`, `2`, `4` | Split-KV factor. `"auto"` (default) picks `4` on H100 for sequences of at least 65536 tokens and `1` elsewhere; `2`/`4` are H100-only |
| `dense_steps` | integer, `>= 0` | Number of early denoise steps kept dense. Default `10` |
| `dense_layers` | selector such as `"0-1,38"` | DiT blocks kept dense. Default `"0-1"` |
| `sink_mode` | `"prefix"`, `"none"` | Keep the published prefix as an exact KV sink (default) or route it like every other block |
| `strict` | bool | Raise on kernel failures, and on a role the kernel cannot serve, instead of silently running dense. Default `false` |
| `dense_backend` | a backend name such as `"CUDNN_ATTN"` | Backend for the warmup steps, dense layers and declined forwards. Defaults to the platform's own choice |

```bash
vllm-omni serve MiniMaxAI/MiniMax-H3 \
  --diffusion-attention-config '{"default":{"backend":"SOL_ATTN",\
    "sol_attn":{"tau":1.0,"dense_steps":10,"dense_layers":"0-1"}}}'
```

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, SolAttnSpec

config = AttentionConfig(
    default=AttentionSpec(
        backend="SOL_ATTN",
        sol_attn=SolAttnSpec(tau=1.0, dense_steps=10, dense_layers="0-1"),
    ),
)
```

Tune `dense_steps` first, then `tau`, then `dense_layers`. Set `strict: true`
while validating a configuration so a run that silently fell back to dense
attention cannot be reported as a sparse one.

## Verifying which backend is in use

A sparse run prints three lines; check all of them before reporting a
speedup, because a configuration that silently ran dense looks like a
plausible measurement.

```text
Resolved diffusion attention backend 'SOL_ATTN' for role='self' via attention_config
SOL_ATTN configured: tau=1.000, thresh_type=diag, kv_splits=auto, dense_steps=10, dense_layers=(0, 1), sink_mode=prefix, strict=False, dense fallback=CUDNN_ATTN.
SOL_ATTN active: tau=1.000, thresh_type=diag, dense_steps=10, dense_layers=[0, 1], sink_mode=prefix, used_len=38247, sink=[0, 951), total_len=38272, heads=7.
```

- The `Resolved ...` line comes from the selector and confirms the role was
  routed to `SOL_ATTN` rather than the platform default. A second such line
  names the dense fallback, resolved for role `sol_attn.dense_fallback`.
- `SOL_ATTN configured` is logged once when the attention layers are built and
  echoes the resolved `sol_attn` block.
- `SOL_ATTN active` is logged on the first forward that actually reaches the
  kernel, with the valid row count and the exact sink range. If it never
  appears, every forward stayed dense.

Each reason for staying dense is logged once as
`SOL_ATTN staying dense: <reason>` (warmup steps and dense layers are
expected and are not logged). With `strict: true`, a kernel failure raises
instead of producing that line.

## Requirements and compatibility

- CUDA only, compute capability >= 8.0, `head_dim=128`, BF16 activations,
  noncausal self-attention with `qkv_layout="BSND"`.
- The dense fallback follows the platform default rather than a fixed
  backend. That matters on consumer Blackwell (SM120), where the common
  FlashAttention wheels carry no kernel image and abort the process; the
  platform routes to `CUDNN_ATTN` there instead. Override with
  `dense_backend` when you want a specific one.
- Ulysses sequence parallelism is supported: after the all-to-all each rank
  holds the whole sequence for its own heads, and routing is decided per
  (query, head). Ring sequence parallelism is rejected at construction
  because the kernel needs the complete key sequence.
- The backend serves one packed document per forward. MiniMax-H3 detects this
  (`supports_multi_doc_packed_varlen` is `False`) and runs co-batched requests
  one forward at a time instead of packing them.
- The exact sink needs `AttentionMetadata.video_layout`. Models that do not
  publish it still run sparse, but without an exact prefix; a warning is
  logged once.

## Geometry handling

The backend reads the valid length of packed document 0 from the packed
padding metadata and hands only those rows to the kernel; trailing alignment
padding never reaches it and its output rows are zero. The sink covers
`[0, prefix_len)` for a legacy `[prefix | t*h*w video]` layout, or
`[0, target.start)` for a multi-span Ref2VA layout, so reference videos and
audio stay exact as keys. Exactness is applied at 64-token block granularity,
rounding outward, so a few target-video keys next to the boundary become
exact as well.

In the Sol-Engine MiniMax-H3 reference run, a 951-row prefix in a 38247-row
packed sequence gives 15 exact sink blocks out of 598 and 951 densely
recomputed query rows, about 1% of the attention on top of the routed blocks.

For common configuration and selector behavior, see the
[attention backend overview](../attention_backends.md) and the
[backend selection design](../../../design/feature/attention_backend_selection.md).
