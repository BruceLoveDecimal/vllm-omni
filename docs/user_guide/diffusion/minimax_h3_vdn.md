# VDN-H3: hybrid attention for MiniMax-H3

[VDN-H3](https://huggingface.co/OpenVDN/vdn-minimax-h3) replaces MiniMax-H3's
dense attention with a hybrid one: a **window softmax** that keeps each video
frame's attention inside its own VAE chunk, and a **linear-attention branch**
that summarises everything the window cannot see. The released checkpoint adds
that branch to the dense backbone and merges two LoRA adapters into it, so
vLLM-Omni serves it through the ordinary `MiniMaxH3Pipeline` with the branch
switched on inside each attention layer.

The 8-step release (`stage-dmd-step-250`) is also a distilled student, so it
samples on a pinned nine-position ladder rather than on a count the request
chooses.

## Serving

```bash
vllm-omni serve /path/to/MiniMax-H3/FL2VA \
  --trust-remote-code --task-type t2va \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --model-config '{"vdn": {"checkpoint": "/path/to/vdn/stage-dmd-step-250"}}'
```

`vdn.checkpoint` is a local VDN release directory (the one holding
`model_spec.json`) or a Hugging Face repository, in which case `vdn.subdir`
names the release inside it:

```bash
--model-config '{"vdn": {"checkpoint": "OpenVDN/vdn-minimax-h3", "subdir": "stage-dmd-step-250"}}'
```

A packaged release can declare the same block itself, under
`_minimax_h3.vdn` in its `model_index.json`, so `vllm-omni serve <dir>` needs no
flags. The server flag wins when both are present.

### Requests

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F model=MiniMax-H3 \
  -F prompt="..." \
  -F width=1344 -F height=768 -F fps=24 \
  -F num_inference_steps=8 \
  -F flow_shift=12 \
  -F 'extra_params={"task":"t2va","duration":14.4,"audio_flow_shift":3}' \
  -o out.mp4
```

`num_inference_steps` counts transformer forwards. The 8-step checkpoint pins
eight, so the field must be `8` or omitted; the 50-step `stage-b-step-2000`
checkpoint is not distilled and keeps the uniform ladder, where the field is the
number of sigma points and drives one fewer forward.

`flow_shift=12` / `audio_flow_shift=3` are the shifts VDN trained at. They are
already H3's defaults, and a request that moves them is refused rather than
sampled off the levels the student saw.

## Options

All under `--model-config '{"vdn": {...}}'`.

| Key | Default | What it does |
|---|---|---|
| `checkpoint` | *required* | Local release directory, or a Hub repository id |
| `subdir` | – | The release inside a Hub repository, e.g. `stage-dmd-step-250` |
| `revision` | – | Hub revision |
| `adapters` | every declared one | Which adapters to merge, by name |
| `window_group_batch` | `4` | Window groups whose keys are gathered per call |
| `window_impl` | `auto` | `auto`, `grouped`, or `varlen` |
| `linear_attention_enabled` | `true` | `false` runs the window alone — an ablation, not the released model |

`window_group_batch` trades transient memory for kernel launches: the window's
keys are gathered per group, and at 768p one group is around half a gigabyte
per rank. Lower it if the branch is the memory peak; raise it if the window is
launch-bound.

`window_impl=auto` sends the whole window through one packed variable-length
call on a backend that isolates multi-document `cu_seqlens` (FlashAttention on
CUDA), and otherwise runs the groups as batched dense calls. The grouped path
works on every backend this repository resolves — including the `CUDNN_ATTN`
that consumer Blackwell cards default to, and the Sage backends, which refuse an
attention mask outright — so nothing depends on the varlen kernel being present.

## What it supports

| | |
|---|---|
| Tasks | T2VA only — VDN distills nothing else |
| Parallelism | Tensor parallel, strict Ulysses sequence parallel, and both together |
| Acceleration | Cache-DiT, TeaCache, HSDP, online fp8 |
| Not supported | Ring / all-gather sequence parallelism, `ulysses_mode=advanced_uaa`, every offload mode, `--lora-path`, FL2VA and Ref2VA |

Each unsupported combination is a startup error rather than a silent fallback:
the offload modes install the transformer without going through `load_weights`,
which is where the branch is assigned, and ring attention dispatches through
kernels that never see the decomposed window. A server that ran anyway would
produce a quietly different sample.

Under step execution (`--step-execution`) requests do not share a forward: the
window plan and the frame recurrence are geometry over one packed sequence.
Requests still batch at the scheduler and take one forward each.

## Memory

The branch adds about 4.3 GiB of weights to H3's 66 GiB, and its per-layer
working set (the gathered window keys plus the two state banks) is a few
gigabytes more. At 768p / 14.4 s that is comfortable on two 96 GiB cards with
`--tensor-parallel-size 2`; a single card is better used for shorter clips or
with fp8.

## How it works

Per attention layer, both branches share the block's QKV projection:

```
softmax = out_proj( softmax_gate(x) * window_softmax(q, k, v) )
linear  = to_out_linear( output_gate(x) * RMSNorm( branch(q_raw, k_raw, v) ) )
out     = softmax;  out[video rows] += linear
```

The window keeps every pair that touches a text or audio row, restricts
video-to-video pairs to the query frame's chunk window, and keeps the first and
last video frames dense in both directions. Because those anchor frames are then
exact, the linear branch drops them from its input entirely and the two branches
remain an exact partition of the sequence.

The window's mask is not arbitrary sparsity: every kept pair lies in one of a
few dense rectangles, so the whole clip decomposes into a handful of ordinary
attention calls — five shapes at the released `chunk=5, radius=1` geometry,
whatever the clip length.

The linear branch is a bidirectional delta-rule recurrence over frames. Each
frame collapses to two `d x d` matrices, two scans build prefix and suffix state
banks, and each query frame reads the state just outside its window, decayed in
through the frames the window already covered. Both scans start from the prompt,
so a frame whose window touches a clip end reads the prompt rather than nothing.

Under Ulysses the branch runs its own two all-to-alls rather than the shared
sequence-parallel strategy: the window needs whole frames and the recurrence
needs every frame, so a rank holding a slice of the rows can compute neither.
One exchange turns rows-sharded/all-heads into all-rows/heads-sharded, and one
turns the two branches' outputs back.
