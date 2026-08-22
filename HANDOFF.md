# Handoff — Mage-VL integration

> Delete this file before opening the PR; it is working state, not repo documentation.

Branch: `claude/model-integration-spec-4602f8`, pushed to fork `BruceLoveDecimal/vllm-omni`.
Base: `02069a5b`. Target runtime: vLLM **0.27.0** (not the newer local checkout — see §6).

---

## 1. Where this stands, in one paragraph

`microsoft/Mage-VL` (codec-native Mage-ViT vision tower + stock Qwen3-4B decoder) is
implemented as a single-stage understanding pipeline and **was working end to end**:
online `/v1/chat/completions` produced token-identical output to the HuggingFace
reference for images and for frame-sampled video, with streaming and 4-way concurrency
clean. Then a reuse audit found three real defects and several reuse opportunities; the
fixes and the reuse have now been **re-verified on the box** (2026-08-19, see §4), with one
real regression found and fixed on the way. Three items remain unrun because the instance
powered off mid-run when the balance ran out again: the `vllm-omni serve` deploy-config
path, `trust_remote_code=False` *serving* (the processor half is proven), and the
profiling comparison in §5. The dev Mac's venv
(`/Users/liuqihao/Developer/vllm-omni/.venv`: torch 2.11.0 CPU/MPS, vllm 0.27.0+cpu,
transformers 5.8.0) runs the CPU tests and settles transformers-runtime questions without
the box -- run it from the worktree with `PYTHONPATH=$PWD`, since the editable install
points at the main checkout.

---

## 2. Do this first

```bash
~/.claude/skills/autodl-gpu/autodl.sh status pro-787016ace1d3
```

If it still reports insufficient balance on power-on, stop and tell the user — nothing
below can be verified without it. Once it boots, run §5 in order.

**Status of the previously-unverified commits**, in commit order:

| Commit | Risk | State |
|---|---|---|
| `8ca45ea8` deploy config `stage_id`/`pipeline` | high | **Verified**: `-m vllm_omni.entrypoints.cli.main serve <model> --omni --deploy-config vllm_omni/deploy/mage_vl.yaml` starts (KV cache 16.45 GiB / 119,760 tokens) and answers 3/3 token-identical. Note **`--deploy-config` is only accepted together with `--omni`** (without it: `unrecognized arguments`). |
| `9128219e` `keep_on_cpu` + `temporal_patch_size` | low | **Verified** -- exercised by every run below, and `temporal_patch_size` is what the reused `_get_vision_info` reads. |
| `e4a1606f` reuse Qwen2-VL merger + MLP | **highest** | **Verified numerically**: fp32 `rel_l2` 4.1865e-06 / cos 1.0000000000, identical to the pre-reuse figure. |
| `a5994c5c` `__init__` exports | none | Verified by import. |
| `efc0a305` test additions | none | 15/15 now, on transformers 5.8.0 (Mac) and 5.8.1 (box). |
| `38744970` reuse Qwen2-VL processor + processing info | **highest** | **Verified** for preprocessing and generation (§4); its profiling change is only half-measured (§5). One regression found and fixed during the run (§8). |
| *(new)* parity driver fixes | none | The committed drivers indexed `mine.encoder` where the tower has `encoder.layers`; **each one failed on first execution**. Fixed in four scripts. |

---

## 3. Machine and environment

AutoDL Pro instance `pro-787016ace1d3` -- RTX 5090, 32 GB. Drive it with the
`autodl-gpu` skill (`~/.claude/skills/autodl-gpu/autodl.sh`), not the console.

```bash
S=~/.claude/skills/autodl-gpu/autodl.sh
$S on   pro-787016ace1d3        # always boots GPU mode; no-GPU boot is not API-supported
$S ssh  pro-787016ace1d3        # host + port + root password, both change every cycle
$S off  pro-787016ace1d3        # <- the box bills while running
$S wait pro-787016ace1d3 running
```

**Power it off when you finish.** It bills whether or not it computes.

Three traps that cost time on this box:

1. **No-GPU boot mode.** The instance was found booted without a GPU: `/dev/nvidia*`
   absent and `/usr/bin/nvidia-smi` a **0-byte placeholder**. Switching modes needs a
   full off → on cycle with `--payload gpu`; a restart is not enough. Check
   `ls /dev/nvidia*` before believing anything else. (The API's `memory` field is
   unreliable — it read "2 GiB" while `free -g` showed 754 GB.)
2. **flashinfer is broken on this box, unrelated to this model.** The installed
   `flashinfer-cubin` version does not match the wheel, and its sampler JIT fails an arch
   check on sm120. Every run needs:
   ```bash
   export FLASHINFER_DISABLE_VERSION_CHECK=1
   export VLLM_USE_FLASHINFER_SAMPLER=0
   ```
   Without the second one the engine dies in kernel warmup with
   `RuntimeError: FlashInfer requires GPUs with sm75 or higher`, which is misleading —
   the failure is in `topk_topp_sampler`, not attention.
3. **Two GPU consumers do not fit at 0.85 memory utilisation.** The video parity script
   loads the HF reference *and* talks to the server, so the server must be started with
   `--gpu-memory-utilization 0.45` for that test.

### Paths on the box

| What | Where |
|---|---|
| Python env | `/root/ming-l3-venv/bin/python` (3.12, torch 2.13.0+cu130, vllm 0.27.0, transformers 5.8.1) |
| vllm-omni (editable install target) | `/root/autodl-tmp/ming-l3/vllm-omni` — **not a git repo**, an extracted copy; sync into it |
| Checkpoint | `/root/autodl-tmp/models/Mage-VL` (complete, incl. `streammind_gate.safetensors` and `examples/{dog.jpg,soccer-broadcast.mp4}`) |
| Reference tensors | `/root/autodl-tmp/mage_ref` (regenerate with `run_hf_baseline.py`) |
| Logs | `/root/autodl-tmp/logs/` |

`/root/autodl-tmp` is the persistent volume and survives power cycles. Missing packages,
if you need them: `mamba_ssm` (gate, M4), `decord` (reference video path), `flash-attn`.

### Sync loop

There is no git remote on the box. Copy changed files in, e.g.:

```bash
K="-i /tmp/autodl_key -o UserKnownHostsFile=/tmp/autodl_known -o LogLevel=ERROR"
R=/root/autodl-tmp/ming-l3/vllm-omni
scp $K -P <port> vllm_omni/model_executor/models/mage_vl/*.py root@<host>:$R/vllm_omni/model_executor/models/mage_vl/
```

Install a key once (`ssh-keygen` + append to `~/.ssh/authorized_keys` via `expect`;
`sshpass` is not available on the dev Mac, `expect` is at `/usr/bin/expect`).

---

## 4. What is verified, and what "verified" means here

The issue asks for SSIM/PSNR. **Those do not apply** — Mage-VL emits text, not images.
The criterion used instead is **token-exact generation**, which is strictly stronger
(bit-level agreement vs SSIM's perceptual tolerance), backed by operator-level `rel_l2`
in fp32.

Measured **2026-08-19, with the processor/processing-info reuse in place**:

| Check | Result |
|---|---|
| Processor vs the checkpoint's own processor | **3/3 elementwise identical**: `input_ids`, `pixel_values` (after the bf16 cast the baseline dumps with), `image_grid_thw`, `patch_positions` |
| Processor load with `trust_remote_code=False` | **loads**; `get_attributes() == [image_processor, tokenizer]` |
| Vision tower, fp32, same bf16 weights upcast | `rel_l2` **4.1865e-06**, cosine **1.0000000000** (was 4.19e-6) |
| Vision tower, bf16, end to end | cosine 0.99980, `rel_l2` 2.0062e-02 — unchanged; accumulation, not a bug (see §6). The driver's own threshold is stricter than this accepted figure and prints FAIL |
| Weight loading | 297/297 tower params, missing 0 |
| Offline generation | **3/3 token-identical**, greedy, at `gpu_memory_utilization=0.85` / `image: 4` / `max_model_len=8192` — the enlarged dummy data did not break startup there |
| Online `/v1/chat/completions` | **3/3 token-identical**, streaming 24 chunks, at `--gpu-memory-utilization 0.45` / `image: 8` |
| Video (frame-sampled multi-image) | token-identical; 4 concurrent requests → 1 distinct output |
| CPU tests | 15/15 on transformers 5.8.0 (Mac) and 5.8.1 (box) |

Also verified in the same session:

| Check | Result |
|---|---|
| `vllm-omni serve --omni --deploy-config` | starts; 3/3 token-identical + streaming |
| Serving with **no** `--trust-remote-code` | starts (`trust_remote_code=False` in the engine config) and answers 3/3 token-identical |
| Profiling at 0.85 / `image: 8` / 32768 | encoder cache budget 3906 tokens profiled with **1** max-size image; KV 16.67 GiB / 121,408 tokens. With `--mm-processor-kwargs '{"max_pixels":150000}'`: budget 2048, KV 16.97 GiB. The 0.3 GiB delta is why the checkpoint default stays -- and it proves `mm_processor_kwargs` passthrough works |
| Online e2e suite (`tests/e2e/online_serving/test_mage_vl.py`) | L2 1/1, L3 3/3 |
| Perf smoke (1 image, 64 tokens, greedy) | TTFT median 76 ms, decode 161.7 tok/s; 4 concurrent: 422.3 tok/s aggregate, TTFT median 124 ms |

**Still never verified: `tp > 1`.** It needs a second GPU, which this box does not have.
Everything else in the M1/M2 acceptance list has been run.

---

## 5. Verification playbook

Scripts live in `tests/model_executor/models/mage_vl/parity/` — read its README first.
Run in this order; each depends on the previous.

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0
V=/root/ming-l3-venv/bin/python
cd /root/autodl-tmp/ming-l3/vllm-omni

# 0. CPU tests (fast, catches the QuickGELU trap in the reuse commit)
$V -m pytest tests/model_executor/models/mage_vl/test_mage_vl_config.py -q

# 1. Reference tensors (only if /root/autodl-tmp/mage_ref is missing)
$V tests/model_executor/models/mage_vl/parity/run_hf_baseline.py

# 2. Vision tower — the reuse commit's blast radius
$V tests/model_executor/models/mage_vl/parity/run_fp32_equivalence.py   # expect ~4e-6
MAGE_FORCE_SDPA=1 $V tests/model_executor/models/mage_vl/parity/run_parity_vision.py

# 3. Offline generation
$V tests/model_executor/models/mage_vl/parity/run_offline_parity.py     # expect 3/3

# 4. Online: start the server, then the two e2e drivers
$V -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/Mage-VL --served-model-name Mage-VL \
    --trust-remote-code --dtype bfloat16 --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":8}' --gpu-memory-utilization 0.45 --port 8077
$V tests/model_executor/models/mage_vl/parity/run_online_parity.py
$V tests/model_executor/models/mage_vl/parity/run_video_parity.py
```

Then the profiling consequence of the processor reuse (**new, and the most likely thing
to break**): dummy data is no longer a hardcoded 448x448 but the checkpoint's own
`max_pixels`, i.e. 2016x1984 -> 3906 vision tokens per image. At `limit_mm_per_prompt
image: 8` that profiles ~31k vision tokens. Check whether the server still starts at
`--gpu-memory-utilization 0.85`, and compare against
`--mm-processor-kwargs '{"max_pixels": 150000}'` (the reference pipeline's own budget,
spec §4.2) -- that kwarg feeds both profiling and runtime, so it is the knob if 4M is too
large a default. Whatever wins goes into `deploy/mage_vl.yaml` + the recipe.

Then the two things that have **never** been run and are the point of the fixes:

```bash
# 5. The deploy-config path that commit 8ca45ea8 fixes (bare `vllm serve` bypasses it)
$V -m vllm_omni.entrypoints.cli.main serve /root/autodl-tmp/models/Mage-VL --omni ...

# 6. Whether vendoring actually removed the trust_remote_code dependency
#    (drop --trust-remote-code and see if config + processor still resolve)
```

If §5.2 regresses, the first suspect is the `act_layer` binding in the reused
`Qwen2VisionMLP` — upstream defaults it to **QuickGELU** while Mage-VL is GELU, and the
mistake is silent (weights still load, nothing raises).

---

## 6. Hard-won findings — read before touching the tower

**The rotary is not any upstream variant, and this was verified the hard way.**
The spec originally claimed `ApplyRotaryEmb(is_neox_style=False)` expressed it. That was
wrong. Mage-VL combines an *interleaved* `rotate_half` with cos/sin built over the **full**
head dim (`cat([freqs, freqs], -1)`), so the two halves of an adjacent pair carry
*different* frequencies, whereas non-neox upstream shares one frequency per pair.
Measured: non-neox cosine **0.699**, neox **0.572**, exact replication **1.0000000000**.
The op is also **not norm-preserving** (‖q‖ 97.034 → 96.827) — the 2×2 matrix is
`[[c₀,−s₀],[s₁,c₁]]`, determinant `c₀c₁+s₀s₁ ≠ 1` — so it is not a rotation at all, it is
a property the checkpoint was trained with. Reproduce it; do not normalise it.
`run_rope_probe.py` re-derives this in about a minute.

**bf16 drift is expected; do not chase it.** End-to-end tower `rel_l2` ≈ 2e-2 in bf16 with
cosine 0.9998. A submodule bisect put patch embedding, both LayerNorms, GELU and the
rotary tables at **exactly 0** difference, leaving only attention at 1.3e-4 — below bf16
single-element precision. Forcing both sides onto SDPA did *not* collapse it, which
initially looked like a bug; fp32 at 4.19e-6 is what settles it.

**The fused qkv layout is favourable.** The checkpoint stores one `[3·1024, 1024]` matrix
in `[all_q | all_k | all_v]` block order (confirmed from the reference's
`reshape(B, L, 3, H, D)`), which is exactly what `QKVParallelLinear`'s loader consumes
with no shard id. Head-interleaved would have needed a de-interleave; it does not.

**No M-RoPE.** `text_config` has no `mrope_section`, so positions are plain 1-D. Do not
add `get_mrope_input_positions`.

**Why the processor is vendored.** The checkpoint's own `MageVLProcessor` deliberately
does not inherit `ProcessorMixin` (its source says so), and vLLM's processor cache
requires it — that is a `TypeError` at startup, not a style preference. `ProcessorMixin`
is a **transformers** class; vLLM has no such class of its own, it only enforces the type.

---

## 7. Code map

```
vllm_omni/model_executor/models/mage_vl/
    vision.py     Mage-ViT. Model-specific: _apply_interleaved_rotary,
                  MageVLVisionRotaryEmbedding (4:6:6 3D RoPE), build_window_cu_seqlens.
                  Everything else is upstream (QKVParallelLinear, MMEncoderAttention,
                  Qwen2VisionMLP, Qwen2VisionPatchMerger).
    mage_vl.py    Top model + vLLM multimodal processor + TensorSchema.
    pipeline.py   Single LLM_AR stage.
vllm_omni/transformers_utils/configs/mage_vl.py      vendored config
vllm_omni/transformers_utils/processors/mage_vl.py   vendored processor + patch positions
vllm_omni/deploy/mage_vl.yaml
tests/model_executor/models/mage_vl/                 CPU tests + parity/ drivers
docs/design/feature/mage_vl_integration.md           the spec; §0.0 records measurements
docs/contributing/model/adding_omni_model.md         generalised guide (weight loading + encoder TP)
```

Registration happens in three places, all required: `models/registry.py` (`_OMNI_MODELS`),
`config/pipeline_registry.py` (`OMNI_PIPELINES`), `engine/arg_utils.py` (AutoConfig).

---

## 7b. Milestone status (2026-08-19)

**M1/M2 complete except `tp > 1`**, which needs a second GPU this box does not have.
Landed in this session: model listing row, `recipes/Microsoft/Mage-VL.md` with measured
accuracy *and* perf numbers, `tests/e2e/online_serving/test_mage_vl.py` (L2 1/1, L3 3/3 on
the box), a CI deploy overlay (`_CI_OVERLAYS["mage_vl"]`), buildkite wiring in
test-ready/test-merge, and CPU guards for the modality declaration and the parallel-axis
fail-fast.

Two traps found while writing those tests, both worth remembering:

- `send_omni_request` forwards only a fixed set of top-level keys. A top-level
  `temperature` is **dropped silently**, so the server sampled with the checkpoint's
  generation_config defaults and every "output is stable" assertion failed for the wrong
  reason. Sampling params must ride in `extra_body`.
- `--deploy-config` is only accepted alongside `--omni`.

**M3's preprocessing is done and verified end to end.** The canvas → model-input half
lives in `vllm_omni/transformers_utils/processors/mage_vl_codec.py`, reachable through
`processor(videos=..., video_backend="codec")` or
`codec_config={"asset_dir": ...}`. Every piece was compared against the checkpoint's own
`codec_video_processing_mage_vl.py`: first on CPU over randomized inputs (all identical),
then on the real 720-frame `examples/soccer-broadcast.mp4` where the engine output
(`src_positions`, canvas pixels), `image_grid_thw`, `pixel_values` (18432, 768),
`patch_positions`, the rewritten prompt (68737 chars) and the prompt token ids (6518
tokens, 4608 visual) are **all identical**. 32 canvases for 256 sampled frames; the codec
pass takes ~1.9 s.

The engine turned out to be obtainable after all: `pip install codec-video-prep` (PyPI,
Linux wheels, needs ffmpeg/ffprobe) provides `cv-preinfer`, so `_run_cv_preinfer` is a
real driver now, with per-(video, budget) caching. It pins `numpy<2.0`, so it lives in
its own venv on the box (`/root/autodl-tmp/codecvenv`) and is reached via
`CV_PREINFER_BIN` / `PATH`; the vllm venv is untouched.

**A real defect in the checkpoint's own defaults, found by running it:** `codec.patch` is
14 while its image processor is `patch_size=16`. The canvases then tile as 36x20 = 720
patches (504x280 px) while the processor reads 527, and `codec_positions_for_processor`
raises a length mismatch -- the reference hits this too. The port derives `patch` from
the image processor and rejects a conflicting override, with the measurement written into
the test.

**Codec generation, measured and attributed.** `run_codec_parity.py` puts the tower on
real codec canvases: fp32 `rel_l2` **7.39e-06** / cos 1.0000000000, bf16 2.05e-02 -- the
same profile as the image path, so sparse `t` and mixed-frame canvases are handled
correctly. `run_codec_generation_parity.py` then drives generation through vLLM
(`prompt_embeds`, since the mm plumbing is not wired) and decomposes the result:

| Comparison, 32 new greedy tokens | Agreement |
|---|---|
| vLLM decoder vs HF decoder, identical embeddings | **32/32 exact** |
| our fp32 tower vs reference fp32 tower (embeds `rel_l2` 7.4e-06) | 5/32 |
| reference fp32 tower vs reference bf16 tower -- *the reference against itself* | 6/32 |

The last row is why M3's "greedy 逐 token 一致" acceptance criterion does not survive
contact with codec inputs: 4608 visual tokens feeding an open-ended description make the
continuation unstable under perturbations the reference itself produces by changing its
own dtype. Token equality cannot discriminate a correct port there, and the spec's
criterion should be replaced by what does: preprocessing identical elementwise, tower
`rel_l2` at 1e-6 in fp32, and a decoder bit-identical given the same embeddings. (The
image path stays token-exact -- short prompts, short answers -- so nothing changes there.)

What M3 still needs: the codec path through **vLLM's multimodal plumbing** (see below),
`dcvc-rt` (needs the checkpoint's `neural_codec` CUDA extensions, not vendored), and L3
codec cases once the serving path exists.

**What the plumbing has to solve, measured rather than guessed.** For the 720-frame clip
the rewritten prompt holds **196 vision blocks** (one per source-frame run) backed by
**32 canvases**, and the run boundaries do *not* line up with canvas boundaries -- a canvas
mixes frames, and a frame's patches span canvases. So a codec video is one mm item whose
embeddings scatter across 196 discontiguous spans; vLLM can express exactly that with
`PromptUpdateDetails.select_token_id`. The harder half is the input side: vLLM decodes
video into frames before a model sees it, while the codec engine needs the encoded
bitstream. Bridging that needs either a model-owned data parser that passes the source
through untouched, or a serving-layer hook -- and the spec's module-boundary rule forbids
the latter. That is a design decision, not an implementation detail.

**M4 has started: the gate's perception encoder is ported and verified.**
`vllm_omni/model_executor/models/mage_vl/gate.py` holds the checkpoint's
``pre_net -> Mamba-1 -> norm -> post_net`` stack with per-session ``(conv_state,
ssm_state)``. Against the checkpoint's own gate on a real weight load
(`parity/run_gate_parity.py`, fp32):

| Comparison | Result |
|---|---|
| ours vs reference, full stream | `rel_l2` **4.55e-07**, cos 1.0000000000 |
| ours fed one segment at a time vs reference | `rel_l2` **5.99e-07** |
| streamed vs one-shot (the streaming contract) | `rel_l2` 4.36e-07 |

Two things worth carrying forward:

- **`mamba_ssm` is not a dependency of the port, only of the reference.** Its CUDA
  extension has no wheel for torch 2.13/cu13. The reference runs via
  `MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation --no-deps mamba-ssm` plus an
  empty `selective_scan_cuda.py` on the path (the package imports it eagerly), with
  `selective_scan_fn` pointed at `selective_scan_ref`. Both are installed on the box.
- **vLLM's Mamba kernels were tried first and do not fit.** `causal_conv1d_fn` /
  `causal_conv1d_update` / `selective_scan_fn` are written for the engine's paged state
  cache: called without the block bookkeeping (`block_idx_*`, `initial_state_idx`) that
  only the engine's state manager produces, they **silently return their input untouched**
  or produce NaN -- measured on a 1-channel micro example, not inferred from a failed
  parity number. The gate runs outside the engine and advances one segment per call, so it
  uses an explicit recurrence instead. That also makes it CPU-testable, which is why the
  gate has 8 CPU tests rather than a GPU-only story.

The classifier head is in too (`MageVLCognitionGate`): 4-layer Qwen3, vocab 2, built from
`transformers` because it runs outside the engine once per segment and matching the
reference exactly matters more than sharing an engine path it never enters. Its output is
the number the policy thresholds, and it lands on the reference:

| Check | Result |
|---|---|
| per-segment `p_speak`, one-shot | max abs diff **2.38e-07** |
| per-segment `p_speak`, fed one segment at a time | max abs diff **2.11e-07** |

The spec's M4 acceptance asks for <= 1e-2, so the trigger sets are identical by a wide
margin.

**The streaming half is built on the repo's own duplex contracts**, not a new mechanism.
`vllm_omni/experimental/fullduplex/mage_vl/` holds a `DuplexAdapter` plus its policy, and
`core/` is untouched -- which is what the package README's recipe asks for. The fit is
close to exact: `core.DuplexRuntime` already starts a response when a session is
`proactive` and `should_respond()` is true, so **the gate is `should_respond`**. Barge-in,
epochs, and stale-output dropping come from `core/` unchanged. Decisions ride the existing
`response.speak` / `response.listen` events (already projected by
`openai/realtime_output.py`) through an injected callback, so no event type was invented.

Two policy details that are ours, because the checkpoint ships no streaming driver
(`inference_streaming.py` is not in the repo -- only the gate module is): a **cooldown**
after answering, without which the gate re-fires on the next segment and the model talks
over itself, and a **bounded prompt window** (the gate's state spans the session; the
prompt cannot). Defaults follow the spec: 8 s segments, tau = 0.5.

12 CPU contract tests drive the **real** `DuplexRuntime` with the model stubbed
(`tests/e2e/features/fullduplex/mage_vl/`), covering proactive firing, cooldown, mid-stream
questions, window eviction, barge-in with no stale deltas, and refused modalities.

What M4 still needs: production wiring of `score_segment` (vision tower -> gate) and the
serving adapter, then the streaming e2e and interrupt tests against a live server -- those
need the box and a `--omni` duplex deployment, not more contract work. The
`image_embeds` passthrough the spec lists for the sliding window is deferred with a reason:
the model does not accept precomputed embeddings as an input today, so the switch would
have nothing to turn on.

The reference implementation's sources are cached at
`<scratchpad>/mage_ref_src/` (checkpoint `.py` + `.json`, no weights) so this reading does
not need the box powered on.

## 8. Reuse done, and what it now needs from the box

Both items the audit left open are **implemented** (the transformers-runtime blockers were
resolved on the dev Mac, not guessed):

- `MageVLProcessor` now subclasses `transformers.Qwen2VLProcessor`. The blocker was real
  but **not for the reason the audit guessed**: the checkpoint *does* ship
  `video_preprocessor_config.json` -- it declares `video_processor_type:
  MageVLVideoProcessor` with an `auto_map` into the checkpoint's own remote code. Since
  transformers derives the attribute list from the **subclass** `__init__` signature
  (`ProcessorMixin.get_attributes`), inheriting Qwen2-VL's signature verbatim puts
  `video_processor` in that list, and resolving it drags in remote code: measured on the
  box, `trust_remote_code=False` raises `ValueError` ("contains custom code which must be
  executed") and `trust_remote_code=True` raises `AttributeError: type object
  'MageVLVideoProcessor' has no attribute 'register_for_auto_class'`. Overriding
  `__init__` without the parameter dissolves both, and is what lets the processor load
  with **`trust_remote_code=False`** -- the first evidence for the vendoring goal, which
  had never been tested. A CPU test pins
  `get_attributes() == ["image_processor", "tokenizer"]`.
  Side effect: upstream also emits `mm_token_type_ids` (transformers defaults
  `return_mm_token_type_ids=True`); vLLM ignores keys absent from `_get_mm_fields_config`
  (`MultiModalKwargsItems.from_hf_inputs`), so it is inert.
- `MageVLProcessingInfo` now subclasses `Qwen2VLProcessingInfo` (image-only limits,
  image-only `get_mm_max_tokens_per_item`, base data parser -- upstream's parser adds video
  and embedding validation this port does not accept). The correctness payoff is the dummy
  size: `get_image_size_with_most_features()` gives 2016x1984 / 3906 tokens from the
  checkpoint's `max_pixels=4,000,000`, where the old hardcoded 448x448 profiled 196.
  `MageVLDummyInputsBuilder` stays on `BaseDummyInputsBuilder` (in-repo precedent:
  `ming_flash_omni_thinker.py:209`) -- with no video modality there is nothing left to
  inherit -- and now also honours `mm_options` overrides.

**The open decision is the profiling default**, and it needs the box: 3906 tokens/image x 8
images is honest about what the processor accepts but ~27x what the reference pipeline
actually runs (spec §4.2 quotes `max_pixels=150000`, ~146 tokens per canvas). Measure
startup at 0.85 utilisation both ways, then either keep 4M or pin `max_pixels` in
`deploy/mage_vl.yaml` (`yaml_engine_args` passes `mm_processor_kwargs` through) and say so
in the recipe. Do not decide this from the source.

**A trap this reuse walked into, worth not repeating.** Deleting the seemingly-dead
`_call_hf_processor` override (its `patch_positions` fallback looked unreachable once the
processor emitted the key itself) broke decoding with `KeyError: 'patch_positions'`. vLLM
selects its mm-only path by *whether that method is overridden*
(`_apply_hf_processor_mm_only`); unoverridden it calls `call_hf_processor_mm_only`, which
bypasses the processor's `__call__` and invokes `processor.image_processor` directly, so
the positions table is never built. Upstream keeps the same override for the same reason
(`nano_nemotron_vl.py`, whose docstring says so outright). The override is restored with
that reason written down, and a CPU test now asserts it stays overridden. Note the failure
surfaces at **decode time**, not startup -- CPU tests and model load both looked healthy.

Net source delta: about -20 lines.

Correctly rejected, with reasons, in the audit: the attention block (upstream needs 3-D
`[s,b,c]`, this tower is 2-D throughout; and `Qwen2VisionAttention` uses
`ColumnParallelLinear` + all-gather, not `QKVParallelLinear`), the encoder layer
(fused-residual RMSNorm signature), the patch embedding (`Conv3dLayer`, 5-D weights vs the
checkpoint's 4-D), and the position tables (no extractable upstream helper).

---

## 9. Roadmap beyond this

M1/M2 image + frame-sampled video are done. Not started, in spec order:

- **M3** codec-native input (H.264/HEVC via cv-preinfer, or DCVC-RT). The whole codec
  path is greenfield — nothing in the repo touches motion vectors or I/P frames on the
  *input* side. Needs optional-deps treatment.
- **M4** StreamMind cognition gate + proactive streaming. The gate is a separate 1.07 GB
  side-car (`streammind_gate.safetensors`, **not** in the model index, so the standard
  loader ignores it) built from a Mamba block; needs `mamba_ssm`. The spec settles on
  hosting it in the serving process behind the duplex `DuplexAdapter.should_respond()`
  seam, mapping decisions onto the existing `response.listen` / `response.speak` events.
- **M5** native full-duplex, **M6** performance.

The spec (`docs/design/feature/mage_vl_integration.md`) has the reasoning, the reuse
mandate (§3.4), the TP-only parallelism scope (§5), and the serving conventions (§4.5.0)
including why the duplex contract was chosen over `/v1/video/chat/stream`.

---

## 10. Decisions worth not re-litigating

- **TP only.** Other parallel axes were explicitly scoped out by the maintainer. The tower
  is nonetheless written with `is_vit_use_data_parallel()` / `disable_tp=` so encoder DP is
  a later switch, and `supports_encoder_tp_data` is deliberately left `False` so vLLM
  downgrades honestly. The audit flagged this as dead code; it is not — there is a comment
  in `mage_vl.py` explaining it.
- **Commit hygiene.** Extremely short subjects, DCO on every commit, **no issue numbers in
  commit messages** (they would ping the tracking issue). The spec file does reference
  issues in its body, which is safe — only commit messages and PR bodies create
  cross-references. Keep this in mind when writing the PR description.
- **Video online = multi-image.** Not a shortcut: the reference's own
  `inference_base.py::run_online` sends sampled frames as multiple images and rejects the
  codec backend online.
