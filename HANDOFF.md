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
fixes and the safe half of the reuse are committed **but could not be run**, because the
GPU box will not power on (account balance). Your first job is to re-verify those five
commits, not to write new code.

---

## 2. Do this first

```bash
python3 .claude/skills/autodl-gpu/scripts/autodl.py status pro-787016ace1d3
```

If it still reports insufficient balance on power-on, stop and tell the user — nothing
below can be verified without it. Once it boots, run §5 in order.

**The last five commits are unverified.** In commit order:

| Commit | Risk | Why it needs a run |
|---|---|---|
| `8ca45ea8` deploy config `stage_id`/`pipeline` | high | Fixes a hard `KeyError`; the yaml path was **never exercised** (all validation went through bare `vllm serve`, which bypasses deploy configs entirely). Verify with `vllm-omni serve`. |
| `9128219e` `keep_on_cpu` + `temporal_patch_size` | low | Behavioural, not numerical. |
| `e4a1606f` reuse Qwen2-VL merger + MLP | **highest** | Touches the numerical path. Structure was verified against v0.27.0 source line by line, but never executed. |
| `a5994c5c` `__init__` exports | none | Import hygiene. |
| `efc0a305` test additions | none | CPU-only. |

---

## 3. Machine and environment

AutoDL Pro instance `pro-787016ace1d3` — RTX 5090, 32 GB. Drive it with the
`autodl-gpu` skill (`.claude/skills/autodl-gpu/scripts/autodl.py`), not the console.

```bash
S=.claude/skills/autodl-gpu/scripts/autodl.py
python3 $S on pro-787016ace1d3 --payload gpu --wait   # ALWAYS pass --payload gpu
python3 $S info pro-787016ace1d3                      # ssh command + password
python3 $S off pro-787016ace1d3 --wait                # ← the box bills while running
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

Measured before the unverified commits:

| Check | Result |
|---|---|
| Vision tower, fp32, same bf16 weights upcast | `rel_l2` **4.19e-6**, cosine **1.0000000000** |
| Vision tower, bf16, end to end | cosine 0.9998, `rel_l2` 2.0e-2 — accumulation, not a bug (see §6) |
| Weight loading | 297/297 tower params; full model no gaps |
| Offline generation | **3/3 cases token-identical**, greedy |
| Online `/v1/chat/completions` | **3/3 token-identical**, streaming 24 deltas |
| Video (frame-sampled multi-image) | token-identical; 4 concurrent requests → 1 distinct output |
| CPU tests | 8/8 (now 11 with the new ones, unrun) |

**Never verified at all:** TP > 1 (single-GPU box), `trust_remote_code=False` (every run
passed `--trust-remote-code`, so the vendoring goal is designed-for but unproven), and
the `vllm-omni serve` deploy-config path.

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

## 8. Reuse left on the table

An audit produced these; the two below were **deliberately not done** because they depend
on transformers *runtime* behaviour that could not be checked without the box:

- `MageVLProcessor` → subclass `transformers.models.qwen2_vl.Qwen2VLProcessor` (≈ −55
  lines). Its `__call__` already does the same placeholder expansion. Blocker to check:
  in transformers 5 the class's `attributes` include `video_processor`, and the Mage
  checkpoint ships no video preprocessor config.
- `MageVLProcessingInfo` / `MageVLDummyInputsBuilder` → subclass the Qwen2-VL ones
  (in-repo precedent: `ming_flash_omni_thinker.py:106`). This is a **correctness** fix,
  not just line count: the current `get_dummy_mm_data` hardcodes 448×448 while upstream
  uses `get_image_size_with_most_features()`, so memory profiling under-provisions the
  encoder (the checkpoint's `max_pixels` is 4,000,000). `temporal_patch_size` was already
  added to the config to unblock this.

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
