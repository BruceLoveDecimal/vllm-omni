# LingBot–VLM integration experiment

This branch checkpoints the GPU-tested semantic-event prototype and its
fixed-prompt comparison. It is an experiment handoff, not a production serving
integration or a full reproduction of LingBot's official Director–Pilot stack.

The Director runs Qwen3-VL-8B-Instruct through Transformers on the same GPU as
LingBot-World-v2-14B-causal-fast. It observes the initial reference and three
frames from the latest generated chunk, proposes a semantic event, and updates
the next ARDiffusionSession prompt. Client-side streaming VAE decoding supplies
the next observation. The fixed arm retains the same initial VLM-generated
prompt; the dynamic arm observes and revises it after each chunk.

## Results and evidence

Start with [visual review](semantic_door_20261101/VISUAL_REVIEW.md), then
[method](semantic_door_20261101/METHOD.md) and
[raw automated assessment](semantic_door_20261101/REPORT.md).

- One door photograph, three paired seeds, two memory policies: 12 videos,
  192 chunks, approximately 11.8 seconds per video.
- With sliding-window and sink memory, dynamic seeds 42 and 2026 visibly open
  at the end; seed 123 has geometry artifacts. Extended-memory trials do not
  show reliable final persistence.
- The same VLM also judges outputs with a separate blinded prompt. It makes
  mistakes, including accepting geometry artifacts and rejecting an open door.
  Its automatic score is not ground truth. Visual inspection covers sampled
  frames, not an independent human evaluation of every video frame.
- The memory follow-up changed several settings together after observing the
  first results. This does not isolate retrieval or establish long-term gains.

The dated directory preserves the original executed scripts and result files
byte-for-byte, including their environment-specific paths and formatting.
`snapshot-sha256.json` records the checked-in snapshot. Full per-chunk PNGs and
unannotated rollout videos remain in the experiment environment; paired videos,
contact sheets, tails, final images, prompts, traces and summaries are included.
Model weights, caches, credentials and machine logs are excluded.

## Runtime prerequisite

**A clean checkout of this branch is not sufficient to reproduce GPU results.**
The experiment ran against a separate WorldKV runtime snapshot at
`/root/worldkv-fixes/repo`, with the vLLM overlay at
`/root/worldkv-fixes/runtime32`. That directory has no verifiable Git HEAD.
The runtime manifest contains inherited commit labels, not proven provenance;
the recorded source hashes are the actual evidence boundary. In particular,
the runtime's `video_memory.py` and related changes are not in this branch.
Do not replace that runtime with this checkout and assume equivalent behavior.

Recorded environment: one RTX PRO 6000 Blackwell Server Edition (96 GB),
Python 3.12.3, PyTorch 2.13.0+cu130, Transformers 5.14.1, Diffusers 0.40.0 and
vLLM 0.32.0 overlay. The Omni/runtime version mismatch remains unqualified.
See [runtime manifest](semantic_door_20261101/runtime.json) for source hashes.

On the existing experiment machine, the original inputs, model weights and
scripts are under `/root/autodl-tmp/t7-semantic-door-20261101` and
`/root/autodl-tmp/models`. The recorded launch command is in METHOD.md.
`semantic.py` runs extended memory; `semantic_sliding.py` runs the sliding-only
directory using its own `deploy.yaml`. Both scripts overwrite result files in
their configured output directory: use a new output directory and copy the
input and configuration there before starting a new run.

Offline assessment can be recomputed without model dependencies:

```sh
python3 benchmarks/lingbot_vlm/semantic_door_20261101/summarize.py
```

This overwrites REPORT.md and metrics.json from the saved traces. `export.py`
requires the full original PNGs and rollout videos, which are not all included
in Git. All photo-derived media use the attributed CC BY 2.0 input documented
in METHOD.md; generated videos are not recordings of a real door opening.

## Handoff validation and remaining work

The original GPU runs completed, applied every event, produced finite images,
and matched first-chunk hashes for all six fixed/dynamic pairs. Packaging checks
verify file hashes, Python syntax, trace consistency and offline summary
reproduction. They do not revalidate this Git revision as a GPU runtime.

Before an upstream PR, consolidate the two experiment launchers, parameterize
paths, validate model response schemas, harden initialization/cleanup, and
port or identify the WorldKV dependencies at a reproducible revision. The
archived scripts use synchronous local inference and CUDA timing calls and
are not concurrency-safe serving implementations. They retain their original
formatting rather than claiming to pass repository lint/pre-commit gates.

This experiment and documentation were produced with AI assistance and have
not undergone independent human code review.
