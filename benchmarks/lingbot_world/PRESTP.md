# LingBot World PreSTP experiment

This branch adapts [PreSTP](https://arxiv.org/abs/2611.07551)
(first submitted November 5, 2026) to LingBot World v2's four-step DMD sampler.
The experiment targets generation latency and checks output consistency against
the same model, camera trajectory, input image, prompt, and random seed.

## Method and adaptation

LingBot already computes `x0 = xt - sigma * flow` before re-noising each DMD
step. This clean estimate is a prerequisite for PreSTP; its presence alone does
not implement token pruning.

The optional `model_config.lingbot_prestp` mapping enables two masks: adjacent
frame similarity identifies static patches, and consecutive clean estimates
identify converged patches. Active queries pass through the transformer;
attention retains full K/V, restoring skipped positions from their last
computed values. Skipped output patches retain their last clean estimate.
Cached keys are stored before RoPE and receive the current frame positions.

The first three-latent-frame block runs densely. Later blocks probe once unless
the preceding static fraction permits mask reuse. Periodic reprobes are counted
in blocks. Reused blocks initialize skipped clean patches and K/V from the
preceding block's final frame. These are explicit block-causal adaptations, not
a claim of bitwise equivalence with the paper's implementation. LingBot's extra
t=0 clean-KV commit always computes every token, including positions skipped
by the denoising probes. It refreshes the pruning K/V snapshots as well as the
committed history. An earlier prototype pruned that commit and produced severe
artifacts; its measurements must be kept separate from this implementation.

Defaults are spatial threshold `0.97`, temporal threshold `0.997`, reuse gate
`0.5`, and reanchor period `3`. The feature is disabled when the mapping is
absent. Dynamic token gathering currently requires eager execution, batch one,
and sequence-parallel size one. Last-step K/V reuse, shared first-block K/V,
and leased post-attention cannot be combined with this experiment.

State is request/session owned. A text-conditioning change discards pruning
state; session cleanup releases it. Paged session admission includes retained
pruning K/V and clean-estimate storage. Camera changes can invalidate the
static-content prediction between reprobes; camera-response quality must be
checked when evaluating mask reuse.

## Matched offline benchmark

Run the same command twice, first without `--prestp`, then with it and a
different result directory:

```bash
python -m benchmarks.lingbot_world.benchmark_prestp \
  --model /path/to/complete/checkpoint \
  --image /path/to/frame.jpg --action-dir /path/to/actions \
  --prompt 'A slow panoramic sweep around Stonehenge.' \
  --height 320 --width 512 --num-frames 81 --seed 42 \
  --enforce-eager --repetitions 2 --result-dir /tmp/lingbot-dense
```

The harness reuses the existing offline recipe and saves lossless pixel arrays,
MP4s, configuration, dependency versions, and generation wall time. Model load
and artifact encoding are excluded; prompt/image encoding and VAE decoding are
included. Keep the first cold request separate from the subsequent warm request.
These offline times do not establish realtime streaming cadence or reproduce
the paper's DiT-only latency. Use the existing realtime benchmark for deployment
claims, and inspect motion/control response before treating pixel similarity as
evidence of acceptable quality.

## Measured pilot (November 7–8, 2026)

Base: `2b610b3e9c5f7d6de7388a5f5394d88e3273f4f3`, plus this branch's
prototype. GPU: one RTX PRO 6000 Blackwell Server Edition, 97,887
MiB. Runtime: PyTorch `2.14.0+cu130`, vLLM `0.33.0`, eager, 320×512, seed 42.
The local checkpoint is `lingbot-world-v2-14b-causal-fast-diffusers` under the
instance's `old-instance-data/models` directory. Its upstream weight revision
was not established; comparisons used the same files throughout.

| Final implementation / reprobe period | Dense generation | PreSTP generation | Speedup | Pixel PSNR vs dense |
| --- | ---: | ---: | ---: | ---: |
| Stonehenge, 81 frames / 3 (default) | 9.683 s | 6.647 s | 1.46× | 17.71 dB |
| Stonehenge, 81 frames / 1 | 9.683 s | 7.165 s | 1.35× | 17.13 dB |
| Alley, 177 frames / 1 | 21.282 s | 19.595 s | 1.09× | 13.30 dB |

Times exclude each process's first request. The Stonehenge baseline has one
warm observation; the other displayed times average two warm observations.
This small pilot does not establish a stable distribution or tail latency.
The two baseline Stonehenge pixel arrays were exactly equal. Both final
scenes retained exact equality for the first nine pixel frames, before
pruning starts. PSNR uses unit-range pixels against the matched dense output;
it measures divergence, not perceptual quality or camera-pose accuracy.

Inspecting eight sampled frames per video found no severe block corruption in
these final runs, but visible scene/geometry changes remain, especially in the
longer alley sequence. Temporal pixel-difference magnitude changed from
0.01583 to 0.01083 (period 3) / 0.01196 (period 1) for Stonehenge and from
0.03208 to 0.05028 for the alley.
These are descriptive differences, not a validated motion-quality score.
The pilot supports content-dependent acceleration, not default production
enablement or equivalence with dense control response.

Earlier runs that also pruned the clean-KV commit are superseded: Stonehenge
with period 3 measured 1.55× and 16.36 dB but exhibited severe block artifacts;
period 1 measured 1.22× and 15.22 dB. The earlier alley period-1 run measured
1.08× and 14.48 dB with visible artifacts. Those numbers describe the earlier
prototype, not the final source. They motivated preserving dense t=0 commits.

Final targeted regressions: **252 passed, 3 skipped** across the pipeline,
attention, transformer, and PreSTP tests. Ruff and `git diff --check` passed.
The live tests covered offline direct-cache generation. Paged realtime
execution, multiple sessions, long-horizon VBench, and quantitative camera-pose
following have not been GPU-validated. This branch remains experimental and
is not declared ready for a production PR.
