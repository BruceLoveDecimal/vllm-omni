# Semantic door event experiment

## Asset

Initial photograph: [Closed white door.jpg](https://commons.wikimedia.org/wiki/File:Closed_white_door.jpg), by [Shixart1985](https://commons.wikimedia.org/wiki/User:Shixart1985), licensed [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Downloaded Wikimedia's 500px thumbnail; resized to 320x480 for inference. Generated videos are model-derived transformations, not recordings of the real door opening.

## Design fixed before results

- Qwen3-VL-8B-Instruct BF16 Director; LingBot-World-v2-14B-causal-fast generator, same existing vLLM-Omni source snapshot and weights as the preceding experiment.
- One real closed-door input; matched seeds 42, 123, 2026; two arms; 16 generated chunks per arm (189 frames, 11.8125 seconds at 16fps). Fixed camera, identical KV settings, resolution and initial conditions.
- Both arms share the same initial VLM-generated event prompt. Fixed retains it throughout. Dynamic rewrites the next event after observing the initial reference and the latest chunk's first/middle/last frames, plus its last three events.
- There is no task counter, scripted opening schedule, oracle action, reference answer, or forced door-state override in the controller. Dynamic prompts are applied through real session prompt updates. All camera actions are neutral.
- Exact first-chunk pixel hashes must match for each pair. This is a check of matched starting generation, not proof of subsequent semantic correctness.
- The full generation budget is used even if the Director claims success, so persistence and false-success claims remain observable. When the Director sees an open door it is instructed to keep it open.

## Evaluation

After generation, a separate stateless evaluation prompt is applied in shuffled order. It sees the reference image and current chunk images, with no arm/seed, intended action, Director claim or event history. It classifies closed/ajar/open/missing/uncertain, identity preservation, visible aperture and visible door leaf. This is a blind prompt using the SAME model, not an independent evaluator or human ground truth.

Strict provisional success requires `open`, preserved identity, visible aperture and visible door leaf in both final chunks. Any two consecutive successful chunks are recorded separately from final success. Contact sheets and disputed frames are inspected visually before reporting results. A disappeared door, changed scene, narrow crack or dark texture does not count as fully open.

Limitations: one scene, three seeds, small sample, no statistical significance claim; no full official deployment reproduction or SAM tracking. Runtime uses the internal ARDiffusionSession API and client-side streaming VAE decode. Prompt edits must be distinguished from visible physical success. Baseline already benefits from the VLM's initial prompt, so the contrast isolates subsequent visual feedback and rewriting as a package, not all possible VLM benefits.

## Follow-up memory ablation

The first six episodes repeatedly opened and closed the door and all failed final persistence. A second, explicitly adaptive six-episode run under `sliding-only/` removes retrieval, world pruning and PC compression settings, retaining only window_frames=18 and sink_frames=3. Seeds, prompts, controller, input and evaluation stay the same. This tests the package of extended memory features, not retrieval alone; any difference cannot be attributed to one component without another ablation. It does not establish full parity with official native inference.

The automated success criterion specifically uses the classified FINAL frame of each of the last two chunks. Six sampled frames from those chunks are also exported as `tail.jpg` to inspect intra-chunk reopening/closing. Two positive endpoints alone do not prove uninterrupted openness.

## Run

```sh
cd /root/autodl-tmp/t7-semantic-door-20261101
setsid nohup env PYTHONPATH=/root/worldkv-fixes/runtime32:/root/worldkv-fixes/repo /root/vllm-omni-env/bin/python semantic.py >experiment.log 2>&1 </dev/null &
```
