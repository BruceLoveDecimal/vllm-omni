# LingBot-World Director Pipeline

A research pipeline that keeps an interactive world on task by composing
several served models. Every model is one `vllm serve` process on its own GPU,
and the Director talks to them over HTTP and WebSocket.

| Role | Model | Endpoint | When it runs |
| ---- | ----- | -------- | ------------ |
| World | LingBot-World 2.0 causal-fast | `WS /v1/realtime/video` | Continuously |
| Standby world | Same | `WS /v1/realtime/video` | Warms up during a restart |
| Sensor | SAM 3.1 | `POST /v1/images/segmentations` | Once per chunk |
| Planner | Qwen3.8-27B | `POST /v1/chat/completions` | Only on events |
| Goal-frame editor | Qwen-Image-Edit | `POST /v1/images/edits` | Only on events |

## What overlaps

- The world never waits for another model. The sensor reads finished chunks on
  its own task, and the VLM and the editor run as background tasks.
- A restart overlaps too when a second world URL is given: the running segment
  keeps streaming until the new segment delivers its first chunk. With one
  world it has to stop first, because a stepwise world serves one session.
- VAE decoding does not overlap the DiT in this deployment. `post_decode` decodes
  inside the same step, so a world's chunk rate already includes its decode.

## Arms

| Arm | Behavior |
| --- | -------- |
| `fixed` | One prompt for the whole rollout. Measures only. |
| `fsm` | The sensor drives the state machine, which switches between the execute and hold prompts. |
| `rebase` | On confirmed success, restarts the world from the best frame showing the goal. |
| `grounded` | Starts the world, edits the goal into the first frame in the background, and restarts from it once the VLM accepts the edit. |

The headline metric is `longest_held_in_one_segment`, which never crosses a
restart, so a restart cannot manufacture a long hold.

## Run

```bash
WORLD_GPU=0 STANDBY_GPU=1 VLM_GPU=2 EDIT_GPU=3 SEGMENT_GPU=3 \
  WORLD_WIDTH=480 WORLD_HEIGHT=832 \
  bash benchmarks/lingbot_director/serve_stack.sh

python benchmarks/lingbot_director/run_director.py \
  --task door-open --arm rebase --seed 11 --chunks 24 \
  --image /path/door.jpg --width 480 --height 832 \
  --world-url ws://127.0.0.1:8000/v1/realtime/video \
  --world-url ws://127.0.0.1:8004/v1/realtime/video \
  --output /path/runs
```

Score the sensor against labelled rollouts before relying on it, because the
ratio limits in `tasks.py` come from SAM 2.1 tracking, not from SAM 3.1:

```bash
python benchmarks/lingbot_director/calibrate_sensor.py \
  --task door-open --runs /path/runs --labels labels.json --mapping mapping.json
```

The first frame of a restarted segment has been through the streamed H.264
encode, so it carries that compression. `facebook/sam3.1` is gated; export
`HF_TOKEN` or point `SEGMENT_MODEL` at a local copy.

## Live agent loop

`live_app.py` turns the same models into an interactive session in the browser,
in the shape of LingBot's Director–Pilot harness:

| Part | Module | What it does |
| ---- | ------ | ------------ |
| User input | `live_static/` | WASD/IJKL camera, U/O/F/G Director cards, 1–9 registered slots, clicking an object card, free-text events |
| Interaction orchestrator | `events.py` | One event queue with priorities (text > object/hotkey/slot > Director); the chunk prompt is background scene + facts + active event; an event's duration counts from the chunk the server reports its prompt applied (`started_event_ids`), and its facts are kept only after the Director confirms it |
| Director (VLM) | `agent.py` | `discover` on scene start, on a clear scene change and every few chunks; `resolve` one trigger into `target / action / preconditions / visual_event / duration_chunks / facts_after / kind / success_check / scene_after / edit_instruction`; `verify` answers the event's `success_check` on the latest frames with yes / partial / no; a failed precondition rejects the event with a reason |
| Event cards | `tracker.py` | SAM keeps each discovered object's box current, with its last box as exemplar, so its card follows it |
| Pilot | LingBot over `WS /v1/realtime/video` | Prompt and camera updates apply at the next chunk boundary; a segment that reaches its frame budget continues from its last frame |

An event goes `sent → playing → verifying`, then:

- **yes**: its `facts_after` and `scene_after` become the new facts and background.
- **partial**, the first time: it plays for one more duration.
- **no**: it retries one rung up. `prompt` switches in scene + facts + event;
  `rewrite` replaces the scene and facts with the Director's `scene_after` and
  `facts_after`, so nothing in the prompt contradicts the event; `reanchor`
  restarts the world from the current frame, edited with `edit_instruction`
  when `--edit-url` points at an image-edit server. A no on `reanchor` drops the
  event with a notice and leaves the facts alone.

Events of kind `appearance` (weather, light, time of day) start at `reanchor`:
on LingBot-World 2.0 causal-fast a mid-session switch from a sunset beach to a
snowy night changed poses but not the look of the scene.

SAM only selects targets and positions cards; it does not condition the world.
Mid-session prompts need the LingBot prompt interaction handler on the server
(`vllm_omni/diffusion/interaction/modality_handlers/lingbot_prompt.py`).

```bash
python benchmarks/lingbot_director/live_app.py --image /path/scene.jpg --port 7860 \
  --record /path/live-runs
```

Super-resolution and frame interpolation are not wired in yet.
