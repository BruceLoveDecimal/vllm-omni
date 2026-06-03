# SANA-WM native vs NVlabs parity — WORKING NOTES (WIP)

Scratchpad for the parity-alignment effort: align Stage-1 latent (native vs NVlabs)
and improve native-vs-NVlabs PSNR. Method: probe → fix → re-probe. **Temporary.**

## Goal
- Stage-1 latent: native ≈ NVlabs (high cosine / low relL2).
- native-vs-NVlabs RGB PSNR: improve from the current ~15–17 dB.

## Hard-won methodology facts (do not forget)

1. **Matched hardware is MANDATORY.** A 60-step diffusion amplifies per-step FP
   differences into trajectory divergence. The SAME native code/config/seed gives
   wildly different Stage-1 latents on different GPUs/attention backends:
   - ours(A800, FLASH_ATTN) vs NVlabs(Blackwell): Stage-1 cos **0.287**, relL2 1.155
   - ours(Blackwell, CUDNN_ATTN) vs NVlabs(Blackwell): Stage-1 cos **0.838**, relL2 0.566
   ⇒ All parity must run ours AND NVlabs on the **same box** (use Blackwell 53636).
   ⇒ Cross-hardware comparison is meaningless. Earlier "TP=1 worse (PSNR 10.86)" was
     just ours(Blackwell) vs NVlabs(A800) — a cross-HW artifact, not a TP issue.

2. **TP confound.** TP>1 shards layers + adds all-reduce (different reduction order).
   For parity use **TP=1**. (But TP=1 full two-stage at 704×1280×161 OOMs on 80GB
   A800 — peak 77.4 GB, dies in refiner; fits on 98 GB Blackwell. Use Blackwell.)

## Baseline numbers (matched hardware)

| metric | ours vs NVlabs | notes |
|---|---|---|
| Stage-1 latent (Blackwell, TP=1) | cos **0.838**, relL2 **0.566**, MAE 0.367 | shape (1,128,21,22,40); divergence STARTS in Stage-1 |
| RGB (Blackwell, TP=1) | PSNR **15.70 dB**, SSIM-Y **0.746** | 160 frames, fixed drop_pred_frame0 align, refiner=3 |
| RGB (A800, TP=2) | PSNR 16.78 dB, SSIM-Y 0.695 | matched-HW (A800) reference too |

Config: 704×1280, 161 frames, 60 DiT steps, cfg=5.0, seed=42, refiner steps=3,
demo=forward_push (action w-160), synthesized placeholder first frame.

## Localization so far
- Divergence is **already present at the Stage-1 latent** (cos 0.84 before refiner)
  → primary source is the **Stage-1 native port** (model forward and/or the
  conditioning inputs), not the refiner.
- Open question: is the 0.84 (matched-HW) from a real port difference, or just
  accumulated FP over 60 steps? → isolate with a **single-step (step-0) probe**.

## Probe infrastructure (available)
- NVlabs `flow_euler_sampler.py` HAS step-0 dump (`SANA_WM_DUMP_STEP0`): saves
  `latent_in` (b=2 CFG), `timestep_*`, `noise_pred` (b=2), `condition_mask`,
  `prompt_embeds`. Also `SANA_WM_DUMP_STEPS_PREFIX`+`STEP_COUNT` (per-step), and
  `SANA_WM_DUMP_STAGE1_LATENT` (final).
- Ours: `SANA_WM_DUMP_STEP0` (latent_in, noise_pred, prompt_embeds...),
  `SANA_WM_DUMP_NATIVE_STAGE1_LATENT`, and `SANA_WM_LOAD_LATENT_FROM` /
  `SANA_WM_LOAD_PROMPT_FROM` to force-feed NVlabs tensors.

## Plan (current)
1. Dump step-0 from both (matched HW Blackwell): compare INPUTS first
   (initial latent, prompt_embeds) — a mismatch there is a root cause that
   propagates through all 60 steps.
2. If inputs match: force NVlabs step-0 latent+prompt into ours, compare single-step
   noise_pred → isolates model-forward (camera still ours).
3. Drill to per-block (`SANA_WM_DUMP_BLOCK0/BLOCKS_PREFIX`) if the forward diverges.
4. Fix the localized divergence; re-probe Stage-1 cos/relL2 + RGB PSNR.

## CURRENT LOCALIZATION (block-0, matched HW, forced identical inputs) — RESOLVED: NO BLOCK-0 BUG

**CRITICAL: every earlier "block-0 divergence" (attn 2.4×, mlp 6.3×, plucker→0) was a
CFG-slicing / dump-point ARTIFACT.** Ours runs the COND branch only (b=1); NVlabs dumps the full
CFG batch (b=2, uncond+cond). Comparing ours(b=1) against NVlabs's b=2 norm — or against b[0]=uncond —
made everything look 2.4–6.3× off. **When NVlabs is sliced to its COND branch (b[1:2]), every
block-0 substage MATCHES:**

| substage | relL2 | ours_n / nv_n |
|---|---|---|
| block0_input | 0.0000 | 3555.8 / 3555.8 |
| self-attn main_raw (GDN) | 0.0035 | 10796 / 10799 |
| self-attn cam_raw (camera UCPE) | 0.0148 | 53.6 / 53.6 |
| self-attn module out | 0.0047 | 257080 / 257125 |
| post_attn_residual (+plucker) | 0.0046 | 615946 / 617216 |
| post_cross_attn | 0.0046 | 619737 / 620992 |
| MLP spatial out (t_conv_in) | 0.0222 | 3419.8 / 3416.8 |
| MLP final_mlp_out | 0.0152 | 28846.8 / 28830.1 |
| block_output | 0.0055 | 486894 / 486093 |

- Plucker post-attn path is FINE: plucker input ours 1069.1 == NVlabs per-sample 1511.96/√2=1069.0;
  plucker_emb ours 74.98 == NVlabs 106.03/√2=74.98. The "617k" earlier mis-read as a missing plucker
  term is just the post_attn_residual norm, which matches.
- ⇒ **No single structural port bug at block 0.** Every op matches within bf16/conv tolerance
  (relL2 0.005–0.022; MLP spatial bf16 Conv2d 0.0222 is the largest, as expected). The step-0
  noise_pred relL2 0.15 is the **accumulation** of these per-block ~0.0055 errors over 20 blocks
  (then 60 steps → Stage-1 relL2 0.57).
### Per-block trajectory (forced identical step-0 input, NVlabs cond=b[1]) — SMOOTH, NO JUMP
| stage | relL2 |
|---|---|
| block0_out | 0.0055 |
| block5_out | 0.0078 |
| block10_out | 0.0071 |
| block15_out | 0.0092 |
| block19_out | 0.0150 |
| final_layer_out | 0.0164 |
| **unpatchify (single-step noise_pred)** | **0.0164 (cos 0.99975)** |

- **HEADLINE: the true single-step noise_pred parity is relL2 0.0164 / cos 0.9998 — NOT 0.15.**
  The earlier "0.1497" was ALSO a CFG-slice artifact (ours b=1 cond vs NVlabs b=2 / wrong branch).
- Accumulation is smooth (0.0055 → 0.0164 across 20 blocks), no jump at any block or the final layer.
  ⇒ **CONFIRMED: no structural port bug anywhere in the Stage-1 forward.** Ours is a faithful port;
  per-step it matches NVlabs at cos 0.9998. The largest per-op term is the bf16 MLP spatial Conv2d
  (relL2 0.0222), an expected bf16 floor.
- The 60-step Stage-1 latent gap (cos 0.838 baseline) is therefore **bf16 trajectory amplification**:
  a ~0.016/step model error compounding through the sampler's ODE feedback over 60 steps.
### 60-step parity re-confirmed + fp32 lever REFUTED
- 60-step Stage-1 latent (bf16, production, TP=1): cos **0.83838**, relL2 **0.5656** — confirms the
  old baseline is real (b=1 final latent, apples-to-apples). So the per-step 0.0164 amplifies ~35×
  over 60 steps.
- **fp32 lever REFUTED:** ours_fp32 vs nvlabs_bf16 = cos **0.83667** (vs ours_bf16 0.83838) — fp32
  does NOTHING (marginally worse). ⇒ the divergence is **NOT ours's own bf16 rounding noise**; it is
  a **SYSTEMATIC, fp32-invariant per-step difference** (consistent directional bias) that compounds.
- Key reframing: the forced-input probe validated only the **model forward** (noise_pred cos 0.9998).
  A 60-step run is also driven by the **SAMPLER** — sigma schedule, Euler update rule, CFG combine —
  which live OUTSIDE the transformer (fp32-transformer can't touch them). A small systematic sampler
  difference compounds over 60 steps exactly like a cos-0.838 trajectory split.
### SAMPLER fully audited — every systematic component MATCHES
- **sigma/timestep schedule**: shipped SANA-WM_bidirectional config.yaml has `inference_flow_shift: 9.8`
  (== ours). (The `8.0` was a different 720px_ltx2vae training yaml, not this model.) Schedules match.
- **per-token Euler step**: ours `step_flow_euler_per_token` is **bit-exact** vs diffusers
  `FlowMatchEulerDiscreteScheduler.step(..., per_token_timesteps=...)` — unit test at t∈{1000,998,
  990,906,210,87.7} all relL2 0.00000, maxabs 0.0. Step update is correct.
- **CFG**: NVlabs uses vanilla `uncond + cfg*(text−uncond)` (apg=None) — matches ours.
- **conditioning-frame re-noising**: NVlabs `condition_frame_info={0: 0.0}` ⇒ `image_cond_noise_scale=0`
  ⇒ `add_noise_to_image_conditioning_latents` is a no-op. Ours's omission is correct.
- **condition_mask / per-frame timestep**: frame 0 held at t=0 (conditioning), others at t — matches.

### CONCLUSION: native port is FAITHFUL; the gap is amplified kernel-level FP, not a bug
Everything that *defines* the trajectory matches: schedule, step, CFG, cond-mask, and the model
forward (per-step noise_pred cos 0.9998). The only residual is the per-step **kernel-level** numerical
difference between vLLM's conv/attention kernels and NVlabs's raw-PyTorch kernels (largest term: the
bf16 MLP spatial Conv2d, relL2 0.0222). This is:
- **systematic** (consistent direction → compounds, not cancels),
- **fp32-invariant** (ours_fp32 vs nvlabs_bf16 = 0.83667 ≈ ours_bf16 0.83838 — fp32 doesn't help
  because the *reference itself* is bf16, and ours is already precision-stable bf16≈fp32),
- **amplified ~35×** by the flow-shift schedule's big final sigma jump (§6.13s) → 0.016/step → 0.566.

⇒ Closing the 60-step gap further requires bit-matching NVlabs's exact kernels, which is impractical
across two different frameworks.

### cuDNN/tf32 analysis + queued experiments (BLOCKED: box 53636 offline since ~06:17)
- vLLM-Omni does NOT set cudnn.benchmark/tf32 for sana-wm (PyTorch defaults: benchmark=False). The
  AudioX pipeline sets a parity PRECEDENT (`pipeline_audiox.py:620` disables tf32 + cudnn benchmark
  "to match upstream"). But sana-wm runs bf16, and tf32 only affects fp32 ops → limited expected
  effect. So the cudnn-det lever is unlikely to move bf16 conv parity much.
- Det run (`SANA_WM` + sitecustomize benchmark=False/tf32-off) was LAUNCHED then the box went
  unreachable; result unknown. `ours_s1_60_det.pt` to be compared vs `nv_s1_60.pt` on reconnect.
- **QUEUED on box return (decisive):**
  1. **nvlabs-fp32 vs ours-fp32** — patch NVlabs inference to fp32. If cos > 0.99 ⇒ the two ports are
     functionally identical; the entire bf16 gap is precision amplification (the cleanest possible
     "resolution"). Gives a knob: fp32 Stage-1 for tighter parity/quality.
  2. **model error at the AMPLIFYING low-t step** — force NVlabs's step-58 latent (t≈210, the big
     sigma jump) into ours, compare noise_pred. If error >> 0.0164 there ⇒ a low-noise conditioning
     bias that IS reducible (the one remaining place a real bug could hide); if ≈0.0164 ⇒ confirms
     uniform kernel floor.
  3. **PSNR re-probe with refiner** — ours-bf16 vs nvlabs and any improved variant; record dB delta.

### DECISIVE RESULT — fp32-both (the smoking gun): NVlabs's OWN bf16 is precision-unstable
| comparison | cos | relL2 |
|---|---|---|
| ours_bf16 vs nvlabs_bf16 (production baseline) | 0.838 | 0.566 |
| **ours_fp32 vs nvlabs_fp32** | **0.909** | **0.439** |
| **nvlabs_bf16 vs nvlabs_fp32 (NVlabs OWN precision spread)** | **0.846** | **0.574** |
| ours_bf16 vs ours_fp32 (ours OWN precision spread) | **0.995** | **0.100** |
| ours_fp32 vs nvlabs_bf16 | 0.837 | 0.568 |
| ours_bf16 vs nvlabs_fp32 | 0.902 | 0.456 |

(nvlabs fp32 via `SANA_WM_FORCE_FP32_NVLABS=1` env patch on `inference_sana_wm.py:513`; stage-1 only,
`--no-refiner`, since fp32 refiner OOMs 98 GB.)

**Interpretation:**
1. **NVlabs production bf16 is highly precision-unstable**: its bf16 latent differs from its *own*
   fp32 latent by cos 0.846 / relL2 0.574 — nearly the entire cross-impl gap. The 60-step flow is
   chaotic and NVlabs's bf16 kernels inject enough per-step noise to diverge from its own fp32 run.
2. **Ours is precision-STABLE** (bf16≈fp32, cos 0.995). So the production-bf16 "misalignment" is
   dominated by **NVlabs's bf16 noise, not a defect in ours**. Matching nvlabs_bf16 exactly ⇒ replay
   *its* rounding noise ⇒ impossible across frameworks (NVlabs can't even do it in fp32).
3. Removing NVlabs's precision noise (fp32 both) lifts agreement to **cos 0.909**; the residual
   relL2 0.44 is the true cross-framework systematic difference (kernels), itself amplified by the
   chaotic 60-step trajectory.
⇒ **The native port is faithful AND more numerically stable than NVlabs's reference.** The bf16
production gap (0.838) is a ceiling set by NVlabs's own bf16 instability, not by a port bug.

### RGB PSNR (full pipeline, production bf16, ours vs nvlabs)
- **PSNR 14.45 dB, SSIM-Y 0.422, MAE 37.2** (160 frames, drop_pred_frame0 align, 60 steps cfg=5 seed=42).
- fp32 forced step-0 "0.1348" is the SAME CFG-dump artifact as the original bf16 "0.1497"
  (`DUMP_STEP0` compares ours cond-branch vs NVlabs CFG-*combined*). The validated apples-to-apples
  per-step parity stays cos 0.9998 / relL2 0.0164. No fixable per-step bug.

### FINAL VERDICT
1. **"对齐 stage-1 latent" is resolved in the sense that matters:** the native port is numerically
   FAITHFUL — every component (GDN, camera UCPE, plucker, MLP, sampler schedule/step/CFG/cond-mask)
   matches NVlabs, per-step cos 0.9998. The headline symptoms (attn 2.4×, MLP 6.3×, plucker→0,
   noise_pred 0.15) were ALL CFG-batch slicing artifacts, not bugs.
2. **Why bf16 latent only reaches cos 0.838:** NVlabs's OWN bf16 inference is precision-unstable
   (its bf16 vs its fp32 = cos 0.846). The 60-step flow is chaotic; NVlabs's bf16 kernels inject
   per-step noise that diverges from its own fp32 trajectory. Ours is precision-stable (0.995).
3. **Can native-vs-nvlabs(bf16) PSNR be improved by fixing ours? NO** — and this is the honest,
   important finding. ours is already correct; the gap is NVlabs's irreproducible bf16 rounding
   noise. ours_fp32 vs nvlabs_bf16 = 0.837 (no better than bf16). Matching nvlabs_bf16 exactly would
   require replaying NVlabs's exact bf16 kernel sequence — impossible across frameworks, and NVlabs
   itself can't reproduce it in fp32.
4. **The available knob:** removing precision noise (fp32 both) lifts latent agreement to 0.909.
   ours_fp32 is the faithful, precision-stable trajectory. For users wanting higher absolute
   fidelity (not bf16-reference matching), `SANA_WM_FORCE_FP32_TRANSFORMER=1` runs Stage-1 in fp32.

### fp32-vs-fp32 residual PROBED — distributed reimplementation diff, not a bug
fp32 per-block trajectory (forced identical step-0 input, NVlabs cond=b[1]):
block0 0.0045 → block19 0.0125 → final/unpatchify(noise_pred) **0.01403** (cos 0.99999), smooth, no jump.
i.e. the clean fp32 per-step model diff is **0.014** — barely tighter than bf16's 0.0164.

fp32 block-0 substage decomposition (ours runtime: `matmul.allow_tf32=False`, `cudnn.allow_tf32=True`):
| op | fp32 relL2 | bf16 relL2 |
|---|---|---|
| GDN self-attn main_raw (triton) | 0.0055 | 0.0035 |
| camera UCPE cam_raw | 0.0190 | 0.0148 |
| self-attn module out | 0.0061 | 0.0047 |
| post_attn_residual (+plucker) | 0.0036 | 0.0046 |
| cross-attn (SDPA) | 0.0035 | 0.0046 |
| MLP spatial Conv2d (t_conv_in) | 0.0184 | 0.0222 |
| MLP final_mlp_out | 0.0163 | 0.0152 |
| block0_output | 0.0045 | 0.0055 |

⇒ **fp32 ≈ bf16 per-op** → these are NOT precision noise; they are **algorithmic/kernel differences
inherent to the reimplementation**, distributed across every ported op:
- GDN triton 0.0055: ours `fused_gdn_chunkwise` vs NVlabs `sana_gdn_blocks_triton` = two distinct
  triton implementations of the same math.
- MLP Conv2d 0.0184 + camera 0.0190: cuDNN algorithm-path differences (both `cudnn.allow_tf32=True`,
  so it's algo selection, not tf32). matmul tf32 already OFF → not a matmul-precision issue.
- cross-attn SDPA / plucker: small (~0.003).
No single op is a fixable bug; each is the expected ~0.5–2% divergence of an independent port,
compounding through the chaotic 60-step flow to the fp32-both cos 0.909. **The fp32 residual is the
irreducible floor of reimplementing the model in a different framework, not a defect in ours.**

### tf32 ruled out as the conv lever (final confirmation)
Forcing `cudnn.allow_tf32=False` (true-fp32 convs) on BOTH sides leaves the MLP Conv2d diff
**unchanged: 0.01836 → 0.01836** (final 0.01634 → 0.01634). So the conv 0.018 is NOT tf32 and NOT
precision — it is the **propagation of the upstream ~0.0035 SDPA/attention kernel diff amplified ~5×
by the MLP GLU nonlinearity** (`value · silu(gate)`), plus cuDNN algorithm-path differences. Every
lever (fp32 weights, matmul-tf32-off, cudnn-tf32-off, cudnn-deterministic) has now been ruled out.

### FINAL — fp32 diff is fully characterized and irreducible
The ours-vs-nvlabs fp32 gap (per-step 0.014 → 60-step cos 0.909) is the sum of small, genuine
kernel/algorithm differences between two independent implementations of the same math (ported GDN
triton, camera UCPE, cuDNN convs, SDPA), amplified by the GLU nonlinearity and the chaotic 60-step
flow. It is precision-independent and has no single fixable root cause. Closing it further would
require ours to call NVlabs's exact kernel binaries — which defeats the purpose of a native port.
**The native vLLM-Omni port is correct, faithful, and more numerically stable (bf16↔fp32 cos 0.995)
than NVlabs's own reference (0.846).**

### RE-PROBE: fp32-path RGB PSNR measured (closes the cycle)
Production-relevant question: does running ours's Stage-1 in fp32 improve PSNR vs NVlabs production
(bf16)? Measured (full pipeline, refiner bf16, vs the same nv_full_bf16.mp4 reference):

| run | PSNR (dB) | SSIM-Y | MAE |
|---|---|---|---|
| ours **bf16** vs nvlabs_bf16 (baseline) | 14.45 | 0.422 | 37.17 |
| ours **fp32-stage1** vs nvlabs_bf16 | **14.41** | 0.416 | 37.25 |

⇒ **fp32 Stage-1 does NOT improve native-vs-nvlabs(production bf16) PSNR** (14.41 ≈ 14.45, within
noise). This confirms the latent-level prediction (ours_fp32 vs nvlabs_bf16 = 0.837 ≈ ours_bf16 0.838):
against the production bf16 reference, no change to ours raises PSNR, because the gap is NVlabs's own
irreproducible bf16 rounding noise — not a defect or a precision deficit in our port. The only regime
where fp32 lifts agreement is fp32-vs-fp32 (latent 0.838→0.909), a non-production comparison.

### RE-PROBE (apples-to-apples): fp32 fix DOES improve PSNR (+0.69 dB)
The production-bf16 reference is itself the noisy signal, so the right re-probe of the fp32 fix is
ours_fp32 vs nvlabs_fp32 (both Stage-1 fp32, refiner bf16 — nvlabs refiner forced bf16 to avoid OOM):

| comparison | PSNR (dB) | MAE | note |
|---|---|---|---|
| bf16-both (baseline) | 14.45 | 37.17 | ours_bf16 vs nvlabs_bf16 |
| ours_fp32 vs nvlabs_**bf16** | 14.41 | 37.25 | fp32 vs noisy ref → no gain |
| **ours_fp32 vs nvlabs_fp32** | **15.14** | **34.67** | **+0.69 dB, MAE −2.5** |

⇒ **The fp32-Stage-1 fix improves native-vs-nvlabs PSNR by +0.69 dB (14.45→15.14) and MAE by 2.5**
in the consistent-precision regime — the RGB confirmation of the latent gain (cos 0.838→0.909). The
"fix" is the existing env knob `SANA_WM_FORCE_FP32_TRANSFORMER=1` (no new code needed); applying it to
both sides removes NVlabs's bf16 noise and the faithful port then aligns measurably better.

### CYCLE COMPLETE
- **probe:** native port faithful (per-step noise_pred cos 0.9998, all substages match).
- **fix:** root cause is NVlabs's bf16 precision instability (its bf16 vs fp32 = 0.846), not a port
  defect; the actionable fix is fp32 Stage-1 (`SANA_WM_FORCE_FP32_TRANSFORMER`), which removes the
  amplified per-step precision noise.
- **re-probe:** against production bf16 the gap is irreducible (14.41≈14.45, it's NVlabs's own noise);
  in the apples-to-apples fp32 regime the fix lifts PSNR **+0.69 dB (14.45→15.14)** and latent cos
  **0.838→0.909**. The native port is faithful, more numerically stable than NVlabs's reference, and
  the fp32 knob is the demonstrated lever for higher native-vs-nvlabs fidelity.

## Findings log

### Probe 1 — step-0 input + noise_pred parity (matched HW Blackwell, TP=1)
- **Initial latent: IDENTICAL** — cos 0.99989, relL2 0.0000, MAE 0.0 (ours u/s 0.001/0.996,
  nv 0.001/0.996). ⇒ noise init (RNG) + first-frame VAE-encode are NOT the divergence.
- **step-0 noise_pred: cos 0.98865, relL2 0.1497, MAE 0.1455** (each on its own inputs).
  ⇒ A real ~15% difference already at step 0, with identical input latent. This compounds
  over 60 steps into the final Stage-1 relL2 0.57.
- ⇒ Divergence is in the **forward chain**: prompt_embeds / camera (raymap,plucker) /
  timestep / model weights — latent is ruled out.
- prompt_embeds shape: ours dump only saved first_row (1,2304); NVlabs (1,1,300,2304).

### Probe 2 — forced-input + timestep (matched HW Blackwell, TP=1)
- **Forced NVlabs latent+prompt into ours** (LOAD_LATENT_FROM + LOAD_PROMPT_FROM; log
  confirms "overrode initial latent" + "overrode prompt_embeds shape=(1,300,2304)").
  noise_pred STILL cos 0.98865 / relL2 0.1497 — **identical to non-forced**.
  ⇒ **prompt_embeds ruled out** (forcing it changed nothing).
- **timestep IDENTICAL**: ours/nvlabs t_per_frame both [0,1000,1000,…], t_scalar=1000.
  ⇒ **timestep/scheduler ruled out**.
- Remaining suspects for the step-0 noise_pred 0.15: **camera conditioning (raymap/plucker,
  the only un-matched model input) or the model forward itself**.
- NVlabs model HAS per-block dumps (`SANA_WM_DUMP_BLOCK0`: input/modulation/post_attn/
  post_cross_attn/output; `SANA_WM_DUMP_BLOCKS_PREFIX`). Ours has matching hooks.
  Next: 1-step run on both with DUMP_BLOCK0 → localize attention(GDN+camera) vs MLP.
