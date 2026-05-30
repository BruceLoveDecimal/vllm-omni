# SANA-WM Architecture Diagrams

Reconstructed from source-reading and parity-probe evidence collected
during the §6.13/§6.14 audit work (see `sana_wm_progress_audit.md`).
Targets the SANA-WM 1600M / 720p release config (`sana_wm_1600m_720p.yaml`).
Shapes use the 9-frame harness as reference: `T_pixel=9`,
`T_latent=2`, `H_pixel=704`, `H_latent=22`, `W_pixel=1280`,
`W_latent=40`, `B=1`, `C_latent=128`, `C_hidden=2240`,
`prompt_tokens=300`, `prompt_dim=2304`.

---

## 1. Top-level pipeline

The `SanaWmTwoStagesPipeline` runs three GPU passes in sequence: Stage-1
DiT denoising, Stage-2 LTX-2 distilled refiner, LTX-2 VAE decode.

```mermaid
flowchart LR
    subgraph Input
        I["RGB first frame<br/>(B, 3, H_pixel, W_pixel)"]
        P["Prompt text"]
        C["Camera trajectory<br/>c2w (T, 4, 4) + intrinsics"]
        A["Action token / k-v"]
    end

    I --> VAE_enc["LTX-2 VAE encode<br/>+ per-channel norm<br/>(latents_mean, latents_std)"]
    P --> TXT["Gemma-2-2b-it<br/>+ y_norm (scale 0.01)<br/>+ chi-prompt prefix"]
    C --> CAM["Camera embedder<br/>Plücker + raymap<br/>+ chunk_plucker_post_attn"]
    A --> ACT["Action injection<br/>(payload schema)"]

    VAE_enc --> z0["z₀<br/>(B, 128, T_lat, H_lat, W_lat)"]
    TXT --> y["prompt_embeds<br/>(B, 1, 300, 2304)"]
    CAM --> raymap["raymap (B, T*H*W, ·)<br/>+ chunk_plucker"]
    ACT --> y

    z0 --> S1
    y --> S1
    raymap --> S1
    S1["Stage-1 DiT<br/>1600M · 20 blocks<br/>20-step flow-matching Euler<br/>flow_shift = 9.8"] -->|"z_stage1<br/>(B, 128, T_lat, H_lat, W_lat)"| S2

    y --> S2
    S2["Stage-2 LTX-2 refiner<br/>distilled · 48 blocks · 3 steps<br/>sink/current split<br/>per-token timestep"] -->|"z_refined"| DEC

    DEC["LTX-2 VAE decode<br/>+ inverse per-channel norm<br/>(framewise tiling)"] --> OUT["RGB video<br/>(B, 3, T_pixel, H_pixel, W_pixel)"]
```

Per `sana_wm_progress_audit.md` §6.14a, the LTX-2 VAE decode is not
the dominant error source (Stage-1 RGB MAE ≈ 3, PSNR ≈ 35). §6.14e
showed the dominant remaining gap is the Stage-2 refiner contract.

---

## 2. Stage-1 DiT block — alternating GDN / softmax hybrid

20 transformer blocks, identical macro structure. The
`softmax_every_n = 4` knob splits the camctrl variant per block index
(0-indexed `(i+1) % 4 == 0` → softmax UCPE, else GDN+UCPE). Main
attention class is also swapped accordingly (despite the misleading
`chunk_size` scaffolding documented in `sana_wm_progress_audit.md`
§6.13r — the chunk-causal mask described in NVlabs' docstring is
unimplemented; both variants run full-bidirectional).

```mermaid
flowchart TB
    X["x: (B, T*H*W, 2240)"] --> N1["LayerNorm (no affine)"]
    T["timestep_modulation<br/>(B, T, 6, 2240)"] --> SST["scale_shift_table<br/>+ per-frame split"]
    SST --> SM1["shift_msa / scale_msa / gate_msa"]
    SST --> SM2["shift_mlp / scale_mlp / gate_mlp"]

    N1 --> MOD1["·(1 + scale_msa) + shift_msa"]
    SM1 --> MOD1

    MOD1 -->|"x_msa_in"| ATTN

    subgraph ATTN["self.attn — see diagram 3"]
        direction TB
        AM["MAIN: GDN or softmax<br/>per block_idx"]
        AC["CAM: UCPE branch<br/>(GDN or softmax variant)"]
        AOG["shared output_gate · proj"]
        AM --> AOG
        AC --> AOG
    end

    ATTN -->|"attn_output"| GATE1["× gate_msa"]
    GATE1 --> RES1[("+")]
    X --> RES1

    RES1 --> XA["x_after_attn"]
    XA --> CA["cross_attn<br/>(Stage-1 SDPA<br/>prompt embeds, 300 tokens)"]
    CA --> RES2[("+")]
    XA --> RES2

    RES2 --> XB["x_after_cross"]
    XB --> N2["LayerNorm (no affine)"]
    N2 --> MOD2["·(1 + scale_mlp) + shift_mlp"]
    SM2 --> MOD2

    MOD2 -->|"x_mlp_in"| MLP

    subgraph MLP["SanaWmMbConvFfn (GLUMBConvTemp)"]
        direction TB
        IC["inverted_conv 1×1 + SiLU<br/>(2240 → 13440)"]
        DC["depth_conv 3×3 dwconv<br/>(per-frame spatial)"]
        GLU["GLU split<br/>value · silu(gate)"]
        PC["point_conv 1×1<br/>(6720 → 2240)"]
        TC["t_conv (3, 1) Conv2d<br/>temporal aggregation<br/>+ residual"]
        IC --> DC --> GLU --> PC --> TC
    end

    MLP -->|"mlp_out"| GATE2["× gate_mlp"]
    GATE2 --> RES3[("+")]
    XB --> RES3
    RES3 --> OUT["x_out"]
```

Notes:
- §6.13j: block-0 sub-stage cosines are essentially perfect
  (attn cos +0.9999, MLP cos +0.998).
- §6.13m: the softmax-UCPE camera branch was missing in early native
  code; landing it pushed 3-step controlled latent cos from 0.968
  → 0.9925.
- §6.13p: at 321-frame length, every block's same-latent cos is
  still ≈ 1.0; the residual is broadly distributed, no single bad
  block.

---

## 3. Self-attention dispatch (SanaWmSelfAttention)

Two branches feed a shared `output_gate + proj`. Branch selection
depends on `block_idx`:

```mermaid
flowchart TB
    HS["hidden_states<br/>(B, N, 2240)"] --> BRANCH{"(block_idx + 1) %<br/>softmax_every_n == 0 ?"}

    BRANCH -->|"No (GDN block:<br/>0, 1, 2, 4–6, 8–10, …)"| GDN
    BRANCH -->|"Yes (softmax block:<br/>3, 7, 11, 15, 19)"| SOFT

    subgraph GDN["GDN main branch — _forward_gdn_raw"]
        direction TB
        GQ["qkv linear → q, k, v"]
        GK["k = bidir temporal short conv<br/>(conv_k)"]
        GG["compute_frame_gates → beta, decay"]
        GN["q_norm / k_norm (RMS)<br/>+ ReLU + k_scale"]
        GR["RoPE (q_rot, k_rot)"]
        GS["Triton bidirectional GDN<br/>fused scan<br/>(or PyTorch fallback)<br/>fp32 internal"]
        GQ --> GK --> GN --> GR --> GS
        GG --> GS
    end

    subgraph SOFT["Softmax main branch — _forward_softmax_raw"]
        direction TB
        SQ["qkv linear → q, k, v"]
        SQN["q_norm / k_norm (RMS)"]
        SR["RoPE"]
        SS["F.scaled_dot_product_attention<br/>(full bidirectional, no mask)"]
        SQ --> SQN --> SR --> SS
    end

    GDN -->|"main_raw<br/>(B, N, 2240)"| COMB[("+")]
    SOFT -->|"main_raw"| COMB

    HS --> CB
    CB["Camera branch dispatch<br/>(diagram 4)"] -->|"cam_contrib =<br/>out_proj_cam(cam_raw)"| COMB

    COMB --> OG["output_gate<br/>silu(linear(x)).fp32 · combined"]
    OG --> PROJ["proj (final linear)"]
    PROJ --> AOUT["attn_output<br/>(B, N, 2240)"]
```

Empirical findings:
- §6.13r: `chunk_size` scaffolding is wired through `forward_frame_aware`
  but every actual forward implementation swallows it via `**kwargs`.
  `_SoftmaxUCPESinglePathLiteLA`'s docstring claims chunk-causal masking
  exists; the implementation runs unmasked SDPA. NVlabs's
  `ChunkCausalSoftmaxUCPESinglePathLiteLA` is an alias of the same class.
- §6.13l: native scheduler matches NVlabs `FlowMatchEulerDiscrete` timestep
  table bit-for-bit (load-bearing "double-shift" of `sigma_min`).

---

## 4. UCPE camera branches

GDN cam branch (`BidirectionalGDNUCPESinglePathLiteLABothTriton`) and
softmax cam branch (`_SoftmaxUCPESinglePathLiteLA`) share the UCPE
prep, differ only in the inner scan.

```mermaid
flowchart TB
    HS["hidden_states<br/>(B, N, 2240)"] --> QKV["q_proj_cam / k_proj_cam / v_proj_cam"]
    QKV --> CONV["conv_k_cam<br/>(bidir temporal short conv,<br/>k_conv_only=True)"]
    CONV --> NORM["q_norm_cam / k_norm_cam (RMS)<br/>+ ReLU + k_scale"]

    RAY["camera_conditions<br/>(B, N, 4, 4)"] --> UCPE["prepare_prope_fns →<br/>apply_fn_q, apply_fn_kv, apply_fn_o<br/>(per-ray 4×4 + sliced RoPE)"]

    NORM --> UQ["apply_fn_q(q)"]
    NORM --> UKV["apply_fn_kv(k, v)"]
    UCPE --> UQ
    UCPE --> UKV

    UQ --> DOWN["_downscale_to_reference_rms<br/>(PostUCPERenorm,<br/>clip back to pre-UCPE RMS)"]
    UKV --> DOWN
    DOWN --> INFL["inflation_sq<br/>= (post_K_norm / pre_K_norm)²"]

    INFL --> DBD["Dynamic Beta Discounting:<br/>β_cam = β / mean(inflation_sq).clamp_min(1)"]
    GATES["β, decay<br/>(from main GDN gates)"] --> DBD

    DOWN --> SCAN{"block uses softmax<br/>variant?"}
    DBD --> SCAN

    SCAN -->|"No (GDN cam)"| GSCAN["cam_scan_bidi_chunkwise<br/>(NVlabs Triton, when<br/>SANA_WM_CAM_TRITON=1)<br/>or PyTorch _bidi_single_path<br/>fp32 internal"]
    SCAN -->|"Yes (softmax cam)"| SSCAN["F.scaled_dot_product_attention<br/>(full bidirectional)"]

    GSCAN --> AFO["apply_fn_o<br/>(inverse UCPE)"]
    SSCAN --> AFO
    AFO --> OPC["out_proj_cam<br/>(input init = base QKV)"]
    OPC --> COUT["cam_contrib<br/>→ main + out_proj_cam(cam_raw)"]
```

§6.12a verified the UCPE math + raw GDN cam branch to `~1e-7` fp32
max-abs vs NVlabs.

---

## 5. Stage-1 multi-step flow-matching Euler

```mermaid
flowchart LR
    Z0["z₀<br/>(B, 128, T_lat, H_lat, W_lat)"] --> LOOP
    SCHED["SanaWmFlowMatchScheduler<br/>flow_shift = 9.8<br/>sigmas = double-shift(linspace)<br/>20-step example:<br/>[1000, 994.4, …, 392.4, 87.7] → 0"] --> LOOP
    Y["prompt_embeds"] --> LOOP
    RAY["raymap / spatial_raymap"] --> LOOP
    PL["plucker / chunk_plucker"] --> LOOP

    subgraph LOOP["per-step loop (for t in timesteps)"]
        direction TB
        TS["model_timestep:<br/>(B, 1, F) per-frame fp32<br/>conditioning frame[0] = 0"]
        TF["transformer.forward<br/>(diagram 2)"]
        TS --> TF
        IN["latents"] --> TF
        TF -->|"noise_pred"| STEP["step_flow_euler_per_token<br/>per-token sigma from<br/>per_token_timesteps<br/>prev = sample + dt · (−noise_pred)"]
        STEP -->|"stepped"| MASK["condition_mask:<br/>frame 0 tokens unchanged"]
        MASK -->|"latents"| IN
    end

    LOOP --> Z1["z_stage1 (pre-refiner latent)"]
```

§6.13d/m: per-token Euler + condition_mask + per-frame timestep are
load-bearing. §6.13l: timestep table reproduced from
`FlowMatchEulerDiscreteScheduler` natively, not via DPMSolver wrapper.

---

## 6. Stage-2 LTX-2 refiner — sink / current split

Native refiner ported in commit `0a0ef672...8951e6a4` to follow the
NVlabs `DiffusersLTX2Refiner` contract (§6.14e).

```mermaid
flowchart TB
    Z1["z_stage1<br/>full latent stream"] --> SPLIT
    SPLIT{"split"}
    SPLIT -->|"sink = z[:, :, :sink_size]<br/>(context frames,<br/>preserved at sigma=0)"| SINK[("sink path")]
    SPLIT -->|"current = z[:, :, sink_size:]<br/>(generation frames)"| INIT

    INIT["seed-42 noise init:<br/>(1−start_sigma)·current<br/>+ start_sigma·ε"] --> RLOOP

    RSCHED["distilled refiner<br/>3-step sigma schedule<br/>(STAGE_2_DISTILLED_SIGMA_VALUES)<br/>+ terminal 0.0"] --> RLOOP

    SINK --> RLOOP
    PE["packed prompt embeds<br/>(refiner connector)"] --> RLOOP

    subgraph RLOOP["per-step refiner loop"]
        direction TB
        PT["per-token timestep:<br/>sink tokens = 0<br/>current tokens = sigma"]
        SM["streaming sink ↔ current<br/>attention mask"]
        RF["video-only forward<br/>(LTX-2 transformer, 48 blocks)"]
        PT --> RF
        SM --> RF
        STATE["packed latent stream"] --> RF
        RF -->|"x0 prediction"| UPD["velocity = (noisy − denoised) / sigma<br/>update current tokens only"]
        UPD --> STATE
    end

    RLOOP --> ZF["z_refined"]
```

§6.14f acceptance target (still open): `latent_cos_generated ≥ 0.98`,
RGB MAE ≤ 5 vs same-source NVlabs refiner.

---

## 7. Current correctness floor (as of §6.14e)

| Layer | 9-frame harness | 321-frame harness | Notes |
|---|---|---|---|
| Stage-1 latent vs NVlabs | cos +0.992 (§6.13m) | cos +0.7575 (§6.13n) | 321f gap is real and on native side (§6.13u NV self-cos bit-exact) |
| Refiner latent (same-source) | cos +0.89 (§6.14e) | (not measured) | §6.14f remaining work |
| VAE RGB direct | PSNR 35 / SSIM-Y 0.99 | (not measured) | not the bottleneck |
| Full-chain RGB | MAE 22.55 / PSNR 17.72 / SSIM-Y +0.587 (§6.14e) | (not measured) | spec gate: PSNR ≥ 30 / SSIM-Y ≥ 0.93 |

Dead ends already ruled out for the 321-frame gap:
- scheduler timestep table (§6.13l)
- camera Triton dispatch (§6.13p path ablation)
- bf16 vs fp32 of the transformer parameters (§6.13p)
- `chunk_size` chunk-causal masking (§6.13r — unimplemented in NVlabs)
- step-18 sigma-jump subdivision (§6.13s)
- single divergent sub-stage within any DiT block (§6.13p, §6.13u block-level diff)

What remains:
- Per-op kernel-level numerical alignment (RMSNorm reduce order,
  Triton GDN accumulation, ColumnParallelLinear bias add ordering).
  Expensive and broadly distributed; each op contributes < 1 % to
  the cumulative 0.04 same-latent `noise_pred` residual.
- Stage-2 refiner same-source parity (§6.14f) — still open and
  actionable; this is the next high-leverage path for the 9-frame
  spec gate.
