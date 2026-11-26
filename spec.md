# TaoMate-R1 接入 vLLM-Omni 实现规格（Roadmap #11195 · Task 10）

> 状态：草案（spec）。本文只描述实现方案，尚未修改任何代码。
> 调研基线：vLLM-Omni `0f55c39`（`claude/beautiful-darwin-kzlvuv`），上游
> `TaoLiveAIGC/TaoMate-H3@fac1a44`，`TaoLiveAIGC/TaoMate-R1@377b70c`，
> `MiniMaxAI/MiniMax-H3@12253298`。
> 所有"上游行为"均来自源码阅读，**未在 GPU 上复现**。
> 权重**不落盘**：网络结构通过 HTTP Range 只读取远端 safetensors 头部获得（每个 LoRA bank 约 43 KB，每个 DiT 分片约 4 KB），
> 另外只下载了 JSON 配置文件。

## 0. 任务来源与认领状态

- Roadmap：[#11195 World Model Roadmap through December 18, 2026](https://github.com/vllm-project/vllm-omni/issues/11195)，Task 10。
- Task 10 原文要点：接入 TaoMate-R1 的 **Ref2VA 基座**、**有序视觉参考（ordered visual references）**、
  **opening/continuation 两组 LoRA bank**；注册**独立的 model 与 deploy**，并提供 **realtime recipe**。
- Task 10 验收标准（来自 Roadmap 表格）：
  1. 向 `/v1/realtime/video` 发送两张可区分的图片，验证其**顺序**被保留；
  2. 跨 **3 个 chunk**，检查 bank 选择依次为 **cold → warm → warm**，且**跳过末尾 chunk 的 clean-history 写入**；
  3. 连续的**音视频同步 chunk 在流结束前**就能被解码出来。
- 认领状态：BruceLoveDecimal 已在 RFC 评论区认领 Task 10，但表格 owner 列仍为空，也没有维护者确认或 assignee。
  按 lessons L005（"Routing is not accepted ownership"），开工前建议在 RFC 下请维护者把 owner 写进表格。
- 依赖：Roadmap 未给 Task 10 列出前置任务。它与 Task 1–8（chunk 执行迁入 main diffusion）是并行关系。
  KV 管理仍留在 `vllm_omni/experimental/ar_diffusion`，所以本 spec 直接复用 TaoMate-H3 当前的 AR-Diffusion 路径。
  若 Task 1/2 先合入并改动 `SupportsStepExecution` / chunk executor 接口，需要相应 rebase。

## 1. 产物与网络结构（仅元数据，未下载权重）

### 1.1 TaoMate-R1 LoRA

只读取了远端文件的 safetensors 头部，权重文件未下载。

```text
TaoMate-R1/
├── config.json        {"model_variant":"Ref2VA","rank":128,"alpha":128,
│                       "banks":{"cold":"cold/adapter_model.safetensors","warm":"warm/adapter_model.safetensors"}}
├── cold/config.json   {"rank":128,"alpha":128}
├── cold/adapter_model.safetensors   2,480,973,992 B  LFS sha256 b15738e9…6a6160
├── warm/config.json   {"rank":128,"alpha":128}
├── warm/adapter_model.safetensors   2,480,973,992 B  LFS sha256 46bb9915…f80af
├── README.md / LICENSE (MiniMax H3 Community License) / NOTICE
```

safetensors 头部检查（两个 bank 相同）：

| 项 | 值 |
|---|---|
| 张量数 | 416 = 52 blocks × 4 projections × {lora_a, lora_b} |
| dtype | 全部 F32（BF16 下每个 bank 约 1.24 GB） |
| blocks | `blocks.0..49` + `token_refiner.blocks.0..1` |
| projections | `attn.qkv_proj`、`attn.out_proj`、`mlp.fc1`、`mlp.fc2` |
| 形状示例 | `qkv_proj.lora_a [128,5376]`、`qkv_proj.lora_b [21504,128]`、`fc1.lora_b [28672,128]`、`fc2.lora_a [128,14336]` |

**结论：** 张量清单与仓库内 `taomate_lora_targets()`（`vllm_omni/diffusion/models/taomate_h3/lora.py:32`）完全一致。
每个 bank 子目录也满足 `_read_adapter_config()` 的约定（`config.json` 含 `rank`/`alpha`）。
因此 `prepare_taomate_lora(bank_dir, …)` 可以**逐 bank 直接复用**，无需新的 key 映射。

### 1.2 MiniMax-H3 Ref2VA 基座

- 只读取了 JSON 配置和 13 个 DiT 分片的 safetensors 头部，未下载权重。
- 体积：Ref2VA 分区约 144 GB，其中 transformer 66.3 GB、text_encoder 66.7 GB、video_vae 10.4 GB、audio_vae 0.6 GB。
- `Ref2VA/model_index.json`：`_minimax_h3.partition = "ref2va"`，`tasks = ["ref2va"]`，`sigma_shift_scales = {video: 12, audio: 3}`。
- `Ref2VA/transformer/config.json` 与 `FL2VA/transformer/config.json` 逐字段一致。两者的 535 个参数名集合也完全相同，只是权重不同。
  因此 `MiniMaxH3DiTModel` 与 TaoMate 的 LoRA target 无需改动。

### 1.3 网络结构（由远端头部和配置得出）

**DiT `MiniMaxH3DiTModel`**（Ref2VA）：535 个张量，约 33.12 B 参数。522 个为 BF16；
`video/audio_patch_proj`、`final_layer.video/audio_out`、`time_embedder.*`、`rope.inv_freq` 等 13 个为 F32。

| 模块 | 形状（out, in） | 说明 |
|---|---|---|
| `video_patch_proj` | [5376, 96] | 24 通道 × patch (1,2,2) → hidden 5376 |
| `audio_patch_proj` | [5376, 32] | 32 维音频 latent |
| `condition_proj` | [5376, 5120] | Qwen3-VL 隐状态 5120 → 5376 |
| `time_embedder.proj_in/out` | [5376, 256] / [2688, 5376] | timestep 256 → 5376 → 2688 |
| `token_refiner.blocks.{0,1}` | 与 DiT block 相同的 attn/mlp，无 adaln | 文本 refiner；带 `final_norm` |
| `blocks.{0..49}.attn.qkv_proj` | [21504, 5376] | 21504 = 3 × 56 头 × 128 |
| `blocks.N.attn.out_proj` | [5376, 7168] | 注意力内维 7168 = 56 × 128（≠ hidden 5376） |
| `blocks.N.attn.q_norm/k_norm` | [128] | per-head QK norm |
| `blocks.N.mlp.fc1` / `fc2` | [28672, 5376] / [5376, 14336] | 门控 FFN：28672 = 2 × 14336 |
| `blocks.N.adaln_proj.linear` | [96768, 2688] | 96768 = 18 × 5376 的逐 token 调制 |
| `blocks.N.norm1/norm2` | [5376] | |
| `final_layer.adaln_proj` / `video_out` / `audio_out` | [10752, 2688] / [96, 5376] / [32, 5376] | 输出头 |

**LoRA 与 DiT 的对应关系**：两个 bank 各有 208 个 target，即 (50 DiT + 2 refiner) × {qkv, out, fc1, fc2}。
逐个核对，A 的形状为 `[128, in]`、B 的形状为 `[out, 128]`，与对应 Linear 的 0 处不匹配，每个 bank 0.62 B 参数。
`adaln_proj`、`condition_proj`、patch/time/final 层**不在 LoRA 范围内**。

**文本编码器**：`Qwen3VLForConditionalGeneration`。语言部分 64 层、hidden 5120、64 头、8 KV 头、FFN 25600，BF16，共 66.7 GB。
视觉部分 27 层、hidden 1152、patch 16、`spatial_merge_size` 2、`temporal_patch_size` 2，输出 5120。
processor 的 `merge_size` 为 2，像素范围 `shortest_edge` 65536 到 `longest_edge` 16777216（以像素面积计）。

**VAE**：video VAE 为 `MiniMaxH3VideoVAE`，24 个 latent 通道，clip 长度 17，tile 256，最小 overlap 64。
audio VAE 为 `MiniMaxH3AudioVAE`，32 个 latent 通道，2 声道，32 kHz。两者都与 TaoMate-H3 使用的 FL2VA 组件同类。

### 1.4 上游参考实现（已 clone）

`/home/user/upstream/TaoMate-H3@fac1a44`，`src/taomate_h3/ref2av/`，共 702 行：

| 文件 | 职责 |
|---|---|
| `conditioning.py` | 读取有序参考、Qwen3-VL 编码 presentation、VAE 编码参考 anchor、构建 ref2va packed sequence |
| `adapters.py` | `BankedLinear`：cold/warm 两组 A/B **同时常驻**，`use_chunk(i)` 切换；每个 bank 单独缓存 refined text |
| `streaming.py` | `chunk_slices`（39/34 帧）、`compact_layout`（保持全局 RoPE）、`CleanHistoryAttention`（sink + 最近 2 chunk） |
| `generation.py` | chunk 循环：3 步去噪；非末尾 chunk 额外做一次 sigma=0 clean forward 写历史 |
| `decoding.py` | 全部 chunk 完成后**一次性**解码 MP4（离线，非流式） |
| `worker.py` / `__main__.py` | TP2 × Ulysses(world/2) 的 torchrun 进程编排；prepare / generate 分两个进程以释放编码器显存 |

## 2. 上游算法契约（必须复现的语义）

### 2.1 输入

- `prompt`：单个文本，整个会话不变。上游没有中途换 prompt 的能力。
- `references`：**有序**列表，每项是 `{"type": "image" | "video", "uri": …}`，至少 1 项。
  - image：`RGB` → `reference_size(w, h, 2048)`。按短边缩放到 2048，两边对齐到 32 的倍数，使用 LANCZOS。
    **注意：这与仓库内 MiniMax-H3 不同**，后者 `resolve_minimax_h3_reference_image_shape` 保留原分辨率。
  - video：ffprobe 校正 SAR 与 rotation 后，用 ffmpeg 重采样到 24 fps，并缩放到
    `resolve_geometry(duration=target_frames/24, aspect=源比例, short_edge=768)`，截取 `target_frames = 会话总帧数` 帧。
    **参考视频自带的音频被丢弃**。这与仓库内 H3 Ref2VA 不同，后者会把视频音轨作为 `<Audio j>` 参考加入。
- 标签：按**参考的原始顺序**逐个生成，image 与 video 各自计数，即 `<Picture i>: ` 和 `<Video k>: `。
  video 的每个时间块前插入 `<{t:.1f} seconds>`。Qwen 采样规则为 `frames[::12]`，时间戳取 `i/2` 的两两均值。
  文末接 prompt。**image 与 video 可交错**；仓库内 H3 则固定为"先全部 image、再全部 video"。

### 2.2 几何（全会话预先确定）

- `resolve_geometry(duration, width, height)`：`frames = align(round(duration*24))`，对齐到 `17j+5`。
  `video_latent_t = 5*(frames-5)/17 + 2`，`audio_latent_t = round(frames/24*40)`（每声道；共 2 声道）。
- 分辨率：宽高均为 32 的倍数，上游默认 `768x1344`。`latent_h = H/16`，`latent_w = W/16`。
  patch 为 (1,2,2)，每个 latent 帧有 `(H/32)*(W/32)` 行，例如 768x1344 对应 1008 行。
- chunk 切分（`chunk_slices`）：第 0 个 chunk 为 **39 帧 = 12 latent**，之后每个 chunk **34 帧 = 10 latent**，
  末尾可能出现 17 帧 = 5 latent 的短 chunk。音频按 `round(end_frame*40/24)` 切分（每声道）。
  例：10 s 对应 243 帧、72 latent、7 个 chunk（12+6×10）、每声道 405 个音频 latent。
- **整段 packed sequence（含全部目标位置的 RoPE）在开始时就按总时长构建**。每个 chunk 用 `compact_layout` 从中 gather
  `text | references | audio_chunk | video_chunk`，**不重置 RoPE**。参考 anchor 的条件噪声也依赖 `target_latent_t`。
  ⇒ **会话总帧数必须在 `session.start` 时确定**，v1 不支持无限流。

### 2.3 条件编码

- 文本：Qwen3-VL `encode_ids(ids, pixel_values…, pixel_values_videos…)` 得到 `text_hidden` 与 `token_tags`。
- 视觉 anchor：每个参考用 Ref2VA video VAE 编码。编码时 `fork_rng` 并固定 seed 42，结果经 `_normalize_keyframe_latent` 归一化。
  然后调用 `minimax_h3_imgvid_cond_noise_aug_rows(noise_aug=0.999)`，**每个参考都用 `seed`（请求 seed）重新播种的 CPU generator**。
  anchor 行在 video 行之前，`video_target_start = len(anchors)`。
- packed：`minimax_h3_packed_sequence_ref2va(text_len, latent_t/h/w, audio_t, ref_blocks)`。
  其中 `ref_blocks` 为 `[{kind: image|video, latent_h, latent_w, (latent_t, ref_audio_t=0)}]`，按参考顺序排列。

### 2.4 两组 LoRA bank

- chunk 0 用 **cold**（opening），chunk ≥1 用 **warm**（continuation）。该 chunk 的 clean commit 也在同一个 bank 下执行。
- refined text（`transformer.refine_prompt_embeds`，token_refiner 也在 LoRA target 内）**每个 bank 计算一次并缓存**。
- `scale = alpha / rank = 1.0`。forward 时 A/B 被 cast 为输出 dtype（BF16），所以 BF16 存储与上游数值等价。

### 2.5 去噪与提交

- sigma：`time_shift_sigmas(50, shift=12)`（video）与 `shift=3`（audio），取下标 **(0→16)、(16→33)、(33→49)**。
  下标 49 的 sigma 为 0，所以 3 步后直接到干净 latent。
- 噪声：每个 chunk 用设备上的 `torch.Generator(seed + chunk_index*97)`，**先抽 video 再抽 audio**，dtype 为 fp32。
- timestep plan：`video/audio_timesteps = 1 - sigma`，`imgvid_cond_noise_aug = 0.999`，`audio_ref_cond_noise_aug = 1.0`。
- Euler：`x ← x + (σ_l − σ_r)·v`。audio 只取 `velocity[1][branch.audio_target_slice]`。
- **clean commit**：若 `index + 1 < len(chunks)`，在 sigma=0 下再跑一次 forward，把本 chunk 的 AV K/V 写入历史。
  **末尾 chunk 不写**（对应验收标准 2）。
- 没有 audio teacher：R1 联合生成音视频，与 TaoMate-H3 依赖 base model 做 audio teacher 不同。

### 2.6 注意力与历史（`CleanHistoryAttention`）

- 行划分：`current` = 本 chunk 的 audio 行 + 目标 video 行；`condition` = text 行 + reference anchor 行，不含 alignment padding。
- condition query 只 attend condition K/V。current query attend `[condition | history | current]`。padding 行输出 0。非因果 SDPA。
- 历史：`sink` = **第一次 commit 中的 video K/V**（去掉 audio 部分）；`recent` = 最近 **2** 个完整 AV chunk。
  `history = ([sink] if commits > 2 else []) + recent`。sink 尚在 recent 中时不会重复。
- 不做音频历史周期性重置（TaoMate-H3 每 12 个窗口清一次，R1 没有）。
- condition K/V **每次 forward 都重新计算，不缓存**。

## 3. 与仓库内 TaoMate-H3 的差异

| 维度 | TaoMate-H3（已接入） | TaoMate-R1（本任务） |
|---|---|---|
| 基座分区 | `FL2VA/`（代码明确拒绝 Ref2VA，`taomate_h3/pipeline.py:773`） | `Ref2VA/` |
| 条件 | 仅文本；有参考输入直接报错（`pipeline.py:1198`） | 文本 + 有序 image/video 参考 |
| LoRA | 单 adapter；teacher 需临时 suspend | cold/warm 两个 bank，按 chunk 切换 |
| 音频 | base DiT audio teacher（9 次 forward/窗口） | 联合生成，无 teacher |
| chunk | 5 s 窗口 × 4 phase（34/34/34/17） | 39 帧开场 + 34 帧续写 |
| RoPE/时长 | 窗口滚动，可无限流 | 全会话一次性定长 |
| 历史保留 | 首 phase video sink + 最近 2 phase；音频每 12 窗口清空 | 首 chunk video sink + 最近 2 chunk；不清音频 |
| 中途交互 | prompt 更新（`TaoMateH3PromptInteractionHandler`） | v1 不支持（上游无此语义） |
| 并行 | 纯 Ulysses，TP=1 | 上游为 TP2 × Ulysses；本接入沿用纯 Ulysses，TP=1 |

**可直接复用、几何吻合的部分：**

- `StreamingVideoDecoder` / `StreamingAudioDecoder`（`taomate_h3/stream_decode.py`）已经按 **12/10/…/5 latent** 的 phase 增量设计：
  首段 39 帧，之后每 10 latent 出 34 帧，保留 5 帧 overlap，最后 flush。与 R1 的 chunk 几何**完全一致**，可原样复用（满足验收标准 3）。
- `TaoMateH3StreamingAttention`（`taomate_h3/attention.py`）的语义与上游 `CleanHistoryAttention` 一致：
  condition 只看 condition，media 看 `[condition | 持久 clean AV | current]`，padding 输出 0。
  只需让 `StreamContext.condition_rows` 覆盖 text **和** reference anchor 行。
- `TaoMateKVHistory`（`taomate_h3/kv_cache.py`）的 sink + 最近 2 段保留策略与 R1 一致。R1 不调用 `drop_audio_history()`。
- `prepare_taomate_lora`（`taomate_h3/lora.py:62`）逐 bank 复用，见 §1.1。
- MiniMax-H3 共享件：`minimax_h3_packed_sequence_ref2va_blocks`（`minimax_h3/packed_sequence.py:341`）；
  `minimax_h3_ref2va_video_presentation(condition_labels=…)`（`model_executor/models/minimax_h3/preprocessing.py:357`），
  它按**传入的 `condition_labels` 顺序**生成标签，天然支持交错；`MiniMaxH3DenoiseBranch`（`minimax_h3/denoise_loop.py`）；
  `condition_noise.py`；`packed_tokens.py`（patchify/unpack）。

## 4. 设计

### 4.1 新增与修改的文件

```text
vllm_omni/diffusion/models/taomate_r1/
  __init__.py
  pipeline.py        TaoMateR1Pipeline(MiniMaxH3Pipeline, SupportsStepExecution)
  conditioning.py    有序参考加载、presentation 和 anchor 编码（按上游 §2.1–2.3 复现）
  chunks.py          chunk 切分、compact layout、bank 选择、terminal 判定（纯 CPU 函数，便于单测）
  lora.py            两个 bank 的准备与切换（复用 taomate_h3.lora.prepare_taomate_lora）
vllm_omni/model_executor/models/taomate_r1/{__init__,pipeline}.py   TAOMATE_R1_PIPELINE（单 stage diffusion）
vllm_omni/deploy/taomate_r1_usp4_realtime.yaml                      （USP8 可选）
recipes/TaoLiveAIGC/TaoMate-R1.md
tests/diffusion/models/taomate_r1/…

修改（注册与 API）：
vllm_omni/diffusion/registry.py          "TaoMateR1Pipeline": ("taomate_r1", "pipeline", "TaoMateR1Pipeline")；
                                          与 TaoMate-H3 一样加入不做 cache 加速的列表；注册 post_process_func
vllm_omni/config/pipeline_registry.py    "taomate_r1": TAOMATE_R1_PIPELINE（不设 default_deploy_config_name，
                                          避免 MiniMax-H3 快照被自动识别为 R1）
vllm_omni/diffusion/model_metadata.py    TaoMateR1Pipeline: supports_multimodal_inputs=True,
                                          max_multimodal_image_inputs=9, final_output_type="video",
                                          attention_mask_free=True
vllm_omni/entrypoints/openai/serving_video_output_stream.py   realtime 支持有序多参考（§4.6）
docs/models/supported_models.md、recipes/README.md
```

公共逻辑尽量**从 `taomate_h3` 导入**，不要复制。如果某个函数需要参数化（例如 attention 的 condition 行来源、
KV history 是否允许清音频），在 `taomate_h3` 内做最小的参数化修改，并保持 TaoMate-H3 测试全绿。

### 4.2 加载

- `--model` 指向 `MiniMax-H3/Ref2VA/`。`load_weights` 校验 partition 为 `ref2va`，否则报错，与 TaoMate-H3 的反向校验对称。
- `--lora-path` 指向 TaoMate-R1 根目录。读取根 `config.json`，校验 `model_variant == "Ref2VA"`，
  并且 `banks` 中恰好有 `cold`、`warm`。对每个 bank 调用 `prepare_taomate_lora(root/bank_dir_of(banks[role]))`。
- 注意力类替换与 TaoMate-H3 相同（`TaoMateH3DiTModel` / streaming attention）。

### 4.3 bank 切换（两种方案，建议先 A 后测）

- **A. 复用 `DiffusionLoRAManager`（推荐首版）**：把 cold、warm 注册成两个 `LoRARequest`（固定 `lora_int_id`），
  要求 `max_cpu_loras ≥ 2`。在 chunk 边界调用 `set_active_adapter(cold|warm)`。
  每次切换会重新绑定 slot 0（约 1.24 GB 的 H2D 拷贝），并调用 `invalidate_denoise_graphs`。
  每个会话只切换两次：开场切到 cold，chunk 1 切到 warm。改动最小，与 TaoMate-H3 的 manager 用法一致。
  `_load_diffusion_lora_adapter` 需要按 `lora_request` 返回对应 bank 的 `LoRAModel`。
- **B. 双 bank 常驻 sidecar**：仿照上游 `BankedLinear`，每层同时持有两组 A/B，只切换指针。
  每 rank 多占约 1.24 GB 显存，切换零拷贝，CUDA graph 也可以按 bank 分别 capture。
  如果 A 在 chunk 1 上的切换延迟或首包抖动不可接受，再在 manager 内实现 B（不要绕开 manager）。
- 每个 bank 缓存一次 refined text（`refine_prompt_embeds` 须在对应 bank 激活后执行）。

### 4.4 Step execution 契约（AR-Diffusion engine）

| 钩子 | 行为 |
|---|---|
| `prepare_encode` | 校验 fps=24、`num_outputs_per_prompt=1`、canvas 与部署一致、至少 1 个参考、`num_frames` 给定。<br>按 §2.1–2.3 编码条件，构建整段 packed，计算 `chunks`。<br>`state.chunk_num_steps = 3`，`state.total_chunks = len(chunks)`。<br>建立 session：几何、anchors、refined text 缓存、KV history、流式 decoder。 |
| `denoise_step` | 当前 chunk 的 bank：`"cold" if chunk_index == 0 else "warm"`。<br>在 `StreamMode.NOISY` 下 forward，用 §2.5 的 sigma 下标做 Euler。 |
| `post_decode` | 若 `chunk_index + 1 < total_chunks`，在 `CLEAN_COMMIT` 模式下用**同一 bank** 跑 sigma=0 forward，<br>再执行 `history.commit()`；否则跳过（terminal）。<br>然后把本 chunk 的 video/audio latent 推给流式 decoder，输出同步的 A/V 帧；最后一个 chunk 调 `flush`。 |
| `ar_diffusion_kv_cache_spec` | segmented KV 容量见下。`session_capacity=1`，`max_num_seqs=1`。 |
| `bind/reset/close_ar_diffusion_session` | 与 TaoMate-H3 相同。 |
| `ar_diffusion_warmup_requests` | 用合成的 2 张参考图跑 **≥ 3 个 chunk**，覆盖 cold、warm、commit 和 terminal。 |

KV 容量估算：retained 包含 sink（chunk 0 video，12 latent）、最近 2 个完整 AV chunk（video ≤ 12 latent，audio ≤ 2×65 行）。
pending 为 1 个 chunk。以 768x1344 为例，sink 有 12×1008 行，单 chunk 约 1.0–1.2 万行。
staging 还要加上 **condition 行数（text + 全部参考 anchor）**。该值依赖参考的 VAE latent 尺寸，必须在 GPU 上实测后写进 spec。

可观测性：每个 chunk 在 `ARDiffusionChunkMetadata` 或 chunk metadata 中带上 `lora_bank` 与 `clean_commit: bool`。
这样验收标准 2 可以从客户端直接断言，不需要读服务端日志。

### 4.5 RNG 与数值对齐

- 噪声：`seed + chunk_index*97`，video 先、audio 后。参考 VAE encode 用 seed 42。cond-noise-aug 每个参考都用请求 seed 重新播种。
- 并行差异：上游为 TP2 × Ulysses4（8 卡），本接入为纯 Ulysses、TP=1。归约顺序不同，预计不能 bit-exact，验收用容差（§6.2）。
- 注意力 kernel：上游用 SDPA（flash/efficient/cudnn），TaoMate-H3 用 FA3（不可用时回退 SDPA）。parity 测试固定使用同一后端。

### 4.6 API：`/v1/realtime/video` 有序多参考

现状：`VideoGenerationRequest.image_reference` 的类型允许 list，`/v1/videos` 也已用 list 顺序表示 Ref2VA 的 `<Picture i>`。
但 realtime 的 `_decode_image_reference`（`serving_video_output_stream.py:1067`）**收到 list 会直接拒绝**。
而且 `_build_prompt_and_sampling_params` 只透传单张 image。

改动：

1. 当模型 metadata 的 `max_multimodal_image_inputs > 1` 时，realtime 接受有序 `image_reference` 列表，
   按顺序解码为 `multi_modal_data["image"] = [img1, img2, …]`。单模型（单图）时保持现有行为和错误信息不变。
2. （可选，v1.1）`video_reference` 列表映射到 `multi_modal_data["video"]`。
   如需 image/video **交错**顺序，增加显式字段（例如 `extra_params.reference_order: ["image","video","image"]`），
   由 R1 conditioning 按此生成 `condition_labels`。无该字段时按"先 image 后 video"处理，与 `/v1/videos` 的 H3 语义一致。
3. `session.start` 上限为 4 MiB（`_MAX_START_MESSAGE_SIZE`）。base64 大图可能超限，recipe 中建议使用 URL 或压缩后的图片。
4. 不注册 interaction handler；R1 会话中发来的 `session.interaction` 应被明确拒绝。

### 4.7 Deploy（`taomate_r1_usp4_realtime.yaml` 草案）

```yaml
pipeline: taomate_r1
async_chunk: false
distributed_executor_backend: mp
dtype: bfloat16
stages:
  - stage_id: 0
    devices: "0,1,2,3"
    max_num_seqs: 1
    enforce_eager: true
    model_class_name: TaoMateR1Pipeline
    engine_backend: vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine
    step_execution: true
    streaming_output: true
    parallel_config:
      tensor_parallel_size: 1
      sequence_parallel_size: 4
      ulysses_degree: 4          # 56 头 / 4 = 14 头每 rank
      ring_degree: 1
      text_encoder_tp_size: 4
      vae_patch_parallel_size: 4
      vae_parallel_mode: tile
    default_sampling_params: {num_inference_steps: 3, output_type: np, fps: 24}
    model_config:
      taomate_r1_width: 768
      taomate_r1_height: 1344
      taomate_r1_seed: 42        # 上游 CLI 默认值；README 示例使用 9111
      ar_diffusion_kv_config: {gpu_memory_fraction: 0.02, warmup_cudagraph: true}
```

显存：DiT 在 BF16 下每 rank 约 66 GB（TP=1），text encoder 在 TP4 下每 rank 约 17 GB，另有 VAE、LoRA（A 方案 1 份 / B 方案 2 份，各约 1.24 GB）、KV 与 scratch。
目标仍是 H200 级 141 GB 显卡。text encoder 可在编码后 offload（上游把 prepare 和 generate 拆成两个进程，就是为了释放它）。
FP8 / USP2 版本留到后续 PR。

## 5. 实施拆分（建议 3 个 PR）

1. **PR-1 模型（request 模式 + parity）**：`taomate_r1/*`、注册、deploy（`step_execution: false` 的 full-request 变体）、CPU 单测、GPU parity 脚本。
   输出与上游 `latents.pt` 对比。标题示例：`[Model] TaoMate-R1: Ref2VA opening/continuation LoRA banks`。
2. **PR-2 realtime**：step execution、流式 A/V 解码、realtime 有序多参考（§4.6）、realtime deploy、recipe、supported_models、e2e 测试（覆盖三条验收）。
3. **PR-3 优化（可选）**：方案 B 双 bank 常驻；**condition K/V 按 bank 缓存**。
   condition 只 attend 自身，且其 timestep 固定（cond aug 0.999 / text），理论上在同一 bank 内跨 step、跨 chunk 不变。
   需先在 GPU 上验证 AdaLN 调制不依赖 step timestep，再考虑复用 `MiniMaxH3ReferenceKVTier2State`。另外还有 CUDA graph 和 FP8。

每个 PR 提交前跑 `precheck-pr` skill。新模型参照 `add-diffusion-model` skill，recipe 参照 `add-recipe` skill，测试分级参照 `vllm-omni-test` skill。
**不要新增模型专属的 Python example**，客户端复用 `examples/online_serving/streaming_video_generation/streaming_video_client.py`。

## 6. 测试与验收

### 6.1 CPU 单测（`tests/diffusion/models/taomate_r1/`）

- `chunks`：243 帧对应 [12,10×6] latent；124 帧对应 [12,10,10,5]；音频切分边界为 65/122/178/…/405；全部 chunk 拼接无重叠、无缺口。
- bank 选择：3 个 chunk 依次为 cold、warm、warm；commit 序列为 True、True、False（terminal 跳过）。
- `compact_layout`：gather 后的 `img_position_ids` 等于全局 packed 中对应行；`update_mask` 排除 reference 行；padding 对齐到 `MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT`。
- presentation 顺序：`[img A, img B]` 与 `[img B, img A]` 生成的 token 序列不同，且标签依次为 `<Picture 1>`、`<Picture 2>`；交错的 image/video 按给定顺序编号。
- KV history：第 3 次 commit 后 retained = sink(video of chunk 0) + chunk 1 + chunk 2；commit 失败时 rollback 不污染历史。
- LoRA：两个 bank 的 inventory 校验；根 `config.json` 缺少 `banks`、`model_variant` 不是 Ref2VA、或 rank 不一致时报错。用小型假 safetensors 测试，不依赖真实权重。
- realtime 协议：单模型时 list 仍被拒绝；R1 metadata 下 list 保序进入 `multi_modal_data["image"]`。

### 6.2 GPU parity（手动或 L4）

- 上游：`python -m taomate_h3.ref2av --model-root … --adapter TaoMate-R1 --prompt-file examples/ref2av_prompt.txt --reference-image presenter.png --reference-image product.png --duration 10 --resolution 768x1344 --seed 9111 --gpus 8`。
  在 `decoding.py` 删除 `latents.pt` 之前保存一份。
- vLLM-Omni：request 模式下输入相同，导出每个 chunk 的最终 latent。
- 指标：逐 chunk 的 video/audio latent cosine 与 max-abs，以及解码后的 PSNR/SSIM。容差在首次实测后写入测试，并注明 TP2 与 TP1 的差异来源。
  按 lessons L002，证据要写清硬件、版本、workload。

### 6.3 Roadmap 验收（e2e，`/v1/realtime/video`）

| 验收 | 方法 |
|---|---|
| 1. 两张可区分图片的顺序 | 发送 `image_reference: [A, B]` 与 `[B, A]` 两个会话（同 seed、同 prompt，prompt 中写明 `<Picture 1>` 与 `<Picture 2>` 的角色）。<br>断言服务端 presentation 标签顺序，并比较两次输出帧：结构上不同，且分别更接近对应的首参考。 |
| 2. 三个 chunk 的 bank 与 terminal | `num_frames = 107`（39+34+34，3 个 chunk）。<br>从 chunk metadata 断言 `lora_bank = cold, warm, warm`，`clean_commit = true, true, false`。<br>服务端 KV ledger 的 commit 计数为 2。 |
| 3. 结束前持续出 A/V | 记录每个 fMP4 chunk 的到达时间。断言第 1 个 chunk 在最后一个 chunk 生成之前到达。<br>每个 chunk 的 `num_audio_samples` 与帧数对齐（800 样本 / latent，32 kHz）。完整 MP4 可解码并含两条轨。 |

## 7. 风险与待定问题

1. **参考分辨率 2048**：上游先把图片缩放到短边 2048 再做 VAE 编码，condition 行数可能很大，而且每次 forward 都要重算。
   需在 GPU 上确认 anchor latent 尺寸，以及 `encode_images` 内部是否还会再缩放。这会决定 KV staging 大小和 PR-3 condition 缓存的收益。
2. **交错顺序**：现有 API 的 image/video 是两个独立列表，交错顺序需要新字段。v1 可以先只支持 image-only（满足验收 1）。
3. **参考视频音频**：上游忽略参考视频的音轨。不能直接复用 H3 的 `prepare_encoder_inputs`，因为它会加入 `<Audio j>`；需要 R1 自己的 conditioning。
4. **会话定长**：RoPE 和 cond-noise 依赖总时长，`session.start` 必须给出 `num_frames`。超过训练分布的长时长效果未知。
5. **TP 差异**：上游以 TP2 验证，本接入用 TP=1 的纯 Ulysses，数值只能容差对齐。
6. **bank 切换开销**：方案 A 在 chunk 1 前有一次约 1.24 GB 的 H2D，会影响第 2 个 chunk 的延迟，需要实测。
7. **许可**：权重与上游代码均为 MiniMax H3 Community License。从上游移植的代码需要保留其 "Modified for … Licensed under …" 头注释（见上游 `conditioning.py`）。
   合入前需要维护者确认是否可以按 Apache-2.0 仓库的惯例引用。
8. **与 Task 1/2 的接口漂移**：chunk 执行迁入 main diffusion 后，`SupportsStepExecution` 的钩子名可能变化。PR-2 需要跟进。

## 8. 复现调研环境

本次调研**未下载任何权重**。结构信息可按以下方式复现，只读取 safetensors 头部：

```python
# 读取远端 safetensors 头部：前 8 字节为小端 u64 头长度，随后是 JSON 头
import json, struct, urllib.request
def header(url):
    get = lambda a, b: urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
    n = struct.unpack("<Q", get(0, 7))[0]
    return json.loads(get(8, 8 + n - 1))
```

需要在 GPU 机器上运行时再下载：

```bash
hf download TaoLiveAIGC/TaoMate-R1 --revision 377b70c6116c5381ddbae542f0ee3e1517c8d30c --local-dir models/TaoMate-R1
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "Ref2VA/*" --local-dir models/MiniMax-H3   # ~144 GB
git clone https://github.com/TaoLiveAIGC/TaoMate-H3 && git -C TaoMate-H3 checkout fac1a44
```
