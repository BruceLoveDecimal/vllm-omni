# Wan2.2-Animate-2 接入实施规格(v2:Milestone、复用边界与验收标准)

> 配套设计文档:[wan_animate_integration.md](wan_animate_integration.md)(架构与技术选型;
> 其中被本版证伪的结论已在该文档开头的"勘误"中标注)。
> 本文回答三个问题:**分几步做**、**每步复用什么 / 新写什么**、**每步做到什么程度算完成**。
> v2 修订日期:2026-09-20。相对 v1 的差异见 §1。

## 0. 使用说明

- 每个里程碑的验收标准编号为 `A<里程碑>.<序号>`,均为**可执行验证**(pytest / 脚本 /
  实测记录),不接受"看起来对了"。
- **门禁(Exit Gate)**:该里程碑全部验收标准通过后方可进入下一里程碑;标记
  `[BLOCKING]` 的项不通过必须触发预案,不得带病推进。
- M0–M5 为主线,M6 为可选独立立项。

## 1. v2 相对 v1 的修订(均有证据,不是重新猜测)

| # | v1 的假设 | 实际情况(2026-09-20 核实) | 对计划的影响 |
|---|---|---|---|
| R1 | Diffusers 仓库只有 modular 格式,无 `model_index.json`,需显式 `--model-class-name` 或改索引探测 | `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` 与 `...-Distilled-Diffusers`(HF 与 ModelScope 同步)**都带标准 `model_index.json`**,`_class_name` 为 `WanAnimate2Pipeline`;组件子目录 `tokenizer / text_encoder / vae / image_encoder / image_processor / scheduler / transformer`,与 Wan2.1-I2V 完全同构 | 注册键 = `WanAnimate2Pipeline`,自动探测;**不改** `data.py`、`entrypoints/utils.py`;设计文档 §8 风险 4 消失;原始格式(`.pth` T5/VAE/CLIP)降为 M6 可选 |
| R2 | transformer 配置需在代码里硬编码 | `transformer/config.json` 齐全,键名为上游命名(`dim / num_heads / in_dim / refer_offset_* / refer_stride`),**不含** `log_scale` | 沿用 I2V 的 `load_transformer_config` 读配置,工厂函数只做键名桥接;蒸馏版由 `modular_model_index.json` 的 `_blocks_class_name == "WanAnimate2DistilledBlocks"` 判定 |
| R3 | scheduler 需移植官方 `FlowDPMSolverMultistepScheduler`(上一版移植为 815 行) | 基础版仓库 `scheduler/` 是 diffusers `DPMSolverMultistepScheduler`(`use_flow_sigmas=True, flow_shift=5.0, prediction_type=flow_prediction`,仓库 pin 的 diffusers 0.38 已支持);蒸馏版仓库是 `FlowMatchEulerDiscreteScheduler(shift=5.0)`,仓库内 `models/schedulers/` 已有同名自研实现 | **不新增 scheduler 文件**。已知差异:diffusers 的 flow sigma 网格首点为 0.999,官方 `get_sampling_sigmas` 为 1.0(见 §M1 A1.2);对拍基线以 diffusers modular 实现为准,该差异对基线为零 |
| R4 | 蒸馏版 `log_scale=-1.3` 作用于参考 token 的注意力分数 | 上游 `_score_mod_impl` 的条件是 `hw <= kv_idx < 2*hw`,即**生成侧 latent 帧 1 的 key 块**;参考块起点在 `target_q_len` 之后,不受影响 | 参考注意力实现按此修正;设计文档 §3.2 第 4 点勘误 |
| R5 | 零填充的 key 槽位会被掩码掉 | 上游 `create_mask` 把填充窗口内全部槽位标为有效;零 key 对任意 query 打分为 0,只扩大 softmax 分母。填充来源两处:末段短于 `origin_len`;letterbox 后真实网格小于 `origin_area` | 注意力必须建模"零 key 分母项",否则每一帧都会静默偏移 |
| R6 | 首版走 `supports_request_batch=True` + 严格 `batch_compatibility_key` | 上游有效 batch=1;仓库现有 `supports_request_batch=False` 即可让引擎不合批 | 首版 `supports_request_batch=False`,少一套 key 逻辑 |
| R7 | 分支从 `8dc3d504d` 出发 | 本分支落后 main 547 个提交;main 已改 `attention/layer.py`、`registry.py`、`model_metadata.py`、S2V 管线(新增 `ChunkedVideoMP4Session` 逐段编码、`_should_release_dit_before_decode`) | 开工第一步是 rebase 到 main,所有复用引用以 main 为准 |
| R8 | 从零实现 | 兄弟 worktree `wan-animate2-model-integration-1e7870` 里有一版**未提交**的首轮移植(模型代码约 2.5k 行 + L1 测试 1.1k 行),含与上游 flex-attention 的数值对拍测试 | M0 的任务从"新写"改为"迁移 + 按 §2 的代码规则重构 + 删除因 R1–R3 而多余的部分" |

## 2. 本次接入遵守的代码规则(每个里程碑的 review 清单)

来自需求方的四条硬约束,逐条落到可检查项:

| 规则 | 检查项 |
|---|---|
| **C1 复用优先** | 凡 §4 复用清单中列出的符号必须 import 使用,不得复制粘贴一份;新写代码只能出现在 §4 "新写"列 |
| **C2 可读性** | 不写多元链式表达式(一个语句一件事);条件分支用 `if/else` 展开而不是嵌套三元;张量形状变换处写明形状注释 |
| **C3 代码质量** | 不引入 ≤5 行且只有一个调用点的 `_helper`;不用 `types.SimpleNamespace` / `type("Config", (), {...})` 造对象,配置用 `dataclass(frozen=True)`;公开签名不写 `Any`(与仓库既有接口对齐处除外,如 `post_process_func(output, output_type)`) |
| **C4 不改核心** | 允许改动的文件:`registry.py` / `model_metadata.py`(纯注册项)、`model_extras/registry.py` + 新增 `model_extras/wan_animate2.py`、`examples/offline_inference/image_to_video/image_to_video.py`(任务中立的 `--video`)、`recipes/`、`docs/`、`tests/`、`.buildkite` yaml。**禁止**改动:`diffusion/data.py`、`entrypoints/`、`attention/`、`distributed/`、`offloader/`、`models/schedulers/`、`models/wan2_2/` 共享模块 |

C4 的两处已知诱惑及处置:
- 共享 `WanImageEmbedding` 的 LayerNorm 未传 eps,继承仓库 `LayerNorm` 默认 1e-6,而上游 `MLPProj` 是 `torch.nn.LayerNorm` 默认 1e-5(dim=128 时约 9e-3 相对偏差)。**在 Animate-2 transformer 里局部覆盖 eps**,不改 `wan2_2_transformer.py`;顺手给 Wan2.2-I2V 提一个独立 issue。
- 统一 `Attention` 层不返回 LSE,参考注意力需要 LSE 合并。**在模型目录内实现带 LSE 的稠密注意力**,不给 `attention/layer.py` 加返回值。

## 3. 全局完成定义(DoD,每个里程碑都要满足)

| # | 条目 | 验证方式 |
|---|---|---|
| G1 | lint / format / typecheck 全绿 | `CONTRIBUTING.md` 的本地检查命令 |
| G2 | 本里程碑新增/修改的测试全部通过,且未跳过任何既有测试 | pytest 输出无 `skipped` 增量 |
| G3 | 无新增 model-specific Python 示例;命令写入 `recipes/Wan-AI/Wan2.2-Animate-2.md`,入口复用共享的 `image_to_video.py` | `precheck-pr` skill 的 Examples policy 维度 ✓ |
| G4 | 无死代码、无预留分支 | `precheck-pr` Dead code 维度无 ✗ |
| G5 | 精度/性能结论附实测数据与环境(卡型、驱动、torch/diffusers 版本、分辨率、帧数) | PR 描述中的表格 |
| G6 | 被本里程碑证伪的假设,同 PR 更新设计文档与本规格 | 人工 review |
| G7 | §2 的 C1–C4 逐条自查,PR 描述列出"本次改动的核心文件 = 无"或给出例外理由 | 人工 review |

## 4. 复用边界(C1 的执行依据)

以 main 为准的复用清单。左列是 Animate-2 需要的能力,中列是直接 import 的符号与位置。

### 4.1 直接复用(import,不复制)

| 能力 | 复用的符号 / 位置 | 说明 |
|---|---|---|
| 组件加载(tokenizer / UMT5 / VAE / CLIP / image_processor) | `pipeline_wan2_2_i2v.py` 构造函数中的 `prefetch_subfolders` + `from_pretrained_with_prefetch` + `DistributedAutoencoderKLWan.from_pretrained` + `CLIPVisionModel` / `CLIPImageProcessor` 模式;`models/utils._load_json` | 仓库布局与 Wan2.1-I2V 逐目录相同 |
| transformer 权重源 | `DiffusersPipelineLoader.ComponentSource(subfolder="transformer", prefix="transformer.")` | 与 I2V 同 |
| transformer 配置读取 | `wan2_2.pipeline_wan2_2.load_transformer_config(model, "transformer", local_files_only)` | 只需一个 ≤20 行的工厂把上游键名映射到构造参数 |
| CLIP 图像特征 | I2V `encode_image`(`hidden_states[-2]`)的实现方式 | M1 用 A1.2 对拍确认与 diffusers `WanImageEncoderStep` 一致 |
| 文本编码 | I2V `encode_prompt`(含 `prompt_clean`)的实现方式 | 参考提示 `prompt_ref` 与负提示同路径 |
| 主干 block 的 TP 化子层 | `wan2_2_transformer.py`:`WanCrossAttention`(含 `add_k_proj / add_v_proj / norm_added_k`)、`WanFeedForward`、`DistributedRMSNorm`、`WanTimeTextImageEmbedding`、`OutputScaleShiftPrepare`、`AdaLayerNorm`、`Conv3dLayer` | 只有 self-attention 需要新类 |
| RoPE 施加核 | `layers/rope.RotaryEmbeddingWanS2V`(接收复数频率表;main 已有 `forward_native` CPU 回退) | 频率表构造(含参考网格偏移)在模型目录内 |
| 分段循环骨架 | S2V `forward()`:段级 generator 派生 `_make_clip_generators`、每段 `scheduler.set_timesteps`、VAE patch-parallel 解码后的 `dist.broadcast`、段末 `current_omni_platform.empty_cache()` | 段长 81、重叠 1、上段末帧回灌 |
| 段级预计算的 offload / FSDP 护栏 | S2V `encode_audio` 调用处的 CPU-offload 搬运与 `unshard()/reshard()` 模式;`_should_release_dit_before_decode` + `selected_offload_components` | `extract_reference` 与 `encode_audio` 同位 |
| 逐段流式 MP4 编码 | `utils/chunked_video.ChunkedVideoMP4Session` + `wan2_2/chunked_mp4.py`(`wan_preencoded_mp4_payload`、`resolve_wan_video_codec_options`) | M5 接入;段数多的请求受益最大 |
| I2V 条件 mask | I2V `prepare_latents` 中 `cat([mask4, latent16])` 的公式 | Animate-2 的 `y` 同式,帧维再拼参考图槽位 |
| latent 归一化 | I2V / S2V 的 `_normalize_latents` / `_denormalize_latents`(`latents_mean / latents_std` 来自 VAE config) | 不再自写 `_latents_mean_std` |
| scheduler | 基础版:diffusers `DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler")`;蒸馏版:`models/schedulers.FlowMatchEulerDiscreteScheduler` | 见 R3 |
| CFG / CFG 并行 | `CFGParallelMixin.predict_noise_maybe_with_cfg(cfg_normalize=False)` | 负分支 kwargs 只换文本 embeds + `is_uncondition=True` |
| LSE 合并 | `attention/backends/ring/ring_utils.update_out_and_lse` | 若签名可直接用于"两分支合并"则 import;否则模型内写 ≤20 行的数值稳定合并,并在注释里说明为何不复用 |
| 驱动视频输入通路 | `multi_modal_data["video"]`(离线:路径;在线:serving 已按 `reference_video_decode_spec` 解码成 PIL 帧列表);`ReferenceVideoDecodeSpec` 与 Cosmos3 的 `reference_video_decode_spec` classmethod 形态 | 零 API 扩展 |
| 示例入口的模型侧钩子 | `model_extras/registry._EXTRA_SPECS`:`reference_image_size_resolver`(管线自己 letterbox,示例不得预缩放)、`video_generation_defaults_builder`(`VideoGenerationDefaults`) | 用这两项替代任何模型专属 CLI flag |
| 测试基建 | `tests/helpers/runtime.OfflineOmniClient.send_diffusion_request`、`tests/helpers/mark.hardware_test`、`tests/e2e/accuracy/helpers.py` 的 SSIM/PSNR | 见各里程碑 |

### 4.2 从上一版移植迁移并重构(R8)

来源:`.claude/worktrees/wan-animate2-model-integration-1e7870`(未提交)。迁移时按 §2 重构。

| 文件 | 处置 | 重构要点 |
|---|---|---|
| `wan_animate2/reference_attention.py` | **保留**(核心难点已通过与上游 flex-attention 的 CPU 对拍) | `attention_with_lse` 改用 `torch.nn.functional.scaled_dot_product_attention` 可返回 LSE 的公开路径或 flash 后端,不直接调 `torch.ops.aten._*` 私有算子;`_zero_key_lse` 并入调用点;`merge_attention_branches` 优先换 `update_out_and_lse` |
| `wan_animate2/reference_kv_cache.py` | **保留** | 保留 `@torch._dynamo.disable` 的理由注释;去掉 `memory_bytes()` 除非 M2 真用到 |
| `wan_animate2/wan_animate2_transformer.py` | **保留主体** | `self.config = type("Config", ...)` 改为 `@dataclass(frozen=True) class WanAnimate2TransformerConfig`;`_project / _modulation / _cross_attention_and_ffn` 三个微 helper 内联回 block 的 `forward`;`_grid_info` 返回 dataclass 而非 `dict[str,int]`;权重重映射表改为 **diffusers 命名**(见 §4.4);`_embed_tokens / _prepare_context` 保留(两阶段共用,调用点 ≥2) |
| `wan_animate2/pipeline_wan_animate2.py` | **保留骨架,删一半** | 删除:`_ANIMATE2_TRANSFORMER_CONFIG`、`load_wan_vae`、`_resolve_is_distilled` 的文件探测、原始格式路径常量、`_latents_mean_std`;`_resolve_request` 返回 dataclass;`_to_pixel_tensor` 内联;`_load_driving_video` 拆成"路径 → 帧"与"帧 → 按 fps 重采样"两个各有测试的函数 |
| `wan_animate2/wan_animate2_clip.py` | **删除**(193 行) | Diffusers 仓库提供 `CLIPVisionModel` |
| `schedulers/scheduling_flow_dpm_solver_multistep.py` | **删除**(815 行) | 见 R3;若 M1 的 A1.2 判定必须逐点复现官方 sigma 网格,再以"仓库已有 `scheduling_flow_unipc_multistep.py` 同款移植"的方式重新引入,并在 PR 中给出数据 |
| 对 `data.py` / `entrypoints/utils.py` / `schedulers/__init__.py` 的改动 | **丢弃** | R1、R3 |
| `tests/diffusion/models/wan_animate2/*` | **保留并扩展** | `test_wan_animate2_attention.py` 的上游 mask / score_mod / RoPE 转写是 A0.1–A0.5 的判定依据,逐行保留 |
| `recipes/Wan-AI/Wan2.2-Animate-2.md` | **重写** | 改为 Diffusers 仓库 id、去掉 `--model-class-name`、去掉 HF_ENDPOINT 说明 |

### 4.3 新写(只有这些)

| 模块 | 内容 | 预估行数 |
|---|---|---|
| `WanAnimate2SelfAttention` | `to_qkv`(`QKVParallelLinear`)+ `norm_q/norm_k` + RoPE + 两阶段:`extract` 存 pre-RoPE K/V,`cached` 调 `reference_context_attention` + `to_out` | ~120 |
| `WanAnimate2TransformerBlock` | 复用 `WanCrossAttention` / `WanFeedForward` / `AdaLayerNorm`,只在 self-attn 处传 `kv_cache` 与阶段标志 | ~90 |
| `WanAnimate2Transformer3DModel` | `extract_reference()` + `forward()`,负分支跳 block 9,`_repeated_blocks / _layerwise_offload_blocks_attrs / _hsdp_shard_conditions / packed_modules_mapping / _sp_plan` 声明,`load_weights` | ~300 |
| 预处理纯函数 | `resize_by_area`(letterbox,只移植闭式 sqrt 分支,见 §4.5)、`get_frame_indices`(fps 重采样)、`zigzag_padding` + `get_padding_len`、`get_i2v_mask` | ~120,全部 L1 覆盖 |
| `Wan22Animate2Pipeline` | 构造(I2V 模式)、`reference_video_decode_spec`、`diffuse`、`forward` 段循环、`_extract_reference` 护栏、`_decode` broadcast | ~450 |
| `model_extras/wan_animate2.py` | `WAN_ANIMATE2_EXTRA_BODY_PARAMS = frozenset({"prompt_ref", "segment_frame_length", "max_driving_frames"})`、`get_wan_animate2_video_generation_defaults` | ~40 |

### 4.4 权重重映射(Diffusers 命名 → vllm-omni 命名)

来自 `transformer/diffusion_pytorch_model.safetensors.index.json`(1303 个张量,4 个分片共 32.8 GB):

| checkpoint 名 | 目标 | 备注 |
|---|---|---|
| `blocks.N.self_attn.{to_q,to_k,to_v}` | `blocks.N.attn1.to_qkv`(stacked) | `packed_modules_mapping` |
| `blocks.N.self_attn.to_out.0` | `blocks.N.attn1.to_out` | |
| `blocks.N.self_attn.norm_{q,k}` | `blocks.N.attn1.norm_{q,k}` | TP 下按本地 head 切分(S2V 同款逻辑) |
| `blocks.N.cross_attn.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}` | `blocks.N.attn2.*` | `WanCrossAttention` 直载 |
| `blocks.N.cross_attn.{add_k_proj,add_v_proj,norm_added_k}` | `blocks.N.attn2.*` | 输入维 = `inner_dim`(CLIP 特征先经 `img_emb` 投影),不是 1280 |
| `blocks.N.ffn.0` / `ffn.2` | `blocks.N.ffn.net_0.proj` / `ffn.net_2` | **裸索引**,不是 `ffn.net.0` |
| `blocks.N.modulation` / `norm3` | `blocks.N.scale_shift_table` / `norm2` | |
| `text_embedding.{0,2}`、`time_embedding.{0,2}`、`time_projection.1` | `condition_embedder.{text_embedder.linear_{1,2}, time_embedder.linear_{1,2}, time_proj}` | |
| `img_emb.proj.{0,1,3,4}` | `condition_embedder.image_embedder.{norm1, ff.net.0.proj, ff.net.2, norm2}` | eps 局部覆盖为 1e-5 |
| `head.head` / `head.modulation` | `proj_out` / `output_scale_shift_prepare.scale_shift_table` | |
| `patch_embedding` | `patch_embedding` | Conv3d(36→5120) |

A1.1 要求零 missing / 零 unexpected。

### 4.5 必须复刻的上游"看似 bug"行为

四条均由上游源码逐行核实,任何一条按"更合理"的方式实现都会静默偏移每一帧:

1. `log_scale` 偏置作用于生成侧 latent 帧 1 的 key 块(R4)。
2. 零填充 key 槽位参与 softmax 分母(R5);`origin_area` 取请求的 `width*height`,不是 letterbox 后的网格。
3. 上游 `calculate_new_size` 因 `check_valid` 参数数目错误必然抛 `TypeError`,`resize_by_area` 的裸 `except` 永远走闭式 sqrt 分支 —— 只移植该分支。
4. Wan VAE 是因果的,驱动视频**逐段**编码,不能整段编码后切 latent。

## 5. 前置资产与环境

| 资产 | 说明 |
|---|---|
| 代码基线 | 本分支 rebase 到 main(R7);上一版移植的 worktree 保持只读作为迁移来源 |
| 权重 | `Wan-AI/Wan2.2-Animate-2-14B-Diffusers`(基础)与 `...-Distilled-Diffusers`(蒸馏),HF 与 ModelScope 均可下载,各约 32.8 GB DiT + 约 10 GB 组件 |
| 对拍基线环境 | diffusers ≥ 0.40(`WanAnimate2ModularPipeline`)+ torch ≥ 2.5(flex-attention);**只用于产出基线视频**,不进入运行时依赖。官方仓库 `Wan-Video/Wan-Animate-2` 作为第二基线(需 `flash_attn`) |
| 测试素材 | 参考人像 ×1;驱动视频 ×2(≤1 段长用于 smoke,≥2 段长用于接缝);官方 demo 素材在 `Wan-Video/Wan-Animate-2/examples` |
| 显存 | 见 §6 估算;M1 单张 80 GB 卡即可(640×800);M2 起需 ≥2 卡或 96 GB 卡验证 720×1280 |

## 6. 显存估算(估算值,M2 以实测替换)

KV cache = `40 层 × T_ref × hw × 5120 × 2(K,V) × 2 B`,`T_ref = 21`(81 帧段):

| 目标面积 | 每帧 token(hw) | 每层 K+V | 40 层 | + bf16 权重 28 GB |
|---|---|---|---|---|
| 640×800(diffusers modular 默认) | 40×50 = 2000 | 0.86 GB | **34 GB** | 62 GB → 单张 80 GB 可装 |
| 720×1280(官方 demo 默认) | 45×80 = 3600 | 1.55 GB | **62 GB** | 90 GB → 单卡不可装 |

720×1280 的可选策略按"改动最少"排序:TP=2(cache 与权重按 head 对半,45 GB/卡,零新代码)→ fp8 KV cache(31 GB,需精度验证)→ 逐层流式 offload(每步 H2D 62 GB,对 40 步基础版不可接受)。M2 只需证明其一,并把结论写进 recipe。

## 7. 里程碑总览

| 里程碑 | 主题 | 核心风险 | 依赖 |
|---|---|---|---|
| **M0** | rebase + 迁移重构 + 参考注意力/两阶段骨架的 CPU 对拍 | 重构后仍与上游数值等价 | 无 |
| **M1** | 单卡最小可运行(蒸馏版、640×800) | 权重重映射、scheduler 网格差异、分段接缝 | M0 |
| **M2** | 显存与全分辨率(720×1280) | 62 GB KV cache | M1 |
| **M3** | TP + CFG 并行 + 基础版(40 步 CFG 3.0) | 负分支跳 block 9 的非对称 | M2 |
| **M4** | SP/USP + HSDP | 参考注意力在序列分片下的正确性 | M3 |
| **M5** | 在线服务 + 流式分段输出 + 文档 + L3/L4 + recipes | 请求级时长上限;在线视频帧率信息 | M3 |
| **M6** | 可选:原始格式 / Cache-DiT / flex 后端 / 真批处理 | — | M5 |

---

## M0 — 迁移重构与数值骨架(无 GPU)

**目标**:在 main 上落下 `vllm_omni/diffusion/models/wan_animate2/` 目录,参考注意力、RoPE、
KV cache、transformer 骨架与上游数值等价,并满足 §2 代码规则。

**范围内**:rebase;§4.2 的迁移与重构;§4.3 中的 transformer 三个类与预处理纯函数;L1 测试。
**范围外**:真权重、pipeline 端到端、并行。

**交付物**:
- `wan_animate2/{__init__,reference_attention,reference_kv_cache,wan_animate2_transformer}.py`
- `tests/diffusion/models/wan_animate2/{test_wan_animate2_attention,test_wan_animate2_transformer}.py`
- `registry.py` / `model_metadata.py` 注册项(`WanAnimate2Pipeline`)

| # | 标准 | 验证方式 |
|---|---|---|
| A0.1 `[BLOCKING]` | 小配置(≤4 层 / 4 头 / 短序列)随机权重,`reference_context_attention` vs 上游 `create_mask + score_mod` 的稠密转写:fp32 max abs diff ≤ 1e-5 | pytest,含无填充 / 末段短填充 / 空间填充 / 双填充四组用例 |
| A0.2 `[BLOCKING]` | `log_scale=-1.3` 时同样满足 A0.1,且偏置命中的是生成帧 1 块(R4) | 参数化用例 |
| A0.3 | 帧对齐语义:帧 f 的 query 对参考帧 g≠f-1 的权重恒为 0;帧 0 不 attend 任何参考 token | 直接检查权重矩阵 |
| A0.4 | KV cache 不在 `state_dict()` / `named_parameters()` 中;`release()` 后无张量引用 | 单测 |
| A0.5 | RoPE 频率表 vs 上游 float64 `torch.polar` 逐样本实现:fp32 max abs diff ≤ 1e-5(含参考网格 t/w 偏移与 `refer_stride`) | 单测 |
| A0.6 | 负分支 `is_uncondition=True` 恰好跳过 block 9(39 vs 40 次 block 调用) | 单测 |
| A0.7 | 权重重映射按 §4.4 覆盖 §4.4 表中全部模式;`load_weights` 对未知键报错 | 单测(不需真权重,构造名单) |
| A0.8 | 上述测试标记 `core_model` + `cpu`,无 GPU 可跑,单次 ≤ 3 分钟 | CI `test-ready.yml` |
| A0.9 | §2 C1–C4 自查通过:模型目录内无 `type(...)` 动态类、无 `SimpleNamespace`、公开签名无 `Any`;`git diff --stat main` 只含 C4 允许的文件 | PR 描述列出 diff 文件清单 |

**门禁**:A0.1–A0.9 全绿。
**预案**:A0.1/A0.2 若在重构后回退,回滚到上一版移植的对应函数逐行比对,不得放宽阈值。

---

## M1 — 单卡最小可运行(蒸馏版、640×800)

**目标**:Diffusers 仓库真权重端到端出片,证明"算得对";用 diffusers modular 默认面积规避显存墙。

**范围内**:pipeline(I2V 式组件加载、pre/post process、段循环)、`--video` 接入共享入口、
`model_extras/wan_animate2.py`、recipe 首版。
**范围外**:720×1280、并行、在线服务、流式输出。

| # | 标准 | 验证方式 |
|---|---|---|
| A1.1 `[BLOCKING]` | 两个 Diffusers 仓库均零 missing / 零 unexpected key;蒸馏版通过 `modular_model_index.json` 自动判定(`log_scale=-1.3`、默认 10 步、`guidance_scale=1.0`),基础版默认 40 步 / 3.0 | 加载日志断言 + 单测 |
| A1.2 `[BLOCKING]` | 逐层激活对拍 vs diffusers modular(block 0/1/2 与末层):bf16 max abs diff ≤ 2e-2,无逐层放大;**CLIP 特征**(`hidden_states[-2]`)与 diffusers `WanImageEncoderStep` 一致;**sigma 网格**:记录 diffusers `DPMSolverMultistepScheduler` / `FlowMatchEulerDiscreteScheduler` 与官方 `get_sampling_sigmas(steps, 5.0)` 的最大差值(预期 ~1e-3),并在 PR 中给出"沿用 diffusers scheduler"或"移植官方 solver"的决定与依据 | 对拍脚本(置于 `tests/` 下),固定 seed |
| A1.3 `[BLOCKING]` | 单段端到端 vs diffusers modular 参考视频:SSIM ≥ 0.94 且 PSNR ≥ 28.0 dB | `tests/e2e/accuracy/helpers.py` |
| A1.4 `[BLOCKING]` | ≥2 段驱动视频:段边界相邻帧 SSIM ≥ 段内相邻帧 SSIM 均值 × 0.98 | 逐帧 SSIM 脚本 |
| A1.5 | 段末释放 KV cache 后显存回落到段初 ±5% | `max_memory_allocated` 逐段记录 |
| A1.6 | 输出帧数 == 驱动视频重采样后帧数;letterbox 与 zigzag padding 均裁回 | 端到端断言 |
| A1.7 | 同 seed 两次运行逐比特一致 | 端到端 |
| A1.8 | 预处理纯函数 L1 全绿(`resize_by_area` 闭式分支、`get_frame_indices`、`zigzag_padding` 反弹而非钳位、`get_padding_len` 末段可用、`get_i2v_mask` 形状) | `core_model` + `cpu` |
| A1.9 | 共享入口:`image_to_video.py --model Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers --image ... --video ...` 不带 `--model-class-name` 即可运行;`--extra-body '{"prompt_ref": ...}'` 经 `apply_declared_extra_args` 到达管线;示例不预缩放参考图(`reference_image_size_resolver`) | 按 recipe 实跑 |
| A1.10 | `supports_request_batch=False`,两个请求串行执行且结果与单独执行一致 | 端到端 |

**门禁**:A1.1–A1.10 全绿。PR 记录实际分辨率、帧数与显存峰值,作为 M2 输入。

---

## M2 — 显存与全分辨率

**目标**:720×1280 / 81 帧段跑通,并给出单卡 80 GB 的明确结论(能装 / 需 TP=2)。

| # | 标准 | 验证方式 |
|---|---|---|
| A2.1 `[BLOCKING]` | 720×1280 在以下之一跑通并写入 recipe:单张 80 GB 卡(需策略)或 TP=2;不依赖 `expandable_segments` 之外的分配器设置 | 端到端 |
| A2.2 `[BLOCKING]` | 相对 M1 无质量回归:SSIM ≥ 0.94 / PSNR ≥ 28.0 | 同 A1.3 |
| A2.3 | 显存分项实测表(权重 / KV cache / 激活),与 §6 估算对照 | `memory_stats` |
| A2.4 | `--enable-cpu-offload` 与 `--enable-layerwise-offload` 均跑通,输出与不开 offload:SSIM ≥ 0.99 / PSNR ≥ 40;`extract_reference` 在 offload 下不触发 hook 外的设备错误 | 端到端 |
| A2.5 | 若采用 fp8 KV cache 等有损策略,必须给出与 bf16 cache 的 SSIM/PSNR | 端到端 |
| A2.6 | 记录端到端耗时作为 M3/M4 基线 | 计时 |

**预案**:单卡 80 GB 无策略可装时,recipe 明确"720×1280 需 ≥2 卡",不得隐瞒。

---

## M3 — TP、CFG 并行与基础版

| # | 标准 | 验证方式 |
|---|---|---|
| A3.1 `[BLOCKING]` | TP=2 / TP=4 vs 单卡:SSIM ≥ 0.99 且 PSNR ≥ 40 dB | 端到端 |
| A3.2 `[BLOCKING]` | CFG 并行(2 卡)vs 单卡 CFG:SSIM ≥ 0.99 / PSNR ≥ 40;负分支确实 39 层 | 端到端 + 单测 |
| A3.3 | KV cache 在 TP 下按本地 head 切分,extract 与 gen 布局一致,无额外通信 | 单测 + 通信 profile |
| A3.4 | 基础版(40 步、CFG 3.0、全分辨率)vs diffusers modular:SSIM ≥ 0.94 / PSNR ≥ 28.0 | 同 A1.3 |
| A3.5 | TP=2 相对单卡加速比 ≥ 1.5×,低于则附 profile | 计时 |
| A3.6 | L3 smoke:`tests/e2e/offline_inference/test_wan_animate2.py`,baseline 用例双标 `core_model` + `advanced_model` | CI `test-merge.yml` |
| A3.7 | 蒸馏版 `guidance_scale=1.0` 时 CFG 并行优雅禁用并给出日志 | 单测 |

---

## M4 — 序列并行与 HSDP

`_sp_plan` 由 `registry.initialize_model` 通过 `apply_sequence_parallel` 装 hook,
`extract_reference` 走 `blocks[i].__call__`,hook 会生效;首版策略:extract 分片跑,
结束后 all-gather 各层 K/V(各 rank 全量持有),去噪期分支 B 无通信。

| # | 标准 | 验证方式 |
|---|---|---|
| A4.1 `[BLOCKING]` | USP=2 / 4 vs 单卡:SSIM ≥ 0.99 / PSNR ≥ 40 | 端到端 |
| A4.2 `[BLOCKING]` | 序列长度不能被 sp_size 整除(`auto_pad`)时仍满足 A4.1 | 非整除分辨率用例 |
| A4.3 | 分片后帧对齐语义正确 | 单测 |
| A4.4 | HSDP 下单卡权重占用下降;`extract_reference` 的 `unshard/reshard` 护栏无 FSDP 状态错误 | 端到端 + 显存 |
| A4.5 | TP + SP 组合跑通,或明确记录约束 | 端到端 |
| A4.6 | HSDP 与 TP 互斥有显式校验与友好报错 | 单测 |

---

## M5 — 在线服务、流式输出、文档与 L4

| # | 标准 | 验证方式 |
|---|---|---|
| A5.1 `[BLOCKING]` | `/v1/videos` 以 `image_reference` + `video_reference` 出片,未新增 API 字段;`model_metadata` 声明 `supports_mixed_reference_inputs=True` | 在线 e2e + `test_serving_api_surface` 无变更 |
| A5.2 `[BLOCKING]` | 在线 vs 离线同参数:SSIM ≥ 0.94 / PSNR ≥ 28.0 | `tests/e2e/accuracy/` |
| A5.3 | 超长驱动视频有明确准入拒绝(`reference_video_decode_spec` 的 `max_frames`,默认 30 s × 24 fps),错误信息可执行 | 在线负例 |
| A5.4 | **在线帧率语义**:serving 交给管线的是解码后的帧列表,不含源 fps;spec 必须写明并验证"在线请求按 `fps` 参数视为已重采样"或在 serving 层拿到源 fps,二选一并记录 | 在线 e2e |
| A5.5 | 逐段流式输出:段解码完成即经 `ChunkedVideoMP4Session` 编码,峰值主机内存不随段数线性增长 | 内存记录 |
| A5.6 | L4:`tests/e2e/online_serving/test_wan_animate2_expansion.py`,参数化覆盖 基础版×40 步 / TP2 / USP2 / cpu-offload 各 ≥1 行,标记 `full_model` + `diffusion` | CI `test-nightly.yml` |
| A5.7 | 文档:`docs/models/supported_models.md` 加行;`docs/user_guide/diffusion_features.md` VideoGen 矩阵加行且每格与实测一致 | 人工对照 |
| A5.8 | recipe 覆盖 蒸馏版单卡 与 基础版多卡,命令可直接复制执行 | 实跑 |
| A5.9 | 设计文档与本规格中被证伪的假设已更新 | 人工 review |

**门禁**:A5.1–A5.9 全绿 → 主线交付完成。

---

## M6 — 可选增强(独立立项)

| 项 | 触发条件 | 粗验收 |
|---|---|---|
| 原始格式 checkpoint(`Wan-AI/Wan2.2-Animate-2-14B`,单文件 DiT + `.pth` T5/VAE/CLIP) | 用户明确要求 | 与 Diffusers 格式输出 SSIM ≥ 0.99;上一版移植的 `wan_animate2_clip.py` 与 S2V `_init_original_format` 模式可在此复活 |
| Cache-DiT | 基础版 40 步有加速诉求 | 加速比 ≥ 1.3× 且 SSIM ≥ 0.94;需 S2V 式定制 `CachedBlocks`;首版把 `WanAnimate2Pipeline` 加入 `_NO_CACHE_ACCELERATION` |
| flex-attention 可选后端 | LSE 双注意力在 M2/M3 profile 中成为瓶颈 | 与 LSE 实现 SSIM ≥ 0.99;torch 内置,不引入新依赖 |
| 真批处理 | 服务吞吐成瓶颈 | KV cache 批维隔离;吞吐实测 |
| Wan2.2-Animate-14B(v1) | 有 v1 需求 | 独立管线,见设计文档附录 A |

---

## 8. 风险项收敛对照

| 风险(设计文档 §8) | 状态 | 收敛于 |
|---|---|---|
| 1. flex-attention 替换的数值等价 | 上一版移植已在 CPU 对拍通过;重构后需重新过 | M0 A0.1/A0.2 |
| 2. scheduler 数值对齐 | 差异已定位(sigma 首点 0.999 vs 1.0) | M1 A1.2 决策 |
| 3. KV cache 显存 | 估算见 §6;640×800 可单卡 | M2 A2.1/A2.3 |
| 4. modular 索引探测 | **消失**(R1) | — |
| 5. 批处理 | `supports_request_batch=False` | M1 A1.10 → M6 |
| 6. 驱动视频时长 | `reference_video_decode_spec` | M5 A5.3 |
| 7.(新)在线帧率信息缺失 | serving 只传帧列表 | M5 A5.4 |
| 8.(新)共享 `WanImageEmbedding` eps 偏差 | 局部覆盖,不改共享代码 | M0(单测断言 eps=1e-5)+ 给 I2V 提 issue |
