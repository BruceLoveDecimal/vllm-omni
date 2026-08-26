# Wan2.2-Animate-2 接入实施规格(Milestone & 验收标准)

> 配套设计文档:[wan_animate_integration.md](wan_animate_integration.md)(架构与技术选型)。
> 本文只回答两个问题:**分几步做**、**每步做到什么程度算完成**。
> 设计文档的 §x.y 引用均指向该文档。

## 0. 使用说明

- 每个里程碑的验收标准编号为 `A<里程碑>.<序号>`,均为**可执行验证**:标注了验证方式
  (pytest / 脚本 / 实测记录),不接受"看起来对了"。
- **门禁(Exit Gate)** 指该里程碑的全部验收标准通过后方可进入下一里程碑;
  标记 `[BLOCKING]` 的项不通过则必须触发预案,不得带病推进。
- 里程碑 M0–M5 为主线(对应设计文档 §9 的 P0–P3 细化),M6 为可选独立立项。

## 1. 全局完成定义(DoD,适用于每一个里程碑)

任一里程碑声明完成时,以下条目必须同时成立:

| # | 条目 | 验证方式 |
|---|---|---|
| G1 | 仓库 lint / format / typecheck 全绿 | 按 `CONTRIBUTING.md` 的本地检查命令 |
| G2 | 该里程碑新增/修改的测试全部通过,且**未跳过**任何既有测试 | pytest 输出无 `skipped` 增量 |
| G3 | **无新增 model-specific Python 示例**——不新建 `examples/offline_inference/wan_animate2/` 一类目录;命令写入 `recipes/Wan-AI/Wan2.2-Animate-2.md`,入口复用共享的 `examples/offline_inference/image_to_video/image_to_video.py` | `precheck-pr` skill 的 Examples policy 维度为 ✓ |
| G4 | 无死代码:未被调用的分支/参数/回退路径不得残留 | `precheck-pr` Dead code 维度无 ✗ |
| G5 | 精度/性能类结论必须附**实测数据与环境**(卡型、驱动、torch/diffusers 版本、分辨率、帧数),不得写"约"、"大致" | PR 描述中的表格 |
| G6 | 设计文档与本规格中被本里程碑证伪的假设,**同 PR 更新**,不留过期描述 | 人工 review |

## 2. 验收所需的前置资产与环境

这些是**开工前必须先备齐**的,否则 M1 起的验收无法执行:

| 资产 | 说明 | 备注 |
|---|---|---|
| 权重 | `Wan2.2-Animate-2-14B-Diffusers` 与 `...-Distilled-Diffusers` | **本调研环境 HF / ModelScope 均被出口代理封锁**,需在可联网环境预先拉取并落到共享存储 |
| 基线环境 | diffusers ≥ 0.40.0 + torch ≥ 2.5(flex-attention 可用) | 仅用于产出对拍基线,不进入运行时依赖 |
| 测试素材 | 参考人像图 ×1;驱动视频 ×2(**一段 ≤1 段长用于 smoke,一段 ≥2 段长用于分段接缝验证**) | 官方 demo 素材在 `github.com/Wan-Video/Wan-Animate-2` |
| 显存 | M2 起需要 ≥1 张 80GB 卡;M3 起需要 ≥2 卡;M4 需要 ≥4 卡 | KV cache 显存墙见设计文档 §8 风险 3 |

## 3. 里程碑总览

| 里程碑 | 主题 | 核心风险 | 依赖 |
|---|---|---|---|
| **M0** | 参考注意力 + 两阶段前向骨架(数值对拍) | 去 flex-attention 化是否数值等价 | 无(不需真权重/大显存) |
| **M1** | 单卡最小可运行(蒸馏版、降分辨率) | 权重重映射、分段接缝 | M0 |
| **M2** | KV cache 显存方案 + 全分辨率 | 62GB 显存墙 | M1 |
| **M3** | TP + CFG 并行 + 基础版(40 步 CFG 3.0) | CFG 负分支跳 block 9 的非对称性 | M2 |
| **M4** | SP/USP + HSDP | 参考注意力在序列分片下的正确性 | M3 |
| **M5** | 在线服务 + 文档 + L4 + recipes | 请求级时长上限与准入 | M3 |
| **M6** | 可选:Cache-DiT / 原始格式 / modular 索引 / 真批处理 | — | M5 |

**排序理由**:M0 把设计中最不确定的一环(§5.3 去 flex-attention 化)提到最前,
且它不依赖真权重与大显存,可最早证伪;M2 独立成里程碑而非并入 M1,是因为
62GB 的 KV cache 显存墙会让全分辨率单卡在 M1 阶段直接跑不起来——**M1 必须降分辨率**,
把"能不能算对"与"能不能装下"两个问题解耦。

---

## M0 — 参考注意力与两阶段前向骨架

**目标**:在不依赖真权重、不依赖大显存的前提下,证明"去 flex-attention 化的帧对齐
参考注意力"与上游数值等价,并确定两阶段前向 + KV cache 的 API 形态。

**范围内**:`wan_animate2_transformer.py` 骨架(层数/头数可配小)、
`reference_kv_cache.py`、LSE 双注意力实现、RoPE 向量化、L1 单测。
**范围外**:真权重加载、pipeline、并行、任何端到端出片。

**交付物**:
- `vllm_omni/diffusion/models/wan_animate2/wan_animate2_transformer.py`(骨架 + 注意力)
- `vllm_omni/diffusion/models/wan_animate2/reference_kv_cache.py`
- `tests/diffusion/models/wan_animate2/test_wan_animate2_attention.py`
- 对拍脚本(置于 `tests/` 下,非 `examples/`)

**验收标准**:

| # | 标准 | 验证方式 |
|---|---|---|
| A0.1 `[BLOCKING]` | 小配置(≤4 层 / 4 头 / 短序列)随机权重下,LSE 合并双注意力输出 vs 上游 flex-attention 参考实现:**fp32 下 max abs diff ≤ 1e-5**,相对误差 ≤ 1e-4 | pytest,两实现同输入同权重 |
| A0.2 `[BLOCKING]` | 蒸馏版 `log_scale=-1.3` 路径同样满足 A0.1 阈值(验证 score bias 与 LSE 权重缩放的等价性) | 同上,参数化用例 |
| A0.3 | **帧对齐掩码语义**:第 f 帧的 query 对第 g≠f 参考帧 token 的有效注意力权重恒为 0;latent 帧 0(参考图槽位)不 attend 任何参考 token | 单测直接检查注意力权重矩阵 |
| A0.4 | **KV cache 生命周期**:cache 不挂在 `nn.Module` 上(不出现在 `state_dict()` / `named_parameters()` 中);段末显式释放后引用归零 | 单测断言 + `gc` 引用检查 |
| A0.5 | RoPE 向量化实现 vs 官方逐样本复数实现:fp32 max abs diff ≤ 1e-5(含参考网格的 t/w 偏移) | 单测 |
| A0.6 | 上述测试标记 `core_model` + `cpu`,**可在无 GPU 环境执行**,单次运行 ≤ 3 分钟 | CI `test-ready.yml` |

**门禁**:A0.1–A0.6 全绿。
**预案**:若 A0.1 或 A0.2 无法达标,不得强行推进——改走"flex-attention 作为可选
backend + 双注意力为默认兜底"的双轨方案(设计文档 §5.3 备选),并同步更新设计文档;
该预案会把 torch ≥ 2.5 与 `torch.compile` 提升为硬依赖,需在 M1 前明确记录。

---

## M1 — 单卡最小可运行(蒸馏版、降分辨率)

**目标**:真权重端到端出片,证明"算得对";**刻意降分辨率**以绕开 KV cache 显存墙。

**范围内**:Diffusers 格式组件加载、权重重映射、pipeline 分段循环、pre/post process、
registry 注册、`--video` 开关接入共享 I2V 入口。优先蒸馏版(10 步,迭代快)。
**范围外**:全分辨率、offload、任何并行、在线服务。

**交付物**:
- `pipeline_wan_animate2.py`、`__init__.py`
- `registry.py` / `model_metadata.py` 注册
- `examples/offline_inference/image_to_video/image_to_video.py` 新增任务中立的 `--video`
- `tests/diffusion/models/wan_animate2/test_wan_animate2_pipeline.py`
- `recipes/Wan-AI/Wan2.2-Animate-2.md`(蒸馏版单卡配方)

**验收标准**:

| # | 标准 | 验证方式 |
|---|---|---|
| A1.1 `[BLOCKING]` | 权重加载**零 unexpected key、零 missing key**(含 `ffn.0/2` 裸索引、`to_qkv` 融合、`img_emb.proj.*`) | 加载日志断言 + 单测 |
| A1.2 `[BLOCKING]` | 逐层激活对拍 vs diffusers 参考实现(取 block 0/1/2 与末层输出):**bf16 下 max abs diff ≤ 2e-2**,且无逐层放大趋势 | 对拍脚本,固定 seed 与输入 |
| A1.3 `[BLOCKING]` | 端到端单段出片,与 diffusers 参考视频:**SSIM ≥ 0.94 且 PSNR ≥ 28.0 dB**(沿用仓库 Wan2.2-I2V 的既有阈值口径) | `tests/e2e/accuracy/helpers.py` 的 ffmpeg SSIM/PSNR |
| A1.4 `[BLOCKING]` | **分段接缝**:≥2 段的驱动视频,段边界处相邻帧的 SSIM 不低于段内相邻帧 SSIM 均值的 0.98 倍(即接缝无可见跳变) | 逐帧 SSIM 序列分析脚本 |
| A1.5 | 段间 KV cache 与中间张量释放后,显存回落到段初水平 ±5%(无跨段累积泄漏) | `torch.cuda.max_memory_allocated` 逐段记录 |
| A1.6 | 输出帧数 == 驱动视频原始帧数(zigzag padding 与 letterbox 均已正确裁回) | 单测 + 端到端断言 |
| A1.7 | 复现性:同 seed 两次运行输出逐比特一致 | 端到端跑两次比对 |
| A1.8 | L1 单测全绿(权重重映射、段数计算、`y` 条件形状、batch 拒绝、`_sp_plan` 结构声明) | pytest `core_model` + `cpu` |

**门禁**:A1.1–A1.8 全绿。本里程碑**必须在 PR 描述中记录实际使用的分辨率与帧数**,
以及该配置下的显存峰值——它是 M2 的输入基线。

---

## M2 — KV cache 显存方案与全分辨率

**目标**:攻克设计文档 §8 风险 3(720×1280/81 帧段约 62GB KV cache),
让全分辨率在单卡跑通。这是**性能设计的第一优先项**。

**范围内**:KV cache 的显存策略(逐层流式 offload / 分片 / 低精度 cache 三选一或组合)、
CPU offload 与 layerwise offload 接入、`SupportsComponentDiscovery` 声明生效。
**范围外**:多卡并行。

**交付物**:KV cache 显存策略实现 + 开关;offload 路径打通;显存实测报告。

**验收标准**:

| # | 标准 | 验证方式 |
|---|---|---|
| A2.1 `[BLOCKING]` | **720×1280 / 81 帧段在单张 80GB 卡上跑通**,不 OOM,不依赖 `expandable_segments` 之外的特殊分配器设置 | 端到端运行 |
| A2.2 `[BLOCKING]` | 相对 M1,输出**无质量回归**:同配置下与 diffusers 参考仍满足 SSIM ≥ 0.94 / PSNR ≥ 28.0 | 同 A1.3 |
| A2.3 | 显存峰值实测并记录:给出 KV cache 部分、权重部分、激活部分的**分项占用**,以及所选策略相对朴素全驻留的**节省比例** | `torch.cuda.memory_stats` 分项统计表 |
| A2.4 | `--enable-cpu-offload` 与 `--enable-layerwise-offload` 两条路径均能跑通且输出与不开 offload 一致(SSIM ≥ 0.99 / PSNR ≥ 40,近似逐比特) | 端到端对比 |
| A2.5 | 显存策略的开关有明确默认值,且默认值在 80GB 卡上开箱可用(用户无需手工调参) | 默认配置端到端 |
| A2.6 | 记录该配置下的端到端耗时,作为 M3/M4 加速比的基线 | 计时记录 |

**门禁**:A2.1–A2.6 全绿。
**预案**:若单卡 80GB 无论如何装不下全分辨率,则**降级目标**为"单卡支持到 X 分辨率、
全分辨率要求 ≥2 卡",并把该结论写回设计文档 §8 与 `recipes`,不得隐瞒。

---

## M3 — 张量并行、CFG 并行与基础版

**目标**:多卡跑通,并让基础版(40 步 + CFG 3.0)达到可用质量。

**范围内**:TP(主干免费获得,需验证)、CFG 并行(注意负分支跳 block 9 的非对称)、
基础版参数路径、L3 smoke 测试。
**范围外**:序列并行、HSDP。

**验收标准**:

| # | 标准 | 验证方式 |
|---|---|---|
| A3.1 `[BLOCKING]` | TP=2 与 TP=4 输出 vs 单卡:SSIM ≥ 0.99 且 PSNR ≥ 40 dB(TP 为无损并行,阈值显著严于跨实现对拍) | 端到端对比 |
| A3.2 `[BLOCKING]` | CFG 并行(2 卡)输出 vs 单卡 CFG:SSIM ≥ 0.99 / PSNR ≥ 40 dB;**且负分支确实跳过 block 9**(非"两分支同构"的错误实现) | 端到端对比 + 单测断言分支层数 39 vs 40 |
| A3.3 | KV cache 在 TP 下按本地 head 切分,extract 与 gen 两阶段布局一致,**无额外跨卡通信** | 单测 + 通信量 profile |
| A3.4 | 基础版(40 步、CFG 3.0、全分辨率)端到端出片,与 diffusers 参考满足 SSIM ≥ 0.94 / PSNR ≥ 28.0 | 同 A1.3 |
| A3.5 | TP=2 相对单卡的加速比 ≥ 1.5×(记录实测;低于该值需给出 profile 与原因分析) | 计时 + torch profiler |
| A3.6 | L3 smoke 测试落地:`tests/e2e/offline_inference/test_wan_animate2.py`,baseline 用例同函数标注 `core_model` + `advanced_model` | pytest,CI `test-merge.yml` |
| A3.7 | 蒸馏版无 CFG,`guidance_scale=1.0` 时 CFG 并行**优雅禁用**并给出明确日志,而非静默跑两遍 | 单测 |

**门禁**:A3.1–A3.7 全绿。

---

## M4 — 序列并行与 HSDP

**目标**:序列并行下参考注意力仍然正确;HSDP 降低单卡权重占用。

**范围内**:`_sp_plan` 生效、KV cache 的 SP 策略(设计文档 §6.2 方案 1:各 rank
全量持有 + extract 后 all-gather)、HSDP 分片与 `extract_reference` 的
`unshard/reshard` 护栏。

**验收标准**:

| # | 标准 | 验证方式 |
|---|---|---|
| A4.1 `[BLOCKING]` | USP=2 / USP=4 输出 vs 单卡:SSIM ≥ 0.99 / PSNR ≥ 40 dB | 端到端对比 |
| A4.2 `[BLOCKING]` | 序列长度**不能被 sp_size 整除**时(`auto_pad` 路径)输出仍满足 A4.1 阈值 | 构造非整除分辨率的用例 |
| A4.3 | 参考注意力在 SP 下的帧对齐语义保持正确(分片后仍只 attend 同帧参考 token) | 单测直接检查分片后的注意力权重 |
| A4.4 | HSDP 开启后单卡权重占用下降,且 `extract_reference`(在 forward 钩子外调用)不报 FSDP 状态错误 | 端到端 + 显存记录 |
| A4.5 | TP + SP 组合跑通(若组合受限,明确记录约束并在文档标注) | 端到端 |
| A4.6 | HSDP 与 TP 的互斥约束(设计文档 §6.6 / skill 已知限制)有显式校验与友好报错,而非运行时崩溃 | 单测 |

**门禁**:A4.1–A4.6 全绿。

---

## M5 — 在线服务、文档与 L4

**目标**:可对外服务,文档与测试矩阵完整。

**范围内**:在线 serving 打通(零 API 扩展)、`ReferenceVideoDecodeSpec` 与请求级
时长上限、L4 扩展测试、文档矩阵、recipes 补全。

**验收标准**:

| # | 标准 | 验证方式 |
|---|---|---|
| A5.1 `[BLOCKING]` | `/v1/videos` 以 `image_reference` + `video_reference` 端到端出片,**未新增任何 API 字段** | 在线 e2e + API surface 守卫测试 `test_serving_api_surface` 无变更 |
| A5.2 `[BLOCKING]` | 在线输出 vs 离线同参数输出:SSIM ≥ 0.94 / PSNR ≥ 28.0(沿用仓库既有在线/离线一致性口径) | `tests/e2e/accuracy/` 相似度用例 |
| A5.3 | 驱动视频超长时有**明确的准入拒绝**(帧数/时长上限),错误信息可执行,而非 OOM 或静默截断 | 在线 e2e 负例 |
| A5.4 | 批准入:不同段数/分辨率/fps 的请求**不被同批**,`batch_compatibility_key` 生效 | 单测 |
| A5.5 | L4 扩展测试落地:`tests/e2e/online_serving/test_wan_animate2_expansion.py`,参数化覆盖 基础版×40步 / TP2 / USP2 / cpu-offload 至少各一行,标记 `full_model` + `diffusion` | pytest,CI `test-nightly.yml` |
| A5.6 | 文档更新齐备:`docs/models/supported_models.md` 新增行;`docs/user_guide/diffusion_features.md` VideoGen 矩阵新增行且**每一格与实测一致**(不得照抄设计文档的"预期") | 人工 review 对照实测 |
| A5.7 | `recipes/Wan-AI/Wan2.2-Animate-2.md` 覆盖 基础版多卡 与 蒸馏版单卡 两个配方,命令可直接复制执行 | 按 recipe 实跑一遍 |
| A5.8 | 设计文档中所有被实现证伪的假设已更新(尤其 §6.5 Cache-DiT 结论、§8 各风险项的实际结论) | 人工 review |

**门禁**:A5.1–A5.8 全绿 → **模型正式可用,主线交付完成**。

---

## M6 — 可选增强(独立立项,不阻塞主线)

以下每项均可独立成 PR,验收标准在立项时细化:

| 项 | 触发条件 | 粗验收 |
|---|---|---|
| Cache-DiT 支持 | 基础版 40 步场景有明确加速诉求 | 加速比 ≥ 1.3× 且 SSIM ≥ 0.94 vs 未开缓存;需定制 CachedBlocks(设计文档 §6.5) |
| 原始格式 checkpoint | 用户要求直接加载官方非 Diffusers 权重 | 与 Diffusers 格式输出一致(SSIM ≥ 0.99);含 xlm-roberta CLIP 加载 |
| `modular_model_index.json` 探测 | 希望免去显式 `--model-class-name` | 不传 `--model-class-name` 可自动解析;不破坏既有 `model_index.json` 路径 |
| 真批处理(多请求同批) | 服务吞吐成为瓶颈 | KV cache 批维隔离正确;吞吐提升有实测 |
| Wan2.2-Animate-14B(v1) | 有 v1 需求 | 独立管线,见设计文档附录 A |

---

## 4. 里程碑与设计文档风险项的对应

设计文档 §8 的六项风险,分别在以下里程碑收敛(不允许悬空到交付):

| 风险 | 收敛于 | 收敛标志 |
|---|---|---|
| 1. flex-attention 替换的数值等价性 | **M0** | A0.1 / A0.2 |
| 2. scheduler 数值对齐 | **M1** | A1.2 / A1.3(sigmas 对拍不通过则先修 scheduler) |
| 3. KV cache 显存 | **M2** | A2.1 / A2.3 |
| 4. modular 索引探测 | M1 规避(显式 `--model-class-name`)→ **M6** 根治 | A1 记录规避方式 |
| 5. 批处理 | M1 严格 batch key 规避 → **M6** 评估 | A5.4 |
| 6. 驱动视频时长 | **M5** | A5.3 |
