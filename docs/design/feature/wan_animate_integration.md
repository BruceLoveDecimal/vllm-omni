# Wan2.2-Animate-2 接入 vLLM-Omni 调研与架构设计

> 状态:调研 / 设计提案(尚未实现)。调研日期:2026-08-26。
> 目标模型:**Wan-AI/Wan2.2-Animate-2-14B**(含 Distilled 蒸馏版)。
> 依据:官方推理仓库 `Wan-Video/Wan-Animate-2`(源码逐行核对)、diffusers v0.40.0
> `WanAnimate2Transformer3DModel` + `WanAnimate2ModularPipeline`(源码逐行核对)、
> 仓库内 Wan2.2 家族实现(T2V / I2V / S2V / VACE)与 `add-diffusion-model` skill。
> 附录 A 保留上一代 Wan2.2-Animate-14B(v1)的接入设计。

## 1. 结论摘要(TL;DR)

1. **vLLM-Omni 当前没有接入任何 Wan-Animate 模型**(v1 与 Animate-2 均无)。
   `vllm_omni/diffusion/registry.py` 的 `_DIFFUSION_MODELS` 中不存在
   `WanAnimatePipeline` / `WanAnimate2*`;唯一相关代码是
   `diffusers_adapter/pipeline_utils.py:56` 对 v1 `WanAnimatePipeline` 的黑盒
   adapter 映射。**Animate-2 连黑盒路径都不可用**:diffusers 只以 modular
   pipeline blocks 形式发布(无 `DiffusionPipeline` 类),而
   `DiffusersAdapterPipeline` 只会委托 `DiffusionPipeline.__call__`。
2. **Animate-2 与 v1 是完全不同的架构**,不要混淆:
   - v1(Wan2.2-Animate-14B):pose 视频 + face 视频 + motion/face encoder +
     face adapter 注入,标准 `DiffusionPipeline`(diffusers v0.36);
   - **Animate-2(Wan2.2-Animate-2-14B)**:输入只有参考图 + **原始驱动视频**
     (无需骨架/人脸预提取);条件机制是 **in-context reference**:每段先对驱动视频
     latent 跑一遍 transformer("extract" 前向)填充**逐层 KV cache**,去噪时每个
     生成 token 通过 flex-attention BlockMask 同时 attend
     `[生成 token | 同帧对齐的参考 K/V]`(diffusers v0.40,modular-only)。
3. **与 Wan2 系列现有四条管线的根本差异**(§2.2):T2V/I2V/S2V/VACE 都是
   "给条件、生成新内容",Animate 是"驱动视频 → 动作/表情迁移";条件注入上,
   前者是特征拼接 / 特征注入 / 旁路分支,Animate-2 是**上下文注意力**;
   长度上 Animate-2 与 S2V 同为分段自回归;拓扑上它是单 transformer
   (无 T2V/I2V 的 MoE 双模型 + `boundary_ratio`);执行上它是
   **两阶段前向**(extract + 去噪),为仓库现有执行模型所无。
4. **必须新增 pipeline,不能复用/继承任何现有 Wan 管线**(§4.1):forward 契约
   (多出 extract 阶段与 KV cache 生命周期)、条件构造、注意力机制三者均不兼容。
   但**复用面很大**(§4.3):组件加载、分段循环骨架、条件 mask 公式、TP 化主干
   block、并行声明均可照搬;真正新写的只有 KV cache 容器 + 两阶段前向 API(§5.2)
   与参考注意力(§5.3,建议用"双注意力 + LSE 合并"替代 flex-attention 硬依赖)。
   Animate-1 与 Animate-2 之间同样无法共用管线类。
5. **新增输入仅一项**:驱动视频(原始视频,无需预提取),走既有
   `multi_modal_data["video"]`;参考图走 `["image"]`。二者恰好落在 serving 层
   **现有的** `image_reference` + `video_reference` 能力内,**在线服务零 API 扩展**
   (§2.4 / §4.5)。相比之下 v1 需要 face/background/mask 三路额外视频输入(附录 A)。

## 2. 仓库现状

### 2.1 注册表与兜底路径

- 原生注册:`registry.py` `_DIFFUSION_MODELS` 以 `model_index.json` 的
  `_class_name` 为 key。已注册 Wan 家族:`WanPipeline`(T2V)、
  `WanImageToVideoPipeline`(I2V)、`WanS2VPipeline`(S2V)、`WanVACEPipeline`
  (VACE),均在 `vllm_omni/diffusion/models/wan2_2/`。
- 黑盒兜底(`--diffusion-load-format diffusers`,`data.py:1244`)只覆盖标准
  `DiffusionPipeline`;Animate-2 的 Diffusers 仓库是 modular 格式
  (`modular_model_index.json`,blocks = `WanAnimate2Blocks`),**兜底路径不可用**。

### 2.2 与 Wan2 系列现有管线的定位与机制差异

一句话:仓库现有四条 Wan2.2 管线都是"给条件、生成新内容",而 Animate 是
"给一段驱动视频、把其中的动作与表情迁移到指定人物",本质是动作重定向
(motion retargeting);Animate-2 相较 Animate-1 又变了一次范式,把条件从
**特征注入**改成了**上下文注意力**。这决定了它无法复用任何一条现有管线(§4.1)。

**任务与条件源**:T2V 只有文本;I2V 多一张首帧图;S2V 多一段音频(驱动口型与
身体节奏);VACE 是结构化视频 + mask 的局部编辑/重绘。Animate 的条件源是**一整段
人物表演视频**,信息量高一个量级——需同时保留驱动视频的逐帧姿态/肢体/表情,
与参考图人物的身份/服装/外貌。

**条件注入机制**(最核心的架构分野):

| 管线 | 条件如何进入 DiT | 机制类别 |
|---|---|---|
| I2V | 首帧 latent + 4 通道 mask 与噪声**通道拼接**(`in_channels=36`),条件与噪声同网格 | 特征拼接 |
| S2V | 音频编码为独立 token 流,每隔几层一次 **cross-attention 注入**(`audio_inject_layers` 默认 8 层) | 特征注入 |
| VACE | 条件走**旁路 `vace_blocks`**,结果以残差加回主干 | 旁路分支 |
| Animate-1 | pose latent 逐元素加到主干;face token 每 5 层一次**逐帧对齐 cross-attention** | 特征注入 |
| **Animate-2** | **驱动视频先跑一遍完整 DiT,存下 40 层 K/V 作参考上下文;去噪 token 直接 attend 这些 K/V** | **上下文注意力** |

前四种都是"把条件压缩成特征,再想办法喂给主干";Animate-2 不压缩,而是让驱动视频与
被生成视频**在同一注意力空间内逐帧对话**,更接近 LLM 的 in-context learning 而非
ControlNet 式条件注入。代价是需维护 40 层 KV cache,显存开销极大(§8 风险 3)。

**输入契约的简化**:Animate-1 要求用户**预先跑骨架提取与人脸裁剪**(pose 视频 +
512×512 face 视频两路),Animate-2 直接吃原始驱动视频,预处理全部内化——这是
Animate-2 对服务端最友好的一点(§4.5)。

**生成长度机制**:T2V / I2V / VACE 单次生成固定帧数(通常 81 帧),长视频靠外部拼接;
S2V 与 Animate 系列均为**分段自回归**(每段解码回像素、取末尾若干帧重新过 VAE 作为
下段条件),可生成任意长度。Animate-2 段长 81 帧、段间重叠 1 帧。这一点直接决定了
多卡下 VAE 解码必须显式 broadcast(S2V 已有该代码),且与连续批处理基本互斥。

**Transformer 拓扑**:T2V / I2V 是 Wan2.2 的 **MoE 双 transformer**
(high-noise + low-noise,靠 `boundary_ratio` 按 timestep 切换);S2V 与 Animate 系列
均为**单 transformer**,无 `boundary_ratio`——因此 I2V 的双模型加载与 Cache-DiT
分段刷新逻辑对 Animate-2 不适用。

**执行模式**:现有全部 diffusion 管线都是单一的"N 步去噪循环"。Animate-2 是
**两阶段**——每段先跑一次 `extract` 前向(timestep 固定 1、用固定参考提示词,
只为填 KV cache),再跑 N 步去噪前向,每段实际前向次数为 `1 + N×CFG分支数`,
中间夹一块段生命周期的大显存。这是 vLLM-Omni 现有执行模型中没有的形态,
最接近的先例是 S2V 的 `transformer.encode_audio()` 段级预计算(但轻量得多)。

### 2.3 可复用的原生实现资产

| 已有机制 | 位置 | Animate-2 对应 |
|---|---|---|
| 分段迭代长视频循环 + 上段末帧回灌 VAE 链式条件 | S2V `pipeline_wan2_2_s2v.py` `forward()` 1316–1460 | 完全同构(Animate-2 official:`start += CLIP_LEN - first_num`,`first_num=1`) |
| 段级 DiT 前向预计算(S2V 的 `transformer.encode_audio`,含 CPU-offload 搬运与 FSDP unshard 护栏) | 1362–1381 / transformer 1491–1523 | Animate-2 的 "extract" KV-cache 前向是同位置、更重的段级预计算 |
| VAE patch-parallel 下解码结果显式 broadcast(自回归循环多卡必需) | 1419–1437 | 相同 |
| 每段重置 scheduler / 段内派生 generator | 1332 / `_make_clip_generators`(76) | 相同 |
| I2V 条件构造:首帧 VAE latent + 4 通道 mask 通道拼接(`in_channels=36`) | I2V `prepare_latents` 876–978 | Animate-2 的 `y = cat([mask4, z16])` 完全同式 |
| CLIP image encoder 组件加载 | I2V(`has_image_encoder` 探测) | Animate-2 Diffusers 格式同用 `CLIPVisionModel` |
| 视频多模态输入 pre-process 模板 | VACE `pipeline_wan2_2_vace.py:108-183` | 驱动视频归一化 |
| `SupportsComponentDiscovery` / `_sp_plan` / `_hsdp_shard_conditions` / TP 化 Wan block | I2V + `wan2_2_transformer.py` | 直接参照(S2V 漏了 ComponentDiscovery 与 packed_modules_mapping,新管线要补) |

### 2.4 输入通路现状

媒体只走 `prompt["multi_modal_data"]`(约定键 `"image"` / `"video"` / `"audio"`),
`OmniDiffusionSamplingParams` 不携带媒体;非标准键由各模型的 pre-process 函数归一到
`prompt["additional_information"]`(S2V 的 `pose_video` / `init_first_frame` 即此模式)。

serving 层 `serving_video.py:187-195` 已映射 `image_reference → image`、
`video_reference → video`、`audio_reference → audio`,即**每种模态各一路引用**。
各 Wan 管线对该通路的占用情况:

| 管线 | image | video | audio | 是否需扩展 API |
|---|---|---|---|---|
| T2V | — | — | — | 否 |
| I2V | 首帧(+可选 `last_image` 走 `multi_modal_data`) | — | — | 否 |
| S2V | 参考图 | (`pose_video` 已预留但管线未实现) | 音频 | 否 |
| VACE | 参考图 | 条件视频 + mask | — | 否(mask 走 `multi_modal_data`) |
| **Animate-2** | **参考图** | **驱动视频** | — | **否——零扩展** |
| Animate-1 | 参考图 | pose 视频 | — | **是**:另需 face / background / mask 三路视频 |

**Animate-2 恰好落在现有能力内**,这是它相对 Animate-1 的一大工程优势;
Animate-1 则必须扩展多视频引用字段(附录 A)。
`ReferenceVideoDecodeSpec`(`models/interface.py:33`)可声明驱动视频解码帧数上限,
Cosmos3 有实现先例。

## 3. Wan2.2-Animate-2-14B 模型架构

### 3.1 权重与仓库布局

**原始仓库 `Wan-AI/Wan2.2-Animate-2-14B`**(HF / ModelScope 同构;本环境出口代理封锁
两站直连,以下由官方推理仓库 `Wan-Video/Wan-Animate-2` 的加载代码
`infer/wan_animate_2.yaml` 还原):

```
ckpts/
├── wan_animate_2/
│   ├── wan_animate_2_bf16.safetensors               # DiT 主权重(基础版,单文件)
│   └── wan_animate_2_bf16_distillation.safetensors  # DiT 蒸馏版
└── videomodel/Wan-AI/
    ├── models_t5_umt5-xxl-enc-bf16.pth              # umt5-xxl 文本编码器(Wan 原始格式)
    ├── umt5-xxl/                                    # tokenizer
    ├── vae.pth                                      # Wan2.1 VAE
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  # CLIP(Wan2.1 I2V 同款)
    └── xlm-roberta-large/                           # CLIP tokenizer
```

**Diffusers 仓库 `Wan-AI/Wan2.2-Animate-2-14B-Diffusers`** 与
`...-Distilled-Diffusers`(modular 格式,`modular_model_index.json` 绑定
`WanAnimate2Blocks` / `WanAnimate2DistilledBlocks`):组件为
`transformer`(`WanAnimate2Transformer3DModel`)、`vae`(`AutoencoderKLWan`)、
`text_encoder`/`tokenizer`(UMT5)、`image_encoder`(`CLIPVisionModel`)、
`scheduler`、`guider`(CFG 3.0 基础版 / 1.0 蒸馏版)。

基础版预设:40 步 + CFG 3.0;蒸馏版:10 步、无 CFG、注意力 score_mod
`log_scale=-1.3`(见 §3.2)。官方 demo 默认 `clip_len=81`、`fps=24`、
720×1280(diffusers modular 默认 800×640 目标面积)、`shift=5.0`。

### 3.2 Transformer(`WanAnimate2Transformer3DModel` / 官方 `WanxiangAnimate2Transformer`)

配置(14B):`in_dim=36`(16 noise + 4 mask + 16 cond)、`dim=5120`、
`ffn_dim=13824`、`num_heads=40`、`num_layers=40`、`out_dim=16`、
`use_img_emb=True`(CLIP 1280 → `MLPProj` → 5120)、
`refer_offset_(t,h,w)=(1,0,-1)`、`refer_stride=1`;蒸馏版另有
`log_scale=-1.3`(基础版 0.0)。

**与基础 Wan DiT 相同**:40 层 block(self-attn + 文本 cross-attn
[+ `add_k_proj/add_v_proj` CLIP image K/V] + FFN + adaLN `modulation [1,6,dim]`)、
`patch_embedding Conv3d(36→5120)`、`head`、umt5 文本嵌入。权重名基本沿用
**原始 Wan 命名**(`time_embedding.0/2`、`blocks.{i}.self_attn.q/k/v/o`、
`ffn.0/2`、`modulation`、`head.head`、`img_emb.proj.{0,1,3,4}`;diffusers 版仅改
注意力投影名 `to_q/to_k/to_v/to_out.0` 与 `add_k_proj/add_v_proj/norm_added_k`)。

**Animate-2 特有机制**:

1. **无 v1 的 motion/face/pose 模块** —— 没有 pose_patch_embedding、
   motion_encoder、face_encoder、face_adapter。
2. **两阶段前向 + 逐层 KV cache**:
   - `forward_ref`("extract"):对驱动视频条件 latent(`[20, T_lat, h, w]`,
     mask 全 1)跑一遍完整 40 层,timestep 固定 1,文本用固定参考提示
     `prompt_ref`(默认 `"人物动作的参考视频"`),CLIP 用**驱动视频首帧**特征;
     每层把 **pre-RoPE 的 K/V** 存入 cache(官方是普通 dict `k_cache/v_cache`,
     diffusers 是 `WanAnimate2KVLayerCache` 对象,`[B,S,H,D]`);
   - `forward_gen`("cached"):去噪主前向,self-attn 的 K/V =
     `[当前生成 token | 缓存参考 K(每步重新施加 RoPE)]`。
3. **参考注意力掩码**(flex-attention `BlockMask`,`create_block_mask(_compile=True)`,
   Q/KV 长度 128 对齐):生成 query 可 attend **全部**有效生成 token
   + **同帧索引**的参考 token(参考帧整体 +1 偏移;q_frame == kv_frame)。
   参考 token 的 RoPE 落在不相交网格(t 从 1 起、w 偏移一个参考网格宽度)。
4. **蒸馏版 score_mod**:`score += log_scale`(条件命中参考 token 区域),
   即对参考注意力分数加常数偏置 —— 移植时等价于给参考分支 logits 加 bias,
   无需 flex 专有能力(§5.3)。
5. **CFG 非对称**:负分支(`is_uncondtion=True`)**跳过第 9 层 block**。
   两分支计算图不同(39 vs 40 层),对 CFG 并行是负载不均而非正确性问题(§6.3)。
6. RoPE 为**原始 Wan 复数实现**(float64 `torch.polar`,按样本循环
   `grid_sizes.tolist()`),非 diffusers cos/sin 版;`_keep_in_fp32_modules`
   含 time_embedding / modulation / norm 系;无 `_cp_plan`、无 `CacheMixin`。
7. 官方实现原生带 **SP(sp_size=8,seq_len 向 sp_size 取整对齐)与 FSDP
   HYBRID_SHARD** 支持 —— 说明并行化路径已被上游验证过,可对照移植。

### 3.3 推理数据流(官方 `pipelines/wan_animate_2_pipeline.py` 核对)

**输入**:参考图 `refer_img_path` + 驱动视频 `tpl_video_path`(原始视频)、
`prompt`(默认 `"static background."` 类简单提示)、`prompt_ref`、中文默认负提示、
`clip_len=81`、`first_num=1`(上段条件帧数,硬编码)、`sample_guide_scale`
(基础 3.0 / 蒸馏 1.0)、`step`(40 / 10)。

**预处理**:参考图按面积 `resize_by_area(width*height, divisor=16)` letterbox
(黑边,记录 padding 信息,输出时裁回);驱动视频按目标 fps 重采样帧索引、
同样 letterbox、尾部 **zigzag padding**(`[...3 4 | 4 3 2]`)补齐整段;
驱动视频的**音轨被抽出**,推理后 mux 回输出视频。

**每段(`while start + first_num < len(frames)`)**:

1. 形状:`T = clip_len + 1`(82),`lat_t = T//4 + 2 = 22`(+1 参考帧槽位),
   噪声 `[16, 22, h/8, w/8]` fp32(逐样本 list,有效 batch=1);
2. 条件 `y`(**帧维**拼接):
   `y_ref = cat([mask4(全1), VAE(参考图)]) [20,1,h,w]` +
   `y_reft = cat([mask4(mask_len = 段0?0:1), VAE([上段末1帧 | 零帧])]) [20,21,h,w]`
   → `[20, 22, h, w]`;transformer 内与噪声通道拼接为 36;
3. 驱动条件(extract 用):`condition_latents = VAE(驱动段像素) [16,21,h,w]`、
   `condition_y = cat([mask4(全1), VAE(驱动段)]) [20,21,h,w]`;注意 **Wan VAE
   因果性**:驱动视频必须逐段编码,不能整段编码后切 latent;
4. `forward_ref` 一次 → 填 KV cache;
5. scheduler 每段重置:官方 `FlowDPMSolverMultistepScheduler` + 自定义 sigmas
   (`get_sampling_sigmas(step, shift=5.0)`);逐步 `forward_gen`(CFG 开启时
   正负两次,负分支 `is_uncondtion=True`)→ `scheduler.step`;
6. 解码 `vae.decode(latents[:, 1:])`(丢参考帧槽位),段 >0 丢首 `first_num` 帧;
   段末 `first_num` 帧像素滚动为下段条件;段间清 KV cache +
   `torch.cuda.empty_cache()`(官方注明防止分配器碎片 OOM);
7. 拼段 → 裁 zigzag padding → 裁 letterbox → mux 音轨。

## 4. 接入总体设计

### 4.1 迁移路径判定:为什么必须新增 pipeline

**结论:必须新增 pipeline + transformer 两个新模块,且不能复用/继承现有任何一条
Wan 管线。**

表层原因是注册机制——`registry.py` 的 `_DIFFUSION_MODELS` 按 `model_index.json` 的
`_class_name` 一对一映射,新模型必然要新 key。但真正的硬约束是下面三条,它们决定了
即便退而求其次去继承 `Wan22S2VPipeline` 也不可行:

1. **`forward()` 契约不同**。Animate-2 每段多出 `extract_reference()` 阶段与 KV cache
   的生命周期管理(创建 → 跨全部去噪步复用 → 段末释放),无法塞进 S2V 单阶段
   forward 的结构;S2V 的 `encode_audio` 虽也是段级预计算,但它输出的是一个小张量,
   不是需要跨步驻留、参与每层注意力的大块状态。
2. **条件构造不同**。S2V 是"参考图 latent 作独立 token 流 + 音频 cross-attn",
   Animate-2 是"参考图占 latent 帧 0 + 驱动视频进 KV cache",两者的张量形状、
   序列组装方式、mask 语义都对不上,没有可共用的 `prepare_latents`。
3. **注意力机制不同**。这不是"在 Wan block 上加几个模块"(那样尚可继承,Animate-1
   就属于此类),而是 self-attention 本身要拼接外部 K/V 并施加帧对齐掩码——属于
   主干级改动,`WanTransformerBlock` 的 forward 必须重写。

反过来说,**Animate-1 与 Animate-2 之间同样不能共用管线类**:两者除同属角色动画
任务外,在输入契约、条件注入、transformer 结构上没有任何共享面(§2.2 对比表)。
若后续同时支持,应是两条独立管线(附录 A)。

按 `add-diffusion-model` skill 的 Step 0 分类,Animate-2 属 **Hybrid**:
Diffusers 仓库组件标准(UMT5 / `AutoencoderKLWan` / `CLIPVisionModel` 均为仓库
已用依赖),但管线与 transformer 无标准 `DiffusionPipeline` 可搬,需自研。

### 4.2 checkpoint 格式选择

**首版建议只支持 Diffusers 格式仓库**(`Wan2.2-Animate-2-14B-Diffusers` 与
Distilled),理由:组件加载与 I2V 现有代码同构、权重为 diffusers 命名的
safetensors 可走标准 loader。原始格式(单文件 bf16 safetensors + Wan 原始
T5/VAE/CLIP `.pth`)按需求二期补,模式照抄 S2V `_init_original_format()`
(含 `_convert_wan_t5_state_dict`;CLIP 需加载 xlm-roberta CLIP,S2V 无先例,
I2V 原始格式亦未支持 —— 这也是推迟原始格式的原因之一)。

### 4.3 文件布局、复用边界与注册

```
vllm_omni/diffusion/models/wan_animate2/
├── __init__.py
├── pipeline_wan_animate2.py       # Wan22Animate2Pipeline + pre/post process funcs
├── wan_animate2_transformer.py    # WanAnimate2Transformer3DModel(vllm-omni 版)
└── reference_kv_cache.py          # 逐层 KV cache 容器(纯 Python 对象)
```

独立目录而非塞进 `wan2_2/`:两阶段前向、参考注意力与 KV cache 使 transformer
结构性偏离基础 Wan DiT。

**复用边界**——尽管必须新增管线,真正从零写的只有两块,工作量远小于"新模型"的直觉:

| 部分 | 处置 | 来源 / 说明 |
|---|---|---|
| 组件加载(tokenizer / UMT5 / VAE / CLIP) | **照搬** | I2V `pipeline_wan2_2_i2v.py` 的 `has_image_encoder` 探测与 `from_pretrained` 模式;全是仓库已有依赖 |
| 分段循环骨架(段级 generator 派生、每段重置 scheduler、VAE broadcast、段间 `empty_cache()`) | **照搬** | S2V `forward()` 1316–1460,含已踩过坑的多卡细节 |
| 段级 DiT 预计算的护栏(CPU-offload 手动搬运、FSDP `unshard/reshard`) | **照搬** | S2V 1362–1381 / transformer 1491–1523 |
| 条件 mask 构造 `cat([mask4, latent16])` | **照搬公式** | I2V `prepare_latents` 876–978,与 Animate-2 的 `y` 同式 |
| 驱动视频 pre-process 归一化 | **改写** | VACE `pipeline_wan2_2_vace.py:108-183` 为模板,换成 letterbox + fps 重采样 + zigzag padding |
| 主干 40 层 block(self-attn / cross-attn / FFN / adaLN) | **继承 + 局部重写** | 与基础 Wan block 同构,TP 化实现(`QKVParallelLinear` / `RowParallelLinear` / `DistributedRMSNorm` / `WanFeedForward`)可直接复用,**TP 支持几乎免费**;仅 self-attn 的 K/V 拼接部分需重写 |
| 并行声明(`_sp_plan` / `_hsdp_shard_conditions` / `SupportsComponentDiscovery` / `packed_modules_mapping`) | **照填** | I2V + `wan2_2_transformer.py`;S2V 漏了后两项,新管线补上 |
| **KV cache 容器 + 两阶段前向 API** | **新写** | §5.2 |
| **参考注意力(帧对齐 + 去 flex-attention 化)** | **新写** | §5.3,核心难点 |

从 `wan2_2` 模块 import 复用的具体符号:`DistributedRMSNorm`、`WanFeedForward`、
`load_transformer_config` 工厂模式、latent 标准化助手。

注册点:

| 文件 | 改动 |
|---|---|
| `registry.py` | `_DIFFUSION_MODELS["WanAnimate2ModularPipeline"] = ("wan_animate2", "pipeline_wan_animate2", "Wan22Animate2Pipeline")`;蒸馏别名 `"WanAnimate2DistilledModularPipeline"` 指向同类;pre/post process 两条 |
| `data.py` `resolve_model_class_name` / `hf_utils.get_diffusion_model_index` | modular 仓库没有 `model_index.json`,需让索引探测兼容 **`modular_model_index.json`** 的 `_class_name`(或文档要求显式 `--model-class-name`,首版可先走显式指定,索引兼容独立小 PR) |
| `model_metadata.py` | `"Wan22Animate2Pipeline": DiffusionModelMetadata(attention_mask_free=True, supports_multimodal_inputs=True)`(registry key 与 pipeline 类名两个键空间都要登记,S2V 先例) |
| `wan_animate2/__init__.py` | 导出 |
| `cache/cachedit/model_specific.py` | 视 §6.5 结论决定是否登记 enabler 或加入 `_NO_CACHE_ACCELERATION` |

### 4.4 Pipeline 设计(`Wan22Animate2Pipeline`)

```python
class Wan22Animate2Pipeline(
    nn.Module,
    SupportImageInput,
    CFGParallelMixin,
    DenoiseProgressMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    supports_request_batch = True          # 批准入靠 batch_compatibility_key 严控
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "image_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    _DEFAULT_SEGMENT_FRAMES = 81           # clip_len
    _DEFAULT_PREV_COND_FRAMES = 1          # first_num,官方硬编码 1
    _DEFAULT_PROMPT_REF = "人物动作的参考视频"
    _DEFAULT_FPS = 24
```

`__init__`(diffusers 布局):tokenizer + `UMT5EncoderModel` +
`DistributedAutoencoderKLWan.from_pretrained(subfolder="vae")` +
`CLIPVisionModel`/`CLIPImageProcessor`(I2V 的 `has_image_encoder` 探测模式);
transformer 经 `create_wan_animate2_transformer_from_config(...)` 实例化,权重走
标准 loader:

```python
self.weights_sources = [DiffusersPipelineLoader.ComponentSource(
    model_or_path=model_path, subfolder="transformer",
    prefix="transformer.", fall_back_to_pt=True)]
```

Scheduler:对齐官方 `FlowDPMSolverMultistepScheduler(shift=1,
use_dynamic_shifting=False)` + `get_sampling_sigmas(steps, shift=5.0)` 自定义
sigmas(sigmas 助手 ~5 行,直接移植;若 Diffusers 仓库 scheduler 配置与官方数值
一致,也可 `from_pretrained(subfolder="scheduler")`,以数值对齐验证为准)。
蒸馏版:默认 `num_inference_steps=10`、`guidance_scale=1.0`,由注册别名在
pipeline 内设默认(模型侧差异仅 `log_scale=-1.3`,进 transformer config)。

**`forward(req: DiffusionRequestBatch, ...)`** 骨架(S2V 结构移植):

1. 逐请求收集 prompt / 负提示(默认官方中文负提示)/ 参考图 / 驱动视频 /
   `prompt_ref` / fps;批内一致性经 `batch_compatibility_key` 拦截
   (段数、分辨率、fps、蒸馏与否);
2. 预处理:参考图与驱动帧 `resize_by_area`(面积 `height*width`、divisor 16、
   letterbox 记录裁剪信息)、驱动视频 fps 重采样 + zigzag padding(纯函数,
   放 pipeline 文件内);音轨抽取/回填首版**不做**进管线(输出仅视频,
   示例脚本演示 `mux_video_audio_bytes` 复用 S2V 的 media_utils);
3. 文本 encode(prompt、prompt_ref、负提示)+ CLIP encode(参考图、每段驱动首帧);
4. **段循环**:构造 `y`(§3.3 步骤 2)与驱动条件(逐段 VAE encode,尊重 VAE
   因果性)→ `transformer.extract_reference(...)` 填 KV cache(段级预计算,
   S2V `encode_audio` 同位;含 FSDP `unshard/reshard` 与 CPU-offload 搬运护栏)
   → `self.diffuse(...)` → 解码(丢参考帧槽位与重复帧;VAE patch-parallel
   broadcast 照抄 S2V)→ 滚动末帧 → **释放 KV cache** + `empty_cache()`;
5. 拼段、裁 padding/letterbox、`split_diffusion_output_by_request`。

**`diffuse()`** 每步:

```python
positive_kwargs = {
    "hidden_states": latents, "timestep": timestep,
    "encoder_hidden_states": prompt_embeds,
    "encoder_hidden_states_image": clip_ref_image_embeds,
    "condition_latents": reference_latents,       # y, [20, 22, h, w]
    "kv_cache": kv_cache, "reference_grid_sizes": grid_sizes_ref,
    "origin_len": segment_frames, "origin_area": [H, W],
    "is_uncondition": False, "return_dict": False,
}
negative_kwargs = {**positive_kwargs,
    "encoder_hidden_states": negative_prompt_embeds,
    "is_uncondition": True}
noise_pred = self.predict_noise_maybe_with_cfg(
    do_true_cfg=guidance_scale > 1.0, true_cfg_scale=guidance_scale,
    positive_kwargs=positive_kwargs, negative_kwargs=negative_kwargs,
    cfg_normalize=False)
```

post process:输出纯视频,套用 I2V 的 `{"payload","metadata"}` 形态,`fps` 随请求。

**批处理决策**:官方与 diffusers 实现均为有效 batch=1(`list[Tensor]` 逐样本)。
首版将 tensor 侧改为标准 batch 维实现,但 `batch_compatibility_key` 先把可同批
条件收紧到"完全同构请求";真正多请求同批(KV cache 逐请求隔离)列为后续优化。

### 4.5 输入 / 服务通路

**新增的输入契约**:相对现有 Wan 管线,Animate-2 只新增一个必填项——
**驱动视频**(原始视频,无需预提取骨架/人脸),走既有的 `multi_modal_data["video"]`;
参考图沿用 `multi_modal_data["image"]`。Animate-2 专有的标量参数
(`prompt_ref`、`segment_frame_length`)经 `sampling_params.extra_args` 透传,
由 pre-process 归一到 `additional_information`(S2V 同模式)。
**无新增媒体模态、无新增 API 字段**(§2.4 对比表)。

离线:

```python
omni = Omni(model="Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
            model_class_name="WanAnimate2ModularPipeline", ...)
out = omni.generate(
    {"prompt": "static background.",
     "multi_modal_data": {"image": ref_image, "video": driving_video}},
    OmniDiffusionSamplingParams(height=1280, width=720, fps=24,
        num_inference_steps=40, guidance_scale=3.0, seed=42,
        extra_args={"segment_frame_length": 81, "prompt_ref": "..."}),
)
```

在线:现有 `/v1/videos` 的 `image_reference` + `video_reference` 直接可用;
pipeline 实现 `ReferenceVideoDecodeSpec`(帧数上限按时长/段数上限设定)。
示例:`examples/offline_inference/wan_animate2/` +
`examples/online_serving/wan_animate2/`(参照 S2V 目录形态)。

## 5. Transformer 移植设计(核心难点)

### 5.1 模块与 TP

主干 40 层与基础 Wan block 同构 → self-attn `QKVParallelLinear` + 统一
`Attention` 层 + `RowParallelLinear`、cross-attn 含 `add_k_proj/add_v_proj`
(I2V 同款)、`WanFeedForward`、`DistributedRMSNorm`,声明
`packed_modules_mapping = {"to_qkv": ["to_q","to_k","to_v"]}`。
`_repeated_blocks=["WanAnimate2TransformerBlock"]`、
`_layerwise_offload_blocks_attrs=["blocks"]`、`_hsdp_shard_conditions` 同 S2V。
fp32 常驻:time embedding / modulation / norm / RoPE 频率(对齐
`_keep_in_fp32_modules`)。`img_emb`(MLPProj)不切分。

### 5.2 两阶段前向 API

```python
class WanAnimate2Transformer3DModel(nn.Module):
    def extract_reference(self, condition_latents, condition_y, context_ref,
                          clip_fea_ref, grid_sizes_ref) -> ReferenceKVCache:
        """段级一次:40 层前向,存各层 pre-RoPE K/V。timestep 固定 1。"""

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_image, condition_latents, kv_cache,
                reference_grid_sizes, origin_len, origin_area,
                is_uncondition=False, ...):
        """去噪前向:self-attn 拼接缓存参考 K/V(每步对参考 K 重施 RoPE)。"""
```

`ReferenceKVCache`:纯 Python 容器(`list[tuple[k, v]]` × 40 层,`[B,S,H,D]`),
由 pipeline 持有并逐段释放;**不挂在 module 上**(diffusers `_skip_keys` 的教训:
挂 module 会污染 state_dict / FSDP / offload 视角)。TP 下 K/V 按本地 head 数
自然切分(extract 与 gen 同一 TP 布局,无需通信);SP 下见 §6.2。

### 5.3 参考注意力:去 flex-attention 化

diffusers/官方实现硬依赖 `torch.nn.attention.flex_attention` + 编译版
`BlockMask`(未编译会物化全量注意力矩阵 OOM),且 `list[Tensor]`、
`.item()`、128 对齐 scatter 打包等对 torch.compile/CUDA graph 不友好。
掩码语义其实很简单:

> 生成 query(帧 f)attend:全部生成 token ∪ 参考帧 f 的 token
> (参考帧索引 +1 偏移;帧 0 = 参考图槽位,无对应参考帧,只做基础注意力)。
> 蒸馏版对参考分支分数加常数 `log_scale`。

**建议实现:双注意力 + LSE(log-sum-exp)合并**,消除 flex 依赖:

1. 分支 A(gen↔gen):现有统一 `Attention` 层稠密注意力,取回 LSE;
2. 分支 B(gen↔ref,逐帧对齐):Q reshape `[B·T, hw, H, D]`、缓存 K/V reshape
   `[B·T_ref, hw_ref, H, D]`(帧对齐后 batch 化),稠密小注意力 + LSE;
   蒸馏版在该分支 logits 上加 `log_scale`(等价于 LSE 合并权重乘
   `exp(log_scale)`,数学上与 score_mod 完全一致);
3. `out = (out_A·exp(lse_A) + out_B·exp(lse_B)) / (exp(lse_A)+exp(lse_B))`
   (数值稳定形式,ring-attention 同款合并,仓库 SP ring 路径已有可借鉴实现)。

该方案:形状静态(zigzag padding 保证段等长)、无 BlockMask/128 对齐打包、
兼容现有 attention backend 选择与 CUDA graph;帧 0 的 query 单独走分支 A。
备选:若后续实测 LSE 合并有性能顾虑,可将 flex-attention 作为可选 backend
(`torch>=2.5` + compile 场景)并保留双注意力为默认与兜底。分支 B 的正确性用
diffusers 实现做数值对拍(逐层 max-diff)。

### 5.4 RoPE

保留官方复数 RoPE 数学(参考 token 不相交网格:`t+1` 偏移、w 偏移参考网格宽),
但实现向量化:预计算两套 cos/sin 表(生成网格、参考网格)缓存于
`{(grid, offsets): freqs}`,去掉逐样本 Python 循环与 float64 `torch.polar`
运行时计算(fp32 预表,数值对拍验证)。缓存参考 K 存 pre-RoPE 值、每步施加
—— 与官方一致,保证与蒸馏权重行为相同。

### 5.5 load_weights

diffusers 格式权重名(§3.2)→ vllm-omni 命名:
`self_attn.{to_q,to_k,to_v} → to_qkv`(stacked mapping,仅 `blocks.`)、
`to_out.0 → to_out`、cross-attn `add_k_proj/add_v_proj` 直载、
`ffn.0/ffn.2 → ffn.net_0.proj/ffn.net_2`(注意 Animate-2 diffusers 版 ffn 是裸
Sequential 索引,**不是** `ffn.net.0.proj`)、`modulation → scale_shift_table`
类推、`head.head → proj_out`、`img_emb.proj.{0,1,3,4}` 直载;TP 下
`norm_q/norm_k` 形状不匹配时切分(S2V 1788 行逻辑)。

## 6. 并行与加速

实现顺序:单卡正确性 → TP → CFG-Parallel → SP/USP → HSDP → 缓存加速评估。
(与 skill 默认顺序相比把 SP 后移:Animate-2 的 SP 需要处理参考注意力分支,
工作量高于 TP/CFG。)

### 6.1 TP

主干免费获得(`num_heads=40` 整除 2/4/8);extract 与 gen 两阶段同布局,
KV cache 天然按 head 分片。参考注意力分支 B 在本地 head 上计算,无额外通信。

### 6.2 SP / USP

生成 token 按序列分片(基础 `_sp_plan` 模式:patch-embed 后分片入口 +
`proj_out` gather,`auto_pad=True`)。参考注意力两个可选方案:
- **方案 1(首版)**:KV cache 各 rank 全量持有(extract 前向本身 SP 分片跑,
  结束后 all-gather K/V 一次),分支 B 无需逐步通信 —— 以显存换通信简单性;
- 方案 2:cache 按帧分片 + 分支 B 前 all-to-all 帧重排,显存友好但复杂,
  留待 KV cache 显存实测(§8 风险 3)后决定。
官方 sp_size=8 + `max_seq_len` 向 sp 取整的做法印证了 padding 对齐路径可行。

### 6.3 CFG-Parallel

`CFGParallelMixin.predict_noise_maybe_with_cfg` 直接可用;负分支 kwargs 仅换
文本 embeds + `is_uncondition=True`。**注意**:负分支跳过 block 9(39 vs 40 层),
两 rank 负载轻微不均但正确性无碍;KV cache 两分支共享(参考 K/V 与 CFG 分支
无关),CFG 并行下两 rank 各自 extract 一遍或 rank0 extract 后广播 ——
首版取"各自 extract"(简单、无额外通信原语)。蒸馏版无 CFG,CFG 并行不适用。

### 6.4 HSDP / Offload

`_hsdp_shard_conditions` 匹配 `blocks.{i}`;`extract_reference` 在 forward 钩子
外调用,需 S2V 同款 `unshard()/reshard()` 护栏。`SupportsComponentDiscovery`
声明齐全(§4.4);layerwise offload 时 `img_emb`、time/text embedding 常驻。
官方在 80GB 卡上依赖分组 offload 才能同时容纳权重 + KV cache(diffusers 文档),
CPU offload 支持应列为 P1 而非可选。

### 6.5 Cache-DiT / TeaCache(需评估,倾向首版不启用)

Animate-2 的 "两阶段 + KV cache + 每 5 步无副作用注入" 与 cache_dit 的残差缓存
假设兼容性未知:gen 阶段 block 调用含 kv_cache 参数与参考注意力,
`BlockAdapter` 直连大概率不工作,需 S2V 式定制 CachedBlocks;且蒸馏版本身
只有 10 步,缓存收益有限。**建议首版将两个注册名加入 `_NO_CACHE_ACCELERATION`**,
待正确性与并行落地后单独立项评估(基础版 40 步场景有收益空间)。

### 6.6 明确不做(首版)

Pipeline-Parallel、step-wise 连续批处理(段循环 + extract 两阶段与 step 化执行
交互复杂)、量化(`packed_modules_mapping` 先声明)、原始格式 checkpoint、
LoRA。

## 7. 测试与文档计划

按 `vllm-omni-test` skill,建议 **Medium priority**:

| 级别 | 内容 |
|---|---|
| L1(`core_model`+`cpu`) | `tests/diffusion/models/wan_animate2/test_wan_animate2_pipeline.py`:权重重映射(ffn 裸索引、to_qkv 融合)、LSE 合并注意力 vs 参考实现数值对拍(小配置随机权重)、蒸馏 log_scale 偏置等价性、zigzag padding / 段数计算、`y` 条件形状(20×(T_lat+1))、KV cache 生命周期(段间释放)、batch 拒绝、`_sp_plan` / `_hsdp_shard_conditions` 结构、CFG 负分支跳 block 9 |
| L3 | `tests/e2e/offline_inference/test_wan_animate2.py`:真权重(优先 Distilled,10 步快)、极短驱动视频单段 smoke,baseline 同函数 `core_model`+`advanced_model` 双标 |
| L4 | `test_wan_animate2_expansion.py`:基础版 CFG3.0×40 步、TP2、USP2、cpu-offload 少量参数化行 |

文档:`docs/models/supported_models.md` 加行;
`docs/user_guide/diffusion_features.md` VideoGen 矩阵加 **Wan2.2-Animate-2** 行
(预期:TeaCache ❌ / Cache-DiT ❌(首版)/ SP ✅ / CFG ✅(仅基础版)/ TP ✅ /
PP ❌ / HSDP ✅ / offload ✅ / VAE-Patch ✅ / 量化 ❌ / Step-Exec ❌);
`recipes/Wan-AI/Wan2.2-Animate-2.md`(基础版 TP=2、蒸馏版单卡两个配方)。

## 8. 风险与开放问题

1. **flex-attention 替换的数值等价性**:LSE 合并数学等价,但需和 diffusers 实现
   逐层对拍(尤其蒸馏 log_scale 路径);对拍脚本进 L1。
2. **scheduler 数值对齐**:官方 FlowDPMSolver+自定义 sigmas 与 Diffusers 仓库
   scheduler 配置是否一致未验证(HF 出口被封,拿不到 scheduler config);
   以官方实现为准绳,加载 Diffusers scheduler 前先做 sigmas 对拍。
3. **KV cache 显存**:40 层 × (T_ref·hw) × 5120 × 2(K/V)× bf16;720×1280、
   81 帧段 ≈ 21×45×80 tokens ≈ 7.6 万 token/层 → 约 40×7.6e4×5120×2×2B ≈ 62GB
   **全量不可行**,必须配合 offload / SP 分片 / 逐层驻留策略 ——
   官方用分组 offload + expandable_segments 也是同因。这是**性能设计的第一优先项**,
   方案(cache 逐层流式 offload、SP 帧分片、fp8 cache)需在 P1 实测定夺。
4. **modular_model_index.json 探测**:`get_diffusion_model_index` 目前只认
   `model_index.json`,首版用显式 `--model-class-name` 绕过,索引兼容另起小 PR。
5. **批处理**:首版有效 batch=1 语义(严格 batch key),多请求真同批需 KV cache
   批维隔离,收益/复杂度待评估。
6. **驱动视频时长**:段数 = 视频长度线性增长,serving 需要请求级帧数/时长上限
   (`ReferenceVideoDecodeSpec` + admission 校验)。

## 9. 实施里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | transformer 移植 + LSE 注意力 + 单卡 pipeline(Diffusers 格式、蒸馏版优先)+ 注册 + 离线示例 | 与官方/diffusers 输出视觉一致、L1 通过 |
| P1 | KV cache 显存方案 + CPU offload + TP + CFG 并行 + 基础版(40 步 CFG3.0)+ 在线 serving | L3 smoke、TP2/offload recipes |
| P2 | SP/USP + HSDP + L4 扩展测试 + 文档矩阵 | 多卡 e2e |
| P3(独立评估) | Cache-DiT、原始格式 checkpoint、modular 索引探测、真批处理 | — |

---

## 附录 A:上一代 Wan2.2-Animate-14B(v1)接入设计(备选)

v1(`WanAnimatePipeline`,diffusers v0.36+)与 Animate-2 定位重叠但输入契约不同
(需预提取 pose 骨架视频 + 512×512 face 视频,支持 replace 换人模式 +
background/mask 输入,relighting LoRA)。若后续有 v1 需求,设计要点:

1. **迁移路径 Path A**,落 `wan2_2/` 目录(`pipeline_wan2_2_animate.py` +
   `wan2_2_animate_transformer.py`),黑盒 adapter 今天即可运行
   (`--diffusion-load-format diffusers`,diffusers ≥ 0.36)。
2. Transformer = 基础 Wan block(TP 免费)+ 四个新模块:
   `pose_patch_embedding`(Conv3d 16→5120,`hidden[:, :, 1:] += pose`)、
   `motion_encoder`(StyleGAN2 EqualConv/EqualLinear,权重存**未缩放**值、
   `act_fn.bias`、fp32 QR 分解)、`face_encoder`(因果 Conv1d,77→20 帧,
   每帧 5 token)、`face_adapter`(8 个逐帧对齐 cross-attn,每 5 层残差注入,
   `to_out` 无 `.0` 后缀)。
3. face_adapter 注入走 S2V `after_transformer_block` 钩子模式(SP 下先 gather),
   Cache-DiT 用 S2V 式定制 CachedBlocks(原方法置 no-op 防双重注入);
   motion/face encoder 提为段级预计算(正/负分支各一份,负分支 face 置 -1,
   CFG 只作用于文本+表情)。
4. 条件构造:参考图 letterbox → VAE + 4 通道 mask 通道拼接 `[20,1,h,w]`,
   段条件 `[20,T_lat,h,w]` 帧维拼接(参考图占 latent 帧 0),噪声
   `[16,T_lat+1,h,w]`,`in_channels=36`;`segment_frame_length=77`、
   `prev_segment_conditioning_frames ∈ {1,5}`、reflect padding、
   UniPC `flow_shift=5.0`、默认 `guidance_scale=1.0`。
5. 输入通路需扩展 serving 多视频引用(`face_video_reference` /
   `background_video_reference` / `mask_video_reference`)—— v1 相比 Animate-2
   的主要工程负担;replace 模式与 relighting LoRA(diffusers 无专门支持,
   通用 Wan LoRA 转换器不覆盖 Animate 专有模块)列为二期。
