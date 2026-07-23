# Mage-Flow 接入 vLLM-Omni 实施规格

> 对应 issue：<https://github.com/vllm-project/vllm-omni/issues/5337>
> 目标模型：`microsoft/Mage-Flow`、`microsoft/Mage-Flow-Edit` 及其 Base / RL / Turbo 变体
> 工作分支：`feat/mega_flow`（沿用请求中的分支名；模型名仍写作 Mage-Flow）
> 文档状态：Implementation Ready
> 更新日期：2026-07-23

## 1. 结论

Mage-Flow 应以原生 diffusion pipeline 接入，不能依赖现有 `DiffusersPipelineAdapter` 直接加载。

首个可合并版本采用“正确性优先、单请求优先”的策略：

- PR 1 先支持 text-to-image；
- PR 2 补齐 image edit 和 1～3 张参考图；
- CUDA + BF16 为首期唯一承诺平台；
- 第一阶段使用 vLLM-Omni `Attention` 的单请求 dense 路径，不引入上游裸
  `flash_attn_varlen_func`，也不降级仓库的 PyTorch / Transformers 版本；
- Base、RL、Turbo 共用一套实现，推理步数和 CFG 默认值从模型变体解析；
- 第一阶段显式关闭 request batching、step execution、TP、SP、CFG parallel 和 cache；
- 上游安全检查及 Gaussian-Shading 水印保留，默认开启，并通过
  `sampling_params.extra_args` 提供模型级开关；
- 完成 T2I + Edit 的工程量预计为 12～18 人日；完成 packed batching、并行和缓存优化后，
  总工程量预计为 27～43 人日，即一名熟悉本仓库的工程师约 5～8 周。

整体接入难度评为“中高”。难点不在注册模型，而在以下四项：

1. Mage-Flow 不是可直接加载的 Diffusers 官方 pipeline；
2. 上游依赖与当前仓库的 Transformers / PyTorch 版本冲突；
3. 模型使用变长 packed joint attention 和按样本二维 RoPE；
4. Mage-VAE 与编辑模型的双路图像条件均需原生移植。

## 2. 背景与上游基线

### 2.1 模型矩阵

| 任务 | 模型变体 | 默认步数 | 默认 CFG | 首期计划 |
| --- | --- | ---: | ---: | --- |
| T2I | Base | 30 | 5.0 | PR 1 |
| T2I | RL | 20 | 5.0 | PR 1 |
| T2I | Turbo | 4 | 1.0 | PR 1 |
| Edit | Base | 30 | 5.0 | PR 2 |
| Edit | RL | 30 | 5.0 | PR 2 |
| Edit | Turbo | 4 | 1.0 | PR 2 |

默认值只在用户没有显式传入 `num_inference_steps` 或 `guidance_scale` 时生效。
不得根据步数反推变体；变体由模型仓库名或配置元数据确定。

### 2.2 组件

T2I 与 Edit 的 `model_index.json` 都声明 `_class_name = "MageFlowPipeline"`，核心组件为：

| 组件 | 上游类型 | 关键配置 |
| --- | --- | --- |
| Transformer | `mage_flow.MageFlow` | 4B NR-MMDiT，12 个 double-stream block |
| VAE | `mage_flow.MageVAE` | 128 latent channels，空间压缩率 16 |
| Text / Vision Encoder | `Qwen3VLForConditionalGeneration` | context dim 2560 |
| Processor | `AutoProcessor` | 文本和编辑参考图输入 |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | shift 6，`z-image` schedule |

Transformer 配置必须从 checkpoint 读取并校验，不在代码中依赖隐式猜测。当前官方 checkpoint 的
关键值为：

```text
in_channels          = 128
out_channels         = 128
context_in_dim       = 2560
hidden_size          = 3072
num_heads            = 24
head_dim             = 128
depth                = 12
depth_single_blocks  = 0
axes_dim             = [16, 56, 56]
patch_size           = 1
packing              = true
static_shift         = 6.0
schedule_mode        = "z-image"
```

初始化时至少校验：

- `hidden_size == num_heads * head_dim`；
- `sum(axes_dim) == head_dim`；
- `in_channels == out_channels == vae.latent_channels`；
- `vae.downsample_factor == 16`；
- 未识别的 `schedule_mode` 必须报错，不能静默回退。

### 2.3 上游推理特性

- 原生覆盖 512～2048 分辨率以及不超过 4:1 的宽高比；
- 不同分辨率样本以 packed sequence 进入同一次 attention；
- 文本流和图像流分别做 QKV 投影，合并后进行 joint attention，再按长度拆回；
- 每个样本使用自己的二维图像 RoPE 坐标；
- CFG 的正、负条件在上游实现中被打包进一次 Transformer forward；
- Edit 参考图同时进入两条条件路径：
  - 经 Qwen3-VL 视觉塔形成语义条件，视觉输入长边默认缩放到 384；
  - 经 Mage-VAE 形成全分辨率 latent 条件；
- 官方训练覆盖最多 3 张参考图；服务层首期同样限制为 3；
- 上游 pipeline 在采样前执行内容检查，并将 Gaussian-Shading 水印写入初始噪声。

上游公开性能仅作为 sanity check：官方报告在 A100、1024×1024、BF16 条件下峰值显存约
18～20 GiB；Turbo 的 T2I / Edit 延迟约 0.59 s / 1.02 s。正式验收必须在同一机器、同一
checkpoint、相同步数与开关下重测，不能直接把公开数字当作回归阈值。

## 3. 目标与非目标

### 3.1 最终目标

- `microsoft/Mage-Flow*` checkpoint 可被 vLLM-Omni 原生识别、加载和服务；
- T2I 与 Edit 的结果在固定 seed 下与官方实现达到可量化的数值和感知一致性；
- 支持离线推理和现有 OpenAI 兼容 diffusion 服务入口；
- 支持 512～2048、宽高比不超过 4:1、宽高均为 16 的倍数；
- 支持 Base / RL / Turbo，不为每个 checkpoint 复制 pipeline；
- 支持 1～3 张编辑参考图；
- 后续能够接入 request batching、step execution、CFG parallel、TP / SP 和缓存，
  首期设计不得阻断这些扩展。

### 3.2 首个 T2I PR 的范围

- CUDA；
- BF16；
- 单个 request、单张输出；
- Base / RL / Turbo；
- prompt、negative prompt、seed、height、width、steps、CFG；
- `pil`、tensor 和 latent 输出；
- 官方安全检查与水印逻辑；
- offline 与 online smoke test；
- Transformer、VAE decode、scheduler、权重加载的单元测试。

### 3.3 Edit PR 的增量范围

- 单请求；
- 1～3 张参考图；
- PIL、NumPy、Tensor 经过统一预处理后进入 pipeline；
- Qwen3-VL 视觉条件与 Mage-VAE latent 条件；
- 不同尺寸参考图；
- 参考图数量和输出尺寸的服务层校验；
- Edit Base / RL / Turbo 的端到端回归。

### 3.4 首期明确不支持

- CPU、NPU、XPU、MUSA；
- FP8、INT8、INT4、量化加载；
- LoRA；
- `tensor_parallel_size > 1`；
- sequence parallel；
- CFG parallel；
- request / continuous batching；
- mixed-resolution packed batching；
- step execution 和中断后续跑；
- diffusion cache；
- VAE patch parallel；
- 多请求共用 prompt embedding cache；
- 超过 3 张参考图；
- 超出官方尺寸范围后的自动缩放。

对这些能力必须 fail fast，并在错误中给出当前限制；禁止无提示地退回慢路径或产生错误结果。

## 4. 为什么不能走 Diffusers Adapter

现有 `pipeline_diffusers_adapter.py` 最终依赖 `DiffusionPipeline.from_pretrained()`。Mage-Flow 的
`MageFlowPipeline` 来自独立的 `mage_flow` 包，不是 Diffusers 内置 pipeline，当前 Diffusers
也没有对应类。因此仅向 registry 增加名字会在加载阶段失败。

即使通过 `trust_remote_code` 绕过类发现，Adapter 仍无法解决：

- 上游自定义 Mage-VAE；
- Qwen3-VL 变长 attention 修改；
- packed mixed-resolution metadata；
- vLLM-Omni 的组件发现、分阶段执行和分布式注入；
- 统一权重加载、attention backend 与 profiling。

因此本方案不增加 Adapter 特例，而是在
`vllm_omni/diffusion/models/mage_flow/` 下提供原生实现。

## 5. 依赖与兼容策略

### 5.1 已知冲突

上游 `mage-flow` 包当前声明：

- `torch >= 2.13`；
- `transformers >= 5.3, < 5.6`；
- 单独安装 `flash-attn == 2.8.3`。

当前 vLLM-Omni 仓库声明的约束为：

- `requirements/common.txt` 要求 `transformers >= 5.5.3`，不设置上限；
- `requirements/common.txt` 固定 `diffusers == 0.38.0`；
- 仓库不直接固定 PyTorch 版本，实际版本由上游 vLLM 依赖和 CI 镜像共同决定。

本地生成的 `uv.lock` 被 `.gitignore` 排除，其解析结果不属于仓库承诺，也不能作为实现中的
版本判断条件。正式兼容基线以提交时的受支持依赖范围和 CI 镜像为准，具体小版本由 Phase 0
的复现记录给出。

此外，上游限制 Transformers `< 5.6` 的直接原因是依赖后来被移除的
`input_embeds` 调用形式。

### 5.2 决策

- 不把 `mage-flow` 加入运行时依赖；
- 不收窄或降级 vLLM-Omni 当前支持的 Transformers 范围；
- 不为 Mage-Flow 增加模型专属 PyTorch 下限，也不强制整个项目升级框架；
- 不直接依赖 `flash-attn` Python 包；
- 在 MIT 许可证允许的范围内移植必要实现，并保留原始版权和来源说明；
- Qwen3-VL 兼容代码优先复用
  `vllm_omni/diffusion/models/internvla_a1/adapter_qwen3_vl.py` 面向当前 Transformers API
  的局部适配方式；
- attention 从 `vllm_omni.diffusion.attention.layer` 导入并统一经过 `Attention`。

若当前 CI 使用的 PyTorch 缺少某个上游算子，只允许在 Mage-Flow 模块内增加等价兼容实现；
不得以本模型接入为由全局升级框架版本。Phase 0 必须记录实际验证的 PyTorch、Transformers、
CUDA 和 attention backend 版本。

## 6. 代码结构

计划新增：

```text
vllm_omni/diffusion/models/mage_flow/
├── __init__.py
├── autoencoder_mage.py
├── mage_flow_layers.py
├── mage_flow_transformer.py
├── pipeline_mage_flow.py
├── prompt_utils.py
└── watermark.py
```

计划修改：

```text
vllm_omni/diffusion/registry.py
vllm_omni/diffusion/model_metadata.py
examples/offline_inference/text_to_image/text_to_image.py
examples/offline_inference/image_to_image/image_edit.py
examples/online_serving/text_to_image/README.md
examples/online_serving/image_to_image/README.md
docs/user_guide/examples/online_serving/text_to_image.md
docs/user_guide/examples/online_serving/image_to_image.md
docs/models/supported_models.md
tests/diffusion/models/mage_flow/...
tests/e2e/offline_inference/test_mage_flow.py
tests/e2e/online_serving/test_mage_flow.py
tests/e2e/online_serving/test_mage_flow_edit.py
tests/e2e/online_serving/test_mage_flow_expansion.py
tests/e2e/online_serving/test_mage_flow_edit_expansion.py
tests/dfx/perf/tests/test_mage_flow_vllm_omni.json
tests/dfx/perf/tests/test_mage_flow_edit_vllm_omni.json
.buildkite/test-ready.yml
.buildkite/test-merge.yml
.buildkite/test-nightly.yml
```

Mage-Flow 属于已有 Text-to-Image / Image-to-Image 分类，优先扩展上述共享示例。只有在共享脚本
无法表达模型必要参数时，才新增 `examples/offline_inference/mage_flow/`；不得在现有分类目录下
新增孤立的 `mage_flow.py`。在线示例复用任务级客户端和启动脚本，在 README 与用户指南中补充
Mage-Flow 命令、模型特有 `extra_args` 和 1～3 图 Edit 用法。

各文件职责如下。

### 6.1 `mage_flow_transformer.py`

提供 `MageFlowTransformer2DModel(nn.Module)`：

- 构造 input / context projection、time embedding、12 个 double-stream block 和 output head，
  其中全部 block 保存在单一 `nn.ModuleList`；
- 声明 `_repeated_blocks = ["MageFlowDoubleStreamBlock"]`，供 regional compilation 识别；
- 接收 `OmniDiffusionConfig`，为将来的分布式与量化扩展保留入口；
- 不继承上游或 Diffusers 的 `ModelMixin`、`ConfigMixin`；
- 暴露 `load_weights()`，使用来自 vLLM 上游
  `vllm.model_executor.models.utils` 的 `AutoWeightsLoader` 和 `WeightsMapper`，且 mapper
  只处理经过确认的命名差异；
- forward 返回图像 token 的 velocity / flow prediction，不返回文本 token；
- 不在 forward 中创建 attention layer、scheduler 或临时 module。

`_repeated_blocks` 只表示可重复编译单元，不等于支持 Cache-DiT。PR 4 若启用缓存，需单独评估
`CachedTransformer`、`_cache_dit_adapter_config` 以及 block 前后 hooks；在此之前将
`MageFlowPipeline` 放入 registry 的 `_NO_CACHE_ACCELERATION` 集合。

建议的首期接口：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    image_grid_hw: list[tuple[int, int]],
    encoder_attention_mask: torch.Tensor | None = None,
    return_dict: bool = False,
) -> tuple[torch.Tensor]:
    ...
```

首期 `hidden_states` 为 `[B, Li, 128]`，且 `B == 1`；`encoder_hidden_states` 为
`[B, Lt, 2560]`。内部投影后，attention 使用 `[B, L, 24, 128]`。

### 6.2 `mage_flow_layers.py`

提供以下最小组件：

- timestep / modulation embedding；
- text / image RMSNorm 或 QK norm；
- 三轴 RoPE，其中一轴用于文本 / 模态位置，两轴用于图像 H/W；
- `MageJointAttention`；
- `MageFlowDoubleStreamBlock`；
- final modulation 和 output projection。

`MageJointAttention` 的固定处理顺序：

1. 文本和图像分别做 Q、K、V 投影；
2. Q/K norm；
3. 对文本和图像应用与官方一致的三轴 RoPE；
4. 单卡路径沿 sequence 维显式拼接 text / image QKV；
5. 调用 `vllm_omni.diffusion.attention.layer.Attention`；
6. 按原始长度将结果拆回 text / image；
7. 分别通过 text / image output projection。

首期只有一个真实样本，不需要 padding，也不需要 `cu_seqlens`。CFG 正负分支顺序执行，从而
避免在第一版修改通用 attention metadata。当前 `AttentionMetadata.joint_*` 只会被 Ring /
Ulysses 等 SP strategy 消费，单卡 `NoParallelAttention` 不会自动拼接；因此 PR 1 参考 LongCat
的 non-SP 显式 concat 路径。PR 3 再处理变长批次，PR 4 实现 SP 时复用 `joint_*`。

### 6.3 `autoencoder_mage.py`

提供 `MageVAE(nn.Module)`，按上游实现完整移植 encode / decode：

- 输入图像范围和 normalization 与官方一致；
- `encode([B, 3, H, W]) -> [B, 128, H/16, W/16]`；
- `decode([B, 128, H/16, W/16]) -> [B, 3, H, W]`；
- `sample_posterior = false` 时禁止引入额外随机性；
- T2I PR 可以先只在主路径调用 decode，但 encode 必须至少有单元测试；
- Edit PR 开启 encode 主路径；
- 首期不实现 distributed VAE，由单卡执行。

Mage-VAE 是一阶段 diffusion codec，不得用 `AutoencoderKL` 或现有 VAE 近似替代。任何近似都会
使 edit 条件和最终图像脱离官方分布。

### 6.4 `prompt_utils.py`

集中管理：

- 官方 system / user prompt template；
- T2I 与 Edit 的模板差异；
- negative prompt；
- Qwen3-VL processor 输出字段兼容；
- 视觉输入长边 384 的保持比例缩放；
- 1～3 张参考图的 placeholder 与顺序；
- 有效 token 提取和 2560 维 context 输出；
- 当前受支持 Transformers API 下替代上游全局 monkeypatch 的局部适配。

禁止在 import 时 monkeypatch 全局 Hugging Face attention。适配必须局限在 Mage-Flow 使用的
module / wrapper 内，避免改变其他模型行为。

### 6.5 `watermark.py`

移植上游 Gaussian-Shading 所需的最小代码，职责仅限：

- 根据 request generator / seed 生成可复现的初始噪声；
- 在开关开启时注入官方参数的水印；
- 在开关关闭时生成与官方“关闭水印”基准完全相同的普通高斯噪声。

不能在模块 import 时修改全局 RNG。所有随机数都必须使用请求级 `torch.Generator`。

### 6.6 `pipeline_mage_flow.py`

提供统一的 `MageFlowPipeline`。是否存在
`prompt["multi_modal_data"]["image"]` 决定 T2I 或 Edit 路径，不为 Edit 注册第二个架构名，
因为两类官方 checkpoint 的 `_class_name` 相同。

类属性首期设置为：

```python
supports_request_batch = False
supports_step_execution = False
support_image_input = True
_dit_modules = ["transformer"]
_encoder_modules = ["text_encoder"]
_vae_modules = ["vae"]
```

Pipeline 在 PR 1 即继承 `nn.Module`、`CFGParallelMixin`、`ProgressBarMixin`、
`DiffusionPipelineProfilerMixin` 和 `SupportsComponentDiscovery`。PR 1 实现稳定的
`predict_noise()` 契约，并通过 `predict_noise_maybe_with_cfg()` 统一处理无 CFG 和单卡顺序
CFG。首期对 `cfg_world_size > 1` fail fast；PR 4 完成分布式正确性与性能验证后再解除该门禁。

构造阶段：

1. 从 `od_config.model` 读取 `model_index.json`；
2. 预取 scheduler、tokenizer / processor、text encoder、transformer、VAE 子目录；
3. 用 `FlowMatchEulerDiscreteScheduler.from_pretrained()` 加载 scheduler；
4. 用当前 Transformers 版本加载 Qwen3-VL；
5. 构造本地 Transformer 和 Mage-VAE；
6. 声明所有 `DiffusersPipelineLoader.ComponentSource`；
7. 初始化 profiler，不执行任何推理。

### 6.7 Registry 与元数据

在 `_DIFFUSION_MODELS` 增加：

```python
"MageFlowPipeline": (
    "mage_flow",
    "pipeline_mage_flow",
    "MageFlowPipeline",
)
```

同时注册：

- `get_mage_flow_pre_process_func`；
- `get_mage_flow_post_process_func`。

在 `model_metadata.py` 增加：

```python
MAGE_FLOW_MAX_INPUT_IMAGES = 3

"MageFlowPipeline": DiffusionModelMetadata(
    supports_multimodal_inputs=True,
    max_multimodal_image_inputs=MAGE_FLOW_MAX_INPUT_IMAGES,
)
```

T2I 请求不带图像时仍合法。元数据表示“可以接收图像”，而不是“必须接收图像”。

## 7. 请求契约与校验

### 7.1 T2I

```python
prompt = {
    "prompt": "a red fox in snow",
    "negative_prompt": "blurry",
}
```

采样参数使用现有 `OmniDiffusionSamplingParams` 字段：

- `height` / `width`；
- `num_inference_steps`；
- `guidance_scale`；
- `seed` / `generator`；
- `num_outputs_per_prompt`；
- `output_type`；
- `latents`；
- `extra_args`。

### 7.2 Edit

```python
prompt = {
    "prompt": "replace the sky with a sunset",
    "negative_prompt": "low quality",
    "multi_modal_data": {
        "image": [image_1, image_2],
    },
}
```

单张图也统一规范为列表。参考图顺序必须在 processor、视觉 embedding 和 VAE latent 三处保持
一致。

### 7.3 模型特有开关

不新增顶层 SamplingParams 字段，先通过 `sampling_params.extra_args` 承载：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mage_enable_safety_check` | bool | `True` | 与官方行为保持一致 |
| `mage_enable_watermark` | bool | `True` | 与官方行为保持一致 |
| `mage_vision_long_edge` | int | `384` | Edit 视觉语义路径尺寸 |

非布尔开关、非正视觉尺寸或未知 `mage_*` 参数必须报错。服务启动参数需允许 sampling
`extra_args` 透传；示例中明确展示所需配置。

### 7.4 输入约束

- 默认输出 1024×1024；
- 高和宽均须位于 `[512, 2048]`；
- 高和宽均须为 16 的倍数；
- `max(height, width) / min(height, width) <= 4`；
- 首期 `num_outputs_per_prompt == 1`；
- 首期 request batch size 必须为 1；
- Edit 参考图数量为 `[1, 3]`；
- T2I 不得传空图像列表以伪装 Edit；
- Turbo 默认 CFG 1.0；当 CFG 为 1.0 时 negative prompt 不参与计算；
- 传入 latents 时，其 shape、dtype、device 和输出尺寸必须一致；
- 同时传 seed 和 generator 时沿用仓库既有优先级，不定义 Mage 特例。

尺寸不合法时直接返回清晰的 `ValueError`，不做静默取整。这样可以避免调用方请求尺寸与实际
结果不一致。

## 8. 推理数据流

### 8.1 T2I

```text
request
  → 校验尺寸、变体和开关
  → 安全检查（默认开启）
  → Qwen3-VL 文本编码
  → 请求级 generator 生成初始 latent / 水印噪声
  → 生成 z-image sigma / timestep
  → MageFlow Transformer 去噪
      ├─ CFG=1：一次 forward
      └─ CFG>1：positive / negative 顺序 forward 后合并
  → scheduler.step
  → Mage-VAE decode
  → 图像后处理
  → DiffusionOutput
```

CFG 合并公式必须逐项对齐上游，包括是否执行 normalization / rescale。不能直接套用其他模型的
默认 CFG 公式而不做对照测试。

### 8.2 Edit

```text
request + 1～3 reference images
  → 校验与统一图像格式
  → 安全检查（文本 + 图像）
  → 路径 A：缩放至视觉长边 384 → Qwen3-VL vision/text context
  → 路径 B：保持官方编辑预处理 → Mage-VAE reference latents
  → 构造 target/reference token、grid 和 RoPE position
  → 初始 target latent / 水印噪声
  → Transformer 去噪
  → Mage-VAE decode
  → DiffusionOutput
```

Edit 的 reference token 排列、position id、mask、latent normalization 必须从官方实现逐行
移植并通过中间张量测试，不允许凭经验重写。

### 8.3 Scheduler

- 使用 checkpoint 的 `FlowMatchEulerDiscreteScheduler` 配置；
- `shift = 6`；
- 按官方 `z-image` 模式生成 sigma / timestep；
- 自定义 `timesteps` 与 `sigmas` 的互斥规则沿用现有 pipeline；
- scheduler 必须按 request 创建或重置内部 step index，连续请求不能互相污染；
- fixed seed parity 测试同时比较初始 noise、timesteps 和每步 latent。

## 9. 权重加载

### 9.1 来源

模型组件按 checkpoint 目录加载：

```text
transformer/
text_encoder/
tokenizer/ or processor files
scheduler/
vae/diffusion_pytorch_model.safetensors
```

`model_index.json` 中的 `_text_encoder_path` 和 `_vae_source` 应被读取和校验。若路径不存在，
错误中输出解析后的实际路径。

### 9.2 加载规则

- Transformer 和本地 Mage-VAE 通过 `ComponentSource` + `AutoWeightsLoader` 加载；
- Qwen3-VL / processor 可复用 Hugging Face `from_pretrained`，但必须走仓库已有的 prefetch
  与 local-files-only 逻辑；
- safetensors 是首选格式；
- 权重保持 checkpoint dtype，在模块放入设备时按 `od_config.dtype` 转换；
- mapper 仅处理已核实的固定前缀或 fused/unfused QKV 映射；
- 不允许用 `strict=False` 吞掉未知缺失；
- 可忽略参数必须维护显式 allowlist，并在测试中锁定。

### 9.3 加载验收

每个官方变体至少执行一次加载审计：

- missing keys = 0，或全部属于评审通过的 allowlist；
- unexpected keys = 0，或全部属于评审通过的 allowlist；
- 没有 meta tensor 残留；
- 关键层抽样 checksum 与原 checkpoint 一致；
- T2I / Edit、Base / RL / Turbo 共用相同命名映射。

安全检查默认开启，因此首期不能删除 Qwen3-VL `lm_head`。若后续提供“服务启动时永久关闭安全
检查”的配置，可在模型仍位于 CPU 时裁掉不用的生成 head；这项优化不进入前两个 PR。

## 10. Attention 演进方案

### 10.1 PR 1 / PR 2：单请求 dense

- `supports_request_batch = False`；
- 每次只有一个样本，无需 padding；
- 使用 `vllm_omni.diffusion.attention.layer.Attention`；
- text / image QKV 在模型层显式拼接，首期不新增 attention metadata 字段；
- CFG 正负分支顺序执行；
- 不依赖上游 raw FlashAttention；
- 输出先追求与官方一致，再优化吞吐。

该路径会比上游“一次 packed CFG forward”慢，但实现边界清晰，不污染通用 attention API。

### 10.2 PR 3：真正的 packed varlen

PR 3 分为两个决策点，不预设必须修改通用 attention API。

第一步实现 padded request batch：

- 每个样本保留独立 text / image 长度；
- 对每个样本拼接 text / image 后构造二维 `attn_mask`；
- 复用现有 FlashAttention backend 的 masked varlen 路径，由 backend 从 mask 内部推导
  `cu_seqlens`；
- 其他 backend 不支持 mask 时使用经过数值验证的 padded fallback 或明确拒绝。

只有 padded 方案不能满足显存或吞吐目标时，才进入第二步：为真正无 padding 的 flat-packed
表示设计显式边界 metadata。以下字段均属于候选的新 API，不是当前 `AttentionMetadata`
已有能力：

```text
cu_seqlens_q
cu_seqlens_kv
max_seqlen_q
max_seqlen_kv
```

`per_sample_image_grid_hw`、`text_lengths` 和 `image_lengths` 保留在 Mage 层，由模型生成 RoPE、
拆分 text / image 输出；通用 attention backend 只接收执行 attention 所需的序列边界。新增
metadata 前必须先提交独立设计与至少两个 backend 的兼容性评估。

目标：

- 同批不同文本长度；
- 同批不同输出分辨率；
- CFG 条件和无条件分支可合并；
- 无跨样本 attention；
- 结果与逐请求 dense 路径一致；
- backend 不支持 varlen 时明确拒绝或使用经过测试的 padded fallback。

不要把 Mage 特有的 grid / RoPE 信息硬编码进通用 backend。通用 attention 只接收序列边界，
Mage 层自己负责位置编码。

### 10.3 PR 4：并行

按以下顺序推进：

1. 验证 CFG collective、解除 `cfg_world_size > 1` 门禁；
2. tensor parallel 的 QKV / MLP；
3. sequence parallel，并按 LongCat 先例用 `AttentionMetadata.joint_*` 表达复制的文本流；
4. VAE parallel；
5. step execution；
6. Cache-DiT / 其他 cache backend。

每项能力独立开关、独立测试；不得在一个 PR 中同时修改 attention、TP、SP 和 scheduler。

## 11. 安全检查与水印

### 11.1 默认行为

为避免接入后悄然改变官方模型语义：

- `mage_enable_safety_check=True`；
- `mage_enable_watermark=True`。

安全检查失败时返回结构化、可识别的请求错误，不返回占位黑图。日志只记录分类结果和 request
ID，不记录用户原始 prompt 或图像内容。

### 11.2 测试行为

数值 parity 测试必须在两端使用完全相同的开关：

- Transformer / scheduler 中间张量测试关闭安全检查和水印；
- 水印单测独立比较初始 noise；
- 默认行为端到端测试验证两个开关确实开启。

### 11.3 待评审的产品边界

`extra_args` 允许单请求关闭安全检查属于本方案的默认提议，但合并前需要维护者确认。如果项目
要求不可绕过，应移除该请求级开关，并改为服务启动时策略；这不影响模型核心实现和测试结构。

## 12. 测试计划

### 12.1 L1 CPU 单元测试

不下载完整 checkpoint，使用 tiny config 或构造权重：

- 配置解析与校验；
- 三轴 RoPE shape 和坐标；
- Mage 单卡 concat / split 与纯 PyTorch joint attention 参考实现数值一致；
- attention mask 不发生跨样本泄漏；
- tiny Transformer forward shape；
- Transformer 权重名映射；
- Mage-VAE encode / decode shape；
- prompt template 和有效 token 截取；
- Base / RL / Turbo 默认参数解析；
- 尺寸、宽高比、参考图数量校验；
- scheduler 参数与 step index 重置；
- registry lazy import；
- 当前 Transformers API 的 import 和调用签名；
- safety / watermark 开关解析；
- 相同 seed 得到相同初始 noise；
- Turbo 在 CFG 1.0 时不构造、也不执行 negative prompt 分支；
- `cfg_world_size > 1` 在首期得到明确的 unsupported error。

这些测试位于 `tests/diffusion/models/mage_flow/`，CPU 可运行用例标记
`core_model and cpu`。仅验证 CUDA kernel 的测试不得混入 CPU marker。

### 12.2 GPU 数值对齐

在同一进程环境无法同时安装冲突版本时，使用两个隔离环境生成和消费基准 tensor：

1. 官方 `mage-flow` 环境保存输入和关键中间张量；
2. vLLM-Omni 环境读取同一输入；
3. 比较：
   - prompt / vision embedding；
   - Mage-VAE latent；
   - 初始 noise；
   - RoPE cos / sin；
   - 第 1 个 block 输出；
   - Transformer 最终 noise prediction；
   - 每步 scheduler latent；
   - VAE decoded tensor。

建议门槛：

- tiny FP32 模型：`rtol <= 1e-4`、`atol <= 1e-5`；
- 完整 BF16 单步：cosine similarity `>= 0.999`，relative L2 `<= 1e-2`；
- 完整图像：LPIPS `<= 0.05`，并人工检查无系统性构图偏移；
- 若 kernel 顺序造成差异，应提供误差定位，不以放宽阈值代替分析。

覆盖组合：

- T2I Base：1024×1024；
- T2I Turbo：1024×1024；
- T2I：512×2048 极端宽高比；
- Edit Base：1 张参考图；
- Edit Turbo：3 张不同尺寸参考图；
- CFG 1.0 与 CFG 5.0；
- safety / watermark 开启与关闭。

### 12.3 服务测试

- 离线 T2I 示例；
- 离线 Edit 示例；
- OpenAI 兼容在线 T2I；
- 在线 multipart / multimodal Edit；
- 重复 seed；
- 同一服务进程连续发送多个请求，验证 scheduler step index、generator 和输出不受前一请求污染；
- 相同 seed 的连续请求结果一致，不同 seed 的请求结果发生变化；
- 非法尺寸；
- 第 4 张参考图被拒绝；
- 无图的 Edit checkpoint 仍按 T2I 路径运行，或给出产品确认后的明确限制；
- batch size > 1 在首期被拒绝；
- TP / SP 参数在首期被拒绝。

### 12.4 CI 分层

测试必须接入仓库 `docs/contributing/ci/CI_5levels.md` 定义的现有体系：

| 级别 | 文件与内容 | Marker | Buildkite |
| --- | --- | --- | --- |
| L1 | `tests/diffusion/models/mage_flow/`：tiny config、纯函数、registry、权重映射 | `core_model and cpu` | `.buildkite/test-ready.yml` 的 CPU 单测步骤 |
| L2 | online `test_mage_flow.py` / `test_mage_flow_edit.py`：dummy / 最小权重服务启动、请求成功与非空图像 | `core_model and diffusion`，用 `hardware_marks` 声明 H100 | `.buildkite/test-ready.yml` |
| L3 | 上述 online 测试与 offline `test_mage_flow.py`：真实权重、尺寸和确定性检查 | `advanced_model and diffusion`，用 `hardware_marks` 声明 H100 | `.buildkite/test-merge.yml` |
| L4 | `test_mage_flow_expansion.py` 与 `test_mage_flow_edit_expansion.py`：变体、边界尺寸、多图、异常路径 | `full_model and diffusion`，用 `hardware_marks` 声明 H100 | `.buildkite/test-nightly.yml` |
| L5 | 长时间并发、故障注入和恢复；首两个模型 PR 不新增 | `slow and diffusion`，附硬件 marker | 需要时接入 `.buildkite/test-weekly.yml` |

具体要求：

- PR 1 合并前至少提供一个接入指南要求的 L4 functionality test；PR 2 增加 Edit L4；
- L2 / L3 的 Buildkite step 配置 `source_file_dependencies`，至少覆盖
  `vllm_omni/diffusion/models/mage_flow/`、registry、metadata 和对应测试文件；
- 同一个在线测试可以同时带 `core_model` 与 `advanced_model`，由 `--run-level` 控制断言深度；
- 完整 checkpoint、模型 revision 和基准 tensor 使用缓存或受控 artifact，并支持离线重跑；
- 完整官方环境的逐层 parity 可作为手动 / nightly 辅助作业，但不能替代 L4 functionality test；
- 固定硬件性能回归放在 `tests/dfx/perf/` 对应配置和 nightly 性能步骤中，不在 L2 / L3
  functionality smoke test 中设置峰值显存硬断言。

## 13. 性能与可观测性

Pipeline profiler 增加或复用以下阶段名：

```text
safety_check
text_vision_encode
vae_encode_reference
prepare_latents
denoise
vae_decode
postprocess
```

每次基准记录：

- checkpoint 加载时间；
- 权重装载后的静态显存；
- peak allocated / reserved memory；
- text / vision encode；
- 每步 denoise；
- reference VAE encode；
- final VAE decode；
- 首张图延迟和端到端延迟；
- 实际 backend、dtype、尺寸、步数和开关。

PR 1 的性能门槛：

- 在同一 A100 环境、相同 checkpoint revision、关闭安全检查和水印并完成 warmup 后，单请求
  T2I 峰值显存相对官方实现的增量不超过 `max(官方峰值的 10%, 2 GiB)`；
- denoise 主循环不比官方 dense 等价基准慢 20% 以上；若因顺序 CFG 超出，应在 PR 中量化，
  并将 packed CFG 列为 PR 3 的阻塞指标；
- 不出现每步 CPU↔GPU 权重迁移或重复 tokenizer / processor 加载。

上述性能门槛在固定硬件的独立 benchmark / nightly 作业中执行，记录 warmup 次数、采样次数和
波动范围；功能 smoke test 只采集指标，不因一次峰值抖动失败。

PR 3 的目标：

- packed batch 的结果与逐请求路径保持测试阈值内一致；
- batch=2 的单位图像吞吐高于逐请求执行；
- mixed-resolution padding 浪费不超过实现文档中声明的 fallback 上限。

## 14. 分阶段交付

### Phase 0：兼容性 Spike，3～5 人日

该工作量包含在 PR 1 的 7～10 人日内，不与后续阶段重复累加。

产物：

- tiny Transformer 可加载官方一层权重；
- Mage-VAE decode 一张官方 latent；
- Qwen3-VL 在仓库当前 CI 依赖环境下输出 context；
- 单次 denoise step 与官方对齐；
- 记录实际显存；
- 记录 PyTorch、Transformers、CUDA、attention backend 和六个 checkpoint revision；
- 复核 VAE channel、context dim、scheduler shift / mode 及上游依赖声明；
- 确认移植文件的许可证头。

退出条件：四个组件均有可运行证据。任何一项失败都先更新本规格，不直接扩写完整 pipeline。

### PR 1：T2I 基线，7～10 人日

内容：

- Transformer、layers、VAE、prompt utils、水印；
- scheduler 与 T2I pipeline；
- registry / post-process；
- `CFGParallelMixin` 单卡顺序 CFG、`ProgressBarMixin` 和 `_repeated_blocks`；
- Base / RL / Turbo；
- 单请求 CUDA BF16；
- 共享 offline / online 示例及 `docs/models/supported_models.md`；
- L1～L4 CI 接线、CPU 单测和 T2I GPU parity。

退出条件：

- 三个 T2I 变体均加载成功；
- 固定 seed parity 达标；
- 1024² 和 512×2048 通过；
- 默认安全检查 / 水印行为已验证；
- 不存在未解释的 missing / unexpected weights；
- L2 / L3 通过，且至少一个 T2I L4 functionality test 可由 nightly 执行。

### PR 2：Edit，5～8 人日

内容：

- multimodal pre-process 和 metadata；
- Qwen3-VL visual path；
- Mage-VAE reference encode；
- 1～3 张参考图；
- Edit 三个变体；
- 共享 Edit 示例、L2～L4 服务测试和 GPU parity。

退出条件：

- 单图与三图 edit 均达标；
- 视觉与 latent 双路径中间张量对齐；
- 第 4 张图在进入模型前被拒绝；
- T2I 无回归；
- Edit L4 expansion test 可由 nightly 执行。

### PR 3：Packed batching，8～12 人日

内容：

- request batching；
- padded mixed-resolution batch + `attn_mask` 原型；
- 在性能数据证明必要时增加 flat-packed sequence 与显式 varlen metadata；
- packed CFG；
- dense fallback；
- 吞吐基准。

退出条件：

- batch 结果与逐请求一致；
- 无跨请求 attention；
- batch=2 吞吐收益可复现；
- OOM / backend 不支持时错误清晰。

### PR 4：并行与服务优化，7～13 人日

内容按收益拆成小 PR：

- CFG collective 验证并解除多卡门禁；
- TP；
- SP；
- step execution；
- `CachedTransformer` / `_cache_dit_adapter_config` 与 cache；
- VAE parallel / offload；
- 可选裁剪 safety generation head。

## 15. 风险登记

| 风险 | 影响 | 预防 / 处理 |
| --- | --- | --- |
| 仓库支持的 Transformers API 与上游 Mage 预期不同 | prompt / vision embedding 不一致 | 复用局部 adapter，在实际 CI 版本增加 embedding golden test |
| 直接移植上游全局 monkeypatch | 污染其他模型 | 禁止 import-time patch，封装在 Mage module |
| 当前 CI PyTorch 缺少 Mage-VAE 所需算子 | 无法解码或性能差 | Phase 0 单独验证，局部兼容实现，不新增模型专属框架下限 |
| dense attention 数值顺序不同 | fixed-seed 图像漂移 | 从 block / noise prediction 逐层定位 |
| sequential CFG 性能较差 | 首期延迟高 | 明确基线，PR 3 packed CFG |
| 权重命名或分片特殊 | 漏加载且不易发现 | 严格 missing/unexpected 审计 |
| 安全检查保留 lm_head | 显存高于预期 | 先保证默认语义，后续启动级裁剪 |
| 水印影响 parity | 误判模型不一致 | 中间张量测试关闭，水印独立测试 |
| Edit 多图 token 顺序错误 | 语义条件错配 | 对每张图加索引化中间张量测试 |
| 通用 attention API 被模型特例侵入 | 维护成本增加 | grid/RoPE 留在 Mage 层，仅通用化 varlen 边界 |
| 上游继续变更 checkpoint | 配置漂移 | 读取 config、校验 invariants、nightly 锁 revision |

## 16. 验收标准

模型接入完成需同时满足：

- [ ] `MageFlowPipeline` 可由 registry lazy load；
- [ ] T2I Base / RL / Turbo 均可离线和在线推理；
- [ ] Edit Base / RL / Turbo 均可处理 1～3 张参考图；
- [ ] 官方 6 个 checkpoint 无未解释的权重缺失；
- [ ] 512～2048 与 4:1 边界校验正确；
- [ ] fixed-seed 数值和感知 parity 达标；
- [ ] 默认步数、CFG、安全检查和水印与官方一致；
- [ ] 当前不支持能力均 fail fast；
- [ ] L1 CPU CI 不下载大模型；
- [ ] L2 / L3 Buildkite step 与 `source_file_dependencies` 已接线；
- [ ] 至少一个 L4 functionality test 可在 nightly 运行；
- [ ] 完整 GPU parity 可在隔离环境复现；
- [ ] 共享 offline / online 示例和用户指南已更新；
- [ ] `docs/models/supported_models.md` 已加入 Mage-Flow 能力与限制；
- [ ] 性能报告包含环境、revision、开关和逐阶段数据；
- [ ] 移植代码包含正确的 MIT 来源和版权说明。

Issue #5337 的最小关闭条件建议设为 PR 1 合并，即 `microsoft/Mage-Flow` T2I 可用；Edit 和性能
优化通过 issue checklist / follow-up PR 跟踪。若维护者要求一次性支持 `Mage-Flow-Edit`，则应将
PR 2 改为关闭 issue 的阻塞项，但不要把 packed batching 和并行能力一起塞入首个模型 PR。

## 17. 实施顺序清单

1. 锁定官方 Mage repo 和 6 个 checkpoint revision；
2. 建立独立官方基准环境，导出一组中间 tensor；
3. 移植 Mage layers 和 tiny Transformer；
4. 接入单卡显式 QKV concat 和 vLLM-Omni `Attention`，完成单步对齐；
5. 移植 Mage-VAE，完成 decode / encode 对齐；
6. 适配 Qwen3-VL + processor 到仓库当前支持的 Transformers API；
7. 实现 scheduler、noise / watermark 和 T2I denoise loop；
8. 接入 CFG mixin、进度条、权重 loader、registry、post-process；
9. 扩展共享示例、支持模型文档和 L1～L4 Buildkite 测试；
10. 完成 T2I GPU parity 和固定硬件性能报告；
11. 合并 PR 1；
12. 增加 Edit 双路图像条件和 multimodal metadata；
13. 完成 1～3 图 Edit parity 与 L4 expansion 后合并 PR 2；
14. 先验证 padded mask batching，再按性能数据决定是否新增 flat-packed metadata；
15. 按 CFG → TP → SP → step/cache 的顺序优化。

## 18. 参考资料

- vLLM-Omni issue：<https://github.com/vllm-project/vllm-omni/issues/5337>
- Mage 官方仓库：<https://github.com/microsoft/Mage>
- T2I 模型：<https://huggingface.co/microsoft/Mage-Flow>
- Edit 模型：<https://huggingface.co/microsoft/Mage-Flow-Edit>
- 本仓库模型接入指南：`docs/contributing/model/adding_diffusion_model.md`
- 本仓库 CI 分层：`docs/contributing/ci/CI_5levels.md`
- 支持模型表：`docs/models/supported_models.md`
- 可参考的 T2I pipeline：`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`
- 可参考的多图处理：`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py`
- 可参考的 Qwen3-VL 兼容层：
  `vllm_omni/diffusion/models/internvla_a1/adapter_qwen3_vl.py`
- 可参考的 joint attention：`vllm_omni/diffusion/models/longcat_image/`
