# SANA-WM 相机分支恒等表机制重设计(GDN identity tables redesign)

> 状态:设计稿(未实现)。配套分支:`feat/sana_wm_integration_optimize`。
> 背景:代码评审发现 `gdn.py` 的 `_cam_identity_tables_cached` 使用
> `lru_cache(maxsize=32)` 缓存按 `(device, B, N, H, D)` 键的全 1 / 全 0 CUDA
> 常量张量,多形状混跑时最多钉死约 0.5GB 不可回收显存(评审编号 G2)。
> 本文给出替代设计:正确性(逐位等价)为前提,最大化性能。

## 0. 设计目标与不变量

**目标排序:** 正确性(逐位等价)> 峰值显存(零额外常驻)> 吞吐(消除无谓带宽/计算)> 代码量。

**必须保持的不变量:**

- I1:主 GDN 分支(真实 RMS/RoPE 表)的行为逐位不变
- I2:相机分支输出与现版本逐位相等(见 §4 数学论证)
- I3:`data_ptr` 在稳态下稳定(CUDA Graph 捕获兼容)
- I4:不引入新的环境变量/配置面

## 1. 问题应该死在哪一层

### 1.1 现状:三层浪费叠加

`phase_a`/`phase_c` kernel 为主 GDN 分支硬编码了 RMS 归一化与 RoPE 旋转;
相机分支的 Q/K 在进 kernel 前已由 `cam_prep_func` 处理完毕,于是用数学恒等元
"骗"过 kernel:全 1 的 `inv_rms`/`norm_weight`(乘 1 不变)+ `cos=1`/`sin=0`
(旋转 0 度不变)。由此产生三层浪费:

1. **分配**:每种 `(B, N)` 形状一组表(N = latent帧数 × H × W,随业务分化);
2. **常驻**:`lru_cache(maxsize=32)` 强引用钉死,`empty_cache()` 无效、无人
   调 `cache_clear()`、profiler 账目不可见,上限 ≈ 0.5GB 且填满的时机恰是
   大请求最缺 headroom 的时候;
3. **带宽**:kernel 每次调用真实地从 HBM 读全 1/全 0 表(逐 head 程序放大)、
   真实地做恒等乘法——存的是信息量为零的常量,读和算全是白做。

`lru_cache` 只处理了第 1 层,而且处理得不好。上游 NVlabs 源码中这是一个
**无上限的模块级 dict**(`_CAM_IDENTITY_CACHE`),其研究脚本单进程只跑一种
固定形状(dict 恒为 1 条目,无害);vLLM-Omni 是多形状常驻服务,语境完全
不同。当前 `lru_cache(32)` 是移植期 review 补丁:把"无上限"改成"上限
0.5GB",没有改变问题的本质。

### 1.2 结论:表不应该存在

```
L1(主设计):IDENTITY_PREP constexpr —— kernel 级消除,表彻底不存在
L2(过渡层):预算化常量池 —— 在 L1 完成 GPU 回归前,O(1) 显存的供给
L0(否决项):专用 cam kernel —— 见 §1.3
```

### 1.3 L0:为什么不写专用 cam kernel(否决)

最激进的方案是为相机分支单独写去掉 RMS/RoPE 参数的 `phase_a_cam`/
`phase_c_cam`。否决理由:constexpr 分支消除后,共享 kernel 的 cam 变体与
专用 kernel 生成的机器码等价(Triton 对 `tl.constexpr` 分支做编译期消除,
死参数指针从不解引用、不占寄存器),性能收益为零,代价是 kernel 体 ×2 的
维护面——这正是本仓库 `gdn.py` 死代码问题的成因模式。性能上限已被 L1
摸到,不为零收益加代码。

## 2. L1 主设计:`IDENTITY_PREP` 编译期开关

### 2.1 接口变化

```python
# phase_a(..., identity_prep: bool = False)
# phase_c(..., identity_prep: bool = False)
# 调用方:
cam_scan_bidi_chunkwise:
    phase_a(qkv, beta, _DUMMY, _DUMMY, _DUMMY, _DUMMY, _DUMMY, _DUMMY,
            ..., skip_relu=True, skip_z=True, identity_prep=True)
    phase_c(qkv, _DUMMY, _DUMMY, _DUMMY, _DUMMY, M_hist, dummy_z,
            ..., num_only=True, identity_prep=True)
```

`_cam_identity_tables`、`_cam_identity_tables_cached`、
`_CAM_IDENTITY_CACHE_MAXSIZE`、`lru_cache` 引用全部删除。表参数位传模块级
单例 `_DUMMY = torch.empty(1)`(每 device 一个),复用 `phase_b_triton` 已有
的 dummy 约定(gdn.py 注释:"hand it a 1-element placeholder for the
inactive buffers"),并遵守已确立的规矩:**dummy 永不解引用**(同
`_phase_c_kernel` 的 `NUM_ONLY` z-load gate 修复)。

### 2.2 kernel 内改动点(以 `_phase_a_kv_kernel` 为例,`_phase_c_kernel` 同理)

```python
IDENTITY_PREP: tl.constexpr,   # 新增 constexpr 形参

# 原:无条件 load + 乘法
#   inv = tl.load(inv_rms_ptr + ...);  nw = tl.load(nw_ptr + ...)
#   Cos = tl.load(rope_cos_ptr + ...); Sin = tl.load(rope_sin_ptr + ...)
#   K_normed = K_raw * inv * nw;  K_rot = K * Cos + K_pair * Sin
# 新:
if IDENTITY_PREP:
    K_normed = K_raw           # 不 load、不乘
    K_rot    = K               # 不 load、不旋转(连 K_pair 的构造都省)
else:
    ...原逻辑原样...
```

关键性质:`tl.constexpr` 分支在 trace 期被消除——`IDENTITY_PREP=1` 变体的
PTX 中那些 `tl.load` 与乘法不存在,指针实参从不被解引用。这与"依赖优化器
DCE 碰运气"(G1 修复前的旧问题)有本质区别:constexpr `if` 是语言保证的
分支消除,不是优化器的可选行为。

### 2.3 性能收益核算(161帧@704×1280,N=18480,D=112)

| 消除项 | 单次调用量级 | 每请求放大(~15 cam块 × ~36步 × CFG2 ≈ 1080 次) |
|---|---|---|
| HBM 读:cos+sin 表(逐 head 程序重复读,×H 放大) | ~8.3MB × 2 × H | 数十 GB 级无谓读流量 |
| HBM 读:inv_rms / norm_weight 表 | MB 级 | 同比例 |
| 计算:恒等乘法 ×2 + 旋转乘加 | 每元素 4 次浮点 | 随表读一起消失 |
| Python 层:cache 查询 + 传参 | µs 级 | 1080 次 |
| 常驻显存 | — | **0.5GB → 0(连 L2 的 32MB 都不需要)** |

A800 HBM ≈ 2TB/s,数十 GB 无谓读 ≈ 每请求秒级的纯带宽时间。这也是
`lru_cache` 在解决错误问题的证据:它优化了表的供给侧,而表的**消费本身**
就是浪费。改完后该路径快于上游 NVlabs(上游真实执行这些恒等运算),
`IDENTITY_PREP` 值得回馈上游。

## 3. L2 过渡层:预算化常量池

L1 须改 kernel、必须 GPU 回归;L2 纯 Python、行为逐位等价、可即刻交付。

### 3.1 数据结构

```python
_CONST_POOL: dict[torch.device, _ConstPool]   # 每 device 一个,无淘汰
class _ConstPool:
    ones:  Tensor   # 一维全1
    zeros: Tensor   # 一维全0
```

取表 = 前缀视图:`pool.ones[: B*N].view(B, N)`、`pool.ones[: N*D].view(N, D)`、
`pool.zeros[: N*D].view(N, D)`。三张 ones 表共享同一块内存。

**硬约束**:kernel 是裸指针算术,表必须 contiguous。一维前缀切片再 view
数学上保证连续;反例是 `(B_max, N_max)` 大矩阵切 `[:B, :N]`——stride 为
`N_max` ≠ `N`,非连续,kernel 会静默读错。实现中在返回点保留
`assert t.is_contiguous()` 作为跨语言边界的契约检查(少数应保留的防御性
代码)。

### 3.2 容量策略:与 token cap 联动的一次性预分配

不做按需几何扩容,而是首次使用时按服务请求包络一次分配到位:

```python
capacity = SANA_WM_NATIVE_MAX_TOKENS × D   # 现 = 36080 × linear_head_dim
```

请求校验已保证 N 不超过该 cap,因此:

- 运行时零增长、零重分配 → `data_ptr` 终身稳定 → CUDA Graph 捕获安全(I3);
  `lru_cache` 版做不到(新形状 = 新条目 = 新指针);
- 显存确定性:`2 × 36080 × 112 × 4B ≈ 32MB`/device,启动即知,可计入显存
  账目(修复 G2 的"账目不可见");
- 若 cap 被 `extra_args` 临时调大:超界时单次重分配并打 WARNING(观测点),
  不静默膨胀。

### 3.3 线程/进程模型

每 worker 进程单线程执行 forward(vLLM-Omni 现状);池子初始化路径加模块级
锁兜底(仅首次分配竞争,稳态零开销)。TP 多卡 = 多进程,各自 32MB,无共享。

## 4. 正确性论证

**数学层(L1/L2 共用):** 替换前后 kernel 消费的值完全相同(全 1/全 0),
差异只在"值从哪来/还读不读"。IEEE-754 行为:`x × 1.0` 对一切有限值与
±inf 逐位保值;`pair × 0.0 = ±0`,`x + (±0) = x`(唯一边角 `-0 + 0 → +0`,
x 为归一化激活,符号翻转无可观测影响);输入经上游 RMSNorm,不含 NaN/Inf。
结论:L1 跳过运算与 L2 更换内存来源,输出均应与现版本 `torch.equal` 级
一致——该判断本身就是测试断言。

**测试矩阵(四组,前两组 CPU 可跑):**

1. **形状/连续性/值**:所有返回视图 `is_contiguous()`、stride 正确、值全 1/
   全 0(守 §3.1 硬约束);
2. **复用/稳定性**:同 device 重复调用 `data_ptr` 不变;混合 ~20 种 `(B, N)`
   形状后池子零增长(对照:旧实现每种形状增长 ~17MB);
3. **逐位 A/B**(GPU):固定 seed 随机 q/k/v/beta/decay,旧 lru_cache 版 vs
   L2 vs L1 三方 `torch.equal`;
4. **隔离性**(GPU):`IDENTITY_PREP=0` 变体(主 GDN 路径)输出与改动前逐位
   一致——证明 I1。

## 5. 风险登记与缓解

| 风险 | 缓解 |
|---|---|
| L1 新 constexpr 使 kernel 变体数 ×2,首次编译时间增加 | 实际只多编译 cam 用的那一种组合;cam 路径目前 env 可选,默认不编译 |
| 与上游 vendored 文件 diff 扩大,未来同步 NVlabs 更难 | 改动点用注释块标记 `# vllm-omni divergence: IDENTITY_PREP`;该优化值得提回上游,合并后 diff 自然消失 |
| L2 的 32MB 在小显存卡上常驻 | 仅在首次走 cam Triton 路径时分配(env 不开 = 0 占用);L1 落地后整层删除 |
| dummy 指针被未来改动误解引用 | 沿用 G1 立的规矩 + 验收时跑 `compute-sanitizer --tool memcheck` |

## 6. 落地顺序

1. **立刻**:L2 常量池(随 gdn.py A/C 类死代码删除同一 commit;CPU 验证
   测试 1/2);
2. **GPU 环境就绪**:跑测试 3 的 A/B + `cam_scan_bidi_chunkwise` vs Python
   参考路径的 parity(后者本就是翻 `SANA_WM_CAM_TRITON` 默认值的前置条件);
3. **同一窗口**:L1 kernel 改动 + 测试 3/4 + compute-sanitizer 验收 → 删除
   L2。最终表机制代码量少于现 lru_cache 版,常驻显存为零,速度优于上游。

---

**设计哲学一句话**:`lru_cache` 在优化"错误需求"的供给侧;正确的设计是
消灭需求本身(L1),过渡期把供给压成一次性预算内的常量池(L2)。两层均以
逐位等价为验收线。
