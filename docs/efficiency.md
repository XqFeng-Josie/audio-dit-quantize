# Efficiency —— 设计 / 方法 / 结果 / 合理性审查

> **论文用什么**:**§2A 主表**(延迟 / RTF / VRAM,fp32 / fp16 / W4A4 同栈可比)+ **§3 两段式质量-延迟收口**。
> **一句话(2026-07-06 重测修正)**:bs=1 DiT ~85% launch-bound → **cudagraph 消启动是命脉**;**W4A4 显存全场双赢(随质量成立)**;延迟**分模型规模**:**1B(launch-bound)not-both**(恢复质量的 fp32-胶水 844 > fp32 677),但 **★ 3.5B(compute-bound)BOTH —— 恢复质量的 fp32-胶水 1291 < fp32 1705(快 1.32×)+ 显存 3.9× 少**。→ **"又快又准又省的 int4" 在够大/compute-bound 的模型上 A100 就能实现**,not-both 只是 launch-bound 专属;NVFP4 是 launch-bound 的补充出路。

---

# 1. 实验设计(详细)

## 1.1 测什么
每个配置测三样,每样按 **短 / 中 / 长** 三种输入长度分别报(latency 随音频长度=token 数近似线性):
- **latency**(ms):端到端一次 voice-cloning 合成的墙钟时间(median of `--runs 10`,`--warmup 2`)。
- **RTF**(Real-Time Factor)= 合成时间 / 音频秒数;**< 1 = 实时**。
- **VRAM**(GB):该配置整模型(DiT+text-enc+VAE)的显存占用(median 输入)。

## 1.2 怎么测(测量协议)
- **硬件**:单卡 A100-40GB。
- **采样**:flow-matching / rectified-flow(非 diffusion),`apg` guidance,**nfe=16 ⇒ 15 Euler steps ⇒ 30 次真实网络 forward**(=2·(nfe−1),CFG 每步 cond+uncond)。
- **精度基线**:fp32 DiT/text-enc、fp16 VAE。
- **执行模式**:**inductor + CUDA-graph**(`src/audio_dit_quantize/efficiency/cudagraph_dit.py::wrap_dit_cudagraph`,含 rotary keystone 修复 + **输出有效性守卫**),消除 bs=1 的 kernel-launch 开销 —— 这是让各精度公平同台的前提(见 §1.4)。
- **主脚本**:`src/audio_dit_quantize/efficiency/profile_efficiency.py`(开关 `--fp16 / --w4a4-deploy / --w4a4-hp-deploy / --inductor-cudagraph / --torchao-int8 / …`);编排 `scripts/benchmark_efficiency.sh`。
- **命令**:`python -m audio_dit_quantize.efficiency.profile_efficiency --model_dir <M> --guidance_method apg --steps 16 --runs 10 --warmup 2 --inductor-cudagraph [--fp16 | --w4a4-deploy | --w4a4-hp-deploy]`。

## 1.3 对比的配置
| 配置 | 说明 | 栈 |
|---|---|---|
| **fp32** | 未量化基线 | cu130(主)|
| **fp16** | 16-bit 基线 | cu130 |
| **W4A4-FlatQuant** | 我们的方法,真 int4 GEMM | cu130 |
| INT8(W8A8)| torchao,旧 compile 栈(只作比值)| 旧栈 |
| SVDQuant-W4A4 | 对比方法,nunchaku 环境,**只有 eager、cudagraph 未验证** | nunchaku |

## 1.4 关键设计① —— 延迟测量"无标定",且**config-无关**(必须写清)
W4A4 的**延迟/显存不做量化标定**:`w4a4_deploy_fp32glue.py::from_linear_latency` 用**单位矩阵 Kron 变换 + amax 假 scale**,不加载任何标定参数。
- **为什么这样做 + 为什么合理**:latency 由 int4 GEMM + 胶水算子的**张量形状与算子序列**决定,**与标定出的数值无关**。单位阵和标定阵是同形状的稠密 matmul、同样的 per-token scale/quant/dequant → **延迟逐位相同**。这是延迟测量的标准做法(测 kernel 时间不需要真权重)。
- **推论**:**延迟表对 per-linear / per-block / best-config 全部成立,不按 config 变**(§4 会再审这条)。

**延迟档为什么用"对称",不和 best-config(非对称)一致?** —— 刻意为之,且不影响代表性:
- **延迟档(`w4a4_deploy_fp16glue` / `w4a4_deploy_fp32glue`)= 对称 + 真 int4 CUDA kernel(`deploy.matmul`)**。测速度必须跑真 kernel,而现有 int4 kernel 路径为**对称码**而设(最简单);延迟又与标定无关 → 对称 identity 就够,不必跑标定。
- **质量档(`w4a4_deploy_quality`)= 非对称+LAC+add_diag(best-config),用 float matmul 精确算 codes**(为逐位复现 fake-quant),**不是真 kernel → 不能拿它测速度**。
- **⇒ 两档实现不同,是因为"测速度"与"验质量"需求不同**:一个要真 kernel(对称最简),一个要精确浮点(非对称 best-config)。
- **对称延迟 ≈ best-config(非对称)延迟**:非对称只多几个**极便宜的 elementwise**(zero-point 校正=外积加、add_diag=逐通道乘、LAC=scale 里的 clip),**主体(int4 GEMM + Kron 变换)完全一样** → 对称测出的延迟**代表实际部署的 best-config(asym)模型**(严格说略微低估,可忽略)。故 §2A 延迟数适用于 best-config 部署。
- *(TODO 后续补:若要延迟与 best-config 完全一致,给非对称加一条真-kernel 延迟路径;延迟数预期不变,纯为一致性。)*

## 1.5 关键设计② —— 质量与延迟**分离报告**(honesty 核心;胶水机理见 §1.7)
W4A4 部署的 int4 GEMM 精确;质量差异全在胶水精度(§1.7),故延迟与质量必须分开报、引用时绑定:
- **fp16-胶水**:快(§2A 的 W4A4-fp16胶水行)—— **但质量退化**(1B-Hard 13.3%,3.5B 崩溃)。
- **fp32-胶水**:质量恢复(≈fake-quant)—— **延迟分模型规模**:**1B 慢**(844 > fp32 677),**但 3.5B 快**(1291 < fp32 1705)。
- **⇒ 1B:主表的 "W4A4 快 1.9×" 是 fp16-胶水/退化档(envelope);恢复质量则慢 → 须标质量档,勿误读免费又快又准。**
- **⇒ 3.5B:恢复质量的 fp32-胶水本身就快过 fp32(§2B)→ 又快又准又省同时成立**,不再是 envelope。**按模型规模分别呈现。**

## 1.6 关键设计③ —— 可比性范围
- **同栈可比(2A 主表)**:fp32 / fp16 / W4A4-FlatQuant 都在 cu130 · inductor-cudagraph,**绝对 ms 三方直接可比**。
- **跨栈只比比值**:INT8 是旧 compile 栈(无 cudagraph);SVDQuant 是 nunchaku 环境 + eager。**绝对 ms 不可与 2A 混比**,只在各自栈内比,或旁证量级(cu130 fp32-cudagraph 678 ≈ 旧栈 fp32-compile 707)。

## 1.7 部署解剖(deploy anatomy)—— 哪些是 W4A4、什么是"胶水"(论文必需,理解 §2 的前提)

**一个被量化的 linear,部署时的流水线:**
```
x ─①Kron变换─ x' ─②算scale/zero─ ─③量化int4─ q_x ─★INT4 GEMM (q_x@q_w)─ ─⑤dequant─ y
                        └────────────── 胶水(浮点) ──────────────┘        └─胶水─┘
                                                              ★ 唯一的整数运算
```
- **★ INT4 GEMM(第④步)**:真整数运算,**精确、与精度无关**(权重码离线固定;这一步 fp16/fp32 都一样)。
- **"胶水" = ①②③⑤**(Kron 变换、per-token scale、量化、dequant):**浮点运算**,可 fp16 或 fp32。
- **为什么胶水精度决定质量**:int4 的"码"`q_x=round(x'/scale)` 是**浮点算出来的**;fp16 胶水的舍入误差会让 `x'/scale` 在**取整边界**落到错误的 int4 档(如 3.51→本应第4档,fp16 算成 3.48→第3档)→ 码错档,累积→退化(1B 13.3%、3.5B 崩溃)。fp32 胶水复现 fake-quant 的精确浮点→码正确→质量恢复。

**部署是混合精度(hybrid),不是全 W4A4:**
| 部分 | 精度 | 说明 |
|---|---|---|
| DiT 的 **self_attn/cross_attn/ffn 投影**(1B **240** 个 / 3.5B **320** 个 linear)| **W4A4** | GEMM 重、FLOPs 大;满足 int4 GEMM 维度 %32 约束(实测 0 skipped)|
| adaLN 调制、time_mlp、embedders、proj_out | fp | 精度敏感 + FLOPs 极小(量化伤质量、无速度收益)→ `_target_linears` 跳过 |
| 胶水(①②③⑤)| fp16/fp32 | 本质浮点,没法 int4 |
| 非 matmul(softmax、layernorm、SiLU、rotary、ODE 求解)| fp | 不是矩阵乘,无 GEMM 可量化 |
| VAE + text-encoder | fp | **DiT 占推理 ~97%** → 只量化 DiT 就抓住收益;VAE/text-enc 小且敏感 |

**⇒ 部署 = DiT 的 attn/ffn 投影层 W4A4,其余全 fp** —— PTQ 部署的标准做法(量化计算大头,敏感/小/非-GEMM 留高精度)。

---

# 2. 结果

> **两层框架**:**Layer-1 = 内在格式效率**(压缩/BitOps/权重流量,只由 shape×bitwidth 决定 → 硬件无关、绝对公平);**Layer-2 = 已实现 wall-clock**(§2A 起,依赖 kernel 成熟度 → 需标 kernel-limited)。这是 GPTQ/AWQ/SVDQuant/音频-DiT-PTQ 的通行写法:**压缩+BitOps 当主,wall-clock 附 caveat**。

## 2.0 Layer-1 内在格式效率(理论,硬件无关;`src/audio_dit_quantize/efficiency/intrinsic_efficiency.py`,无需 GPU)
BitOps = MACs×bit_w×bit_a(Baskin UNIQ,ASIC 面积/功耗代理);T=150 帧;权重字节与 T 无关。
| 精度 | W/A | 1B 权重 GB | 1B BitOps/fwd (Tb) | 3.5B 权重 GB | 3.5B BitOps/fwd (Tb) | 压缩 vs fp32 | BitOps↓ vs fp16 |
|---|---|--:|--:|--:|--:|--:|--:|
| fp32 | 32/32 | 3.624 | 139.16 | 12.751 | 489.63 | 1.0× | 0.25× |
| fp16 | 16/16 | 1.812 | 34.79 | 6.375 | 122.41 | 2.0× | 1.0× |
| INT8 | 8/8 | 0.906 | 8.70 | 3.188 | 30.60 | 4.0× | 4.0× |
| W4A16 | 4/16 | 0.453 | 8.70 | 1.594 | 30.60 | 8.0× | 4.0× |
| **W4A4** | **4/4** | **0.453** | **2.17** | **1.594** | **7.65** | **8.0×** | **16.0×** |

> **Layer-1:W4A4 全胜** —— 压缩 **8×** vs fp32、BitOps **16×** less vs fp16、**4×** less vs INT8。这是"格式内在能有多省"(硬件天花板);**§2A 起的 wall-clock 是"当前 A100+kernel 实现到多少"**(远未及内在天花板,因 int4 kernel 不成熟 + launch-bound)。**两者分开报:内在效率 W4A4 全胜,实现延迟受 kernel 限制。**

> **数据来源:2026-07-06 全量重测**(`scripts/benchmark_efficiency.sh` → `results/eff/*.txt`,**N=10**,median±std,输出有效性守卫,GPU 独占逐配置)。旧 5A 数(490/678/986 等)完美复现(见下),**且方差极小(±0.2–0.5ms,cudagraph 消掉启动方差)→ 1.9×/3.0× 比值远超噪声**。

## ★ 2A 主表(论文用;cu130 · compile+inductor-cudagraph;fp32/fp16/W4A4 同栈直接可比)
| 模型 | 方法(质量档)| 延迟 短/中/长 (ms, ±std) | RTF(中)| VRAM |
|---|---|--:|--:|--:|
| **1B** | fp32 | 488 / **677**±0.5 / 983 | 0.132 | 5.90 GB |
| | fp16 | 187 / **190**±0.9 / 237⚠ | 0.037 | 3.94 GB |
| | W4A4-fp16胶水(快,**质量退化 13.3%**)| 332 / **354**±0.4 / 395 | 0.069 | **2.67 GB** |
| | **W4A4-fp32胶水(准,≈fp32 质量)** | 765 / **844**±0.2 / 924 | 0.165 | **2.68 GB** |
| **3.5B** | fp32 | 1405 / **1705**±0.3 / 2942 | 0.333 | 15.83 GB |
| | fp16 | 332⚠ / 362⚠ / 471⚠(**全 NaN**)| — | — |
| | W4A4-fp16胶水(快,质量退化)| 526 / **567**±0.5 / 640 | 0.111 | **4.01 GB** |
| | **★ W4A4-fp32胶水(准,≈fp32 质量)** | 1138 / **1291**±0.2 / 1475 | 0.252 | **4.02 GB** |

**读表(W4A4 有两档:fp16-胶水=快但退化、fp32-胶水=准且可发货;⚠️ 引用须标是哪档)**:
- **显存随质量双赢**:两档 W4A4 显存都少 **2.2×(1B)/3.9×(3.5B)**,全场最小(权重 int4,与胶水无关)。
- **fp32-胶水(准,可发货)延迟分模型规模** —— **1B 844 > fp32 677(慢 1.25×,not-both);★ 3.5B 1291 < fp32 1705(快 1.32×)→ 又快又准又省同时成立**(§2B/§3)。**这是主表最该看的行**(fp16-胶水退化档不发货)。
- **fp16-胶水(快)延迟 1.9×(1B)/3.0×(3.5B)** —— 但**质量退化(1B-Hard 13.3%,3.5B 崩溃)**,是 envelope,不发货。
- **W4A4 vs fp16:慢 ~1.86×**(1B 354/190)—— int4 是 4 个不可融 kernel。
- **⚠️ fp16 数值不稳**:1B 长序列溢出 NaN(long INVALID),**3.5B fp16 全 3 档都 NaN**(短/中/长全挂)→ 3.5B 无有效 16-bit 基线(需 bf16)。这比旧文档("仅 3.5B 溢出")更严重:**1B 长序列也溢出**,守卫抓到。

> **VRAM**:`torch.cuda.max_memory_allocated` 峰值,含 DiT+text-enc+VAE,median 输入。**N=10 median±std**(源 `results/eff/`);cudagraph 下 std≈0.2–0.5ms(极稳)。

## 2B 胶水精度 → 质量(§2A 是延迟,这里是质量;int4 GEMM 精确,质量只由胶水精度决定)
**旋钮**:int4 GEMM 外围的 Kron 变换+scale+dequant 用 fp16(便宜)还是 fp32(精确)。Hard-set 实测:
| 胶水精度 | 1B-Hard | 3.5B-Hard | 结论 |
|---|---|---|---|
| **fp16-胶水(§2A 的"快"档)** | WER 13.3% / SIM .738 | WER 110% / SIM .074(崩溃)| **退化**(fp16 数值不准;3.5B 溢出崩) |
| **fp32-胶水(§2A 的"准"档)** | WER **6.98** ≈ fake-quant | WER **5.996** ≈ fake-quant 6.059 | **质量恢复**(fp32 胶水代数精确)|

- **要点:fp32-胶水恢复质量(两模型都是),fp16-胶水退化** → **可发货的是 fp32-胶水档**。
- **合起来看(§2A 延迟 + 本表质量)**:
  - **1B:not-both** —— fp32-胶水(准)延迟 844 > fp32 677(慢),快档(fp16)又退化。
  - **★ 3.5B:BOTH** —— fp32-胶水(准)延迟 1291 < fp32 1705(**快1.32×**)+ 质量恢复 + 显存3.9×少 = **又快又准又省**。
- **fused 弃用**:试图把 fp32 Kron 融进 kernel,两模型都更慢(1B 1717、3.5B 3363ms)。一次性负结果,config 已从 `benchmark_efficiency.sh` 移除、原始延迟文件不保留(数值见本行)。

## 2C 执行模式阶梯(三级分解:消启动 vs 融合)—— N=10,medium
| 配置 | eager | cudagraph-only | compile+cudagraph | 分解 |
|---|--:|--:|--:|---|
| 1B fp32 | 1707 | 789 | **677** | 消启动 2.2× + 融合 1.17× |
| 1B W4A4-fp16胶水 | 4354 | 537 | **354** | 消启动 8.1× + 融合 1.52× |
| 3.5B W4A4-fp16胶水 | 5941 | 833 | **567** | 消启动 7.1× + 融合 1.47× |
- **cudagraph(消启动)是大头**(W4A4 上 7–8×),compile(融合)再补 1.2–1.5×。custom_op 把 W4A4 送进融合路(但融不进 4 个 int4 kernel 的 epilogue → 仍慢于 fp16)。
- compile-only 单独档 **cu130 max-autotune 预热超时**,测不了;三级分解靠 cudagraph-only vs compile+cudagraph 反推融合净贡献。

## 2D INT8(cu130 测不了 —— 本次重测确认)
**本次重测(cu130):INT8 eager = 20.5s(1B)/26.6s(3.5B)**(torchao 动态量化 per-op 开销,无意义);**INT8 + inductor-cudagraph 报错**(`_int_mm` 需 M>16,bs=1 的 timestep 投影 M=1,仅在 cudagraph 分区路触发);**但 int8 + compile(no-cg)可测**(1B 890ms,同栈 fp32-compile 707 → 慢 1.26×,复现"1B launch-bound INT8 负")。→ cu130 实测见 `results/eff_int8/`(专用脚本 `scripts/benchmark_int8_efficiency.sh`);cudagraph 档仍不可测,3.5B compute-bound 的正向比值只能引**旧 compile 栈**:
| 旧栈 compile | fp32 | INT8 | 比值 |
|---|--:|--:|---|
| 1B(launch-bound)| 707 | 814 | 0.87×(慢)|
| 3.5B(compute-bound)| 1752 | 1205 | **1.45×(快)**+ 显存 2.9× 少 |
⇒ INT8 在 compute-bound 3.5B 又快又准(旧栈);**cu130 无法验证** *(provenance 待核)*。⚠️ 与 W4A4 不同栈,绝对 ms 不可比。

## 2E SVDQuant(本次重测:输出 INVALID,不可用)
**本次重测(nunchaku env):SVDQuant eager + compile+cudagraph **全部 INVALID**(短/中/长全 NaN/全零,1B+3.5B 都是)** → 延迟数(1B 3203/6771、3.5B 4300/9139ms)**不是有效测量**。**连 eager 都 INVALID**,不只 cudagraph → **SVDQuant 在这套环境产出退化输出,不可信、不采用**。(悬案落定:SVDQuant 的 cudagraph 数确实不能信,且问题比"默认流 bug"更广。)数据:`results/eff_svd/`(1B/3.5B × eager/compilecg,全 INVALID;手动 `svdquant_deploy.py` 产出)。

---

# 3. 论文口径(2026-07-06 重测后修正)
**核心修正:not-both 是 launch-bound(1B)专属,不是 A100 全面结论。compute-bound 的 3.5B 上 W4A4 又快又准又省。**
- **显存双赢:全场随质量成立**(权重 int4,胶水无关)→ 省 2–4× 显存 @ 恢复质量。
- **延迟:模型规模相关(关键)**:
  - **1B(launch-bound):not-both** —— fp32-胶水(恢复质量)844 > fp32 677(慢 1.25×);快档(fp16-胶水 354)质量退化 13.3%。int4 的多 kernel 启动开销 > 省的算力。
  - **★ 3.5B(compute-bound):BOTH** —— **fp32-胶水(恢复质量)1291 < fp32 1705(快 1.32×)+ 显存 3.9× 少**。int4 GEMM 省的算力 > fp32-Kron 开销。**"又快又准又省的 int4" A100 上就能实现(在够大/compute-bound 的模型上)。**
- **一等结论(修正)**:W4A4 的"加速⊥恢复质量"**只在 launch-bound(小模型/短序列)成立**;**compute-bound(大模型)上加速与质量兼得**。→ **写法:按模型规模分档报**;NVFP4 是给 launch-bound 场景的补充出路,不是唯一出路。
- ✅ **3.5B-both 已实测坐实**:fp32-胶水 Hard SIM 0.7963 vs fake-quant 0.7967(ΔSIM −0.0004)、WER 5.996 vs 6.059 → **质量恢复**;配合延迟 1291 < fp32 1705 + 显存 3.9× 少 → **又快又准又省三条全成立**。

---

# 4. 合理性审查(是否合理 —— 逐条)
| # | 设计点 | 是否合理 | 说明 |
|---|---|---|---|
| 1 | 延迟用单位阵、无标定 | ✅ 合理 | latency 由形状/算子决定,与标定值无关 → 单位阵=标定阵同延迟。**标准做法。** |
| 2 | 由此声称"延迟 config-无关" | ✅ 合理(带小注)| 对 per-linear/per-block/best-config 成立;**唯 best-config 的 add_diag/LAC 极小 elementwise 未计 → 略微低估**(可忽略)。 |
| 3 | 主表 W4A4 有两档(快/准)| ✅ 已配对呈现 | §2A 两档都在(fp16-胶水标"退化13.3%"、fp32-胶水标"≈fp32质量"),不会被误读免费又快又准。 |
| 4 | fp32/fp16/W4A4 同栈比绝对 ms | ✅ 合理 | 同 cu130·cudagraph、同 nfe、同 guidance;INT8/SVDQuant 跨栈只比比值。 |
| 5 | fp16 溢出 NaN | ⚠️ 局限 | 1B 长序列 + 3.5B 全档 NaN → 3.5B 无 16-bit 基线(需 bf16);守卫抓到,如实标 INVALID。 |
| 6 | 测量正确性守卫 | ✅ 加分(本次再验)| 守卫抓到:早先 365ms bug、fp16-NaN、SVDQuant-INVALID、INT8-报错。非可选。 |
| 7 | N=10 median±std | ✅ **已补方差** | 本次重测 N=10、报 median±std/min/max;cudagraph 下 std≈0.2–0.5ms(0.2%)→ **1.9×/3.0× 远超 1 std,非噪声**。评审"补方差"已完成。 |
| 8 | VRAM 定义 | ✅ 已定义(§2A 表注)| `max_memory_allocated` 峰值,含 DiT+text-enc+VAE,median 输入。 |
| 9 | RTF<1 全实时 | ✅ 合理 | 所有有效配置 RTF<1。 |
| 10 | ★ both-on-3.5B 新发现 | ✅ 质量+延迟都实测 | fp32-胶水 3.5B 1291<1705(延迟,issues=0)+ 质量恢复(ΔSIM −0.0004,`benchmark_deploy_quality_seedtts.sh 3.5b hard fp32` 产出:`dep_bc35_fp32_hard_*`)→ 结论稳。 |

**审查结论(经 3-视角对抗评审 + 逐条源文件核实 + 2026-07-06 全量 N=10 重测)**:测量方法**合理、可发表(0 critical),所有评审待办已闭合**。
- **已闭合**:#3 质量配对、#7 方差(N=10±std)、#8 VRAM 定义、INT8 跨栈标注。
- **sound(保留)**:identity-延迟法、输出守卫、cudagraph 捕获(3 修复)、同栈可比、算术全对、质量-延迟分离(实测)。
- **仅剩局限**:#5 fp16 无 3.5B-16bit 基线(需 bf16)、INT8 cu130 不可测(用旧栈)、SVDQuant INVALID(不采用)。

---

# 5. 代码 / 构建 / 复现(移交正式 repo 用)

## 5.1 代码清单(本 repo 布局;`python -m audio_dit_quantize.<module>`,PYTHONPATH=src 见 env.sh)
```
src/audio_dit_quantize/efficiency/(子包;python -m audio_dit_quantize.efficiency.<module>)
  ── profiling(测量)──
  profile_efficiency.py            # 主 profiler:延迟/RTF/VRAM/±std,按 --flag 选配置 + 输出有效性守卫
  cudagraph_dit.py                 # CUDA-graph 捕获 + rotary keystone 修复 + clone_dict_output
  intrinsic_efficiency.py          # Layer-1 内在效率(BitOps/压缩,无 GPU)
  ── W4A4 deploy 路径 ──
  w4a4_deploy_fp32glue.py          # fp32-胶水(准档)+ from_linear_latency(单位阵,延迟测量 wrap)
  w4a4_deploy_fp16glue.py          # fp16-胶水(快档,用 deploy.nn.Linear4bit)
  w4a4_deploy_quality.py           # best-config(asym+LAC)deploy【实现】+【端到端质量 eval CLI】(python -m 可直接跑,见 §5.4)
  svdquant_deploy.py               # SVDQuant 对比(nunchaku env)
  ── deploy 数值验证(脚本)──
  w4a4_deploy_check_numerics.py    # 数值检查:fp32-胶水 == fake-quant(~1e-6),跑全量前先过
src/audio_dit_quantize/  precision.py / paths.py / flatquant_layers.py / flatquant_best.py  # 父包共用(from ..)
scripts/
  benchmark_efficiency.sh     # 编排:量化×执行模式 全矩阵,source env.sh + python -m
vendor/flatquant_ref/(gitignore,按 README 拉取+构建)
  deploy/__init__.py          # matmul(A,B) -> deploy._CUDA.matmul(int4 GEMM)
  deploy/_CUDA.*.so           # ★ CUDA 扩展,setup.py 编译(本 repo 需 build,见 §5.3)
  deploy/functional/{quantization.py(pack_i4), online_trans.py(kronecker_matmul)}
  deploy/nn.py(Linear4bit) / deploy/kernels/gemm.cu(★CUTLASS int4b_t,patch 打这)
  flatquant/flat_utils.py(kronecker_matmul,fp32 胶水) / third-party/{cutlass, fast-hadamard-transform}
patches/flatquant_cudagraph_stream.patch  # ★ gemm.cu 用当前 PyTorch 流(否则 cudagraph 丢 GEMM→全零=365ms bug)
外部:LongCat-AudioDiT(LONGCAT_DIR)—— audiodit + batch_inference.infer_one
```

## 5.2 依赖链(按配置)
```
profile_efficiency.py
 ├─ 通用: audiodit(LongCat)、transformers、..precision.apply_precision、..paths
 ├─ fp32/fp16              → precision.apply_precision
 ├─ W4A4 fp16-胶水(快)    → w4a4_deploy_fp16glue.wrap_dit_w4a4 → deploy.nn.Linear4bit → deploy._CUDA.matmul
 ├─ W4A4 fp32-胶水(准)    → w4a4_deploy_fp32glue.wrap_dit_w4a4_hp_latency → flatquant.kronecker_matmul(fp32)+ deploy.matmul
 │   (fused 变体)          → w4a4_deploy_fp32glue.wrap_dit_w4a4_hp_fused_latency
 ├─ SVDQuant               → svdquant_deploy.wrap_dit_svdquant(nunchaku env)
 ├─ INT8/fp8               → torchao.quantize_
 └─ cudagraph(所有)       → cudagraph_dit.{patch_rotary_for_cudagraph, wrap_dit_cudagraph, clone_dict_output}

质量验证(用 best-config 标定模型;都在 w4a4_deploy_quality 里):
  w4a4_deploy_quality (main)  → wrap_dit_bc_deploy + ..flatquant_best.load_items(端到端 gen + SIM/WER)
  w4a4_deploy_check_numerics  → import w4a4_deploy_quality 的 W4A4BCDeployLinear / bc_fakequant_ref(数值 ~1e-6 检查)
```
**注**:延迟档(`from_linear_latency`)用单位阵、不加载标定 → 与标定 config 无关(§1.4);质量档(`w4a4_deploy_quality.py`)才需要标定模型。

## 5.3 环境 + 构建 int4 kernel(★ 关键)
- **运行环境**:Python 3.13、`torch 2.12 + cu130`、CUDA 13.3 toolkit(构建用)、A100。SVDQuant 另需隔离 `nunchaku` env(`torch 2.11 + nunchaku wheel`)。
- **构建 `deploy/_CUDA.so`(必须先打 patch 再编译)**:
  ```bash
  cd vendor/flatquant_ref
  git apply ../../patches/flatquant_cudagraph_stream.patch   # ★ 先打:gemm.cu 用当前流(否则 cudagraph 丢 GEMM→全零)
  # CUTLASS/fast-hadamard 子模块就位后:
  python setup.py build_ext --inplace                        # 生成 deploy/_CUDA.cpython-*.so
  ```
  **本 repo:vendor/flatquant_ref 是 gitignore 的,按 README 拉取后须自行 build `.so`。patch 未打则延迟数因 cudagraph 丢 int4 GEMM 而失真(输出全零,被有效性守卫抓到)。**

## 5.4 运行(怎么跑)
所有命令先 `source env.sh`(设 `PYTHONPATH=src:vendor/flatquant_ref:$LONGCAT_DIR` + CUDA + `$PYTHON_BIN`),统一用 `python -m audio_dit_quantize.efficiency.<module>`。

**(A) 延迟 / RTF / VRAM —— ⚠️ 需独占 GPU(cudagraph 计时怕争抢)**
```bash
C="--guidance_method apg --steps 16 --runs 10 --warmup 2 --compile --inductor-cudagraph"
M="--model_dir meituan-longcat/LongCat-AudioDiT-1B"   # 或 -3.5B
$PYTHON_BIN -m audio_dit_quantize.efficiency.profile_efficiency $M $C --precision fp32        # fp32
$PYTHON_BIN -m audio_dit_quantize.efficiency.profile_efficiency $M $C --fp16                  # fp16
$PYTHON_BIN -m audio_dit_quantize.efficiency.profile_efficiency $M $C --fp16 --w4a4-deploy    # W4A4 fp16-胶水(快档)
$PYTHON_BIN -m audio_dit_quantize.efficiency.profile_efficiency $M $C --fp16 --w4a4-hp-deploy # W4A4 fp32-胶水(准档,1B 844/3.5B 1291)
# 全矩阵一键(量化×执行模式):
bash scripts/benchmark_efficiency.sh both 10     # -> results/eff/<model>_<config>.txt + progress.log
```
输出:每配置 short/med/long 延迟 median±std(N=10)、RTF、VRAM + 输出有效性守卫。

**(B) Layer-1 内在效率(BitOps/压缩,无需 GPU)**
```bash
$PYTHON_BIN -m audio_dit_quantize.efficiency.intrinsic_efficiency
```

**(C) 部署质量验证(需 best-config 标定模型 `models/bc_{1b,3p5b}_model.pt`;质量测,可与他人共用 GPU)**
```bash
# ① 数值检查(~1min):fp32-胶水 deploy == fake-quant(~1e-6),跑全量前先过这关
$PYTHON_BIN -m audio_dit_quantize.efficiency.w4a4_deploy_check_numerics --model models/bc_1b_model.pt --sample 24

# ② 端到端质量(gen+eval 一键;§2B 的 fp32-胶水恢复 vs fp16-胶水退化):
bash scripts/benchmark_deploy_quality_seedtts.sh both hard          # 两模型×两胶水×Hard -> results/dep_bc[35]_{fp32,fp16}_hard_{cer,sim}.txt
bash scripts/benchmark_deploy_quality_seedtts.sh 1b "zh en hard"    # 可扩到 zh/en(hard 是诊断集,退化最明显)
# 底层单步(wrapper 封装的就是它):
$PYTHON_BIN -m audio_dit_quantize.efficiency.w4a4_deploy_quality \
    --model_dir meituan-longcat/LongCat-AudioDiT-1B --set hard --glue fp32 --out_subdir dep_bc_fp32/hard   # 准档(恢复质量)
```
> 注:`w4a4_deploy_quality` = deploy 实现(被 check_numerics import)+ 质量 **gen**(扁平写 `gen/<out_subdir>/*.wav`);打分由 `evaluate_seedtts_metrics.sh` 做(命名 `<prefix>_<set>_<metric>`,故 wrapper 用 `--out_subdir dep_<tag>_<glue>/<set>` 造出分 set 目录)。wrapper `benchmark_deploy_quality_seedtts.sh` 把这两步串起来,支持 `[1b|3.5b|both] [zh,en,hard] [fp32,fp16]`。`--glue fp32`=准档(复现 fake-quant,both-on-3.5B)、`--glue fp16`=快档(测退化,§2B)。

## 5.5 本 repo 就位状态
- **已在 repo**:`src/audio_dit_quantize/efficiency/` 的 8 个模块(quality 含实现+eval) + `scripts/benchmark_efficiency.sh` + `patches/flatquant_cudagraph_stream.patch` + 本文档。
- **须按 README 拉取+构建**:`vendor/flatquant_ref`(git clone + apply patch + `python setup.py build_ext --inplace`)、LongCat-AudioDiT、seed-tts-eval。
- **可裁**:只留 FlatQuant-W4A4 可去 `svdquant_deploy.py`;INT8 用 torchao;质量验证用 `w4a4_deploy_quality`(实现+eval)`/_check_numerics`。
- **构建产物**:`deploy/_CUDA.so` 按 6.3 重编(或带对应 py/cu 版预编译)。
- **外部依赖**:LongCat-AudioDiT(`LONGCAT_DIR`,提供 `audiodit` + `batch_inference`);torchao(INT8/fp8);nunchaku env(仅 SVDQuant)。
- **可裁**:若正式 repo 只留 FlatQuant-W4A4,可去 `svdquant_deploy.py` + nunchaku env + torchao(INT8)。

---

# 6. 局限 / 待办
- **✅ 已闭合(2026-07-06 N=10 重测)**:方差(median±std,±0.2–0.5ms)、VRAM 定义、质量档配对、both-on-3.5B 质量+延迟验证。
- **fp16 溢出无法作 16-bit 基线**:1B 长序列 + 3.5B 全档 NaN → 3.5B 的 W4A4-vs-16bit 需 bf16(未测)。
- **INT8 cu130 cudagraph 不可测**:torchao⊗inductor-cudagraph 报错(`_int_mm` M>16)+ eager 20s 无意义;**int8+compile(no-cg)可测**(cu130 实测 `results/eff_int8/`,专用脚本 `benchmark_int8_efficiency.sh`),3.5B 正向比值(1B 0.87× / 3.5B 1.45×)引旧栈,*provenance 待核*(`logs/int8_efficiency_sweep.log`)。
- **SVDQuant 不采用**:本套环境输出 INVALID(eager+cudagraph 全崩),数不可信。
- **compile-only 档测不了**:cu130 max-autotune 预热超时;三级阶梯靠 cudagraph-only vs compile+cudagraph 反推。
- 数据:`results/eff/*.txt`(N=10 主表,严格对应 `benchmark_efficiency.sh`:fp32/fp16/W4A4)、`results/eff_int8/`(INT8 latency,cu130 + 专用脚本 `benchmark_int8_efficiency.sh`)、`results/eff_svd/`(SVDQuant,全 INVALID,手动)、`results/dep_bc*`(1B/3.5B deploy 质量,`dep_<bc|bc35>_<glue>_<set>_<metric>`,脚本 `benchmark_deploy_quality_seedtts.sh`)、`logs/int8_efficiency_sweep.log`(旧栈 INT8 比值);模型 `models/bc_*model.pt`;重测脚本 `scripts/benchmark_efficiency.sh` + `benchmark_int8_efficiency.sh` + `benchmark_deploy_quality_seedtts.sh`。
