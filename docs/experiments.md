# 实验文档：面向任务特征的音频 DiT 量化标定数据选择

> 本文档是本项目的**长期维护实验档案**：研究目标、实验方法、实验记录、结果分析都在这里。
> 面向读者：不要求语音背景。每次新增实验请按 [维护指南](#8-维护指南) 追加记录，不要删除历史结论。
>
> 创建：2026-07-18。状态：**从头实验**（旧标定集已于 2026-07-18 删除，见 §5.3）。

---

## 1. 一句话概述

我们研究：把一个大型语音生成模型压缩到 4-bit（W4A4 量化）时，**用哪些数据做"标定"效果最好**——并提出从任务特征（文本难度、说话人、prompt/生成区域、扩散时间步）出发**自动选择标定数据**的方法。

---

## 2. 背景速览（写给非语音方向的读者）

### 2.1 任务：零样本声音克隆 TTS

给模型两样东西：一段某说话人的**参考音频**（prompt，约几秒）+ 一句**目标文本**，模型直接合成"这个人说这句话"的音频——不需要针对该说话人训练。衡量好坏主要看两点：**说得对不对**（合成音频转回文字后错字率低）和**像不像这个人**（音色相似度高）。

### 2.2 被量化的模型：LongCat-AudioDiT

美团 LongCat 团队的开源扩散 TTS 模型（[arXiv:2603.29339](https://arxiv.org/abs/2603.29339)，HF: `meituan-longcat/LongCat-AudioDiT-{1B,3.5B}`），在 Seed 基准上是 SOTA 水平。关键构件：

| 构件 | 说明 |
|---|---|
| Wav-VAE | 把 24 kHz 波形压缩成"波形隐空间"序列（不经过 mel 频谱，这是该模型的核心创新） |
| DiT 主干 | Diffusion Transformer，在隐空间做扩散去噪。1B 版：24 层 block，隐维 1536；每 block = self-attn + cross-attn + FFN，AdaLN 调制 |
| 文本编码器 | UMT5-base，把目标文本编成 cross-attn 的条件 |
| 采样 | 流匹配 ODE，**16 步**去噪（nfe=16），APG 引导（强度 4.0） |

推理时序列由两段拼接：**prompt 区**（参考音频的隐向量，负责"像谁"）+ **生成区**（被去噪合成的部分，负责"说什么"）。这个双区结构 + 16 个去噪时间步，是本项目"任务特征"的主要来源。

**量化只针对 DiT 主干的 GEMM 线性层**（self_attn / cross_attn / ffn 的投影矩阵）；VAE、文本编码器、embedding 等保持原精度。

### 2.3 量化与"标定"

- **W4A4**：权重（Weight）和激活（Activation）都用 4-bit 整数表示。数值只剩 16 个档位，误差主要来自少数数值极大的"离群通道"。
- **PTQ（训练后量化）**：不重训模型，只用**少量数据**（标定集，本项目 32 条）跑前向，据此确定量化参数（缩放、裁剪阈值、旋转/变换矩阵等）。
- **标定数据的作用**：量化参数是对着标定数据的激活分布优化出来的。标定集覆盖不了真实使用场景的分布，量化模型就会在没见过的分布上出错。**"选哪 32 条数据"因此是一个自由度很大、但长期被随意处理的设计选择——这正是本项目的研究对象。**

---

## 3. 研究目标与故事

### 3.1 研究问题

> 在音频 DiT 的 PTQ 中，能否**从模型和任务的角度自动选择标定数据**（替代随机抽取/手工启发式），稳定提升量化后的任务指标？

### 3.2 贡献主张（目标形态）

1. **主线：任务感知的标定数据选择**。定义便宜的逐条/逐集合代理分数（模型侧：激活统计、任务代理梯度影响力；任务侧：音素覆盖、文本难度、说话人多样性、时间步覆盖），做集合级（多样性感知）选择，超过随机与手工基线。
2. **副线：任务敏感度加权的标定损失**。用任务代理 loss 的梯度统计给 block 重建损失做 token×通道联合加权（region 加权与通道加权的统一框架）。
3. 两条线共用同一套**敏感度统计基础设施**（Fisher/梯度平方），按通道聚合 → loss 权重，按样本聚合 → 数据影响力分数。

### 3.3 冻结基线原则（重要纪律）

改进必须建立在强基线上：FlatQuant 冻结为论文最优配置（见 `docs/paper-best-alignment.md`），SVDQuant 为第二基线（验证选择方法与量化方法正交），QuaRot 仅作方法选型表参考。**做数据选择实验时，一切贡献旋钮（loss 加权、region 权重）保持关闭**；增益只作为冻结基线之上的增量报告。

### 3.4 为什么这个方向成立

- LLM 侧已证明 PTQ 对标定数据敏感（Williams & Aletras 2024 等），扩散模型侧只有时间步采样类工作（Q-Diffusion、PTQ4DM），**音频 DiT/TTS 上的任务感知标定数据选择是空白**。
- TTS 的选择空间比 LLM 丰富：文本难度 × 说话人 × prompt/生成双区 × 扩散时间步。
- 科学风险由判定门把守（§6 的 GATE-B）：若标定集之间的差异不超过随机种子噪声，则内容维度无杠杆，转向时间步维度选择。

---

## 4. 实验方法与协议

### 4.1 量化方法（均已对齐论文最优配置，详表见 `docs/paper-best-alignment.md`）

| 方法 | 类型 | 角色 | 入口 |
|---|---|---|---|
| FlatQuant | 梯度训练（逐 block 重建 + 可学习 Kronecker 变换 + LWC/LAC/diag） | **主基线**（唯一有显式 loss、对标定数据敏感的方法） | `flatquant_best.py` / `scripts/benchmark/benchmark_flatquant_best_seedtts.sh` |
| SVDQuant | 闭式（低秩分支 + 平滑网格搜索） | 第二基线（泛化性验证） | `svdquant_pipeline.py` / `scripts/benchmark/benchmark_svdquant_seedtts.sh` |
| QuaRot(-GPTQ) | 免训练（Hadamard 旋转 + GPTQ） | 方法选型表参考 | `quarot_linear.py` / `scripts/benchmark/benchmark_quarot_gptq_seedtts.sh` |

FlatQuant 标定流程：32 条标定数据各跑一次全精度推理（16 步 ODE），钩取 block-0 输入，每条沿轨迹取 2 个时间步快照（共 64 条序列）→ 逐 block 训练量化参数（200 步 × mb4，块输出 MSE）→ 冻结。

### 4.2 评测协议

**测试集**：Seed-TTS 官方测试集，三个子集——zh（中文常规，2020 条）、en（英文常规，1088 条）、hard（中文困难：绕口令/生僻组合，400 条）。

**指标**（都是"量化后 vs 全精度"越接近越好）：

| 指标 | 含义 | 工具 | 方向 |
|---|---|---|---|
| CER / WER | 合成音频经 ASR 转文字后的字/词错误率——**"说得对不对"** | Paraformer(zh) / Whisper(en) | ↓ |
| SIM | 合成音频与参考说话人的音色相似度（说话人向量余弦）——**"像不像这个人"** | WavLM 微调模型 | ↑ |
| MOS | 无参考音质分 | UTMOS + DNSMOS | ↑ |

**统计纪律**：逐条固定种子（seed = base + 全局条目序号），量化前后逐条严格配对；结论必须过 `paired_bootstrap.py` 的配对 bootstrap 置信区间，不在噪声内的差异才算数。标定自身用 `--calib_seed` 固定。

### 4.3 数据治理（本项目的核心协议）

**三层严格分离**：

| 层 | 用途 | 要求 |
|---|---|---|
| 候选池（candidate pool） | 标定数据的来源，选择算法在此池内挑选 | 与测试集**说话人级不相交**（L2，见下） |
| dev 划分 | 选择信号 / 实验间比较。**已冻结（2026-07-18，seed=20260718）**：zh 300/2020 + hard 100/400 + en 200/1088。uid 清单入库 `data/splits/*_dev_uids.txt`（冻结语义：清单存在时 `dev_split.py` 只物化不重采样）；gen/eval 全链路以 `zh_dev/hard_dev/en_dev` 集名使用 | 测试集的固定子集，仅用于开发期 |
| untouched test | 最终报告（集名 `*_heldtest`：zh 1720 / hard 300 / en 888） | 测试集其余部分，报告前绝不触碰 |

⚠ **配对协议注意**：生成的逐条种子 = base + 该列表内索引，dev 列表索引 ≠ 全集索引——dev 运行只与其他 dev 运行（含 `p0_fp32` 参照）逐条配对，绝不与旧全集运行配对。

**污染等级定义**（数据选择研究必须 L2；方法对比允许 L1 + 记录 caveat）：

- **L0 配对级**：无标定条目与测试条目 (prompt, 文本) 完全相同；
- **L1 成分级**：prompt 音频、目标文本均不与测试集共享；
- **L2 说话人级**：prompt 说话人来自与测试集不同的语料库。

**候选池建设要求**（约 200–500 条）：zh prompt 取自与 Seed-TTS zh 来源不同的语料（如 AISHELL-3），en 少量混入（如 LibriTTS）；目标文本用全新来源（新闻/口语句 + 程序合成绕口令），长度拉开分布；**每条记录元数据**（说话人、性别、文本来源、字数、音素覆盖、prompt 时长）——该元数据表即后续特征回归（P1）的任务侧特征矩阵。建成后须通过重叠审计（全零）。

#### 4.3.1 候选池 v1 规格（2026-07-18，实现：`calib/pool.py` 构建 / `calib/audit.py` 审计）

**测量依据**（决定各筛选窗口）：Seed-TTS 测试 prompt 时长 zh p10/p50/p90 = 4.1/4.5/5.5s（几乎全部 ≥4.0s）、en = 3.4/4.5/6.0s；目标文本长度 zh 常规 18–28 字、hard 中位 51 字长尾至 385、en 41–86 字符。AISHELL-3 单句中位仅 3.2s → zh prompt 只取 ≥4s 的长句（≥4s 供给约 1.3 万条，无需拼接）；模型侧 `load_audio` 自动重采样，但池子仍写出裁剪后 24k 单声道副本（去首尾静音 top_db=35、自包含、路径稳定）。

**构成**（`pool_v1`，seed=20260718，旋钮均可调）：

| 子集 | 条数目标 | prompt 来源 | 目标文本来源 | 文本长度窗 |
|---|---|---|---|---|
| zh_normal | 120 说话人 × 2 | AISHELL-3，裁剪后 4.0–9.5s，性别轮转分层，每人取最接近 4.5s 的句子 | AISHELL-3 转写（**非 prompt 说话人**的句子，自带拼音）+“。” | 15–35 字 |
| zh_hardlike | 60 | 复用 zh prompt（打乱） | **按真实 hard 集分类学设计的确定性生成器**（三子风格，素材全部来自 AISHELL-3）：twister 词级混淆绕口令（高频二字词 + 同音/z-zh/n-l/in-ing 等混淆配对 + 句法模板，功能字停用表过滤）/ repeat 句子与片段重复 4–6 次 / concat 多句拼接长句；子风格记入元数据 `hard_substyle` | 25–120 字 |
| en_normal | 39 说话人 × 2 | LibriTTS dev-clean，裁剪后 3.0–9.5s | LibriTTS normalized 文本（非 prompt 说话人） | 40–90 字符 |

**交叉配对协议**（镜像测试集结构）：目标文本绝不来自 prompt 自身的句子或说话人；每个 prompt 音频至多用于一个 normal 条目（hardlike 可复用 prompt，审计仅提示不报错）；目标文本池内唯一。

**元数据表**（`pool_v1_meta.csv`，即 P1 任务侧特征矩阵）：说话人/性别/年龄段/口音（AISHELL-3 spk-info 免费提供）、prompt 时长与 RMS 响度、文本来源、字数、音节数、去调唯一音节数、最大同音节重复次数、唯一声母/韵母数、唯一字符比。

**审计政策**（`calib/audit.py`，不通过则池子不可用）：A 音频不得位于 seedtts_testset 内；B prompt 文件名不得与测试 prompt 碰撞；C 归一化文本不得与任何测试 prompt/目标文本相等；D ≥10 字的归一化文本不得与测试文本互为包含；E 池内卫生（uid/目标文本唯一、wav 存在、字段完整）。说话人级（L2）隔离由语料库来源保证（AISHELL-3/LibriTTS vs Seed-TTS 的 DiDiSpeech/Common Voice）并记录于此。

**GATE-B 抽样器**：`python -m audio_dit_quantize.calib.pool sample --pool data/calib_pool/pool_v1.lst --n 32 --seed <k> --out data/calib_pool/sets/rand32_s<k>.lst`（wav 路径按输出目录重写，`load_items` 直接可用）。

**已知局限（记录在案）**：① hardlike 为合成文本：twister 子风格的"词"来自 bigram 挖掘，仍有少量跨词边界的伪词（发音混淆属性正确，语义通顺性不保证）；真实 hard 集 max_char_run p90=6 的字符连跑尾部未覆盖；② en 无 hard 子集（测试集也没有）；③ AISHELL-3 转写无标点，构建时统一补句号；④ 语料域差异（朗读风格 vs 测试集口语域）是 E1 锚点实验要量化的因素之一。

---

## 5. 历史记录与已知结论

### 5.1 已确立的实验结论

| 日期 | 结论 | 证据 | 对后续的意义 |
|---|---|---|---|
| ≤2026-07 | 三方法均可跑通 W4A4 并对齐论文最优配置 | `docs/paper-best-alignment.md` | 基线冻结的前提 |
| ≤2026-07 | **chanbal loss（1/方差通道加权）在逐 block 粒度下全面差于普通 MSE；在逐 linear 粒度下明显更好**（均为 best config 下） | 历史实验（旧标定集） | 通道加权是真实杠杆但 1/var 在残差流上方向反了；催生 §6 的 GATE-A（敏感度加权） |
| ≤2026-07 | 晚期 ODE 步保持激活全精度可恢复 SIM（step-axis 实验，LATE 有效、EARLY 同预算无效） | `scripts/benchmark/benchmark_step_axis_seedtts.sh` | 时间步敏感度不均匀——GATE-B 失败时的备选方向（时间步选择）已有先验支持 |

### 5.2 chanbal 现象的机制假设（待 GATE-A 验证）

块输出是残差流，方差被少数功能上极重要的离群通道主导（量化领域共识：这些通道最需保护）。1/var 恰好把它们**降权**→ 逐 block 下系统性变差；逐 linear 的输出无此残差离群结构 → 平衡有正收益。推论：换成**下游敏感度权重**（任务代理 loss 对通道的梯度平方，Fisher 风格，预期给离群通道**升权**）应能取回收益。可零成本先验证：各 block 上 Fisher 权重与 1/var 权重应**负相关**。

### 5.3 旧标定集的污染审计与删除（2026-07-18）

旧标定集 `data/calib_heldout_hardlike32.lst`（32 条，zh prompt × 绕口令风格文本）的重叠审计：

| 重叠层面 | 结果 |
|---|---|
| 精确 (prompt, 文本) 配对 vs 测试条目 | 0/32（"heldout"仅是配对级） |
| prompt 音频 | 16 个中 **8 个**同时是 zh 测试条目的 prompt |
| 说话人 | 16 个中 **8 个**出现在 zh 测试集 |
| 目标文本 | 32 条中 **9 条**是测试集真实文本 |

即：**L0 干净、L1/L2 污染**。对（同一列表下的）方法间对比是内部公平的；但作为数据选择研究的候选池会让"选好数据"退化为"选和测试集像的数据"，且可能通过说话人泄漏抬高 SIM。**处置**：2026-07-18 从工作区删除，从头实验；如需复现旧 baseline，git 历史可找回（`git show b746150:data/calib_heldout_hardlike32.lst`）。旧 baseline 数字仅作历史参考，不进论文主表。

**当前仓库状态**（2026-07-18 更新）：标定列表已参数化——`flatquant_best.py` / `generate_seedtts.py`（svdquant、quarot_gptq 两条路径）支持 `--calib_lst`，`paths.py` 的默认值可被 `SEED_CALIB_LST` 环境变量覆盖（shell 启动器零改动换列表）。默认路径仍指向已删除的旧文件——**新实验必须显式传入干净池生成的列表**，否则报错并提示传参。注意：列表中 prompt 音频路径按"相对列表文件所在目录"解析（`load_items` 行为），建池时保持这一约定。

---

## 6. 实验路线图

两个**判定门**先行，通过后才重投入；全部判定在 1B 模型 + dev 划分上做，3.5B 只在最终确认用。

| ID | 实验 | 内容 | 判定标准 | 状态 |
|---|---|---|---|---|
| **E0** | 工程解锁 | `--calib_lst` 参数化；候选池构建（§4.3）+ 重叠审计脚本；dev/test 划分冻结；Phase 0 批量驱动脚本 | 全管线用新池冒烟跑通 | **进行中**（2026-07-18：参数化已完成——三个 python 入口支持 `--calib_lst`，shell 启动器支持 `SEED_CALIB_LST` 环境变量；语料已就位：`data/calib_corpora/aishell3/`（88,035 wav，218 说话人，含 spk-info 性别/年龄/口音元数据与拼音转写）+ `data/calib_corpora/LibriTTS/dev-clean/`（5,736 wav，带 normalized 文本）。**候选池 v1 已建成并通过审计**（§4.3.1）；**dev/heldtest 划分已冻结**（§4.3）；**Phase 0 驱动就绪**：`scripts/calib/phase0_sensitivity.sh`（fp32 参照 + K 组随机标定 + seed 重复，GPU 池波次调度、幂等续跑、自动评测收集，汇总 `phase0_collect.py`）。仅剩 GPU 冒烟：`LIMIT=2 K=2 SEED_REPEATS="1" EVAL_METRICS="cer wer" bash scripts/calib/phase0_sensitivity.sh 1b`） |
| **E1** | 污染溢价锚点 | 干净池随机 32 条标定一次 vs 旧列表基线（dev，配对） | 差距是否集中于 SIM（说话人泄漏证据） | 未开始 |
| **GATE-B** | 敏感性研究（Phase 0） | K=8–12 组随机 32 条集各标定评测 + 同一集 3–4 个 calib_seed 作噪声基线。**驱动就绪**：`GPUS=... K=10 bash scripts/calib/phase0_sensitivity.sh 1b`；汇总表 `results/p0_summary.csv`（`phase0_collect` 直接打印 组间std/seed std 比值 + oracle gap 的门判读） | **组间方差 ≫ seed 方差**且 oracle gap（最好-最差随机集）值得追 → 主线开绿灯；否则转时间步选择 | 待启动（等 GPU 冒烟通过） |
| **E2** | 敏感度统计模块 | 捕获态上 Hutchinson 探针反传单步 DiT 输出 → 逐 block 逐通道梯度平方；支持 prompt/生成区分开；按样本聚合出数据影响力分数 | 基础设施，无判定 | 未开始 |
| **GATE-A** | 损失加权三组对照 | var 加权（反向 chanbal）/ Fisher 通道加权 / Fisher token×通道，各 vs uniform-MSE（配对 bootstrap） | 任一显著优于 uniform → 副线成立；全在噪声内 → 副线收缩为消融 | 未开始 |
| **P1** | 代理分数回归 | 用 GATE-B 的 K 组完整实验做"代理分数→真实收益"相关性分析（零额外算力） | 找到可预测收益的分数 | 未开始 |
| **P2** | 集合级选择 | 多样性感知贪心（quality + coverage），对照：随机 / 朴素 top-k / 手工启发式；集合大小扫描 8/16/32/64 | 选出的集显著优于随机；理想主张"选 16 ≥ 随机 32" | 未开始 |
| **P3** | 泛化确认 | 选中集合迁移 SVDQuant；上 3.5B；untouched test 报最终数 | 增益跨方法/跨规模成立 | 未开始 |

依赖关系：E0 → {E1, GATE-B} 可立即排队；E2 与 GATE-B 并行开发；GATE-A 依赖 E2；P1 依赖 GATE-B 完成；P2 依赖 P1；P3 最后。

---

## 7. 实验记录（living，按时间追加）

> 每行一个实验运行。**约定**：结果一律写"vs 对照组的配对差值 + bootstrap 区间"，不只写绝对值；标定用 `--calib_seed`，生成种子协议默认 base=1024。

| 日期 | ID | 配置摘要 | dev 结果（CER/WER, SIM, MOS） | 结论/分析 |
|---|---|---|---|---|
| 2026-07-18 | E0 | 候选池 v1 构建：`calib/pool.py build`（seed=20260718）→ 376 条（zh_normal 240 / zh_hardlike 60 / en_normal 76），158 说话人，71MB 自包含 | （非量化实验，无指标） | 审计 PASS（0 污染）。prompt 时长 p10/50/90 = 4.4/4.7/5.2s（目标 4.1/4.5/5.5）、zh 语速 p50=3.8 字/秒；zh 性别 202F/98M（AISHELL-3 男性供给上限）、口音 north 224/south 76；en 缺 4 条（2 个说话人无合格时长句）。抽样器 + `load_calib_items` 端到端验证通过 |
| 2026-07-18 | E0 | **dev/heldtest 划分冻结**（seed=20260718）：zh 300/1720、hard 100/300、en 200/888；uid 清单入库 `data/splits/`，冻结语义验证通过（二次运行仅物化）。gen（`SETS`）、sharding（`set_meta`）、eval（`run_set`）三处注册 `*_dev/*_heldtest` 集名 | （非量化实验） | Phase 0 驱动 `scripts/calib/phase0_sensitivity.sh` + 收集器 `phase0_collect.py` 就绪：fp32 参照 + K 随机组 + seed 重复，逐 GPU 波次调度，幂等可续跑 |
| 2026-07-18 | E0 复查 | **hard 子集推翻重做**：初版字符汤生成器与真实 hardcase 结构严重错位（max_char_run p50 4 vs 1、标点密度减半、缺高唯一比类型）。量化真实 hard 集分类学（词级绕口令/短语句子重复/长句）后重写为三子风格生成器 | （结构指标对比，非量化实验） | 最终 p50 对齐：len 42.5↔51、max_char_run 1↔1、uniq_ratio 0.39↔0.44、phrase_rep 4↔5、punct 0.10↔0.10。教训入档：**合成难例必须先量化真实难例的结构分布再设计生成器**，"难"的来源是词/短语级重复与混淆，不是字符连跑 |

### 分析笔记（living）

- （待填：GATE-A 的 Fisher vs 1/var 相关性图、GATE-B 的方差分解与 oracle gap 等）

---

## 8. 维护指南

- **加一个实验**：先在 §6 路线图领一个 ID（或为新方向新增行），跑完在 §7 追加记录行；改变项目认知的结论提升到 §5.1。
- **不许删历史**：错误结论标注"已推翻 + 指向推翻它的实验"，保留原文。
- **命令速查**：环境 `source env.sh`；冒烟 `LIMIT=1 bash scripts/benchmark/benchmark_flatquant_best_seedtts.sh 1b hard`；评测 `bash scripts/evaluate_seedtts_metrics.sh <gen_dir> <tag> "zh en hard"`；SIM 需 `EVAL_METRICS="wer cer mos sim"`。
- **关键文件索引**：方法配置对齐 `docs/paper-best-alignment.md`；FlatQuant 标定 `src/audio_dit_quantize/flatquant_best.py`；路径/数据集常量 `src/audio_dit_quantize/paths.py`；配对统计 `src/audio_dit_quantize/paired_bootstrap.py`；模型仓库 `../LongCat-AudioDiT`（`audiodit/` 为架构定义）。
