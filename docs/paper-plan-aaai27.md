# AAAI-27 论文路线（证据全部指向 results-consolidated.md，旧 paper-outline.md 作废、仅其"弹药库/风险清单"两节继续引用）



## 1. 标题与论题

**标题（候选 A，推荐）**:Task-Structure-Aligned Precision Allocation for Quantizing Text-to-Speech Diffusion Transformers
**候选 B**:Where 4-Bit Hurts: Anatomy and Structure-Aligned Repair of W4A4 Quantization in TTS Diffusion Transformers

**一句话论题**:TTS DiT 的 W4A4 退化沿任务轴各向异性,其结构(哪个轴疼/哪段时间疼/哪类模块疼)是可测且稳定的——跨标定数据、跨尺度成立,但**不能从视觉域搬运**(双翻转)、**不能从数据侧修复**(语言效应=容量副现象);损伤源二分(权重↔可懂度、激活↔音色)直接给出结构对齐的精度分配配方。

**贡献列表(4 条,引言用)**:
1. **首个 TTS DiT 的系统性 W4A4 PTQ 研究**与任务轴退化解剖,建立在 **10 组独立随机标定集的分布形态**上(mean±std,而非单点):损伤集中于重复结构可懂度(严格 CER 分类学揭示原始 CER 低估真实损伤约一半)、SIM 成本小而普遍并带稳定脆弱声音尾部、zh 存在 4-bit 特有量化红利——结构在第二个尺度(3.5B,同批 K=10)上复现。〔总编 §1〕
2. **权重↔CER / 激活↔SIM 二分归因**:激活精度 0→100% 任意分配不动 CER(双尺度全 ns + W8A8 印证 + W16A4 交叉);SIM 模块特异且非单调——量化注意力激活中性甚至有益(双尺度 SIG)。**与视觉 DiT 共识双翻转**(护晚期步而非早期;cross-attn 与内容无关)。〔总编 §3.3/3.4〕
3. **late_ffn 配方**:仅晚期步×仅 FFN 保激活(1/6 预算),两个尺度的 SIM 最优点,严格优于两个均匀极端;绝对收益随尺度递减而原则不反转("模型越大越不需要配方,但配方方向不变")。〔总编 §3.5〕
4. **数据侧系统性答案**:语言构成是唯一因果因子(1B 预注册剂量链:伤 zh 显著/助 hard 方向性),**预注册的 3.5B 复检判定其为容量副现象**(红利消失→效应消失)——数据侧没有跨尺度稳定杠杆,修复必须走结构侧。〔总编 §2〕

## 2. 章节结构(7 页预算)

| § | 页 | 内容 | 证据(总编) |
|---|---|---|---|
| 1 引言 | 1.0 | 动机(TTS 部署成本;8-bit 免费、4-bit 才是边界);无先例声明(措辞:first systematic W4A4 PTQ study of a TTS DiT);贡献 4 条 | §1.2;弹药库 |
| 2 相关工作 | 0.5 | 三线:视觉 DiT PTQ(ViDiT-Q/MixDQ 系,全部护早期步→foil);音频 PTQ 边界(WASPAA'25 W4A8 非 TTS/BitTTS QAT);标定数据线(LLM 系 + arXiv:2607.00908 concurrent) | 弹药库(旧 outline) |
| 3 实验设置 | 0.75 | 模型/FlatQuant 论文最优/seed-tts-eval 三集/逐条配对协议(统一 calib_seed=0,配对 bootstrap 95% CI);**标定候选池的制定**(考虑维度:zh/en 双语来源 AISHELL-3+LibriTTS、与测试集说话人级不相交、污染审计、376 条规模)与 **K=10 随机 32 条子集协议**(seed 0–9,每组独立走完标定→生成→评测) | §0;统计口径 |
| 4 退化解剖 | 1.5 | T1 主表(fp32/W8A8/W4A4 **10 组 mean±std**×6 轴×双尺度);**组间散布本身是发现**(hard 极差 0.845=平均成本 5 倍→引§6);位宽二分;严格 CER 分类学(1B +0.13→+0.40;**3.5B 双尺度确认**+0.246→+0.324,低估幅度随尺度收窄=红利机制衰减)与重复串定位(1B 独扛→3.5B 摊平);声音尾部(**双尺度结构常数**,split-half 0.6–0.7、尾部 7–10×,均值随尺度减半);尺度效应(同批 K=10:成本稳定/散布收缩 2.5×/红利消失)。**全解剖轴现均为双尺度** | §1.1–1.7,1.3 |
| 5 归因与配方(主贡献) | 1.75 | 同模型配对设计;2×2 归因(fp32/noact/full/W16A4);T3 全配置双尺度表;SIM 非单调结构;late_ffn 配方+尺度递减;损失侧负结果一段(M2/M3+副作用);(若 W1 阳性:深度对齐权重配方小节→"双配方对称") | §3.1–3.6;W16A4/W1 在跑 |
| 6 数据侧:没有稳定杠杆 | 1.0 | 管线三步走:**10 子集分布(§4 已建立)→探索性相关筛假设(2.2 表)→单因子对照逐一验证**;F1 剂量双图(1B 因果链 vs 3.5B 消失);容量副现象判定;其余因素负结果压缩成一段+表;跨尺度零转移(10 集名次重洗 rho=0.03,重复类 0.60 例外) | §2.1–2.7 |
| 7 讨论与结论 | 0.5 | 双翻转的含义(任务实测不可搬运);实践指南三行;限制(单模型家族/fake-quant/MOS 代理无听测);未来(F5-TTS、per-linear、真 kernel) | 风险清单 |

**顺序理由**:解剖(什么坏了)→ 归因与修复(主贡献,先亮建设性结果)→ 数据侧(为什么不能靠数据,负结果垫在后)。备选是 4→6→5(先排除数据侧再上方法),叙事更"侦探"但把负结果放中间——冲刺版取前者。

## 3. 表图规划(正文 3 表 2 图,其余附录)

| 编号 | 内容 | 来源 |
|---|---|---|
| T1 | 主退化表:fp32/W8A8/W4A4(10 组 mean±std)× zh/hard/en CER+SIM × 1B/3.5B | §1.1/1.2/1.3 |
| T2 | 严格 CER 分类学压缩表(同音同调假错误 0/10 ↔ 严格口径 9/10;+0.13 vs +0.40) | §1.4 |
| T3 | 全配置总表双尺度(9+7 配置,预算/CER/SIM/配对 Δ,SIG 标注) | §3.2 |
| F1 | 剂量双图:1B 三点剂量链(zh 升/hard 降,SIG 标注) vs 3.5B 同集(两线全平)——"容量副现象"一图讲完 | §2.3 |
| F2 | step×module 结构示意 + late_ffn 位置(16 步×3 模块网格,染色=保精度收益方向) | §3.4/3.5 |
| 附录 | **10 子集逐集全表(1B+3.5B,含六轴综合排名与 s4 选定)**、候选池构建细节与污染审计、深度剖面、声音尾部、跨尺度排名表、M2/M3 全表、2.2 假设筛选表、L2/W1 全数据、效率(显存 4× 理论值+kernel 引用) | §1.1/1.3/1.6/1.7/2.6/3.6 |

## 4. 统计口径段(直接进 §3,已是卖点)

全文统一单 calib_seed=0;逐条固定种子(seed=base+条目索引)严格配对;方法系=同一固定标定模型逐条配对 bootstrap(最强内部效度);跨标定系=逐条配对 bootstrap 95% CI;"方向一致但 ns"永不表述为"验证有效";关键主张(语言轴、W5、W16A4/W1/L2)全部**预注册**。领域惯例不报 CI——这是超出规范的可信度卖点。

## 5. 在跑实验的插槽(全文截稿前的落点)

| 实验 | 预计落地 | 进论文位置 | 若赶不上 |
|---|---|---|---|
| W16A4 | 1–2 天(已给命令) | §5 归因 2×2 的第四角 | 用现有三角表述(0→100% 全 ns + W8A8),完全成立,只是少一角 |
| 3.5B 配方 zh/en 补全 | 生成中,数天 | T3 的 3.5B zh/en 行 | 3.5B 侧只报 hard(现状已自洽) |
| W1 深度混精 | 1–2 天 | 阳性→§5 新小节"双配方";阴性→§5 边界段一句话("护最难 6 块亦不动 CER,损伤不可约") | 附录或删,不影响主线 |
| L2 快照构成 | 1–2 天 | 阳性→§6 或 §5 一段;阴性→附录一行 | 同上 |
| heldtest 终报 | 1 作业+过滤 | 理想:正文表加 heldtest 列;现实:附录"冻结划分复核" | dev/全 test 已是官方测试集,非阻塞 |
| F5-TTS 第二家族 | **本轮不可能** | 限制段 + rebuttal 预案 | —— |
| 真人听测 | 本轮不可能 | 限制段 | —— |

## 6. 摘要草稿(明天注册用,~170 词,可改)

> Post-training quantization (PTQ) of diffusion transformers has been studied extensively for images and video, but never systematically for text-to-speech (TTS). We present the first systematic W4A4 PTQ study of a TTS diffusion transformer (1B/3.5B), and show that 4-bit degradation is sharply anisotropic along speech-specific task axes: intelligibility damage concentrates on repetition-structured text (a strict-CER error taxonomy reveals standard CER underestimates true damage by ~2x), while speaker-similarity cost is small, universal, and carried by a stable tail of fragile voices. Built on ten independent random calibration sets per scale, this structure is stable across calibration data and model scales, yet defies transfer from vision: protecting late—not early—diffusion steps helps, and cross-attention is irrelevant to content fidelity. We causally attribute intelligibility damage to weight quantization and timbre damage to activation quantization, with a counterintuitive non-monotonicity: quantizing attention activations is neutral-to-beneficial. This yields a targeted recipe—full-precision activations only for late-step FFNs (1/6 budget)—that beats both uniform extremes at both scales. Finally, preregistered calibration-data experiments show the only causal data-side factor (language composition) is a capacity epiphenomenon that vanishes at 3.5B: repair must come from structure-aligned allocation, not data selection.

## 7. 8 天写作日程

| 日期 | 交付 |
|---|---|
| 7/22 | §4 解剖节 + T1/T2;夜批结果回收分析 |
| 7/23 | §5 归因与配方 + T3(并入 W16A4;W1/L2 判定入位) |
| 7/24 | §6 数据侧 + F1;3.5B zh/en 回收入 T3 |
| 7/25 | §1 引言 + §2 相关工作(弹药库落引用;逐条核对 arXiv ID)+ §7 |
| 7/26 | F2 成图;全文首次通读;附录组装 |
| 7/27 | 内部评审(全文红笔一轮)+ 修订;数字逐一对表复核 |
| 7/28 | 提交全文 |
| 7/29–31 | 附录/代码打包,补充材料提交 |

## 8. 红线(沿用既定口径)

- 不做 best-of-K/构成规则等选择类正向主张(实例噪声档案 cs/W1' 不进论文);
- **SVDQuant 跨方法结果不进论文**(留内部档案,总编 §1.8;论文不做方法比较主张);
- "方向一致但 ns"只写方向性,不写"验证有效";
- 新颖性措辞:first systematic W4A4 PTQ study of a **TTS** DiT(不 claim first audio DiT quantization);
- 单模型家族/fake-quant/无听测在限制段主动披露。
