# Quality metrics matrix —— baseline(full) / early / late(ODE 步轴激活精度)

> best-config FlatQuant W4A4 fake-quant(`models/bc_1b_model.pt` / `bc_3p5b_model.pt`,一次标定,per-block+LWC+LAC+add_diag,seed 0)。
> `full`/`early`/`late` = 同一模型、同一 seed(base=1024),只在推理时逐步门控激活精度:
> - **full**(baseline)= 全 15 步激活 int4
> - **early** = 前 5 步(0–4)激活 fp16、其余 int4(**等预算对照**)
> - **late** = 后 5 步(10–14)激活 fp16、其余 int4(**方法**)
> 权重全程 int4。指标:WER/CER(内容)、SIM(音色,per-item paired 可显著性检验)、UTMOS/DNSMOS(自然度,**仅聚合**,无 per-item)。

## 复现(怎么跑)
**固定标定模型是唯一的一个**(`models/bc_{1b,3p5b}_model.pt`,seed 0);full/early/late **加载同一个模型**、只换步门控 —— 这个"同模型"是对照成立的前提。
> **一个 canonical 目录**:`generate_step_axis` / `w4a4_deploy_quality` / `w4a4_deploy_check_numerics` 都默认从 `paths.bc_model_path()`(= `$SEED_MODELS_DIR`,默认 `models/`)取模型,`--model` 可显式覆盖。想复用已有的 bc_*.pt(免拷 5–15GB):`export SEED_MODELS_DIR=~/workspace/seed_repro/models`。
```bash
source env.sh
# 一次性:产出固定标定模型(或 export SEED_MODELS_DIR 指向已有 bc_*.pt)
MODE=calibrate bash scripts/benchmark_step_axis_seedtts.sh 1b
# 生成 full/early/late × 三集 + 评 WER/CER/SIM/MOS + 配对 ΔSIM(early|late vs full):
bash scripts/benchmark_step_axis_seedtts.sh 1b "zh en hard" "full early late"
bash scripts/benchmark_step_axis_seedtts.sh 3.5b "zh en hard" "full late"
```
底层:`generate_step_axis.py`(步门控 gen,`_ACT_QUANT` per-step 钩子)+ `evaluate_seedtts_metrics.sh`(WER/CER/SIM/MOS)+ `paired_bootstrap.py`(配对 ΔSIM,10000 次自助,已对齐历史数复现本表数值)。

---

## 1B best-config
| set | cfg | WER | SIM | UTMOS | DNSMOS | ΔSIM vs full (P) |
|---|---|--:|--:|--:|--:|---|
| **zh** | full | 1.207 | 0.8096 | 3.105 | 3.396 | (baseline) |
| | early | 1.211 | 0.8093 | 3.108 | 3.395 | −0.0002 (P=0.21) n.s. |
| | **late** | 1.208 | **0.8108** | **3.142** | 3.401 | **+0.0013 (P=1.00) ✅** |
| **en** | full | 2.310 | 0.7553 | 3.755 | 3.274 | (baseline) |
| | early | 2.215 | 0.7554 | 3.765 | 3.275 | +0.0001 (P=0.56) n.s. |
| | **late** | 2.248 | **0.7571** | **3.771** | 3.279 | **+0.0018 (P=1.00) ✅** |
| **Hard** | full | 7.003 | 0.7882 | 2.932 | 3.398 | (baseline) |
| | early | 6.673 | 0.7870 | 2.941 | 3.400 | −0.0013 (P=0.03) n.s. |
| | **late** | 6.912 | **0.7894** | **2.977** | 3.402 | **+0.0011 (P=1.00) ✅** |

## 3.5B best-config(只跑 full/late)
| set | cfg | WER | SIM | UTMOS | DNSMOS | ΔSIM vs full (P) |
|---|---|--:|--:|--:|--:|---|
| **zh** | full | 1.174 | 0.8172 | 3.129 | 3.392 | (baseline) |
| | early | 1.096 | 0.8173 | 3.130 | 3.391 | +0.0001 (P=0.70) n.s. |
| | **late** | 1.130 | **0.8183** | **3.157** | 3.395 | **+0.0011 (P=1.00) ✅** |
| **en** | full | 1.869 | 0.7793 | 3.753 | 3.273 | (baseline) |
| | early | 1.925 | 0.7794 | 3.754 | 3.272 | +0.0001 (P=0.58) n.s. |
| | **late** | 1.873 | **0.7809** | **3.764** | 3.275 | **+0.0016 (P=1.00) ✅** |
| **Hard** | full | 6.059 | 0.7967 | 2.972 | 3.403 | (baseline) |
| | early | 6.057 | 0.7968 | 2.973 | 3.404 | +0.0001 (P=0.55) n.s. |
| | **late** | 6.108 | **0.7973** | **3.007** | 3.406 | **+0.0006 (P=0.99) ✅** |

---

## 关键观察

1. **SIM(音色)—— late 在 6/6(两模型×三集)全部显著恢复**(+0.0006~+0.0018,P≥0.99),幅度小但稳健、代表性(全集)、跨尺度。
2. **等预算 early 对照全空(两模型 6/6,已闭环)**:1B −0.0002 / +0.0001 / −0.0013、**3.5B +0.0001 / +0.0001 / +0.0001**,全 n.s. → 排除"多比特就变好",**是真定位:音色精度集中在晚去噪步**。**跨尺度完整闭环:1B+3.5B 都是 early 空 / late 显著。**
3. **UTMOS(自然度)—— late > full 在 6/6 全部成立**(1B: +0.037/+0.016/+0.045;3.5B: +0.028/+0.011/+0.035),early 居中(略高于 full)。方向与 SIM 一致,**corroborate**。⚠️ 但 UTMOS **仅聚合、无 per-item**,不能做 paired 显著性 —— 只作方向性佐证,不作显著性声明。
4. **DNSMOS ~ 平**(late−full 仅 +0.003~0.006)—— 对该效应不敏感。
5. **WER/CER —— 噪声,不受影响**(方向混、CI 全跨 0;之前 paired bootstrap 已确认;Hard per-item std ~3.2)。→ 方法**只动音色,不动内容**。

---

## 说明 / 注意

- **baseline = `full`**(标定模型的标准 W4A4 fake-quant)。它是本次 one-calibration 管线里**重 gen** 的,和旧 canonical(on_1b_mse)是两次标定 draw(SIM 差 ~0.001、Hard WER 差 ~0.5,GPU 非确定性)。**方法提升(ΔSIM)用 `full` 作 reference 是正确的**(同模型同 seed);绝对质量数以此表(bc_*model)为准,已定为唯一 canonical。
- **MOS 仅聚合**:MOS 后端(`seedtts_mos`,经 `evaluate_seedtts_metrics.sh`)只输出 mean UTMOS / DNSMOS-ovrl + n,无 per-item。若要 UTMOS 的 paired 显著性,需改脚本输出 per-item 再重跑。
- ~~3.5B 未跑 early~~ **已补(2026-07-06):3.5B early 三集全空(+0.0001,n.s.)→ 等预算闭环两模型都成立。**
- 数据(已同步入本 repo):`results/step_{full,early,late}_{zh,en,hard}_{sim,cer,wer,utmos,dnsmos}.txt`(**3.5B 加 `_3.5b`**:`step_<cfg>_3.5b_<set>_<metric>`;en 文本指标为 `wer`、zh/hard 为 `cer`);固定标定模型 `models/bc_1b_model.pt` / `bc_3p5b_model.pt`(唯一一个,seed 0)。*(源 seed_repro `results/oc_*`(1B)/`oc35_*`(3.5B),数值逐位一致——本表即由其 per-item SIM 经 `paired_bootstrap.py` 复现;同步时重命名对齐 `benchmark_step_axis_seedtts.sh` 的 `step_<cfg>` 口径。)* 另:fp32 基线(reproduce)= `results/pf_fp32*`(`benchmark_fp32_seedtts.sh`)、INT8 = `results/pf_int8*`(`benchmark_int8_seedtts.sh`),MOS 为 `mos_{fp32,int8}*`。
- 脚本:`generate_step_axis.py`(步门控 gen,加载固定模型 + `_ACT_QUANT` per-step 钩子)、`evaluate_seedtts_metrics.sh`(WER/CER/SIM/MOS)、`paired_bootstrap.py`(配对 ΔSIM/ΔCER 自助)、编排 `scripts/benchmark_step_axis_seedtts.sh`。
