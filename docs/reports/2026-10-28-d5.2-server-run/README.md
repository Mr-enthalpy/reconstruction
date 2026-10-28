# D5.2 单卡 A6 服务器运行（24,800 步）

日期：2026-10-28
运行标识：`server_a6_24800`
分支：`codex/d5.2-server-training`
设备：NVIDIA GeForce RTX 3090（C-4 验收测试服务器 `kunqian`）

## 1. 目的与范围

在实验室资源系统（Citespec Resource）的一次真实租用中，用**单张 GPU**完成 D5.2
A6 基线的一次可复现训练，回答一个具体问题：

> D5 报告中"高频恢复不足"（HF relative L2 ≈ 0.97）是结构性问题，还是训练预算不足？

范围限制（与 D5/D5.1 一致）：

- 21 波段 450–650 nm，512×512 场景，39 train / 9 val；
- measured PSF 仅作 **signed shape-only** 近似，不是标定过的物理模型；
- 不加载冻结测试集，不做分类、不做 RGB 前端、不做压缩、不做联合优化；
- 不涉及硬件采集，不修改 `optic_system`，不修改 Meta pin。

## 2. 环境与预算

| 项 | 值 |
|---|---|
| 代码 | 干净提交 `69bd221`（`reconstruction-69bd221.tar.gz`），工作区未提交改动未进入实验 |
| Python | 3.11.13 |
| PyTorch | 2.7.1+cu128，`cuda.is_available()=True`，可见 device 数 = 1 |
| 依赖补充 | `scipy 1.17.1` + `opencv-python-headless 5.0.0.93`（容器根只读，装在 `/workspace/.local`） |
| 数据 | `train.npz` 629,637,033 B；`val.npz` 138,099,071 B；`kernels.npz` 152,546 B；`manifest.json` 43,041 B |
| 设备显存/算力 | 单卡整租，8 CPU，32 GiB 内存 |
| 租约 | 两段 4 h 租约（中间释放，workspace 保留后原地续训） |
| 实际计算 | 128.4 + 861.6 + 4,624.8 = **5,614.8 s ≈ 93.6 min**，0.23 s/step |

数据出处（服务端实测，与准备阶段记录一致）：

```text
train.npz     024d2b4c8956707387289242ddbadce08e3379a51ed6e0752a167666dbb32908
val.npz       e61ac4f102d58a44bdb8f7174a960d13f2e102c5db00cc0bc408ea5583ff9247
kernels.npz   f07d9bbb7a4ff870433e8e72dcd19298d642e639c499afe1057a97eeb9206d4b
manifest.json cfe89c34868b7eece4c6089d238d79fe224cd62c54db23df0a81ba7e40ef02f3
```

冻结测试集 `test.npz` 未上传、未加载（容器内 `find` 仅命中 scipy 自带同名测试文件）。

## 3. 配置

`experiments/d5_2/configs/a6_formal.json`，未改动：

- 模型 A（CNN-PGD），6 stages，width 64，depth 4，**98,139 参数**；
- batch 1，100 步/epoch，目标 300 epoch（30,000 步），`patience=40`，`eval_every=5`；
- AdamW，lr 2e-4，cosine 退火到 1e-5，weight decay 1e-5，grad clip 1.0；
- 训练噪声 RMS 比例 0.5%，亮度抖动 [0.9, 1.1]；
- 损失：`lambda_grad=0.2`、`lambda_spectral=0.05`、`lambda_data=0.02`；
- 选模：最小验证重建目标；seed 20261021。

实际运行分两段，第二段从第一段 `last.pt` 续训（含 optimizer/scheduler/RNG 状态）：

```bash
python3 -X utf8 experiments/d5/train.py \
  --prepared /workspace/data/prepared512 \
  --output   /workspace/runs/a6_4000 \
  --config   experiments/d5_2/configs/a6_formal.json \
  --device cuda --resume /workspace/runs/a6_4000/last.pt --stop-after 300
```

## 4. 结果

实际完成 **248 epoch / 24,800 步**（占 300 epoch 的 83%），结束原因是配置自带的
`patience=40` 早停（epoch 208 最优后连续 40 个 epoch 无改善），**不是时限或步数上限**。
早停时 lr 仍有 2.4e-5，未完全收敛。

### 4.1 训练过程（每 20 epoch 抽样，9 个验证场景平均）

| epoch | step | train_loss | val_loss | PSNR (dB) | SSIM | SAM (°) | HF rel-L2 |
|---|---|---|---|---|---|---|---|
| 5 | 500 | 0.02757 | 0.016532 | 29.055 | 0.9042 | 12.341 | 0.9703 |
| 40 | 4000 | 0.02579 | 0.014809 | 29.777 | 0.9202 | 11.000 | 0.9424 |
| 80 | 8000 | 0.02681 | 0.013768 | 30.262 | 0.9320 | 10.345 | 0.8722 |
| 120 | 12000 | 0.02487 | 0.013766 | 30.359 | 0.9331 | 11.078 | 0.8334 |
| 160 | 16000 | 0.02638 | 0.013274 | 30.515 | 0.9377 | 10.168 | 0.7851 |
| 200 | 20000 | 0.02305 | 0.013069 | 30.606 | 0.9391 | 10.116 | 0.7676 |
| 240 | 24000 | 0.02313 | 0.013021 | 30.663 | 0.9393 | 10.113 | 0.7513 |
| **208（best）** | 20800 | 0.02325 | **0.012924** | —¹ | —¹ | —¹ | —¹ |

¹ `eval_every=5`，epoch 208 不是 5 的倍数，因此 `history.json` 该行没有 metrics 字段；
最优 checkpoint 的指标由 `validation.json` 单独复算（见 4.2）。这是当前记录方式的缺陷，
应把选模 epoch 与指标记录 epoch 对齐。

### 4.2 最优 checkpoint（epoch 208）验证集指标

`best.pt` 在 9 个验证场景上的平均：

| 指标 | 值 |
|---|---|
| PSNR (unit range) | **30.725 dB** |
| SSIM (band mean) | **0.9400** |
| SAM | **9.899°** |
| cube relative L2 | 0.2493 |
| global spectrum relative L2 | 0.1009 |
| high-frequency relative L2 | **0.7619** |
| measurement relative L2 | 0.0110 |

逐场景：

| scene | dataset | PSNR | SSIM | SAM |
|---|---|---|---|---|
| fake_and_real_lemon_slices | CAVE | 31.474 | 0.9391 | 10.019 |
| fake_and_real_lemons | CAVE | 28.956 | 0.9454 | 10.064 |
| fake_and_real_strawberries | CAVE | 31.613 | 0.9051 | 13.353 |
| fake_and_real_sushi | CAVE | 32.916 | 0.9670 | 7.032 |
| fake_and_real_tomatoes | CAVE | 29.092 | 0.9607 | 9.285 |
| kaist_25 | KAIST | 28.385 | 0.9223 | 11.235 |
| kaist_26 | KAIST | 31.577 | 0.9571 | 7.691 |
| kaist_28 | KAIST | 30.468 | 0.9210 | 10.483 |
| kaist_29 | KAIST | 32.041 | 0.9426 | 9.929 |

## 5. 结论

1. **高频不足主要是预算问题，不是结构性失败。** 从 4,000 步到 24,800 步，
   HF relative L2 从 0.942 降到 0.762（相对下降约 19%），同期 PSNR +0.95 dB、
   SSIM +0.020、SAM −1.10°、cube relative L2 0.275 → 0.249。
   而在 500→4,000 步区间，HF 只从 0.970 降到 0.942。也就是说，**高频指标的改善
   主要发生在训练后期**；若只用 4,000 步的短跑就下"高频无法恢复"的结论，会明显误判。
2. **A6 基线在该数据与 PSF 设定下可用但仍有明显短板。** 24,800 步后 HF 仍为 0.762，
   远未接近 1.0 以下的目标区间；空间梯度项与全局光谱项（0.101）也说明结构性误差仍在。
   单纯加预算不能解决，需要在结构/先验层面继续（与 D5 的失败分析一致）。
3. **单点指标有噪声，不能按 epoch 挑数。** SAM 在 epoch 120 反弹到 11.078 后才继续下降；
   选模必须依据验证损失或固定评测点，不能凭单个 epoch 的 PSNR/SAM 判断。

## 6. 明确不支持的结论

- **不是** 冻结测试集结果：全程只用 train/val，`test.npz` 未加载，不能据此宣称泛化性能。
- **不是** 与 A-large、C6-SFAT、Achromatic 对照的结论：本次只跑了 A6 单模型单 seed。
- **不是** "D5.2 S1 完成"：正式 S1 指 300 epoch / 30,000 步，本次 24,800 步且提前早停。
- **不是** 硬件或光学验证：PSF 仍是 signed shape-only 近似，未经辐射度/Bayer/视场标定。
- **不是** 设备无关结论：本次在 RTX 3090 上运行；与 amax 的 4090 数值不可直接等同。
- 不涉及分类、RGB 前端、压缩、联合优化、PSF 掩模优化。

## 7. 产物

服务端（`/workspace` = `/home/data1/citespec-users/udcagent`，租约释放后仍保留）：

```text
runs/a6_4000/best.pt        89,448,811 B  sha256 d92b8ed9891f03d6da47735a104dc02bae38f0096e69abe54f3818d47f339e0f  (epoch 208)
runs/a6_4000/last.pt        89,448,811 B  sha256 40de8e5f07145545bfd9e799f8a138919715a3c62cf44155f38cd28642843ca0  (epoch 248)
runs/a6_4000/history.json      174,810 B  sha256 0c5446018887ba21683fe2cb2cbb14f1f36de95e75178a66291cde2882e2b3a9
runs/a6_4000/validation.json    19,477 B  sha256 87bc4e314f834dd559945795c2d127f19d0c65c3d5854421f1c643052b3669d1
runs/a6_4000/execution.json      1,224 B  sha256 65dc0e65fc317793e700ce74b0a021d8d48fa9b7c381210f4cc2646c12424db9
```

本目录内的可提交证据：

- `server_run_metrics.json`：本文所有数值的机器可读版本（进度抽样、最优指标、逐场景、来源哈希）；
- `server_execution.json`：训练时写入的 provenance（config、seed、torch、设备、数据哈希）。

checkpoint 本身不入 Git（体积与数据策略），以路径 + SHA256 引用。

## 8. 复现

同一 tar 包 + 同数据 + 同 config/seed 可复现；续训等价性由
[`resume_equivalence.json`](../2026-10-21-d5.2/resume_equivalence.json) 保证
（`passed=true, differences=[]`）。本次实测的连续性证据：第一段 epoch 40 的
val loss 0.014809 被第二段 epoch 41 直接接上，曲线无断点。

## 9. 后续

按 D5.2 计划，仍需在正式环境完成：

1. S1 完整 30,000 步（建议同时修 `eval_every` 与选模 epoch 对齐问题）；
2. S3 Achromatic 对照（同预算、同 seed）；
3. S2 A-large 与容量匹配 C6-SFAT；
4. PSF 失配与先验基线诊断；
5. 以上冻结后，再一次性评测冻结测试集。

本次运行的价值是把"预算不足"这一项从 D5 的高频失败解释中排除，并为后续对照提供
同配置、可复现的单卡基线。
