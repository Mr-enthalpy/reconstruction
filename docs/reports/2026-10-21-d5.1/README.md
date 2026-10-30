# D5.1 — 有限 PSF 支撑与低积分倍率审计

日期：2026-10-21  
分支：`codex/d5.1-psf-support`  
基线：D5 `codex/d5-hsfaut-reconstruction` @ `7962126`

## 结论

D5.1 没有找到采用更低积分倍率的证据：

- 1× 原生 PSF 受 CFA 相位支配，相位差异远大于同波长重复性，**拒绝作为正式物理基准**。
- 2× 在数学上消除 CFA 相位，但高频重复性差（450 nm HF 0.265），且 1000 步 A6 pilot 没有重建增益。
- S1/S2/S3 的前向截断误差 6.2%–7.9%，高于 baseline 512/8 的 3.2%；验证指标也更差。
- baseline 512/8/64 在 512 场景、相同 A6/预算下仍最好：PSNR 29.12 dB、SAM 11.45°、HF 0.9695。
- **建议暂不冻结 PSF 策略，不启动 G3**；baseline 仅作为继续离线实验的数值对照。

完整决定：[psf_policy_decision.md](psf_policy_decision.md)。

## 修正的 D5 统计语义

原始 PSF 和 D5 报告未修改。D5.1 新增修正：

1. 区分投影前/后的负像素比例；投影后为 0，并单独记录 shape renormalization 后比例。
2. 负绝对质量比例在重新归一化前计算。
3. 512/1024 中央比较明确标记为 `processing consistency only`，不是截断无损证据。
4. signed 绝对半径明确为 `physical_energy_enclosure=false` 的描述性指标。
5. 回归测试覆盖以上语义、2×2 CFA 相位不变性和固定中心裁剪。

回归：`29 passed`。

## 支撑/积分扫描

| ID | 支撑 | 积分 | 核 | 平均绝对质量 | 平均 signed 质量 | 平均波段相关 | 前向误差（shape） |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 512 | 8 | 64 | 0.883 | 0.963 | 0.921 | 3.24% |
| S1 | 256 | 4 | 64 | 0.813 | 0.904 | 0.911 | 6.19% |
| S2 | 192 | 2 | 96 | 0.777 | 0.866 | 0.904 | 7.92% |
| S3 | 256 | 2 | 128 | 0.813 | 0.904 | 0.906 | 6.24% |
| S4 | 192 | 1 | 192 | 0.777 | 0.866 | 0.505 | 7.94% |
| S5 | 256 | 1 | 256 | 0.813 | 0.904 | 0.512 | 6.26% |

细节：[psf_support_sweep.json](psf_support_sweep.json)。

1× 的低相关不是“更多可重复光谱信息”，而是 CFA 相位。2×/4×/8× 的相关差异很小；8× 反而有更好的高频重复性。

## 重复性

同波长 2× 总形状重复性为 0.63%–1.65%，但高频重复性为 6.5%–26.5%；8× 的高频重复性为 0.8%–1.8%。1× 的 CFA 相位相对 2× 块平均为 0.18–1.16，远大于重复性。

只有 450/550/650 有独立重复；其他 18 波长只有一次扫描。详见 [psf_repeatability.md](psf_repeatability.md)。

## 前向截断与尺度

- 真实有限核前向误差：baseline 3.24%，S1 6.19%，S2 7.92%，S3 6.24%；amplitude-preserving 模式单独报告。
- 场景先高分辨率模拟后降采样 vs 先缩放场景/核：factor 2 总误差 0.90%、HF 6.25%；factor 4 为 2.59%、23.8%。
- 因此 pilot 统一使用 512×512 场景，baseline 也在 512 重跑。

文件：[forward_truncation.json](forward_truncation.json)、[scale_consistency.md](scale_consistency.md)。

## A6 pilot

1000 步、6 阶段、batch 1、相同 seed/损失/噪声，仅 validation 选模。测试集未加载。

| Run | PSNR | SSIM | SAM° | Cube rel L2 | HF rel L2 | 局部 SAM° |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **29.12** | **0.906** | **11.45** | **0.297** | **0.9695** | **10.67** |
| S1 | 27.99 | 0.871 | 12.09 | 0.337 | 0.9944 | — |
| S2 | 26.60 | 0.832 | 13.05 | 0.395 | 0.9970 | 14.62 |
| S3 | 26.52 | 0.830 | 13.00 | 0.399 | 0.9970 | — |

详见 [pilot_comparison.md](pilot_comparison.md) 与 [pilot_summary.json](pilot_summary.json)。

## 计算成本

512×512、batch 1、A6 单步约 227–229 ms，峰值 CUDA 约 4.33 GiB。所有候选均可运行；未因显存问题删波段或裁 PSF。详见 [operator_benchmark.json](operator_benchmark.json)。

## 证据文件

- `psf_support_sweep.json`
- `psf_repeatability.json` / `psf_repeatability.md`
- `forward_truncation.json`
- `scale_consistency.json` / `scale_consistency.md`
- `operator_benchmark.json`
- `pilot_summary.json` / `pilot_comparison.md`
- `local_spectra_baseline.json` / `local_spectra_S2.json`
- `psf_policy_decision.md`

## 复现命令

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path

python -X utf8 experiments/d5_1/psf_sweep.py --psf <spectral21-root> `
  --output <new-sweep>

python -X utf8 experiments/d5_1/repeatability.py --psf <spectral21-root> `
  --output <new-repeatability.json>

python -X utf8 experiments/d5_1/truncation.py --psf <spectral21-root> `
  --cave <cave-npz> --output <new-forward-truncation.json>

python -X utf8 experiments/d5_1/scale_consistency.py --psf <spectral21-root> `
  --cave <cave-npz> --output <new-scale.json>

python -X utf8 experiments/d5/prepare.py --cave <cave> --watercolors <zip> `
  --kaist-train <official-nonselected> --kaist-test <official-selected> `
  --psf <spectral21-root> --output <prepared512> --size 512

python -X utf8 experiments/d5/train.py --prepared <prepared512> `
  --output <new-run> --config experiments/d5/config_pilot_a6_long.json `
  --kernel-file <sweep>/kernels/baseline_shape.npz

python -X utf8 -m pytest -q -p no:cacheprovider
```

## 边界

- 未修改硬件采集代码、原始 PSF、D5 历史报告、checkpoint 或 Meta pin。
- 未执行分类、压缩、联合优化或 G3 正式训练。
- 未使用冻结测试集选择 PSF 策略。
- 所有结论限于当前 21 波段、实测 PSF、shape-only 数值模型和 1000 步 A6 pilot。
