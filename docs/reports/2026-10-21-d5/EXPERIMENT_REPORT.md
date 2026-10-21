# D5 实验报告：HSFAU 风格通用空间—光谱重建

记录日期：2026-10-21  
分支：`codex/d5-hsfaut-reconstruction`  
报告覆盖：G0 环境/代码审计、G1 数据与物理验收、G2 模型与短 pilot  
当前结论：**G0/G1 通过数值验收；G2 已执行但未通过质量验收；G3/G4 未启动。**

## 1. 结论摘要

本轮完成了 D5 的可复现实验骨架：

- 本地 GPU 环境、D4 代码复用和官方数据访问已完成审计。
- 冻结了 62 个独立场景：39 train / 9 validation / 14 test，统一为 21×256×256。
- 21 波段实测 PSF 的 forward/adjoint 通过 16 项数值检查，回归测试 23 passed。
- 发现并修复了滤波分裂算子的 Lipschitz 上界错误（441 → 21），该错误曾使 B/C 的数据一致性步长缩小 21 倍。
- 完成 A/B/C 三个模型、单样本过拟合、1000 步 pilot、checkpoint 恢复和显存/吞吐基准。

但高质量通用重建尚未成立：

- 验证集上 A6/B6 为 PSNR 28.24 dB、SAM 12.32°、cube relative L2 0.330；
- C6（SFAT）为 27.96 dB、SAM 14.24°、0.342，**未优于** CNN 先验；
- 高频相对误差仍约 0.974，空间梯度误差约 0.84，纹理/细节没有得到有效恢复；
- 单样本拟合损失下降但地板约 0.033–0.039，不能称为紧拟合。

更重要的物理限制是：8×8 空间积分使 21 个波段核的平均相关从 0.521 升到 0.921，接近无色核的 0.961。当前 8:1 仿真尺度可能在进入网络前就削弱了波长编码。因此，即使 D5 后续得到更好的重建分数，也不能直接解释为强波长编码增益。

## 2. 范围与停止边界

本阶段只研究一般空间—光谱重建。

已执行：
- 离线数据读取、PSF 分析、数值算子测试、短时 GPU 训练与验证集评估。

明确未执行：
- 未加载 `test.npz` 做评估；
- 未做 Face/Non-face 分类、RGB 人脸检测、低秩压缩、联合优化；
- 未修改 `optic_system`；
- 未更新 Meta Submodule pin；
- 未使用共享服务器或部署远端训练。

测试集在训练前冻结。G2 未通过前，不启动 G3 正式训练。

## 3. 代码、环境与可追溯性

### 3.1 本地环境

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA RTX 3060 Ti，8 GiB |
| 驱动 | 630.20 |
| PyTorch | 2.7.1+cu128 |
| CUDA | 12.8，可用 |
| BF16 | 支持（本轮未启用 AMP） |
| Python | 3.13.5 |
| CPU / RAM | i7-12700K，12 核 / 20 线程；31.7 GiB |
| 本地磁盘 | D: 约 675.8 GiB 可用 |

未验证共享 GPU 分配；本轮全部为本地单卡运行。

### 3.2 代码复用

| 文件 | 本轮用途 |
|---|---|
| `operators/measured_psf.py` | 21 波段 NumPy forward/adjoint 参考 |
| `unrolling/spatial_pgd.py` | FFT 卷积/伴随、Lipschitz 上界 |
| `unrolling/full_spectral.py` | 已有 `SinglePlanePhysics` 与 D4 基线 |
| `unrolling/general_pgd.py` | Model A 对照 |
| `unrolling/filter_split.py` | 新增逐波段卷积与滤波分裂 |
| `unrolling/hsfaut.py` | 新增 HSFAU 风格展开与重建损失 |
| `priors/sfat.py` | 新增空间—频率聚合先验 |
| `experiments/d5/*` | 数据准备、预检、训练、基准、汇总 |

### 3.3 提交与测试

| 提交 | 内容 |
|---|---|
| `94577a2` | D5 G0/G1 数据协议与 forward 验证 |
| `9bee8b1` | preflight provenance hash |
| `4b6be21` | 在已提交代码状态下记录 G1 验证 |
| `ebae592` | G2 模型、pilot 与验证证据 |

发布前回归：`python -X utf8 -m pytest -q -p no:cacheprovider` → **23 passed**。

## 4. 数据协议

### 4.1 数据集与划分

| 来源 | 原始内容 | 本轮用途 |
|---|---|---|
| CAVE 本地 31 场景 | `21×512×512 uint16`，直接 450–650 nm / 10 nm | 23 train / 5 val / 3 test |
| CAVE `watercolors` | 31 波段，8-bit 灰度-in-RGB，alpha=255 | 1 个独立 8-bit test |
| KAIST 官方 non-selected | 20 个 `2704×3376×28 float64` | 16 train / 4 val |
| KAIST 官方 selected | 10 个 `586×586×28 float64` | 10 test |

最终 prepared 集：

| Split | CAVE | KAIST | 合计 |
|---|---:|---:|---:|
| train | 23 | 16 | 39 |
| validation | 5 | 4 | 9 |
| test | 4 | 10 | 14 |

D4 的 6 个 development-test 场景在 D5 中进入训练，D4 结果不作为 D5 测试；D5 的 CAVE test 是 3 个从未训练的 reserve 场景加 `watercolors`。测试 ID 在训练前写入 `data_split_manifest.json`。

### 4.2 官方数据验证

实际下载并校验了全部 30 个 KAIST 官方文件。单文件抽样：

| 样本 | SHA256 | 变量 | Shape | Dtype |
|---|---|---|---|---|
| `cave_1024_28_scene1.mat` | `8EA19395871017643EA42068236CF4DD1140E84109E4D45A41E71D3DB18C7CAD` | `img_expand` | 1024×1024×28 | uint16 |
| `kaist_selected_scene1.mat` | `3E3EDDD69101D24EF66EDE0A76DB6415A78EDCB82FA36AB8367C8B49A4CEFC1D` | `HSI_crop_586` | 586×586×28 | float64 |

处理约定：

- KAIST 非方场景先做中心方形裁剪，再缩放到 256×256；
- KAIST selected 的少量负值裁剪到 0，并记录裁剪前负值比例；
- 28 个谱样本没有随文件提供波长向量，D5 声明假设为 450–650 nm 均匀分布，再线性插值到 21 个 10 nm 通道；
- CAVE 直接选择 450–650 nm 的 21 个实际通道，不做插值；
- CAVE 使用 `/65535`，KAIST 使用原始相对强度，均不做逐场景最大值归一化；
- `watercolors` 使用 `/255`，作为 8-bit 量化压力测试单独披露；
- 官方 `cave_1024_28` 的 205 个文件没有原始场景映射，公开参考实现把它们当作独立输入，存在场景组泄漏风险，因此未作为 D5 主训练集。

### 4.3 数据哈希

canonical prepared 目录（`prepared02`）记录：

| 文件 | SHA256 |
|---|---|
| `manifest.json` | `78ecf24fdaa6627b810aa74297b27aaac50d2b0dc24b4efd7f6fa1bfaa9cbdf1` |
| `physics.json` | `24f6191221aab5fa909d79d6300fd931f789465041323f013f99327e9970e2eb` |
| `kernels.npz` | `f07d9bbb7a4ff870433e8e72dcd19298d642e639c499afe1057a97eeb9206d4b` |
| `train.npz` | `92aa29658b79ac21ce3cd4ef05b29c6db2acfe25e010da52a3c4b518bd415470` |
| `val.npz` | `2d331758c32361aef232c25e693e18a543608dba908b9e91a714f7b87f35e11f` |

`prepared01` 是同一数据的早期目录；在 audit schema 重构后以 `prepared02` 为 canonical。两者不混用。

## 5. G1 物理算子与 PSF 验收

### 5.1 Mode S：signed shape-only

主模式沿用：

1. 读取 `roi1024_{nm}.npz` 与元数据；
2. 减去 matched pre/post dark；
3. 外圈 40 像素估计并扣除背景；
4. 取中心 512×512 sensor ROI；
5. 按 signed sum 归一化；
6. 8×8 积分到 64×64；
7. 单位波长权重、零边界 full convolution、整数中心裁剪。

审计范围内，21 个核的负值比例为 **0.745–0.874**，积分后每个核之和为 1。该模型是 shape-only 数值近似，不是绝对辐射、Bayer RAW 或非负光学 PSF。

### 5.2 Mode P：非负投影敏感性

对每个积分核裁剪负值后重新归一化：

- 相对 L2 变化：0.0010–0.0176；
- 被裁掉的绝对能量比例：0.1%–3.4%；
- 裁剪后负值比例仍为 0.26–0.75。

Mode P 仅作敏感性分析，不进入主训练或主指标。

### 5.3 关键发现：积分削弱波段可分性

| 比较 | 平均非对角相关 |
|---|---:|
| signed 512 支撑核 | **0.521** |
| 8×8 积分后的 64×64 核 | **0.921** |
| 无色平均核（512） | 0.735 |
| 无色平均核（积分后） | 0.961 |

512 支撑与 1024 支撑中心 512 的归一化差异约 `5.8e-8`，说明问题主要不是裁剪原点，而是 8:1 积分/尺度本身。99% 绝对能量半径约 32 个仿真像素（256 个 sensor 像素），因此 256×256 场景的边缘 32 像素存在 halo/上下文损失。

这直接限制了 D5 的科学解释：在当前算子下取得的高分不能单独证明模型利用了波长编码。

### 5.4 forward/adjoint 数值验收

16 项检查全部通过，包括：

- 单波段脉冲与参考卷积一致；
- 21 波段线性叠加；
- CPU/GPU forward 与 adjoint 一致；
- 前向/伴随内积一致；
- FP64 方向导数与 autograd 一致；
- 非对称偶数核；
- 四个边界脉冲；
- batch 一致性；
- 实测 64×64 核与 SciPy 参考一致。

实测核误差：

| 检查 | 相对误差 |
|---|---:|
| measured-kernel adjoint | 4.37e-7 |
| FFT vs SciPy | 1.28e-7 |
| CPU vs GPU | 1.19e-7 |

### 5.5 算子资源基准

64×64 核，forward+adjoint，batch 2：

| 空间尺寸 | 时间 | 峰值显存 |
|---|---:|---:|
| 128 | 0.66 ms | 53 MiB |
| 256 | 2.46 ms | 213 MiB |
| 512 | 10.55 ms | 848 MiB |

这是算子级基准，不是端到端模型吞吐。

## 6. G2 模型、过拟合与短 pilot

### 6.1 模型

| 模型 | 结构 | 参数量（6-stage pilot） |
|---|---|---:|
| A | D4 `GeneralSpectralPGD`，共享 CNN 先验 | 98,139 |
| B | `FilterSplitPhysics` + 共享 CNN 先验 | 98,139 |
| C | `FilterSplitPhysics` + 3 层 SFAT | 60,699 |

统一损失：

`Charbonnier + 0.2 × spatial-gradient + 0.05 × spectral-angle + 0.02 × data-consistency`

N1 主输出为 `[B,21,H,W]`，并导出重投影、全局谱、残差和每阶段残差。训练不读取 Face 标签、分类分数或测试集。

### 6.2 实现错误与修复

首版 `FilterSplitPhysics.lipschitz_bound` 把已经按波段求和的频谱再次按波段求和，得到 441 而不是 21。结果是 B/C 的数据一致性步长被缩小 21 倍，表现为测量残差显著差于 A。

修复为：

`L = max sum_b w_b^2 |K_b|^2`

修复后 B6 与 A6 的验证指标在数值精度内完全一致，说明滤波分裂在单位权重下正确退化为已验证的 A/At。早期 buggy B/C 结果不作证据。

### 6.3 单样本过拟合

固定 `balloons` 全场景，无增强，重建损失：

| 模型 | 步数 | 初始损失 | 最终损失 | checkpoint 重载最大差 |
|---|---:|---:|---:|---:|
| C3 | 1000 | 0.12578 | 0.03518 | 0.0 |
| C5 | 600 | 0.11538 | 0.03908 | 0.0 |
| A6 | 600 | 0.04824 | 0.03349 | 0.0 |

梯度、阶段参数和 checkpoint 重载均正常；但损失地板约 0.033–0.039，不能称为紧拟合。这反映单平面 21→1 问题的病态性和先验容量限制，而不是继续增加 epoch 的理由。

### 6.4 1000 步 pilot

预算：10 epochs × 100 steps，batch 1，256×256，5% 训练测量噪声，所有波段共享几何增强。仅用 validation loss 选 checkpoint。

整体 9 个验证场景：

| Run | PSNR dB | SSIM | SAM° | Cube rel L2 | Global rel L2 | HF rel L2 | Grad rel L2 | Meas rel L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | **28.24** | **0.868** | **12.32** | **0.330** | **0.105** | 0.9741 | 0.8335 | 0.0484 |
| B6 | **28.24** | **0.868** | **12.32** | **0.330** | **0.105** | 0.9741 | 0.8335 | 0.0484 |
| C6 | 27.96 | 0.863 | 14.24 | 0.342 | 0.107 | 0.9748 | 0.8387 | 0.0450 |

分数据源：

| Run / 来源 | PSNR dB | SSIM | SAM° | Cube rel L2 | HF rel L2 |
|---|---:|---:|---:|---:|---:|
| A6 CAVE | 28.83 | 0.887 | 11.93 | 0.341 | 0.9732 |
| A6 KAIST | 27.51 | 0.843 | 12.81 | 0.317 | 0.9752 |
| C6 CAVE | 28.45 | 0.882 | 14.21 | 0.357 | 0.9741 |
| C6 KAIST | 27.34 | 0.840 | 14.28 | 0.324 | 0.9758 |

A6 与 B6 的完全一致验证了算子实现，但也说明：在单位滤波权重下，B 不是一个独立于 A 的算法；它只是同一物理算子的显式分裂写法。C6 是当前唯一真正不同的先验，但它没有带来验证收益。

### 6.5 资源与恢复

| 模型 | 尺寸 | Batch | 单步时间 | 峰值显存 |
|---|---:|---:|---:|---:|
| A6 | 256 | 1 / 2 | 64.6 / 118.9 ms | 1.10 / 2.14 GiB |
| B6 | 256 | 1 / 2 | 75.2 / 152.4 ms | 1.23 / 2.39 GiB |
| C6 | 256 | 1 / 2 | 93.2 / 165.6 ms | 1.22 / 2.34 GiB |
| C6 | 512 | 1 | 302.4 ms | 4.75 GiB |

断点恢复测试：先训练 3 epochs，再从 `last.pt` 恢复到 5 epochs；第 4–5 epoch、scheduler、最佳验证指标和采样状态均正确延续，未重置。

## 7. 验收状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| G0 环境/复用 | 通过 | 本地 GPU、PyTorch、磁盘、D4 代码表已核实 |
| G1 数据协议 | 通过（有声明假设） | 62 场景冻结；KAIST 28→21 为假设；官方 CAVE-1024 未用 |
| G1 forward/adjoint | 通过 | 16 项检查；23 tests |
| G1 PSF 物理解释 | 未通过“强波长编码” | 8×8 积分显著提高波段相关 |
| G2 机制/梯度/resume | 通过 | 全状态、梯度、checkpoint 恢复 |
| G2 单样本拟合 | 部分通过 | 损失下降但不紧 |
| G2 pilot 质量 | 未通过 | HF 误差约 0.974；SFAT 无增益 |
| G3 正式训练 | 未启动 | 被 G2 质量门阻塞 |
| G4 独立测试 | 未启动 | test 仍冻结 |

## 8. 失败原因与限制

1. **积分尺度可能先验地削弱波长信息。** 8×8 积分后波段核相关 0.921，接近无色 0.961；这比网络结构更基础。
2. **21→1 单平面问题是病态的。** 测量残差可以很低，但 truth cube 仍可能误差很大；数据一致性不等于可辨识性。
3. **当前先验容量/结构不足。** 3/5/6 阶段与 60k–98k 参数在 1000 步内无法恢复高频纹理；SFAT 版本反而略差。
4. **B 在单位滤波下等价于 A。** 真正的 HSFAU 滤波分裂需要非均匀空间—光谱响应；本轮没有标定响应，不能假装 B 是独立算法。
5. **KAIST 波段映射是假设。** 28 个样本没有随文件提供波长向量；均匀 450–650 nm 假设可能引入系统误差。
6. **watercolors 是 8-bit。** 它只能作为量化压力测试，不能与 16-bit CAVE 混合成单一“高精度”结论。
7. **无绝对辐射/Bayer。** 所有结果仍是 shape-only 标量测量仿真，不是 RAW8 Bayer 相机模型。

## 9. 建议的最小下一步

在启动 G3 前，只做两个受控实验：

1. **匹配容量比较。** 将 C 的 SFAT 宽度提高到与 A/B 约 98k 参数，保持 6 阶段、同数据、同预算，只在 validation 上比较 A/B/C。若 SFAT 仍无增益，停止该先验路线。
2. **积分尺度/可辨识性审计。** 在相同物理校正下比较 8×8、4×4、2×2 积分的波段相关、有效秩和重建上界；确认 8:1 是否应保留。若积分本身抹掉波长差异，应先改物理/尺度，而不是继续加网络。

只有上述两项给出正证据后，才冻结 G3 配置并启动正式训练；测试集继续保持冻结。分类、压缩、联合优化仍不进入 D5。

## 10. 复现入口

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path

python -X utf8 experiments/d5/prepare.py --cave <cave> --watercolors <zip> `
  --kaist-train <official-nonselected> --kaist-test <official-selected> `
  --psf <spectral21-root> --output <new-prepared> --size 256 --include-1024

python -X utf8 experiments/d5/preflight.py --prepared <prepared> `
  --output <new-forward-validation.json> --sizes 128,256,512 --batches 1,2

python -X utf8 experiments/d5/psf_modes.py --psf <spectral21-root> `
  --output <new-psf-comparison.json>

python -X utf8 experiments/d5/train.py --prepared <prepared> `
  --output <new-run> --config experiments/d5/config_pilot_c6_long.json

python -X utf8 -m pytest -q -p no:cacheprovider
```

证据文件：

- `forward_validation.json`
- `psf_mode_comparison.json`
- `psf_policy.md`
- `data_split_manifest.json`
- `prepared_checks.json`
- `g2_pilot_summary.json`
- `overfit_summary.json`
- `resume_summary.json`
- `model_benchmark.json`

本报告只包含聚合指标、哈希和研究结论；原始数据、模型权重、私有路径和机器配置不进入 Git。
