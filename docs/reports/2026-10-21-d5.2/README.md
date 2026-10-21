# D5.2 — 实验室 GPU 正式训练与空间-光谱机理诊断

日期：2026-10-21  
分支：`codex/d5.2-server-training`  
基线：D5.1 commit `ae17ff4`，D5.1 暂定数值配置 512/8/64。

## 当前状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| S0 服务器、数据、resume | 已完成 | [s0_environment.md](s0_environment.md), [resume_equivalence.json](resume_equivalence.json) |
| S1 A6 正式训练 | 运行中 | 服务器 run `a6_formal_s1_gpu3` |
| S3 Achromatic 对照 | 运行中 | 服务器 run `a6_achromatic_s1_gpu5` |
| S2 容量/SFAT 对照 | 未开始 | 等 A6 基线 |
| S4 冻结测试 | 未开始 | 测试集未上传 |

## S0 结论

- 实验室共享服务器：2 × RTX 4090 48 GiB 实际使用；GPU 3/5 分配。
- 服务器 Python 3.10.12、PyTorch 2.2.1+cu121、CUDA 可用；数据盘 4.5 TiB free。
- 只上传 train/val/manifest/kernels；`test.npz` 未上传。
- 修复 checkpoint 保存顺序，新增 `next_epoch`、`global_step`、`best_epoch`。
- 严格 resume 等价性测试：**passed=true, differences=[]**。
- 100-step smoke 正常：PSNR 28.44 dB、HF 0.9709、measurement rel L2 0.0333。

## 正在执行

- `a6_formal_s1_gpu3`：Measured chromatic PSF，A6，30,000 steps 上限。
- `a6_achromatic_s1_gpu5`：21 波段同一平均核，A6 从头训练，相同数据/预算/seed。

两者均只使用 train/validation；validation loss 每 epoch 记录，昂贵指标每 5 epochs 记录。正式结论要等训练完成，不以当前中间曲线代替。

## 边界

- 512/8/64 signed shape-only 是暂定数值模型，不是绝对辐射/Bayer 标定。
- 未执行 G4、分类、压缩或联合优化。
- 未修改 D5/D5.1 历史报告、`optic_system` 或 Meta pin。
- 服务器共享且 CPU 负载较高；实际吞吐受其他任务影响。
