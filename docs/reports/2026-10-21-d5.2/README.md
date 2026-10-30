# D5.2 — 实验室 GPU 正式训练与空间-光谱机理诊断

日期：2026-10-21（2026-10-28 更新 S1/S3 结论）
分支：`codex/d5.2-server-training` / `codex/d5-hsfaut-reconstruction`
基线：D5.1 commit `ae17ff4`，D5.1 暂定数值配置 512/8/64。

## 当前状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| S0 服务器、数据、resume | 已完成 | [s0_environment.md](s0_environment.md), [resume_equivalence.json](resume_equivalence.json) |
| S1 A6 正式训练（Chromatic） | **已完成**（248 epoch，早停） | [2026-10-28 S1/S3 报告](../2026-10-28-d5.2-s1-s3/README.md) |
| S3 Achromatic 对照 | **已完成**（300 epoch） | 同上 |
| S2 容量/SFAT 对照 | 未开始 | 等 S1/S3 结论稳定 |
| S4 冻结测试 | 未开始 | 测试集未上传 |
| C-4 验收（测试服务器单卡） | 完成 24,800 步 | [2026-10-28 report](../2026-10-28-d5.2-server-run/README.md) |
| C-4 复测（测试服务器双卡并行） | 完成，同租约 S1+S3 | 同上（S1/S3 报告 §4） |

## S0 结论

- 实验室共享服务器：2 × RTX 4090 48 GiB 实际使用；GPU 3/5 分配。
- 服务器 Python 3.10.12、PyTorch 2.2.1+cu121、CUDA 可用；数据盘 4.5 TiB free。
- 只上传 train/val/manifest/kernels；`test.npz` 未上传。
- 修复 checkpoint 保存顺序，新增 `next_epoch`、`global_step`、`best_epoch`。
- 严格 resume 等价性测试：**passed=true, differences=[]**。
- 100-step smoke 正常：PSNR 28.44 dB、HF 0.9709、measurement rel L2 0.0333。

## amax 上的 S1/S3 旧运行（已停止，被取代）

2026-10-21 在 amax（RTX 4090）启动的两条运行没有跑完，2026-10-28 只读核对的结果是：

| run | 停在哪 | best | 状态 |
|---|---|---|---|
| `a6_formal_s1_gpu3` | epoch 109 / 10,900 步 | 0.013354 @107 | 自 10-21 21:35 无进展 |
| `a6_achromatic_s1_gpu5` | epoch 99 / 9,900 步 | 0.013934 @96 | 自 10-21 21:31 无进展 |

10-21 当晚 21:53–23:26 有 6 次重启尝试，日志最多打印到 `resumed 109` 就结束，**没有任何
traceback**；容器内外都没有可读的 OOM/kill 记录（`dmesg`/`journalctl` 拒绝非 root）。
因此"为什么停"**未确认**，只能确认它们不是在跑。宿主此后在 10-22 与 10-26 各重启一次。
checkpoint、history 与日志全部保留在 `/data1/udc-d5.2/`，未删除。

这两条中途中止的运行不作为任何结论的证据；S1/S3 的正式结论来自 kunqian 上的完整匹配对
（见上表）。同一份配置在 3090 上复现了 amax 曲线的形状（10k 步附近 PSNR 30.25/30.33、
HF 0.849/0.846），说明设备差异不是此前曲线不一致的原因。

## C-4 验收运行（2026-10-28，不同机器）

在实验室资源系统的验收测试服务器（`kunqian`，RTX 3090）上，用同一份干净提交
`69bd221`、同一份数据与配置，完成了一次单卡真实训练：**248 epoch / 24,800 步**，
`patience=40` 早停，累计 GPU 计算 93.6 分钟。

- 最优验证损失 0.012924（epoch 208）；最优 checkpoint 验证集 PSNR 30.725 dB、
  SSIM 0.9400、SAM 9.899°、HF relative L2 0.7619。
- 主要结论：充分训练显著缓解了高频恢复不足（4,000 步 0.942 → 24,800 步 0.762），
  但剩余误差由哪些因素主导尚未确定；用 4,000 步短跑下"高频无法恢复"的结论会误判。
- 该运行是按既定早停规则结束的基线，不需要为凑满 30,000 步而续训。

详见 [2026-10-28 报告](../2026-10-28-d5.2-server-run/README.md) 与
[机器可读指标](../2026-10-28-d5.2-server-run/server_run_metrics.json)。

## S1/S3 匹配对照结论（2026-10-28，kunqian 双卡）

同一租约内两臂并行，同数据、同配置、同 seed，只换 PSF：

- S1 Chromatic：248 epoch / 24,800 步（patience 早停），best 0.012924 @208。
- S3 Achromatic：300 epoch / 30,000 步（未早停），best 0.013399 @296。
- 色散臂 PSNR **+0.42 dB**、高频相对误差 **−0.063**（**9/9** 场景一致）、空间梯度
  **−0.065**、测量残差 **−0.0075**；SAM 未改善（9.73° vs 9.90°，方向对消色差略有利，
  在平台期波动内）。
- 优势在约 100 epoch 后才稳定出现；≤40 epoch 两臂无差别，因此不是"预算不对称"的假象
  （240 epoch 公共点上差距与 best-vs-best 一致）。

详见 [S1/S3 报告](../2026-10-28-d5.2-s1-s3/README.md)。

## 边界

- 512/8/64 signed shape-only 是暂定数值模型，不是绝对辐射/Bayer 标定。
- 未执行 G4、分类、压缩或联合优化。
- 未修改 D5/D5.1 历史报告、`optic_system` 或 Meta pin。
- 单 seed；平台期波动是 epoch 间变异，不等于种子方差。
- 服务器共享；amax 的实际吞吐曾受其他任务影响。
