# D5.3 S0 — 环境、代码与证据审计

日期：2026-10-29（UTC，对应本地 2026-10-30 凌晨）
分支：`codex/d5.3-chromatic-mechanism`，起点 `30cd8d4`；训练代码 `1203ac9`，分析与报告代码 `351dff1`
设备：`kunqian` 4 × RTX 3090 24 GiB，租约 `3d3e5397eb17409c9a81b1499c66c637`（4 GPU / 32 CPU / 48 GiB / 8 h）

## 1. 代码复用与改动

只复用 D5 训练入口与 D5.2 诊断脚本，不新建训练框架。

| 文件 | 状态 | 说明 |
|---|---|---|
| `experiments/d5/train.py` | 改动 | 新增 `--seed`（默认仍取 config，记录 `seed_source`）；训练开始后把 `measured_forward_rms`、`train_noise_std_absolute`、`cosine_epochs`、`steps_per_epoch` 追加写入 `execution.json` |
| `experiments/d5_2/frequency_diagnostics.py` | 改动 | 接入 `resolve_psf_policy`；此前是全仓库最后一个仍会静默覆盖 `kernel_fft` 的脚本 |
| `experiments/d5_3/*` | 新增 | 反事实构造、算子级可观测性、冻结网络响应、OTF 对照、多种子汇总、控制核生成、启动脚本 |
| `tests/test_d5_3_*` | 新增 | 27 + 5 个测试（见 §4） |

**数值不变性证据**：用 `git show 30cd8d4:experiments/d5/train.py` 与改后版本在同一 tiny 数据集上各跑 3 epoch（CPU），`history.json` 的 SHA256 完全相同：

```text
history_identical = true
history_sha256    = e06720c57c66acbdfe6e25d5e32ee32c176e2ce57b8d2f51b8d35494f707237a
```

`resume_equivalence.py` 仍为 `passed=true, differences=[]`。

## 2. 训练噪声的自参照问题（实测）

`train.py` 的噪声标准差是 **该臂自身** forward 在前 8 个训练场景上的 RMS × 0.005，同一标量还用于 `data` 项归一化。本轮把实测值写进了 `execution.json`：

| 臂 | measured_forward_rms | 绝对噪声 std | 相对差 |
|---|---|---|---|
| Chromatic (seed 20261021) | 3.4658713340759277 | 0.017329356670379638 | — |
| Achromatic (seed 20261021) | 3.46728515625 | 0.01733642578125 | **+0.04%** |

结论：两臂的绝对噪声与损失归一化只差 0.04%，混杂可忽略；但**严格意义上验证损失仍不能跨臂直接比较**，报告只用固定指标（PSNR/SSIM/SAM/HF/…）做配对检验。

## 3. 数据、核与旧 checkpoint 复核

| 项 | 实测 SHA256 | 与 D5.2 记录 |
|---|---|---|
| `train.npz` | `024d2b4c8956707387289242ddbadce08e3379a51ed6e0752a167666dbb32908` | 一致 |
| `val.npz` | `e61ac4f102d58a44bdb8f7174a960d13f2e102c5db00cc0bc408ea5583ff9247` | 一致 |
| `manifest.json` | `cfe89c34868b7eece4c6089d238d79fe224cd62c54db23df0a81ba7e40ef02f3` | 一致 |
| `s1_chromatic_20261028/best.pt` | `a2153a7e47b52e9f88ee2b08f025b9fbad83ba560ae157c60c180440a0e58e1d` | 一致 |
| `s1_chromatic_20261028/last.pt` | `d190ed4a629b9e93257a6254ed6feda722cbb4696426e1debd07ce62123ce1c8` | 一致 |
| `s3_achromatic_20261028/best.pt` | `3b11db24bd9f53327a14d149765c7ec06c31d0c048442c9848e3126944873868` | 一致 |
| `s3_achromatic_20261028/last.pt` | `52c9cb759c0e48995c539f03313477e51e438a0d22f8e5f0a7055e756857c46c` | 一致 |

核的**数组摘要**（shape + float64 C-order bytes）：

| 条件 | 数组摘要 | 说明 |
|---|---|---|
| Chromatic | `070346a50f89735d4b17790221aeecbae41e25bbc1fad7df55deeb6da6eecf04` | 实测 21 波段 |
| Achromatic（平均核） | `eadb74005a2ba12237bbe814d8deeb732b4a78ed5dc286f2e456e7f7528d33d0` | 与 D5.2 一致 |
| Achromatic-550 | `87073bc41592e69fb7c1f6a295373494d56c8ab11bac607d7ac81c60a205375a` | 550 nm 实测核复制 21 份，波段和 1.0 |

550 nm 核由仓库内 `experiments/d5_3/make_achromatic_control.py --kind band --band-nm 550` 生成，生成报告落盘为 `kernels_achromatic_550.json`。

**PSF 来源实测（负向）**：用消色差核 resume 色散 checkpoint，训练在构建模型后立即中止：

```text
PSFProvenanceError: requested kernel file does not match the checkpoint operator;
refusing to substitute silently. Requested {L=838b57eb1f57,
physics.operator.kernel_fft=2101f7565fce} vs checkpoint {L=838b57eb1f57,
physics.operator.kernel_fft=176c6433d973}. ...
Recorded training kernel: kernels.npz (array 070346a50f89).
```

这正是本轮 6 个 run 共用同一个 workspace 时防止两个臂串台的关键保险。

## 4. 测试

本地（`d4_general_20261020` 环境，同一提交 `1203ac9`）：

```text
68 passed, 1 skipped        # D5.3 新增 27 项 + 控制核 5 项，其余为既有测试
```

容器内**未运行 pytest**（镜像没有 pytest，且未安装）：容器内只做真 GPU 训练与算子级检查，不把本地测试记作容器内通过。

## 5. 与计划不同的两处环境事实

1. **客户端文件传输被策略关闭**。服务端已升级到 `0.2.0`，`status` 返回
   `capabilities: {file_transfer: false, vscode: false, development_profile: false}`；
   `push`/`pull` 均返回 `{"code":"forbidden","message":"file transfer is disabled by policy"}`。
   - 应对：代码通过**公开仓库的 commit tarball**（`codeload.github.com/.../tar.gz/1203ac9`）
     用 `curl` 在容器内取回；产物通过 `job logs` 读取（训练日志本身每个 epoch 一行紧凑
     JSON，是完整的指标载体）。未使用任何提权或绕过手段。
   - 影响：D5.2 验证过的"释放后 pull"路径本轮无法复核。
2. **`git clone` 在容器内挂起**（`.git` 已建、工作树为空），`curl` 正常。改用 tarball。

## 6. 单步耗时与容量

4 卡并发、每进程 4 线程 OMP、32 核配额：

| 观测 | 值 |
|---|---|
| 预热后每 epoch | ≈ 30 s（0.30 s/step，含 best-epoch 指标复算） |
| D5.2 双卡并发基线 | 22.7 s/epoch（0.227 s/step） |
| 400 epoch 预计 | ≈ 3.3 h/run，两波 ≈ 6.7 h |
| 显存 | 5.1 GiB/进程，四卡独立物理卡（PCI 18/3B/86/AF） |

4 卡并发比 D5.2 双卡慢约 32%，因此把 6 个 S1 run 分成两波，并在第二波期间完成 S2 冻结网络推理。

## 7. 会话后段的客户端故障

租约内最后约 30 分钟，`job start` 开始连续失败：

```text
$ citespec-resource job start <id> --name secs3 -- grep -h seconds /workspace/.citespec-logs/s3a21.log
broker_unavailable: Unix domain sockets are not supported on this platform   (exit 6)
```

同一时刻 `status` 仍然正常（返回租约、GPU、内存、workspace）。客户端目录显示本会话中发生了自动更新：

| 版本 | 时间 |
|---|---|
| 0.1.1 | 2026-10-28 07:30（会话开始时在用） |
| 0.2.0 | 2026-10-29 22:00 |
| 0.2.1 | 2026-10-29 22:13 |
| 0.2.2 | 2026-10-29 22:47（当前 `current.json` 指向） |

**影响**：报告的 6 个 run、S2 的 810 个响应用例、675 个可观测性用例、OTF 与多种子汇总都在
故障前完成并取回；但 S3 的 550 nm 训练无法在本租约内启动，S4 也因此未进入。

**未做的尝试**：没有改用直连 SSH 或其它旁路继续跑训练——那会绕过作业管理路径，超出本轮授权。
问题按"说明现象、给出命令与完整错误、停止"的方式处理。
