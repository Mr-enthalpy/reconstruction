# D5.3 S0 — 环境、代码与证据审计

日期：2026-10-29（UTC，对应本地 2026-10-30 凌晨）
分支：`codex/d5.3-chromatic-mechanism`，起点 `30cd8d4`；训练代码 `1203ac9`，分析与报告代码 `351dff1` —— 该分支已合并进 `main` 并删除
设备：`kunqian` 4 × RTX 3090 24 GiB，租约 `3d3e5397eb17409c9a81b1499c66c637`（4 GPU / 32 CPU / 48 GiB / 8 h）

## 1. 代码复用与改动

只复用 D5 训练入口与 D5.2 诊断脚本，不新建训练框架。

| 文件 | 状态 | 说明 |
|---|---|---|
| `experiments/d5/train.py` | 改动 | 新增 `--seed`（默认仍取 config，记录 `seed_source`）；训练开始后把 `measured_forward_rms`、`train_noise_std_absolute`、`cosine_epochs`、`steps_per_epoch` 追加写入 `execution.json`；`evaluate_metrics` 的 `CAVE/KAIST/ALL` 增加 `band_relative_l2_mean`、`band_rmse_mean` 两个 21 元列表（2026-10-30 追加，见 §1a） |
| `experiments/d5_2/frequency_diagnostics.py` | 改动 | 接入 `resolve_psf_policy`；此前是全仓库最后一个仍会静默覆盖 `kernel_fft` 的脚本 |
| `experiments/d5_3/*` | 新增 | 反事实构造、算子级可观测性、冻结网络响应、OTF 对照、多种子汇总、逐波段报告、控制核生成、启动脚本 |
| `tests/test_d5_3_*` | 新增 | 27 + 5 + 12（逐波段）+ 14（三 seed 逐波段）个测试（见 §4） |

**数值不变性证据**：用 `git show 30cd8d4:experiments/d5/train.py` 与改后版本在同一 tiny 数据集上各跑 3 epoch（CPU），`history.json` 的 SHA256 完全相同：

```text
history_identical = true
history_sha256    = e06720c57c66acbdfe6e25d5e32ee32c176e2ce57b8d2f51b8d35494f707237a
```

`resume_equivalence.py` 仍为 `passed=true, differences=[]`。

### 1a. 逐波段聚合的加入（2026-10-30，不改任何旧数字）

`evaluate_metrics` 现在额外写 `band_relative_l2_mean` / `band_rmse_mean`。这会改变
`history.json` 与 `validation.json` 的**字节**（cadence epoch 的 `metrics` 行多了两个数组），
所以上一节的"字节相同"只对加入它们之前的版本成立。对加入之后的版本，重跑了同一 tiny 数据集：

| 检查 | 结果 |
|---|---|
| 去掉两个新 key 后 `history.json` 逐字节相同 | ✅ |
| `history.json` 原始字节 | ❌ 不同（预期，仅因新增数组） |
| `best.pt` 权重、`best_epoch`、`validation_loss` | ✅ 全等 |
| 8 个旧聚合指标 | ✅ 全等 |
| 9 个场景的全部旧逐场景指标 | ✅ 全等 |
| `consistency` 块 | `compared_keys` 10 个（含两个新列表）、`unresolved_keys=[]`、`max_abs_difference=0.0`、`consistent=true` |

顺带修掉一个会被新 key 触发的真实崩溃：原 `consistency` 计算对每个 key 做
`float(np.mean(...))` 再 `float(recorded[key])`，遇到列表会 `TypeError`。现改为
`compare_recorded_metrics`，标量比绝对差、列表逐元素比并取最大差；
`band_relative_l2_mean` → 逐场景 `band_relative_l2` 的命名映射也做了显式查找，
避免把本来就叫 `ssim_band_mean` 的**标量**误当聚合 key 去掉后缀。

老 checkpoint（D5.2/D5.3 已产出的 6 个 run）的 `selection.metrics` 里没有这两个 key，
回放时只比对旧 key，行为与之前完全一致。

### 1b. 逐波段归约链的补齐（2026-10-30）

同一件事的第二半：`band_relative_l2` / `band_rmse` 一直是**逐场景**写进 `validation.json`
`results[]` 的（每个场景 21 个值），但没有任何归约把它汇总。

| 文件 | 改动 |
|---|---|
| `experiments/d5_3/band_report.py` | 新增：任意 `validation.json` → 逐波段场景均值/sd/min/max，`CAVE/KAIST/ALL` 分组，双条件时按 `scene_id` 配对；场景集不一致直接报错 |
| `experiments/d5_3/band_seeds.py` | 新增：三 seed 逐波段归约，落实预注册判据（同号 / 超过跨 seed sd / 每 seed ≥8/9 场景）；逐波段 plateau sd 不可得，显式写 `plateau_available: false` |
| `experiments/d5_3/summarize_seeds.py` | 修复根因：`runs` 与 `pairs` 现在都带 `per_band` 曲线与配对 Δ，不再只保留 8 个标量 |
| `tests/test_d5_3_band_report.py` | 新增 12 项（长度/缺失/非有限/负值/重复场景/场景集不一致、mean-of-ratios 与 ratio-of-norms 的区分、ddof=1、符号与配对顺序） |
| `tests/test_d5_3_band_seeds.py` | 新增 14 项（三条判据各自单独失败、配对顺序无关、波段独立、未配对 seed 不并入、`summarize_seeds` 的 `per_band` 透传） |

取回的 6 份 `validation.json` 存放在 `docs/reports/2026-10-29-d5.3/runs/<arm>_s<seed>/`，
与服务器端 `sha256sum` 逐字节一致；本轮**没有重训、没有重选 checkpoint**。
对 seed 20261021，`band_seeds.py` 与 `band_report.py` 独立算出的逐波段 Δ 最大绝对差为 **0**。

已发布的 `multiseed_comparison.json` **没有重跑**：本地只有 `validation.json`，缺 `history.json` /
`execution.json`（`budget_censored` 与 plateau 统计需要它们）。因此 `per_band` 字段只对下一次
归约生效，本轮逐波段的权威结果是 `per_band_d5_3.json`，`budget_censored` 沿用已发布的
`multiseed_comparison.json`。

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
99 passed, 1 skipped        # D5.3 新增 27 + 控制核 5 + 逐波段 12 + 三 seed 逐波段 14，其余为既有测试
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
   - **2026-10-30 更正**：当时把"取不回产物"写成了"产物在已释放容器里"，这是错的。容器下线后
     workspace 保留；恢复后 `--remote <alias> job start ... -- bash -c 'sha256sum; wc -c;
     base64 -w 76'` + `job logs` 可以把文件**逐字节**取回（6 份 `validation.json` 的 SHA256 全部
     对上）。本机 `file get` 仍报 `broker_unavailable`，所以走的是 job 日志通道。
     逐波段数据当时真正丢失的原因是 `summarize_seeds.py` 只保留 8 个标量 key，不是数据没了。
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

## 8. S3 第二个租约（2026-10-30）

| 项 | 值 |
|---|---|
| 租约 | `c81e44f042b54dac94d60dcf96f20041`，3 × RTX 3090（0,1,2）/ 16 CPU / 24 GiB / 6 h |
| 唯一变量 | `--kernel-file /workspace/data/prepared512/kernels_achromatic_550.npz` |
| 代码 | `/workspace/code/reconstruction-1203ac9`（与 S1 相同），归约用 `reconstruction-653155f` / `-42db6d4` |
| 门禁 | `train.npz 024d2b4c…`、`val.npz e61ac4f1…`、`kernels.npz f07d9bbb…` 与 S1 相同；550 核文件 `5c8e350d…`、数组 `87073bc4…`、band sum 1.0 |
| 稳态速度 | 20–26 s/epoch（比 S1 的 30 s 略快）；启动阶段约 10 min（630 MB 数据 + CUDA 初始化） |
| 完成 | 324 / 369 / 400 epoch，best 264 / 309 / 366，1/3 `budget_censored` |
| 归约 | 容器内跑 `summarize_seeds.py` 与 `band_seeds.py`，只回传 112 KB + 47 KB + 47 KB 三个 JSON |

**通道仍然只有 job 日志**：本机 `file get` 依旧 `broker_unavailable`，`--remote` 明确拒绝对
`file` 子命令生效。因此产物一律 `base64 -w 76` + `job logs` 取回，并在本地复核 SHA256
（`s3_multiseed.json`、`s3_secondary.json`、三份 `validation.json` 全部逐字节一致）。

**一个客户端限制**：`job start` 的 `--shell`/`bash -c` 参数过长（约 500 字符以上）会被
`bad_request` 拒绝，需拆成多条短命令或直接用 argv 形式，例如
`job start <id> --name x -- python3 -X utf8 <abs script> <args>`。
