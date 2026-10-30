# D5.2 S0：服务器接入、数据与训练可靠性

日期：2026-10-21。范围：实验室 GPU 服务器上的只读环境核验、数据上传、smoke 与断点等价性测试。未更新 Meta pin。

## 1. 计算资源

使用实验室授权的共享 GPU 服务器，单卡起步，后续按需使用第二卡。实际分配由研究负责人在群内协调；本报告不记录登录地址、端口或凭据。

| 项目 | 实际值 |
|---|---|
| GPU | 2 × NVIDIA RTX 4090，49140 MiB/卡（实际使用） |
| GPU 分配 | GPU 3 主训练；GPU 5 独立机理对照 |
| CPU | 128 logical CPUs，共享负载较高 |
| RAM | 251 GiB，约 89 GiB available |
| 数据盘 | 4.5 TiB available |
| Python | 3.10.12 |
| PyTorch | 2.2.1+cu121，CUDA available |
| 依赖 | numpy 1.26.4、scipy 1.15.3、opencv 5.0.0、h5py 可用 |

未安装系统包、未修改其他用户环境、未终止任何非本项目的 GPU 进程。GPU 2/4/6/7 属于其他任务。

## 2. 数据上传

只上传训练和验证所需文件；`test.npz` 未上传，G4 前不进入服务器训练目录。

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `train.npz` | 629,637,033 B | `024d2b4c8956707387289242ddbadce08e3379a51ed6e0752a167666dbb32908` |
| `val.npz` | 138,099,071 B | `e61ac4f102d58a44bdb8f7174a960d13f2e102c5db00cc0bc408ea5583ff9247` |
| `manifest.json` | 43,041 B | `cfe89c34868b7eece4c6089d238d79fe224cd62c54db23df0a81ba7e40ef02f3` |
| `kernels.npz` | 152,546 B | `f07d9bbb7a4ff870433e8e72dcd19298d642e639c499afe1057a97eeb9206d4b` |

Achromatic 对照核：

`3f28af69003f451e90f9220aee433091535bc76d0de04d7987cade37349de61a`

由 21 个 baseline shape-only 核取平均后复制 21 份，每份和为 1。

## 3. Resume 等价性

修复了 `last.pt` 在 `best/stale` 更新和 `scheduler.step()` 之前保存的问题。现在每个 epoch 的顺序为：

1. validation loss；
2. 更新 best/stale/best_epoch；
3. `scheduler.step()`；
4. 保存 `last.pt` 与按需 `best.pt`；
5. 写 history；
6. 检查 early stopping。

checkpoint 增加 `next_epoch`、`global_step`、`best_epoch`；旧 checkpoint 仍兼容。

严格测试：同一 seed、样本顺序、噪声下，比较“连续 4 epochs”与“2 epochs + resume 到 4 epochs”。比较模型权重、optimizer、scheduler、epoch、global step、RNG、best 选择。

结果：**passed=true，differences=[]**。  
证据：[resume_equivalence.json](resume_equivalence.json)。

## 4. 服务器 smoke

GPU 3 上运行 100 steps（1 epoch）的 A6：

- 分辨率 512×512，21 通道，6 阶段；
- batch 1；
- 100 steps 用时约 33.3 s；
- 首轮 validation：PSNR 28.44 dB、SSIM 0.897、SAM 14.04°、HF 0.9709、measurement relative L2 0.0333；
- checkpoint、history、execution 正常写入。

## 5. 已启动的正式运行

| Run | GPU | 条件 | 状态 |
|---|---|---|---|
| `a6_formal_s1_gpu3` | 3 | Measured chromatic PSF | 运行中 |
| `a6_achromatic_s1_gpu5` | 5 | 21 个波段使用同一平均核，从头训练 | 运行中 |

两者均为 A6、30,000 steps 上限、相同 seed/损失/数据/验证规则，测试集未加载。正式训练曲线、best checkpoint 与最终 validation 指标在 S1/S3 报告中汇总。

## 6. 边界

- 512/8/64 signed shape-only 仍是暂定数值模型，不是绝对辐射/Bayer 标定。
- 服务器资源为共享环境；实际吞吐会受其他任务影响。
- 没有执行 G4；没有使用冻结测试集。
- 没有修改 `optic_system`、D5/D5.1 历史报告或 Meta pin。
