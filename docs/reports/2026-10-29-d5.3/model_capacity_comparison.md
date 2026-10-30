# S4 — A-large / C6 结构对照（本轮未执行）

状态：**⛔ 未执行**。

## 决定

按 D5.3 的阶段门槛，S4 只有在 S1–S3 形成清楚结论、且资源允许时才投入。本轮：

- S1（6 个 400 epoch run）占满了 8 h 租约；
- S3 的 550 nm 训练对照尚未执行（见 `README.md` 的执行记录）；
- 租约后段客户端 `job start` 失效，无法再申请作业。

因此没有训练 A-large 或 C6，也不报告它们的任何数字。

## 已有的、可直接复用的资产

| 配置 | 参数 | 说明 |
|---|---|---|
| `experiments/d5_2/configs/a6_formal.json` | 98,139 | 本轮实际使用的结构（400 epoch 版本见 `d5_3/configs/d5_3_a6.json`） |
| `experiments/d5_2/configs/a_large_formal.json` | ≈285k（width 96 / depth 5） | 容量瓶颈对照 |
| `experiments/d5_2/configs/c6_matched_formal.json` | ≈101k（SFAT，sfat_width 64） | 与 A6 参数量大致匹配的先验对照 |

## 启动条件

1. S1 的 HF / PSNR 结论在 seed 层面稳定（本轮为"方向一致但未达严格判据"）；
2. 550 nm 对照完成，平均核 OTF 混杂被排除或被量化；
3. 至少在 Chromatic 条件下完成 A-large 与 C6 的**同预算**训练，再决定是否为有希望的结构
   补做 Achromatic 匹配；
4. 不得只比较"结构在 Chromatic 观测上的表现"就声称"更强的网络更会用色散"。

## 预注册的做法（若后续执行）

- 预算、seed、数据、选模规则与 S1 完全一致（同一 `d5_3_a6.json` 的预算字段）；
- 先跑 1 epoch 探针测真实步时，再决定租约规格；
- 结论按 A/B/C 三类分开：容量是否改善**空间**指标、是否改善**光谱角**、是否在**反事实**任务
  上提高恢复比例。
