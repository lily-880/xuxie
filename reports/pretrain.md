# 预训练 · d24

底座是自己训的，不是下载的官方权重。单张 RTX PRO 5000（48GB）从零训练 nanochat d24，训完再做读后续写。

## 结果

| 项 | 数值 |
|---|---|
| 步数 | 5568 / 5568 |
| 验证 bpb | **0.714** |
| CORE | **0.2620** |
| GPT-2 的 CORE | 0.2565 |
| nanochat 官方 Run 6 均值 | 0.2626 |
| 墙钟 | 3284 分钟（约 54.7 小时） |
| 峰值显存 | 37.6 GB |
| 结束时间 | 2026-09-21 |

CORE 过了 GPT-2，和官方同配方的均值持平。官方榜比的是 8 卡 H100 上的墙钟时间；这里是单卡，比的是能力，不是速度。

训练结束时抽了 CORE 里的几项（多数是 10-shot 或 0-shot 的子集，不是完整榜单）：

| 任务 | 准确率 |
|---|---|
| ARC-Easy | 0.720 |
| PIQA | 0.750 |
| HellaSwag（10-shot） | 0.580 |
| LAMBADA | 0.446 |
| COPA | 0.620 |
| WinoGrande | 0.538 |

## 单卡上改过的地方

官方配方在这张卡上跑不动，改了三处，模型尺寸和数据比例没改：

- depth 24，`target-param-data-ratio=8`（GPT-2 CORE 的欠训练配方）。
- `device-batch-size=8`。16 在 48GB 上会 OOM，8 大约占 38GB。
- `window-pattern=L`。这张 Blackwell 没有 FA3，官方的短窗加长窗太慢，改成全长窗口。
- 没有开 fp8。

配置是 24 层、隐层 1536、12 头、词表 32768。checkpoint 在 `base_checkpoints/d24_core_attempt1` 的 step 5568。读后续写从这里开始，没有换过底座。
