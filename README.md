# 读后续写

在自己预训练的 GPT-2 量级模型上，做高考英语读后续写。底座是 nanochat d24（CORE 约 0.262）。后训练用 SFT 和两轮 on-policy DPO，评测用留出的 80 道题，两段开头写死、不让模型改。

正式模型是 **DPO v10 step 50**。模拟题干净率 **70.8%**，词数 **100%** 合格，重讲原文从 SFT 的 10% 降到 **0.3%**。完整数字和后来没超过它的实验在 [reports/summary.md](reports/summary.md)。

## 做成了什么

- **评测尺子。** 80 道留出题（20 道高考、59 道模拟、1 道联考），训练时没用过。DeepSeek 裁判在 80 篇人写参考文上判干净 96.2%，均分 19.3（满分 25）。
- **写死开头的解码。** 模型只写两段正文。词数要落在 100–220，不在这个区间就重写。部署设置是 temperature 0.6、top_k 50、重复惩罚 1.2。
- **后训练。** SFT 按生成质量选 step 80，不按验证 loss。DPO 的好篇和差篇都由当时的模型自己写，开头不参与训练。干净率从 37.2% 到 50.6% 再到 **58.4%**。

| 阶段 | 全量干净率 | 模拟题 | 高考真题 | 重讲原文 |
|---|---|---|---|---|
| SFT step 80 | 37.2% | 49.2% | 3.8% | 10.0% |
| DPO v8 | 50.6% | 62.3% | 17.5% | 0.6% |
| **DPO v10** | **58.4%** | **70.8%** | **23.8%** | **0.3%** |

干净 = 没有重讲原文，也没有严重情节矛盾。和 v8 逐题比，v10 高 7.8 个点（95% 置信区间 2.8–13.8）。

## 权重

正式权重是 DPO v10 step 50 的 bf16 推理副本，约 2.6GB，放在 `weights/parts/`。GitHub 不允许单个超过 100MB 的文件，所以拆成了 90MB 一段。拼回 nanochat 能直接加载的 checkpoint：

```bash
python fetch_weights.py
```

写到 `~/.cache/nanochat/dpo_checkpoints/d24_xuxie_dpo_v10_lr3e-6/model_000050.pt`。需要本机已安装 PyTorch。

## 怎么跑

脚本依赖 [nanochat](https://github.com/karpathy/nanochat)。放进一份检出：

```bash
cp -a xuxie xuxie_eval xuxie_amplify  /path/to/nanochat/dev/d24_core_run/
cp nanochat_changes/dpo_train.py      /path/to/nanochat/scripts/dpo_train.py
cp nanochat_changes/continuation.py   /path/to/nanochat/tasks/continuation.py
cd /path/to/nanochat && git apply /path/to/this/repo/nanochat_changes/engine_on_token.patch
python fetch_weights.py
```

评测：

```bash
cd /path/to/nanochat
.venv/bin/python dev/d24_core_run/xuxie/eval_xuxie.py \
  --kind dpo --tag d24_xuxie_dpo_v10_lr3e-6 --step 50 --name dpo10_s50
```

裁判要 DeepSeek 的 key。复制 `xuxie_amplify/.env.example` 为 `.env`，不要把 key 提交进仓库。

## 仓库里有什么

| 路径 | 作用 |
|---|---|
| `xuxie/decode_locked.py` | 写死开头的批量解码 |
| `xuxie/eval_xuxie.py` | 留出集评测、置信区间 |
| `xuxie/build_dpo_pairs_v2.py` | 自采样、裁判、配偏好对 |
| `xuxie/xuxie_common.py` | 提示、规则、裁判 |
| `xuxie_eval/eval.jsonl` | 80 道留出题 |
| `nanochat_changes/` | 放回 nanochat 的训练脚本和 `Engine.generate` 补丁 |
| `fetch_weights.py` | 下载正式权重 |
| `reports/summary.md` | 结项报告 |

## 这个规模的上限

高考真题干净率是 23.8%。同一底座上再做一轮 DPO、把温度降到 0.4 或 0.5、用模型自己当裁判挑篇、以及 60 步强化学习，都没有超过 v10。形式已经稳定，剩下的错主要是情节前后矛盾。人写参考文的干净率是 96.2%、均分 19.3。再往上要换更大的底座。细节在结项报告里。
