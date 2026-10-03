# 读后续写专家

在一个 GPT-2 级别的 nanochat d24 底座上，做读后续写（高考英语续写）的后训练。后训练到此结束。正式模型是 **DPO v10 step 50**。

完整结论、数字和失败实验见 [reports/summary.md](reports/summary.md)。

## 结果

同一把尺子：留出的 80 道题，开头写死，每题 4 篇。干净 = 没有重讲原文、没有严重情节矛盾。

| 阶段 | 全量干净率 | 高考真题 | 模拟题 |
|---|---|---|---|
| SFT step 80 | 37.2% | 3.8% | 49.2% |
| DPO v8 | 50.6% | 17.5% | 62.3% |
| **DPO v10** | **58.4%** | **23.8%** | **70.8%** |
| 人写参考文 | 96.2% | 90.0% | 98.3% |

形式已经合格：词数、分段、不乱码、几乎不抄原文。短板是情节前后矛盾。在这个底座上继续做 DPO、调温度、自己训打分模型、强化学习，都没有超过 v10。

## 这个仓库里有什么

| 路径 | 作用 |
|---|---|
| `xuxie/decode_locked.py` | 部署解码：两段开头写死，词数不对就重写 |
| `xuxie/eval_xuxie.py` | 留出集评测 |
| `xuxie/xuxie_common.py` | 提示、规则、DeepSeek 裁判、DPO 用的 token |
| `xuxie/build_dpo_pairs_v2.py` | 模型自己写、裁判打分、配偏好对 |
| `xuxie/best_of.py` | 可选：写多篇，让 DeepSeek 挑一篇。写作时要调用 API |
| `xuxie/xuxie_rl.py` | 试过的 GRPO。60 步后全量 55.3%，不用 |
| `xuxie_eval/eval.jsonl` | 80 道留出题，训练时没有用过 |
| `nanochat_changes/` | 要放进 nanochat 仓库的三个文件，以及 `Engine.generate` 的补丁 |
| `reports/summary.md` | 结项报告 |

权重不在这里。正式权重在训练机上：

`~/.cache/nanochat/dpo_checkpoints/d24_xuxie_dpo_v10_lr3e-6/model_000050.pt`（4GB）

底座是 `d24_core_attempt1` step 5568，CORE 约 0.262。API key 不在这里，复制 `xuxie_amplify/.env.example` 为 `.env` 再填写。

## 怎么接到 nanochat 上

这些脚本依赖 [nanochat](https://github.com/karpathy/nanochat)。放到一份 nanochat 检出里：

```bash
cp -a xuxie xuxie_eval xuxie_amplify  /path/to/nanochat/dev/d24_core_run/
cp nanochat_changes/dpo_train.py      /path/to/nanochat/scripts/dpo_train.py
cp nanochat_changes/continuation.py   /path/to/nanochat/tasks/continuation.py
cd /path/to/nanochat && git apply /path/to/xuxie/nanochat_changes/engine_on_token.patch
```

评测正式模型：

```bash
cd /path/to/nanochat
.venv/bin/python dev/d24_core_run/xuxie/eval_xuxie.py \
  --kind dpo --tag d24_xuxie_dpo_v10_lr3e-6 --step 50 --name dpo10_s50
```

解码参数：temperature 0.6，top_k 50，重复惩罚 1.2，正文 100–220 词。
