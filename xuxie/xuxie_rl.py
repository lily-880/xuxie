#!/usr/bin/env python3
"""GRPO for 读后续写, same simplification as scripts/chat_rl.py.

On-policy, no KL term, no PPO clip. Advantage of a sample is its reward minus
the mean reward of the samples written for the same stem. Only tokens the model
actually sampled get gradient; the prompt and the forced openings do not.

Sampling uses the locked-opening decoder, because that is how the model is
deployed. Temperature is 1.0 so the group of samples disagrees often enough
for the advantage to be nonzero. There is no word-count retry: a short or long
essay is penalized by the reward instead of being replaced.

    R = format + words + repetition + 0.5 * harmony + 0.5 * link

format and words are 1 or 0. repetition is -1 when the essay repeats itself.
harmony is 1 / 0.5 / 0 for no conflict / a minor conflict / a replay or a major
conflict. link is fits_openings / 2. The two judge terms together are at most 1,
the two rule terms at most 2.

    python dev/d24_core_run/xuxie/xuxie_rl.py --run-tag d24_xuxie_rl_v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xuxie_common as xc  # noqa: E402
from decode_locked import _sample_rows  # noqa: E402

from nanochat.checkpoint_manager import save_checkpoint  # noqa: E402
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0  # noqa: E402
from nanochat.engine import Engine  # noqa: E402

parser = argparse.ArgumentParser(description="GRPO on locked-opening continuations")
parser.add_argument("--model-kind", default="dpo", choices=["sft", "dpo"])
parser.add_argument("--model-tag", default="d24_xuxie_dpo_v10_lr3e-6")
parser.add_argument("--model-step", type=int, default=50)
parser.add_argument("--run-tag", default="d24_xuxie_rl_v1")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--stems-per-step", type=int, default=4)
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--clip", type=float, default=1.0)
parser.add_argument("--save-every", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def reward_parts(rules: dict, judge: dict | None) -> dict:
    fmt = 1.0 if rules["format_ok"] else 0.0
    words = 1.0 if rules["word_ok"] else 0.0
    rep = -1.0 if rules["heavy_rep"] else 0.0
    if judge is None:
        harmony, link = 0.0, 0.0
    else:
        if judge["replay"] or judge["contradiction"] == "major":
            harmony = 0.0
        elif judge["contradiction"] == "minor":
            harmony = 0.5
        else:
            harmony = 1.0
        link = judge["fits_openings"] / 2
    total = fmt + words + rep + 0.5 * harmony + 0.5 * link
    return {"R": total, "format": fmt, "words": words, "rep": rep, "harmony": harmony, "link": link}


def main() -> None:
    device_type = autodetect_device_type()
    compute_init(device_type)
    device = torch.device(device_type)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    policy, tokenizer, _ = xc.load_checkpoint(args.model_kind, args.model_tag, args.model_step, device, phase="train")
    policy.transformer.wte.weight.requires_grad_(False)
    for ve in policy.value_embeds.values():
        ve.weight.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable)
    print0(f"trainable params {sum(p.numel() for p in trainable):,} (embeddings frozen)")
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    pad = tokenizer.encode_special("<|assistant_end|>")
    engine = Engine(policy, tokenizer)
    judge = xc.Judge()

    stems = xc.load_train_stems()
    rng.shuffle(stems)
    order = list(stems)
    print0(f"stems {len(stems)} steps {args.steps} per step {args.stems_per_step}x{args.samples} "
           f"lr {args.lr} temperature {args.temperature}")

    out_dir = os.path.join(get_base_dir(), "dpo_checkpoints", args.run_tag)
    cfg = policy.config

    def meta(step: int, extra: dict) -> dict:
        return {
            "step": step,
            "start": f"{args.model_kind}:{args.model_tag}:{args.model_step}",
            "lr": args.lr, "temperature": args.temperature, **extra,
            "model_config": {
                "sequence_len": cfg.sequence_len, "vocab_size": cfg.vocab_size, "n_layer": cfg.n_layer,
                "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head, "n_embd": cfg.n_embd,
                "window_pattern": cfg.window_pattern, "softcap": cfg.softcap,
            },
        }

    decode = dict(temperature=args.temperature, top_k=xc.DECODE["top_k"],
                  repetition_penalty=xc.DECODE["repetition_penalty"],
                  min_para_words=xc.DECODE["min_para_words"])
    history = []
    saved: list[int] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = step / max(args.steps, 1)
        for g in opt.param_groups:
            g["lr"] = args.lr * max(0.0, 1.0 - frac)
        batch = []
        while len(batch) < args.stems_per_step:
            if not order:
                order = list(stems)
                rng.shuffle(order)
            batch.append(order.pop())

        policy.eval()
        groups = []
        jobs, where = [], []
        for item in batch:
            rows = _sample_rows(
                engine, tokenizer, xc.prompt_tokens(tokenizer, item), item["opening1"], item["opening2"],
                num_samples=args.samples, seed=rng.randrange(1_000_000),
                budget=xc.DECODE["max_body_tokens"], **decode,
            )
            samples = []
            for text, ids, mask in rows:
                rules = xc.rule_metrics(item, text)
                samples.append({"text": text, "ids": ids, "mask": mask, "rules": rules, "judge": None})
                if rules["format_ok"] and not rules["garbage"]:
                    jobs.append((item, text))
                    where.append((len(groups), len(samples) - 1))
            groups.append({"item": item, "samples": samples})
        verdicts = judge.many(jobs, workers=16, log_every=0) if jobs else []
        for (gi, k), j in zip(where, verdicts):
            groups[gi]["samples"][k]["judge"] = j

        policy.train()
        parts = []
        total_valid = 0
        prepared = []
        for g in groups:
            scored = [reward_parts(s["rules"], s["judge"]) for s in g["samples"]]
            parts.extend(scored)
            mean_r = sum(p["R"] for p in scored) / len(scored)
            for s, p in zip(g["samples"], scored):
                adv = p["R"] - mean_r
                n = sum(s["mask"])
                total_valid += n
                prepared.append((s["ids"], s["mask"], adv, n))
        total_valid = max(total_valid, 1)
        for ids, mask, adv, n in prepared:
            if n == 0 or adv == 0.0:
                continue
            logp, _ = xc.seq_logprob(policy, [(ids, mask)], pad)
            (-(logp[0] * adv) / total_valid).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.clip))
        opt.step()
        opt.zero_grad(set_to_none=True)

        def avg(key: str) -> float:
            return sum(p[key] for p in parts) / len(parts)

        record = {
            "step": step, "R": avg("R"), "format": avg("format"), "words": avg("words"),
            "rep": avg("rep"), "harmony": avg("harmony"), "link": avg("link"),
            "grad_norm": grad_norm, "judged": len(jobs),
        }
        history.append(record)
        print0(
            f"step {step:03d}/{args.steps} R {record['R']:.3f} format {record['format']:.2f} "
            f"words {record['words']:.2f} rep {record['rep']:.2f} harmony {record['harmony']:.2f} "
            f"link {record['link']:.2f} judged {record['judged']} gnorm {grad_norm:.2f} "
            f"{(time.time() - t0) / 60:.1f}m"
        )
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(out_dir, step, policy.state_dict(), None, meta(step, {"train": record}))
            saved.append(step)
            print0(f"saved step {step} -> {out_dir}")

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history, "saved": saved}, f, indent=1)
    print0(f"done saved={saved} dir={out_dir}")
    compute_cleanup()


if __name__ == "__main__":
    main()
