#!/usr/bin/env python3
"""DPO for 读后续写 on pairs the start model wrote itself.

Each pair has a clean continuation (chosen) and a replayed / contradicting one
(rejected), both sampled from the start SFT checkpoint with locked openings.

    loss = -log sigmoid(beta * margin) + alpha * (chosen NLL per token)
    margin = (log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))

Log-probabilities are summed over the body tokens; the openings are never trained.
Reference log-probabilities come precomputed in the pair files, so only the
policy sits on the GPU.

    python -m scripts.dpo_train --run-tag d24_xuxie_dpo_v7 --lr 1e-6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "d24_core_run" / "xuxie"))
import xuxie_common as xc  # noqa: E402
from decode_locked import generate_locked_batch  # noqa: E402

from nanochat.checkpoint_manager import save_checkpoint  # noqa: E402
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0  # noqa: E402
from nanochat.engine import Engine  # noqa: E402

parser = argparse.ArgumentParser(description="DPO on self-written continuation pairs")
parser.add_argument("--pairs", type=str, default=str(xc.TASK_DATA / "dpo_pairs_v2.jsonl"))
parser.add_argument("--val-pairs", type=str, default=str(xc.TASK_DATA / "dpo_pairs_v2_val.jsonl"))
parser.add_argument("--model-kind", type=str, default="sft", choices=["sft", "dpo"], help="start from an SFT or an earlier DPO checkpoint")
parser.add_argument("--model-tag", type=str, default="d24_core_attempt1")
parser.add_argument("--model-step", type=int, default=80, help="start step; must match the pairs' reference")
parser.add_argument("--run-tag", type=str, default="d24_xuxie_dpo_v7")
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--alpha", type=float, default=0.2, help="weight of the chosen NLL per token")
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--warmup-frac", type=float, default=0.1)
parser.add_argument("--epochs", type=float, default=2.0)
parser.add_argument("--pairs-per-step", type=int, default=32)
parser.add_argument("--clip", type=float, default=1.0)
parser.add_argument("--max-steps", type=int, default=-1)
parser.add_argument("--val-every", type=int, default=5)
parser.add_argument("--guard-every", type=int, default=10)
parser.add_argument("--guard-items", type=int, default=8)
parser.add_argument("--canary-items", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()


def load_pairs(path: str, start: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for row in rows:
        assert row.get("ref") == start, f"pair reference {row.get('ref')} != start model {start}"
    return rows


def prepare(rows: list[dict], tokenizer) -> list[dict]:
    out = []
    for row in rows:
        c = xc.render_answer(tokenizer, row, row["chosen"])
        r = xc.render_answer(tokenizer, row, row["rejected"])
        assert sum(c[1][1:]) == row["n_chosen"] and sum(r[1][1:]) == row["n_rejected"], f"token count drift in {row['id']}"
        out.append({"id": row["id"], "c": c, "r": r, "ref_c": row["ref_chosen"], "ref_r": row["ref_rejected"],
                    "n_c": row["n_chosen"]})
    return out


def pair_terms(policy, pair: dict, pad: int):
    logp, _ = xc.seq_logprob(policy, [pair["c"], pair["r"]], pad)
    d_c = logp[0] - pair["ref_c"]
    d_r = logp[1] - pair["ref_r"]
    margin = d_c - d_r
    dpo = -F.logsigmoid(args.beta * margin)
    nll = -logp[0] / pair["n_c"]
    return dpo, nll, margin, d_c, d_r


@torch.no_grad()
def evaluate_pairs(policy, pairs: list[dict], pad: int) -> dict:
    acc = margin = d_c = d_r = 0.0
    for p in pairs:
        _, _, m, c, r = pair_terms(policy, p, pad)
        acc += float(m > 0)
        margin += float(m)
        d_c += float(c)
        d_r += float(r)
    n = max(len(pairs), 1)
    return {"acc": acc / n, "margin": margin / n, "chosen_delta": d_c / n, "rejected_delta": d_r / n}


def guard_check(policy, tokenizer, items: list[dict], canary: list[dict]) -> dict:
    """Full-length writing on held-out eval items, plus canaries that also write the openings."""
    policy.eval()
    engine = Engine(policy, tokenizer)
    texts = []
    for k, item in enumerate(items):
        texts.append(generate_locked_batch(
            engine, tokenizer, xc.prompt_tokens(tokenizer, item), item["opening1"], item["opening2"],
            num_samples=1, seed=500 + k, **xc.DECODE,
        )[0])
    rules = [xc.rule_metrics(it, t) for it, t in zip(items, texts)]
    canary_rules = []
    for item in canary:
        toks = xc.prompt_tokens(tokenizer, item)
        res, _ = engine.generate_batch(toks, num_samples=1, max_tokens=300, temperature=xc.DECODE["temperature"],
                                       top_k=xc.DECODE["top_k"], seed=7,
                                       repetition_penalty=xc.DECODE["repetition_penalty"])
        canary_rules.append(xc.rule_metrics(item, tokenizer.decode(res[0][len(toks):]).strip()))
    policy.train()
    bad = sum(r["garbage"] or r["cut"] or r["heavy_rep"] or not r["word_ok"] for r in rules)
    return {
        "bad": bad,
        "garbage": sum(r["garbage"] for r in rules),
        "long_copy": sum(r["long_copy"] for r in rules),
        "o2_leak": sum(r["o2_leak"] for r in rules),
        "canary_garbage": sum(r["garbage"] for r in canary_rules),
        "canary_format": sum(r["format_ok"] for r in canary_rules),
        "sample": texts[0],
    }


def guard_failed(now: dict, base: dict) -> str:
    if now["garbage"] > base["garbage"] or now["canary_garbage"] > base["canary_garbage"]:
        return "garbled English"
    if now["bad"] > base["bad"] + 1:
        return f"{now['bad']} broken samples vs {base['bad']} at start"
    return ""


def main() -> None:
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    compute_init(device_type)
    device = torch.device(device_type)
    torch.manual_seed(args.seed)
    start = f"{args.model_kind}:{args.model_tag}:{args.model_step}"

    policy, tokenizer, _meta = xc.load_checkpoint(args.model_kind, args.model_tag, args.model_step, device, phase="train")
    policy.transformer.wte.weight.requires_grad_(False)
    for ve in policy.value_embeds.values():
        ve.weight.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable), "trainable params must be fp32"
    print0(f"trainable params {sum(p.numel() for p in trainable):,} (embeddings frozen)")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    pad = tokenizer.encode_special("<|assistant_end|>")

    train = prepare(load_pairs(args.pairs, start), tokenizer)
    val = prepare(load_pairs(args.val_pairs, start), tokenizer)
    steps_per_epoch = math.ceil(len(train) / args.pairs_per_step)
    total = math.ceil(args.epochs * steps_per_epoch)
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    warmup = max(1, round(args.warmup_frac * total))
    print0(f"pairs train={len(train)} val={len(val)} steps={total} warmup={warmup} beta={args.beta} "
           f"alpha={args.alpha} lr={args.lr} pairs/step={args.pairs_per_step}")

    eval_items = xc.load_eval_items()
    guard_items = xc.spread(eval_items, args.guard_items)
    canary = xc.spread(eval_items[1:], args.canary_items)
    out_dir = os.path.join(get_base_dir(), "dpo_checkpoints", args.run_tag)

    def meta(step: int, extra: dict) -> dict:
        cfg = policy.config
        return {
            "step": step, "start": start, "beta": args.beta, "alpha": args.alpha, "lr": args.lr,
            "pairs": args.pairs, **extra,
            "model_config": {
                "sequence_len": cfg.sequence_len, "vocab_size": cfg.vocab_size, "n_layer": cfg.n_layer,
                "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head, "n_embd": cfg.n_embd,
                "window_pattern": cfg.window_pattern, "softcap": cfg.softcap,
            },
        }

    v = evaluate_pairs(policy, val, pad)
    print0(f"step 0000 val acc {v['acc']:.3f} margin {v['margin']:.3f} (should be ~0: reference check)")
    base_guard = guard_check(policy, tokenizer, guard_items, canary)
    print0(f"step 0000 guard bad {base_guard['bad']}/{len(guard_items)} garbage {base_guard['garbage']} "
           f"canary format {base_guard['canary_format']}/{len(canary)} garbage {base_guard['canary_garbage']}")

    order: list[dict] = []
    rng = random.Random(args.seed)
    epoch = 0
    t0 = time.time()
    saved: list[int] = []
    stop_reason = ""
    history = []
    for step in range(1, total + 1):
        lr = args.lr * (step / warmup if step <= warmup else max(0.0, (total - step + 1) / (total - warmup + 1)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        batch = []
        while len(batch) < args.pairs_per_step:
            if not order:
                epoch += 1
                order = list(train)
                rng.shuffle(order)
            batch.append(order.pop())
        stats = {"loss": 0.0, "dpo": 0.0, "nll": 0.0, "acc": 0.0, "margin": 0.0, "d_c": 0.0, "d_r": 0.0}
        for pair in batch:
            dpo, nll, margin, d_c, d_r = pair_terms(policy, pair, pad)
            loss = dpo + args.alpha * nll
            (loss / len(batch)).backward()
            for k, val_ in (("loss", loss), ("dpo", dpo), ("nll", nll), ("margin", margin), ("d_c", d_c), ("d_r", d_r)):
                stats[k] += float(val_) / len(batch)
            stats["acc"] += float(margin > 0) / len(batch)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print0(
            f"step {step:04d}/{total} ep {epoch} lr {lr:.2e} loss {stats['loss']:.4f} dpo {stats['dpo']:.4f} "
            f"acc {stats['acc']:.2f} margin {stats['margin']:.2f} chosen {stats['d_c']:+.2f} "
            f"rejected {stats['d_r']:+.2f} nll {stats['nll']:.3f} gnorm {grad_norm:.2f} "
            f"time {(time.time() - t0) / 60:.1f}m"
        )
        record = {"step": step, **stats, "grad_norm": grad_norm}
        if step % args.val_every == 0 or step == total:
            v = evaluate_pairs(policy, val, pad)
            record["val"] = v
            print0(f"step {step:04d} val acc {v['acc']:.3f} margin {v['margin']:.2f} "
                   f"chosen {v['chosen_delta']:+.2f} rejected {v['rejected_delta']:+.2f}")
        if step % args.guard_every == 0 or step == total:
            g = guard_check(policy, tokenizer, guard_items, canary)
            record["guard"] = {k: g[k] for k in g if k != "sample"}
            print0(f"step {step:04d} guard bad {g['bad']}/{len(guard_items)} garbage {g['garbage']} "
                   f"long_copy {g['long_copy']} o2_leak {g['o2_leak']} canary format {g['canary_format']}/"
                   f"{len(canary)} garbage {g['canary_garbage']}")
            print0("  sample: " + g["sample"].replace("\n", " / ")[:400])
            why = guard_failed(g, base_guard)
            if why:
                stop_reason = f"guard failed at step {step}: {why}"
                print0("stop: " + stop_reason)
                history.append(record)
                break
            save_checkpoint(out_dir, step, policy.state_dict(), None, meta(step, {"val": record.get("val")}))
            saved.append(step)
        history.append(record)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "base_guard": {k: base_guard[k] for k in base_guard if k != "sample"},
                   "history": history, "saved": saved, "stop": stop_reason}, f, indent=1)
    print0(f"done saved={saved} stop={stop_reason or 'none'} dir={out_dir}")
    compute_cleanup()


if __name__ == "__main__":
    main()
