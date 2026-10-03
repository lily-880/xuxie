#!/usr/bin/env python3
"""Pick the best of N locked continuations with the graded judge.

On the 20 held-out gaokao items, DPO v8's single sample is clean 17.5% of the
time. Among 4 samples, a clean one exists for 35% of items, and the judge's
ranking picks that one whenever it exists.

    python dev/d24_core_run/xuxie/best_of.py --kind dpo --tag d24_xuxie_dpo_v8_lr3e-6 --step 50 --source gaokao --n 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xuxie_common as xc  # noqa: E402
from decode_locked import generate_locked_batch  # noqa: E402
from nanochat.common import autodetect_device_type, compute_init  # noqa: E402
from nanochat.engine import Engine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="dpo", choices=["sft", "dpo"])
    ap.add_argument("--tag", default="d24_xuxie_dpo_v8_lr3e-6")
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--source", default="gaokao")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    items = xc.load_eval_items()
    if args.source:
        items = [it for it in items if it.get("source") == args.source]
    device_type = args.device or autodetect_device_type()
    compute_init(device_type)
    model, tokenizer, _ = xc.load_checkpoint(args.kind, args.tag, args.step, torch.device(device_type))
    engine = Engine(model, tokenizer)
    judge = xc.Judge()
    picked_clean = 0
    for n, item in enumerate(items, 1):
        texts = generate_locked_batch(
            engine, tokenizer, xc.prompt_tokens(tokenizer, item),
            item["opening1"], item["opening2"],
            num_samples=args.n, seed=3000 + item.get("_index", n),
            **dict(xc.DECODE, retries=1),
        )
        verdicts = [judge(item, text) for text in texts]
        best = max(range(len(texts)), key=lambda i: xc.quality(verdicts[i]))
        ok = xc.is_clean(verdicts[best])
        picked_clean += int(ok)
        print(f"[{n}/{len(items)}] {item['id']} clean={ok} score={verdicts[best]['score']} "
              f"any={any(xc.is_clean(v) for v in verdicts)}", flush=True)
    print(json.dumps({"items": len(items), "n": args.n, "picked_clean_rate": picked_clean / len(items)}))


if __name__ == "__main__":
    main()
