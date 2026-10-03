#!/usr/bin/env python3
"""DPO pairs v2: both sides are written by the start model itself.

    sample   start model writes N continuations per training stem (locked openings)
    judge    rules + graded DeepSeek judge on every sample that passes the rules
    pairs    clean sample vs replay / major-conflict sample, similar length, big score gap
    reflogp  start model's summed log-probabilities for both sides, stored in the pair files

    python dev/d24_core_run/xuxie/build_dpo_pairs_v2.py sample --start-step 20 --shard 0 --num-shards 2
    python dev/d24_core_run/xuxie/build_dpo_pairs_v2.py judge
    python dev/d24_core_run/xuxie/build_dpo_pairs_v2.py pairs
    python dev/d24_core_run/xuxie/build_dpo_pairs_v2.py reflogp --start-step 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xuxie_common as xc  # noqa: E402

RAW_GLOB = "dpo_v2_raw.shard*.jsonl"
JUDGED = xc.TASK_DATA / "dpo_v2_judged.jsonl"
PAIRS = xc.TASK_DATA / "dpo_pairs_v2.jsonl"
PAIRS_VAL = xc.TASK_DATA / "dpo_pairs_v2_val.jsonl"
REPORT = xc.REPORTS / "dpo_pairs_v2.md"


def set_prefix(prefix: str) -> None:
    """dpo_v2 keeps the round-1 file names; later rounds get their own."""
    global RAW_GLOB, JUDGED, PAIRS, PAIRS_VAL, REPORT
    RAW_GLOB = f"{prefix}_raw.shard*.jsonl"
    JUDGED = xc.TASK_DATA / f"{prefix}_judged.jsonl"
    if prefix != "dpo_v2":
        PAIRS = xc.TASK_DATA / f"{prefix}_pairs.jsonl"
        PAIRS_VAL = xc.TASK_DATA / f"{prefix}_pairs_val.jsonl"
        REPORT = xc.REPORTS / f"{prefix}_pairs.md"


def _hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def is_val(stem_id: str) -> bool:
    return _hash("val:" + stem_id) % 20 == 0


def chosen_stems(n: int) -> list[dict]:
    stems = xc.load_train_stems()
    return xc.spread(stems, n) if n > 0 else stems


def load_stems(args) -> list[dict]:
    if getattr(args, "items", ""):
        return [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    return chosen_stems(args.n_stems)


def read_raw() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(xc.TASK_DATA.glob(RAW_GLOB)):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def cmd_sample(args) -> None:
    from nanochat.common import autodetect_device_type, compute_init
    from nanochat.engine import Engine
    from decode_locked import generate_locked_batch
    import torch

    stems = [s for i, s in enumerate(load_stems(args)) if i % args.num_shards == args.shard]
    out_path = xc.TASK_DATA / f"{args.prefix}_raw.shard{args.shard}.jsonl"
    done = set(read_raw())
    todo = [s for s in stems if s["id"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: stems={len(stems)} todo={len(todo)}", flush=True)
    device_type = args.device or autodetect_device_type()
    compute_init(device_type)
    model, tokenizer, _ = xc.load_checkpoint(args.start_kind, args.tag, args.start_step, torch.device(device_type))
    engine = Engine(model, tokenizer)
    start = f"{args.start_kind}:{args.tag}:{args.start_step}"
    decode = dict(xc.DECODE, temperature=args.temperature, retries=1)
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for n, item in enumerate(todo, 1):
            texts = generate_locked_batch(
                engine, tokenizer, xc.prompt_tokens(tokenizer, item), item["opening1"], item["opening2"],
                num_samples=args.samples, seed=_hash(item["id"]) % 100000, **decode,
            )
            f.write(json.dumps({"id": item["id"], "start": start, "temperature": args.temperature,
                                "samples": texts}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"sampled {n}/{len(todo)} {rate:.1f}s/stem eta {rate * (len(todo) - n) / 60:.0f}m", flush=True)


def passes_rules(r: dict) -> bool:
    return r["word_ok"] and not r["garbage"] and not r["cut"]


def cmd_judge(args) -> None:
    stems = {s["id"]: s for s in load_stems(args)}
    raw = read_raw()
    jobs, where = [], []
    records = []
    for sid, rec in raw.items():
        item = stems[sid]
        samples = []
        for k, text in enumerate(rec["samples"]):
            r = xc.rule_metrics(item, text)
            samples.append({"text": text, "rules": r, "judge": None})
            if passes_rules(r):
                jobs.append((item, text))
                where.append((len(records), k))
        records.append({"id": sid, "start": rec["start"], "samples": samples})
    print(f"stems={len(records)} samples={sum(len(r['samples']) for r in records)} to_judge={len(jobs)}", flush=True)
    verdicts = xc.Judge().many(jobs, workers=args.workers, log_every=200)
    for (ri, k), j in zip(where, verdicts):
        records[ri]["samples"][k]["judge"] = j
    with JUDGED.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {JUDGED}", flush=True)


def chosen_ok(s: dict) -> bool:
    r, j = s["rules"], s["judge"]
    return (j is not None and xc.is_clean(j) and passes_rules(r) and not r["heavy_rep"]
            and not r["long_copy"] and not r["o2_leak"] and j["language"] >= 1)


def rejected_ok(s: dict) -> bool:
    r, j = s["rules"], s["judge"]
    if j is None or not passes_rules(r) or j["language"] < 1:
        return False
    return j["replay"] or j["contradiction"] == "major" or r["heavy_rep"] or r["long_copy"] or r["o2_leak"]


def why_rejected(s: dict) -> str:
    r, j = s["rules"], s["judge"]
    tags = []
    if j["replay"]:
        tags.append("replay")
    if j["contradiction"] == "major":
        tags.append("major")
    if r["heavy_rep"]:
        tags.append("heavy_rep")
    if r["long_copy"]:
        tags.append("long_copy")
    if r["o2_leak"]:
        tags.append("o2_leak")
    return "+".join(tags)


def cmd_pairs(args) -> None:
    stems = {s["id"]: s for s in load_stems(args)}
    records = [json.loads(l) for l in JUDGED.open(encoding="utf-8") if l.strip()]
    train, val = [], []
    n_samples = n_judged = n_clean = stems_with_clean = stems_with_both = 0
    reasons: dict[str, int] = {}
    for rec in records:
        item = stems[rec["id"]]
        samples = rec["samples"]
        judged = [s for s in samples if s["judge"] is not None]
        n_samples += len(samples)
        n_judged += len(judged)
        good = sorted((s for s in samples if chosen_ok(s)), key=lambda s: -xc.quality(s["judge"]))
        bad = sorted((s for s in samples if rejected_ok(s)), key=lambda s: xc.quality(s["judge"]))
        n_clean += sum(xc.is_clean(s["judge"]) for s in judged)
        stems_with_clean += bool(good)
        stems_with_both += bool(good and bad)
        used: set[int] = set()
        made = 0
        for c in good:
            if made >= args.max_pairs_per_stem:
                break
            for r in bad:
                if id(r) in used:
                    continue
                wc, wr = c["rules"]["words"], r["rules"]["words"]
                if abs(wc - wr) / max(wc, wr) > args.max_len_gap:
                    continue
                if xc.quality(c["judge"]) - xc.quality(r["judge"]) < args.min_gap:
                    continue
                used.add(id(r))
                why = why_rejected(r)
                reasons[why] = reasons.get(why, 0) + 1
                row = {
                    "id": rec["id"],
                    "start": rec["start"],
                    "passage": item["passage"],
                    "opening1": item["opening1"],
                    "opening2": item["opening2"],
                    "chosen": c["text"],
                    "rejected": r["text"],
                    "chosen_judge": c["judge"],
                    "rejected_judge": r["judge"],
                    "chosen_words": wc,
                    "rejected_words": wr,
                    "rejected_why": why,
                }
                (val if is_val(rec["id"]) else train).append(row)
                made += 1
                break
    for path, rows in ((PAIRS, train), (PAIRS_VAL, val)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_stems = len(records)
    lines = [
        "# DPO 配对 v2 · 好篇和差篇都由起点模型自己写",
        "",
        f"- 起点：`{records[0]['start'] if records else ''}`；每题写死开头采样，temperature={args.temperature_note}",
        f"- 题干 {n_stems} 个，样本 {n_samples} 篇，过规则后送裁判 {n_judged} 篇",
        f"- 送审样本里干净的比例：{100 * n_clean / max(n_judged, 1):.1f}%",
        f"- 至少有一篇可当好篇的题干：{stems_with_clean}/{n_stems}（{100 * stems_with_clean / max(n_stems, 1):.1f}%）",
        f"- 好篇、差篇都有的题干：{stems_with_both}/{n_stems}",
        f"- 留下的对：训练 {len(train)}，验证 {len(val)}（验证按题干分，不和训练重叠）",
        f"- 选对条件：两篇词数相差 ≤{int(100 * args.max_len_gap)}%，裁判综合分差 ≥{args.min_gap}，每题最多 {args.max_pairs_per_stem} 对",
        "- 差篇原因：" + "，".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])),
        f"- 文件：`{PAIRS}`、`{PAIRS_VAL}`",
        "",
    ]
    for row in train[:3]:
        lines += [
            f"## {row['id']}",
            "",
            "```",
            row["passage"].strip(),
            "```",
            f"开头 1：{row['opening1']}",
            "",
            f"开头 2：{row['opening2']}",
            "",
            f"**好篇**（分 {row['chosen_judge']['score']}）",
            "",
            "```",
            row["chosen"],
            "```",
            f"**差篇**（{row['rejected_why']}，分 {row['rejected_judge']['score']}）：{row['rejected_judge']['reason']}",
            "",
            "```",
            row["rejected"],
            "```",
            "",
        ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]), flush=True)


def cmd_reflogp(args) -> None:
    from nanochat.common import autodetect_device_type, compute_init
    import torch

    device_type = args.device or autodetect_device_type()
    compute_init(device_type)
    model, tokenizer, _ = xc.load_checkpoint(args.start_kind, args.tag, args.start_step, torch.device(device_type))
    start = f"{args.start_kind}:{args.tag}:{args.start_step}"
    pad = tokenizer.encode_special("<|assistant_end|>")
    for path in (PAIRS, PAIRS_VAL):
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        for n, row in enumerate(rows, 1):
            assert row["start"] == start, f"pair from {row['start']} but reference is {start}"
            c = xc.render_answer(tokenizer, row, row["chosen"])
            r = xc.render_answer(tokenizer, row, row["rejected"])
            with torch.no_grad():
                logp, counts = xc.seq_logprob(model, [c, r], pad)
            row["ref"] = start
            row["ref_chosen"], row["ref_rejected"] = float(logp[0]), float(logp[1])
            row["n_chosen"], row["n_rejected"] = int(counts[0]), int(counts[1])
            if n % 100 == 0:
                print(f"{path.name}: {n}/{len(rows)}", flush=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote ref log-probs into {path} ({len(rows)} pairs)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "judge", "pairs", "reflogp"])
    ap.add_argument("--prefix", default="dpo_v2", help="file prefix; one per sampling round")
    ap.add_argument("--items", default="", help="jsonl of passages to sample; default is the training stems")
    ap.add_argument("--start-kind", choices=["sft", "dpo"], default="sft")
    ap.add_argument("--tag", default="d24_core_attempt1")
    ap.add_argument("--start-step", type=int, default=20)
    ap.add_argument("--n-stems", type=int, default=1200, help="0 = every training stem")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-len-gap", type=float, default=0.25)
    ap.add_argument("--min-gap", type=float, default=4.0)
    ap.add_argument("--max-pairs-per-stem", type=int, default=2)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    args.temperature_note = args.temperature
    set_prefix(args.prefix)
    {"sample": cmd_sample, "judge": cmd_judge, "pairs": cmd_pairs, "reflogp": cmd_reflogp}[args.cmd](args)


if __name__ == "__main__":
    main()
