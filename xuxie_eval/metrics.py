#!/usr/bin/env python3
"""Automatic metrics for Gaokao-style continuation writing (Phase 0 freeze)."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EVAL = HERE / "eval.jsonl"


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def strip_opening(paragraph: str, opening: str) -> str:
    p = paragraph.strip()
    o = opening.strip()
    if p.startswith(o):
        return p[len(o) :].lstrip(" \t\n\r.,;:!-")
    o2 = o.rstrip(".!? ")
    if p.startswith(o2):
        return p[len(o2) :].lstrip(" \t\n\r.,;:!-")
    return p


def split_two_paragraphs(text: str) -> list[str] | None:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) == 2:
        return parts
    parts = [p.strip() for p in text.strip().split("\n") if p.strip()]
    if len(parts) == 2:
        return parts
    return None


def heavy_repetition(text: str, min_repeats: int = 3) -> bool:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if len(toks) < 15:
        return False
    grams = [" ".join(toks[i : i + 5]) for i in range(len(toks) - 4)]
    counts = Counter(grams)
    return any(c >= min_repeats for c in counts.values())


def score_continuation(item: dict, continuation: str) -> dict:
    openings_exact = False
    two_paragraphs = False
    body_words = 0
    paras = split_two_paragraphs(continuation)
    if paras is not None:
        two_paragraphs = True
        o1, o2 = item["opening1"].strip(), item["opening2"].strip()
        p1, p2 = paras
        starts1 = p1.startswith(o1) or p1.startswith(o1.rstrip(".!? "))
        starts2 = p2.startswith(o2) or p2.startswith(o2.rstrip(".!? "))
        openings_exact = starts1 and starts2
        body = strip_opening(p1, o1) + " " + strip_opening(p2, o2)
        body_words = word_count(body)
    else:
        body_words = word_count(continuation)

    lo = item.get("min_continuation_words", 100)
    hi = item.get("max_continuation_words", 220)

    return {
        "id": item["id"],
        "two_paragraphs": two_paragraphs,
        "openings_exact": openings_exact,
        "continuation_words": body_words,
        "word_count_ok": lo <= body_words <= hi,
        "passage_words": word_count(item.get("passage", "")),
        "heavy_repetition": heavy_repetition(continuation),
        "format_ok": two_paragraphs and openings_exact,
    }


def load_eval(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def aggregate(results: list[dict]) -> dict:
    n = len(results) or 1

    def rate(key):
        return sum(1 for r in results if r[key]) / n

    words = [r["continuation_words"] for r in results]
    words_sorted = sorted(words)
    mid = words_sorted[len(words_sorted) // 2] if words_sorted else 0
    passages = [r.get("passage_words", 0) for r in results]
    return {
        "n": len(results),
        "format_rate": rate("format_ok"),
        "word_count_ok_rate": rate("word_count_ok"),
        "heavy_repetition_rate": rate("heavy_repetition"),
        "median_continuation_words": mid,
        "mean_continuation_words": sum(words) / n if words else 0.0,
        "median_passage_words": sorted(passages)[len(passages) // 2] if passages else 0,
        "mean_passage_words": sum(passages) / n if passages else 0.0,
    }


def self_check(eval_path: Path) -> None:
    items = load_eval(eval_path)
    results = []
    missing_ref = 0
    for item in items:
        ref = item.get("reference_continuation")
        if not ref:
            missing_ref += 1
            continue
        results.append(score_continuation(item, ref))
    print(f"items={len(items)} with_reference={len(results)} missing_ref={missing_ref}")
    if not results:
        return
    print(json.dumps(aggregate(results), indent=2))
    bad = [r for r in results if not r["format_ok"] or not r["word_count_ok"]]
    if bad:
        print("references failing format/word checks:")
        for r in bad:
            print(f"  {r['id']}: body_words={r['continuation_words']} format={r['format_ok']}")
    pass_out = [
        r for r in results if not (270 <= r.get("passage_words", 0) <= 370)
    ]
    if pass_out:
        print("passages outside 270-370:")
        for r in pass_out:
            print(f"  {r['id']}: passage_words={r['passage_words']}")


def score_file(eval_path: Path, preds_path: Path) -> None:
    items = {it["id"]: it for it in load_eval(eval_path)}
    results = []
    with preds_path.open(encoding="utf-8") as f:
        for line in f:
            pred = json.loads(line)
            item = items[pred["id"]]
            results.append(score_continuation(item, pred["continuation"]))
    print(json.dumps({"per_item": results, "aggregate": aggregate(results)}, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--self-check", action="store_true", help="score reference_continuation fields")
    ap.add_argument("--preds", type=Path, default=None, help="jsonl of {id, continuation}")
    args = ap.parse_args()
    if args.self_check:
        self_check(args.eval)
    elif args.preds:
        score_file(args.eval, args.preds)
    else:
        ap.error("use --self-check or --preds")


if __name__ == "__main__":
    main()
