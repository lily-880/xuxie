#!/usr/bin/env python3
"""One ruler for 读后续写 checkpoints.

Writes K continuations per eval item with locked openings (deployment decoding),
scores them with rules and the graded DeepSeek judge, and compares against a
baseline run item by item.

    python dev/d24_core_run/xuxie/eval_xuxie.py --kind ref --name ref
    python dev/d24_core_run/xuxie/eval_xuxie.py --kind sft --step 20 --name sft20
    python dev/d24_core_run/xuxie/eval_xuxie.py --kind sft --step 20 --seeds 8 --n-items 40 --canary 0 --name headroom_sft20
    python dev/d24_core_run/xuxie/eval_xuxie.py --kind dpo --tag d24_xuxie_dpo_v7 --step 40 --name dpo7 --baseline sft20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xuxie_common as xc  # noqa: E402

OUT_DIR = xc.REPORTS / "eval"


def keep_two_paragraphs(text: str) -> str:
    import re

    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) >= 2:
        return parts[0] + "\n\n" + parts[1]
    lines = [p.strip() for p in text.split("\n") if p.strip()]
    if len(lines) >= 2:
        return lines[0] + "\n\n" + lines[1]
    return text.strip()


def generate(args, items: list[dict]) -> tuple[list[dict], list[dict], str]:
    from nanochat.common import autodetect_device_type, compute_init
    from nanochat.engine import Engine
    from decode_locked import generate_locked_batch
    import torch

    device_type = args.device or autodetect_device_type()
    compute_init(device_type)
    device = torch.device(device_type)
    model, tokenizer, _meta = xc.load_checkpoint(args.kind, args.tag, args.step, device)
    engine = Engine(model, tokenizer)
    label = f"{args.kind}:{args.tag}:{args.step}"
    decode = dict(xc.DECODE, temperature=args.temperature)

    samples = []
    t0 = time.time()
    for n, item in enumerate(items, 1):
        texts = generate_locked_batch(
            engine, tokenizer, xc.prompt_tokens(tokenizer, item), item["opening1"], item["opening2"],
            num_samples=args.seeds, seed=1000 + 17 * item["_index"], **decode,
        )
        for s, text in enumerate(texts):
            samples.append({"id": item["id"], "seed": s, "text": text})
        print(f"[{n}/{len(items)}] {item['id']} {time.time() - t0:.0f}s", flush=True)

    canary = []
    for item in xc.spread(items, args.canary) if args.canary > 0 else []:
        results, _ = engine.generate_batch(
            xc.prompt_tokens(tokenizer, item), num_samples=1, max_tokens=300,
            temperature=decode["temperature"], top_k=decode["top_k"], seed=7,
            repetition_penalty=decode["repetition_penalty"],
        )
        n_prompt = len(xc.prompt_tokens(tokenizer, item))
        text = keep_two_paragraphs(tokenizer.decode(results[0][n_prompt:]))
        canary.append({"id": item["id"], "text": text})
    return samples, canary, label


def summarize(items_by_id: dict, samples: list[dict], canary: list[dict]) -> dict:
    per_item_clean: dict[str, list[float]] = {}
    per_item_score: dict[str, list[float]] = {}
    for s in samples:
        j = s.get("judge")
        if j is None:
            continue
        per_item_clean.setdefault(s["id"], []).append(float(xc.is_clean(j)))
        per_item_score.setdefault(s["id"], []).append(float(j["score"]))
    judged = [s for s in samples if s.get("judge") is not None]
    clean_items = [xc.mean(v) for v in per_item_clean.values()]
    score_items = [xc.mean(v) for v in per_item_score.values()]

    def rate(key, pool):
        return xc.mean(float(bool(key(s))) for s in pool) if pool else float("nan")

    out = {
        "n_samples": len(samples),
        "n_judged": len(judged),
        "n_items": len({s["id"] for s in samples}),
        "clean": xc.bootstrap_ci(clean_items),
        "score": xc.bootstrap_ci(score_items),
        "pass_at_k": xc.mean(float(any(v)) for v in per_item_clean.values()) if per_item_clean else float("nan"),
        "k": max((len(v) for v in per_item_clean.values()), default=0),
        "replay": rate(lambda s: s["judge"]["replay"], judged),
        "major": rate(lambda s: s["judge"]["contradiction"] == "major", judged),
        "minor": rate(lambda s: s["judge"]["contradiction"] == "minor", judged),
        "advances_plot": xc.mean(s["judge"]["advances_plot"] for s in judged) if judged else float("nan"),
        "fits_openings": xc.mean(s["judge"]["fits_openings"] for s in judged) if judged else float("nan"),
        "resolution": xc.mean(s["judge"]["resolution"] for s in judged) if judged else float("nan"),
        "language": xc.mean(s["judge"]["language"] for s in judged) if judged else float("nan"),
        "word_ok": rate(lambda s: s["rules"]["word_ok"], samples),
        "heavy_rep": rate(lambda s: s["rules"]["heavy_rep"], samples),
        "long_copy": rate(lambda s: s["rules"]["long_copy"], samples),
        "copy8": xc.mean(s["rules"]["copy8"] for s in samples),
        "o2_leak": rate(lambda s: s["rules"]["o2_leak"], samples),
        "garbage": rate(lambda s: s["rules"]["garbage"], samples),
        "cut": rate(lambda s: s["rules"]["cut"], samples),
        "median_words": statistics.median(s["rules"]["words"] for s in samples) if samples else 0,
    }
    by_source: dict[str, dict] = {}
    for src in sorted({items_by_id[s["id"]].get("source", "?") for s in judged}):
        pool = [s for s in judged if items_by_id[s["id"]].get("source", "?") == src]
        by_source[src] = {
            "n": len(pool),
            "clean": rate(lambda s: xc.is_clean(s["judge"]), pool),
            "score": xc.mean(s["judge"]["score"] for s in pool),
        }
    out["by_source"] = by_source
    if canary:
        out["canary_format"] = rate(lambda c: c["rules"]["format_ok"], canary)
        out["canary_garbage"] = rate(lambda c: c["rules"]["garbage"], canary)
        out["canary_n"] = len(canary)
    return out


def per_item(samples: list[dict], fn) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for s in samples:
        if s.get("judge") is not None:
            acc.setdefault(s["id"], []).append(fn(s))
    return {k: xc.mean(v) for k, v in acc.items()}


def compare(new: list[dict], old: list[dict], items_by_id: dict) -> dict:
    out = {}
    for name, fn in (
        ("clean", lambda s: float(xc.is_clean(s["judge"]))),
        ("score", lambda s: float(s["judge"]["score"])),
        ("replay", lambda s: float(s["judge"]["replay"])),
        ("major", lambda s: float(s["judge"]["contradiction"] == "major")),
    ):
        out[name] = xc.paired_delta_ci(per_item(new, fn), per_item(old, fn))
    clean = lambda s: float(xc.is_clean(s["judge"]))  # noqa: E731
    source_of = {it["id"]: it.get("source") for it in xc.load_eval_items()}
    for src in ("gaokao", "mock"):
        pick = lambda pool, src=src: [s for s in pool if source_of.get(s["id"]) == src]  # noqa: E731
        out[f"clean_{src}"] = xc.paired_delta_ci(per_item(pick(new), clean), per_item(pick(old), clean))
    return out


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def ci_pct(t) -> str:
    m, lo, hi = t
    return "n/a" if m != m else f"{100 * m:.1f}% ({100 * lo:.1f}–{100 * hi:.1f})"


def ci_num(t) -> str:
    m, lo, hi = t
    return "n/a" if m != m else f"{m:.2f} ({lo:.2f}–{hi:.2f})"


def write_markdown(path: Path, data: dict, items_by_id: dict, baseline: dict | None) -> None:
    s = data["summary"]
    lines = [
        f"# 评测 · {data['name']}",
        "",
        f"- 模型：`{data['label']}`",
        f"- 题目：{s['n_items']} 道 × 每题 {data['seeds']} 篇，共 {s['n_samples']} 篇；裁判判了 {s['n_judged']} 篇",
        f"- 解码：开头写死，temperature={data['decode']['temperature']}，top_k={data['decode']['top_k']}，"
        f"重复惩罚 {data['decode']['repetition_penalty']}，正文不在 {data['decode']['min_body_words']}–"
        f"{data['decode']['max_body_words']} 词就重写，最多 {data['decode']['retries']} 次",
        "- 干净 = 裁判判定没有重讲原文、没有严重冲突。括号里是 95% 置信区间（按题目重抽样）。",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 干净率 | {ci_pct(s['clean'])} |",
        f"| 至少一篇干净的题目（{s['k']} 篇里） | {pct(s['pass_at_k'])} |",
        f"| 重讲原文 | {pct(s['replay'])} |",
        f"| 严重冲突 | {pct(s['major'])} |",
        f"| 轻微冲突 | {pct(s['minor'])} |",
        f"| 裁判分（满分 25） | {ci_num(s['score'])} |",
        f"| 推进情节 / 接开头 / 收尾 / 语言（各 0–2） | {s['advances_plot']:.2f} / {s['fits_openings']:.2f} / {s['resolution']:.2f} / {s['language']:.2f} |",
        f"| 词数合格（100–220） | {pct(s['word_ok'])} |",
        f"| 中位词数 | {s['median_words']} |",
        f"| 严重复读 | {pct(s['heavy_rep'])} |",
        f"| 抄原文 12 词以上 | {pct(s['long_copy'])} |",
        f"| 原文 8-gram 抄写率（平均） | {pct(s['copy8'])} |",
        f"| 第二段开头漏进第一段 | {pct(s['o2_leak'])} |",
        f"| 句子被截断 | {pct(s['cut'])} |",
        f"| 乱码 | {pct(s['garbage'])} |",
    ]
    if "canary_format" in s:
        lines.append(f"| 金丝雀：自己写开头且格式对（{s['canary_n']} 题） | {pct(s['canary_format'])} |")
        lines.append(f"| 金丝雀：乱码 | {pct(s['canary_garbage'])} |")
    source_names = {"gaokao": "高考真题", "mock": "模拟题", "joint": "联考题"}
    for src, v in s.get("by_source", {}).items():
        lines.append(f"| {source_names.get(src, src)}（{v['n']} 篇）：干净率 / 裁判分 | {pct(v['clean'])} / {v['score']:.1f} |")
    if data.get("compare"):
        c = data["compare"]
        lines += [
            "",
            f"## 和 `{data['baseline_name']}` 逐题对比（新 − 旧）",
            "",
            "| 指标 | 变化 |",
            "|---|---|",
            f"| 干净率 | {ci_pct(c['clean'])} |",
            f"| 裁判分 | {ci_num(c['score'])} |",
            f"| 重讲原文 | {ci_pct(c['replay'])} |",
            f"| 严重冲突 | {ci_pct(c['major'])} |",
            f"| 干净率 · 高考真题 | {ci_pct(c['clean_gaokao'])} |",
            f"| 干净率 · 模拟题 | {ci_pct(c['clean_mock'])} |",
        ]
    base_by_id = {}
    if baseline:
        for b in baseline["samples"]:
            if b["seed"] == 0:
                base_by_id[b["id"]] = b
    shown = 0
    lines += ["", "## 样例（每题第一篇）", ""]
    for sample in data["samples"]:
        if sample["seed"] != 0 or shown >= 6:
            continue
        item = items_by_id[sample["id"]]
        shown += 1
        lines += [
            f"### {item['id']} · {item.get('theme', '')}",
            "",
            "原文：",
            "```",
            item["passage"].strip(),
            "```",
            f"开头 1：{item['opening1'].strip()}",
            "",
            f"开头 2：{item['opening2'].strip()}",
            "",
        ]
        for tag, smp in (("本模型", sample), (f"基线 {data.get('baseline_name', '')}", base_by_id.get(sample["id"]))):
            if smp is None:
                continue
            j = smp.get("judge")
            verdict = (
                f"干净={xc.is_clean(j)} · 重讲={j['replay']} · 冲突={j['contradiction']} · 分={j['score']} · {j['reason']}"
                if j else "未评"
            )
            lines += [f"**{tag}**：{verdict}", "", "```", smp["text"].strip(), "```", ""]
            if j and j.get("evidence"):
                lines += [f"> 裁判摘出的句子：{j['evidence']}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sft", "dpo", "base", "ref"], default="sft")
    ap.add_argument("--tag", default="d24_core_attempt1")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=4, help="continuations per item")
    ap.add_argument("--n-items", type=int, default=0, help="0 = all 80, else spread this many")
    ap.add_argument("--source", default="", help="only items with this source, e.g. gaokao")
    ap.add_argument("--canary", type=int, default=8, help="items where the model also writes the openings itself")
    ap.add_argument("--temperature", type=float, default=xc.DECODE["temperature"])
    ap.add_argument("--baseline", default="", help="name of an earlier run in reports/eval")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--rejudge-only", action="store_true", help="reuse saved generations")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{args.name}.json"
    out_md = OUT_DIR / f"{args.name}.md"
    all_items = xc.load_eval_items()
    if args.source:
        all_items = [it for it in all_items if it.get("source") == args.source]
    for i, it in enumerate(all_items):
        it["_index"] = i
    items = xc.spread(all_items, args.n_items) if args.n_items else all_items
    items_by_id = {it["id"]: it for it in all_items}

    if args.rejudge_only:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        samples, canary, label = data["samples"], data["canary"], data["label"]
    elif args.kind == "ref":
        samples = [{"id": it["id"], "seed": 0, "text": it["reference_continuation"].strip()} for it in items]
        canary, label = [], "human reference"
        args.seeds = 1
    else:
        samples, canary, label = generate(args, items)

    for s in samples:
        s["rules"] = xc.rule_metrics(items_by_id[s["id"]], s["text"])
    for c in canary:
        c["rules"] = xc.rule_metrics(items_by_id[c["id"]], c["text"])
    data = {
        "name": args.name,
        "label": label,
        "seeds": args.seeds,
        "decode": dict(xc.DECODE, temperature=args.temperature),
        "samples": samples,
        "canary": canary,
    }
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.no_judge:
        judge = xc.Judge()
        verdicts = judge.many([(items_by_id[s["id"]], s["text"]) for s in samples], workers=args.workers)
        for s, j in zip(samples, verdicts):
            s["judge"] = j

    data["summary"] = summarize(items_by_id, samples, canary)
    baseline = None
    if args.baseline:
        baseline = json.loads((OUT_DIR / f"{args.baseline}.json").read_text(encoding="utf-8"))
        data["baseline_name"] = args.baseline
        if not args.no_judge:
            data["compare"] = compare(samples, baseline["samples"], items_by_id)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(out_md, data, items_by_id, baseline)

    s = data["summary"]
    print(json.dumps({k: v for k, v in s.items()}, ensure_ascii=False))
    if data.get("compare"):
        print("compare", json.dumps(data["compare"]))
    print(f"wrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
