"""Shared pieces for 读后续写 evaluation and DPO.

Prompts, checkpoint loading, rule checks, the graded DeepSeek judge, bootstrap
intervals, and the token rendering used for DPO log-probabilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

XUXIE = Path(__file__).resolve().parent
D24 = XUXIE.parent
REPO = D24.parents[1]
for _p in (str(REPO), str(XUXIE), str(D24 / "xuxie_amplify"), str(D24 / "xuxie_eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tasks.continuation import _user_content  # noqa: E402
from metrics import score_continuation, split_two_paragraphs, strip_opening  # noqa: E402

EVAL_PATH = D24 / "xuxie_eval" / "eval.jsonl"
REPORTS = XUXIE / "reports"
TASK_DATA = Path.home() / ".cache" / "nanochat" / "task_data" / "xuxie"
TRAIN_PATH = TASK_DATA / "train_v1_clean.jsonl"
JUDGE_CACHE = TASK_DATA / "judge_cache_plot_v2.jsonl"

# Deployment decoding. Evaluation, guards and the headroom test all use this.
DECODE = dict(temperature=0.6, top_k=50, repetition_penalty=1.2, max_body_tokens=110,
              min_body_words=100, max_body_words=220, min_para_words=50, retries=4)


# -----------------------------------------------------------------------------
# items and prompts

def load_eval_items() -> list[dict]:
    return [json.loads(l) for l in EVAL_PATH.open(encoding="utf-8") if l.strip()]


def parse_train_row(row: dict) -> dict:
    user = row["messages"][0]["content"]
    o1 = o2 = ""
    for ln in user.split("\n"):
        if ln.startswith("Paragraph 1 opening:"):
            o1 = ln.split(":", 1)[1].strip()
        elif ln.startswith("Paragraph 2 opening:"):
            o2 = ln.split(":", 1)[1].strip()
    return {
        "id": row.get("stem_id") or row["id"],
        "theme": row.get("theme"),
        "passage": user.split("Paragraph 1 opening:")[0].strip(),
        "opening1": o1,
        "opening2": o2,
        "teacher": row["messages"][1]["content"].strip(),
    }


def load_train_stems() -> list[dict]:
    """One item per stem, in file order."""
    seen: dict[str, dict] = {}
    for line in TRAIN_PATH.open(encoding="utf-8"):
        if line.strip():
            item = parse_train_row(json.loads(line))
            seen.setdefault(item["id"], item)
    return list(seen.values())


def spread(items: list, n: int) -> list:
    if n >= len(items):
        return list(items)
    if n <= 1:
        return items[:n]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def user_text(item: dict) -> str:
    return _user_content(item["passage"], item["opening1"], item["opening2"])


def prompt_tokens(tokenizer, item: dict) -> list[int]:
    return [
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<|user_start|>"),
        *tokenizer.encode(user_text(item)),
        tokenizer.encode_special("<|user_end|>"),
        tokenizer.encode_special("<|assistant_start|>"),
    ]


# -----------------------------------------------------------------------------
# checkpoints

def load_checkpoint(kind: str, tag: str, step: int, device, phase: str = "eval"):
    """kind: sft | dpo | base. Returns (model, tokenizer, meta)."""
    import torch
    from nanochat.checkpoint_manager import load_model, load_model_from_dir
    from nanochat.common import get_base_dir

    if kind == "dpo":
        out = load_model_from_dir(os.path.join(get_base_dir(), "dpo_checkpoints"), device,
                                  phase=phase, model_tag=tag, step=step)
    else:
        out = load_model(kind, device, phase=phase, model_tag=tag, step=step)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # drop the throwaway init weights the loader allocated first
    return out


# -----------------------------------------------------------------------------
# rule checks

_SENT_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*$")
_ALLOWED_NON_ASCII = set("\u201c\u201d\u2018\u2019\u2014\u2013\u2026")


def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def split_bodies(item: dict, text: str) -> tuple[str, str]:
    """Body text after each opening. The locked decoder writes 'opening body'."""
    o1, o2 = item["opening1"].strip(), item["opening2"].strip()
    paras = split_two_paragraphs(text)
    if paras is None:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        paras = (parts + ["", ""])[:2] if parts else ["", ""]

    def body(p: str, o: str) -> str:
        return p[len(o):].strip() if p.startswith(o) else strip_opening(p, o)

    return body(paras[0], o1), body(paras[1], o2)


def rule_metrics(item: dict, text: str) -> dict:
    sc = score_continuation(item, text)
    b1, b2 = split_bodies(item, text)
    pw, bw = _toks(item["passage"]), _toks(b1 + " " + b2)
    p8 = {tuple(pw[i:i + 8]) for i in range(len(pw) - 7)}
    g8 = [tuple(bw[i:i + 8]) for i in range(len(bw) - 7)]
    copy8 = sum(g in p8 for g in g8) / max(len(g8), 1)
    p12 = {tuple(pw[i:i + 12]) for i in range(len(pw) - 11)}
    long_copy = any(tuple(bw[i:i + 12]) in p12 for i in range(len(bw) - 11))
    o2w = _toks(item["opening2"])[:6]
    b1w = _toks(b1)
    o2_leak = len(o2w) >= 3 and any(b1w[i:i + len(o2w)] == o2w for i in range(len(b1w) - len(o2w) + 1))
    odd = sum(1 for ch in text if ord(ch) > 126 and ch not in _ALLOWED_NON_ASCII)
    garbage = (
        "<|" in text
        or odd > 2
        or bool(re.search(r"\.\d+\.", text))
        or bool(re.search(r"[A-Za-z]{25,}", text))
    )
    cut = not (b1 and b2 and _SENT_END.search(b1.rstrip()) and _SENT_END.search(b2.rstrip()))
    return {
        "format_ok": sc["format_ok"],
        "words": sc["continuation_words"],
        "word_ok": sc["word_count_ok"],
        "heavy_rep": sc["heavy_repetition"],
        "copy8": round(copy8, 4),
        "long_copy": long_copy,
        "o2_leak": o2_leak,
        "garbage": garbage,
        "cut": cut,
    }


# -----------------------------------------------------------------------------
# graded judge

JUDGE_VERSION = "plot_v2"
JUDGE_SYSTEM = """You grade one student continuation for a Gaokao English continuation-writing task.
The student read a PASSAGE (the first part of a story) and two fixed paragraph openings, then wrote two paragraphs that should carry the story on from where the passage stops.

Judge these points:

1. replay (true/false): true only if the continuation narrates again events that the passage has ALREADY narrated, as if they were happening again or for the first time (for example the passage already found the lost dog, and the continuation finds the same dog again from the start), or if it copies whole sentences of the passage. Recalling or briefly mentioning earlier events, thanking someone for them, or describing feelings about them is NOT replay. Picking up a scene the passage left unfinished is NOT replay.

2. contradiction ("none" / "minor" / "major"): "major" when the continuation clearly conflicts with facts fixed by the passage, by the openings, or by its own earlier sentences: who someone is, what has happened, where people or objects are, what is physically possible (a phone left behind is suddenly in her hand, a person who left is present with no explanation, an animal changes kind, a finished event is undone). "minor" for small slips that do not break the story. "none" otherwise.

3. advances_plot (0-2): 0 = no real progress (stalls, loops, only description); 1 = some progress; 2 = clearly moves the story on from where the passage stops.

4. fits_openings (0-2): 0 = the paragraphs ignore their openings; 1 = loose link; 2 = paragraph 1 grows out of opening 1 and leads into opening 2, and paragraph 2 grows out of opening 2.

5. resolution (0-2): 0 = no ending or a broken one; 1 = some ending; 2 = a clear ending that fits the story and its theme.

6. language (0-2): 0 = many errors or hard to follow; 1 = understandable with errors; 2 = mostly correct and natural.

7. score (0-25): the Gaokao mark for the whole continuation. Band 5 = 21-25, band 4 = 16-20, band 3 = 11-15, band 2 = 6-10, band 1 = 0-5.

Reply with JSON only:
{"replay": true or false, "contradiction": "none" or "minor" or "major", "evidence": "the offending sentence copied exactly from the continuation, or an empty string", "advances_plot": 0, "fits_openings": 0, "resolution": 0, "language": 0, "score": 0, "reason": "one short sentence"}"""


def judge_user(item: dict, text: str) -> str:
    return (
        f"PASSAGE:\n{item['passage'].strip()}\n\n"
        f"OPENING 1: {item['opening1'].strip()}\n"
        f"OPENING 2: {item['opening2'].strip()}\n\n"
        f"STUDENT CONTINUATION:\n{text.strip()}"
    )


def _as_bool(x) -> bool:
    return x is True or str(x).strip().lower() in ("true", "yes", "1")


def _as_int(x, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(x)))))
    except (TypeError, ValueError):
        return lo


def normalize_judgment(d: dict) -> dict:
    c = str(d.get("contradiction", "none")).strip().lower()
    if c not in ("none", "minor", "major"):
        c = "major" if _as_bool(d.get("contradiction")) else "none"
    return {
        "replay": _as_bool(d.get("replay")),
        "contradiction": c,
        "evidence": str(d.get("evidence") or "")[:400],
        "advances_plot": _as_int(d.get("advances_plot"), 0, 2),
        "fits_openings": _as_int(d.get("fits_openings"), 0, 2),
        "resolution": _as_int(d.get("resolution"), 0, 2),
        "language": _as_int(d.get("language"), 0, 2),
        "score": _as_int(d.get("score"), 0, 25),
        "reason": str(d.get("reason") or "")[:400],
    }


def is_clean(j: dict | None) -> bool:
    return j is not None and not j["replay"] and j["contradiction"] != "major"


def quality(j: dict) -> float:
    """Ranking score for picking the better of two samples."""
    return j["score"] + 2 * (j["advances_plot"] + j["fits_openings"] + j["resolution"]) + j["language"]


class Judge:
    """DeepSeek judge with an on-disk cache keyed by (version, model, inputs)."""

    def __init__(self, cache_path: Path = JUDGE_CACHE, model: str | None = None):
        from client import DeepSeekClient, load_env

        env = load_env()
        if model:
            env["DEEPSEEK_MODEL"] = model
        self.client = DeepSeekClient(env)
        self.model = self.client.model
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["judgment"]

    def key(self, item: dict, text: str) -> str:
        blob = "\x1f".join([JUDGE_VERSION, self.model, item["passage"].strip(), item["opening1"].strip(),
                            item["opening2"].strip(), text.strip()])
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def __call__(self, item: dict, text: str) -> dict:
        from prompts import parse_json_object

        k = self.key(item, text)
        if k in self.cache:
            return self.cache[k]
        last: Exception | None = None
        for _ in range(3):
            try:
                raw = self.client.chat(
                    [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_user(item, text)}],
                    temperature=0.0,
                    max_tokens=320,
                )
                j = normalize_judgment(parse_json_object(raw))
                break
            except Exception as e:  # malformed JSON or network trouble
                last = e
        else:
            raise RuntimeError(f"judge failed: {last}")
        with self.lock:
            self.cache[k] = j
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "judgment": j}, ensure_ascii=False) + "\n")
        return j

    def many(self, jobs: list[tuple[dict, str]], workers: int = 24, log_every: int = 50) -> list[dict | None]:
        out: list[dict | None] = [None] * len(jobs)
        done = 0

        def run(i: int):
            item, text = jobs[i]
            try:
                return i, self(item, text)
            except Exception as e:
                print(f"judge error on {item.get('id')}: {e}", flush=True)
                return i, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, j in ex.map(run, range(len(jobs))):
                out[i] = j
                done += 1
                if log_every and done % log_every == 0:
                    print(f"judged {done}/{len(jobs)}", flush=True)
        return out


# -----------------------------------------------------------------------------
# statistics

def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(per_item: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over items with a 95% interval from resampling items."""
    if not per_item:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(per_item)
    boots = sorted(mean(per_item[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return mean(per_item), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def paired_delta_ci(new: dict[str, float], old: dict[str, float], n_boot: int = 2000, seed: int = 0):
    ids = sorted(set(new) & set(old))
    return bootstrap_ci([new[i] - old[i] for i in ids], n_boot=n_boot, seed=seed)


# -----------------------------------------------------------------------------
# DPO rendering and log-probabilities

def render_answer(tokenizer, item: dict, text: str, max_tokens: int = 2048) -> tuple[list[int], list[int]]:
    """Prompt and openings are mask 0. Body words, the paragraph break and the end token are mask 1,
    matching what the model itself decides under locked-opening decoding."""
    o1, o2 = item["opening1"].strip(), item["opening2"].strip()
    b1, b2 = split_bodies(item, text)
    ids = prompt_tokens(tokenizer, item)
    mask = [0] * len(ids)

    def add(tok_ids: list[int], m: int):
        ids.extend(tok_ids)
        mask.extend([m] * len(tok_ids))

    add(tokenizer.encode(o1), 0)
    if b1:
        add(tokenizer.encode(" " + b1), 1)
    add(tokenizer.encode("\n\n"), 1)
    add(tokenizer.encode(o2), 0)
    if b2:
        add(tokenizer.encode(" " + b2), 1)
    add([tokenizer.encode_special("<|assistant_end|>")], 1)
    return ids[:max_tokens], mask[:max_tokens]


def seq_logprob(model, seqs: list[tuple[list[int], list[int]]], pad_id: int):
    """Summed log-probability and token count of the mask-1 tokens of each sequence."""
    import torch

    device = model.get_device()
    T = max(len(ids) for ids, _ in seqs)
    idx = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    tgt = torch.full((len(seqs), T), -1, dtype=torch.long)
    for b, (ids, mask) in enumerate(seqs):
        idx[b, :len(ids)] = torch.tensor(ids)
        t = torch.tensor(ids[1:] + [pad_id])
        m = torch.tensor(mask[1:] + [0], dtype=torch.bool)
        tgt[b, :len(ids)] = torch.where(m, t, torch.full_like(t, -1))
    idx, tgt = idx.to(device), tgt.to(device)
    nll = model(idx, tgt, loss_reduction="none").view(len(seqs), T)
    counts = (tgt >= 0).sum(1)
    return -nll.sum(1), counts
