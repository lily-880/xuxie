"""Keep the two given openings fixed. The model only writes the words after them.

All samples for one prompt are written in a single batched `Engine.generate` call.
When a row finishes its first body, opening 2 is forced into that row, so the
repetition penalty sees both bodies.
"""

from __future__ import annotations

import re

from nanochat.tokenizer import SPECIAL_TOKENS

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_BREAK = re.compile(r"[ \t]*\n+\s*")
_SENT_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*$")


def _words(text: str) -> int:
    return len(_WORD.findall(text))


def _body_after(text: str, opening: str) -> str:
    text = text.strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    if opening and text.startswith(opening):
        text = text[len(opening):].lstrip()
    return text.strip()


def body_word_count(text: str, opening1: str, opening2: str) -> int:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) >= 2:
        return _words(_body_after(parts[0], opening1.strip())) + _words(_body_after(parts[1], opening2.strip()))
    return _words(_body_after(text, opening1.strip()))


def _cut_at_break(text: str, min_para_words: int) -> tuple[str, bool]:
    """Line breaks before `min_para_words` words are joined into the paragraph.
    Returns the paragraph text and whether a later break ended it."""
    parts = _BREAK.split(text)
    acc = parts[0].strip()
    for nxt in parts[1:]:
        if _words(acc) >= min_para_words:
            return acc, True
        acc = (acc + " " + nxt.strip()).strip()
    return acc, False


def _sentence_done(text: str) -> bool:
    t = text.rstrip()
    if not _SENT_END.search(t):
        return False
    return t.count('"') % 2 == 0 and t.count("\u201c") == t.count("\u201d")


class _Row:
    __slots__ = ("phase", "ids", "bodies", "done")

    def __init__(self):
        self.phase = 0  # 0 = first body, 1 = second body
        self.ids: list[list[int]] = [[], []]
        self.bodies = ["", ""]
        self.done = False


def _sample_rows(
    engine,
    tokenizer,
    prompt_tokens: list[int],
    opening1: str,
    opening2: str,
    *,
    num_samples: int,
    seed: int,
    budget: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    min_para_words: int,
    hard_extra: int = 40,
) -> list[str]:
    o1, o2 = opening1.strip(), opening2.strip()
    o2_ids = tokenizer.encode(o2)
    break_ids = tokenizer.encode("\n\n")
    end_id = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    special = {tokenizer.encode_special(s) for s in SPECIAL_TOKENS}
    rows = [_Row() for _ in range(num_samples)]

    def close_body(row: _Row, text: str, forced: list[int]) -> list[int]:
        row.bodies[row.phase] = text
        if row.phase == 0:
            row.phase = 1
            return forced + o2_ids
        row.done = True
        return [end_id]

    def on_token(i: int, tok: int):
        row = rows[i]
        if row.done:
            return None
        if tok in special:
            row.bodies[row.phase] = _cut_at_break(tokenizer.decode(row.ids[row.phase]), min_para_words)[0]
            row.done = True
            return None if tok in (end_id, bos) else [end_id]
        ids = row.ids[row.phase]
        ids.append(tok)
        text, at_break = _cut_at_break(tokenizer.decode(ids), min_para_words)
        if at_break:
            return close_body(row, text, [])
        if len(ids) >= budget and (_sentence_done(text) or len(ids) >= budget + hard_extra):
            return close_body(row, text, list(break_ids))
        return None

    context = list(prompt_tokens) + tokenizer.encode(o1)
    max_tokens = 2 * (budget + hard_extra) + len(break_ids) + len(o2_ids) + 2
    seqs = [list(context) for _ in range(num_samples)]
    masks = [[0] * len(context) for _ in range(num_samples)]
    closed = [False] * num_samples
    for token_column, token_masks in engine.generate(
        context,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        repetition_penalty=repetition_penalty,
        on_token=on_token,
    ):
        for i, (tok, m) in enumerate(zip(token_column, token_masks)):
            if closed[i]:
                continue
            if tok == end_id:
                closed[i] = True
                continue
            seqs[i].append(tok)
            masks[i].append(m)
        if all(closed):
            break

    out = []
    for i, row in enumerate(rows):
        if not row.done:
            row.bodies[row.phase] = _cut_at_break(tokenizer.decode(row.ids[row.phase]), min_para_words)[0]
        b1, b2 = (b.strip() for b in row.bodies)
        p1 = f"{o1} {b1}" if b1 else o1
        p2 = f"{o2} {b2}" if b2 else o2
        out.append((f"{p1}\n\n{p2}", seqs[i], masks[i]))
    return out


def generate_locked_batch(
    engine,
    tokenizer,
    prompt_tokens: list[int],
    opening1: str,
    opening2: str,
    *,
    num_samples: int = 1,
    max_body_tokens: int = 110,
    temperature: float = 0.6,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    seed: int = 42,
    min_body_words: int = 100,
    max_body_words: int = 220,
    min_para_words: int = 50,
    retries: int = 4,
) -> list[str]:
    """prompt_tokens must already end at <|assistant_start|>.

    A line break ends a body only after it already has about `min_para_words`
    words. A body that reaches `max_body_tokens` ends at the next sentence end.
    Samples whose two bodies fall outside the word band are written again after
    the same openings, up to `retries` attempts in total.
    """
    kwargs = dict(
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        min_para_words=min_para_words,
    )
    texts = [t for t, _, _ in _sample_rows(
        engine, tokenizer, prompt_tokens, opening1, opening2,
        num_samples=num_samples, seed=seed, budget=max_body_tokens, **kwargs,
    )]
    counts = [body_word_count(t, opening1, opening2) for t in texts]

    def in_band(n: int) -> bool:
        return min_body_words <= n <= max_body_words

    for attempt in range(1, retries):
        todo = [i for i, n in enumerate(counts) if not in_band(n)]
        if not todo:
            break
        fresh = [t for t, _, _ in _sample_rows(
            engine, tokenizer, prompt_tokens, opening1, opening2,
            num_samples=len(todo), seed=seed + 10 * attempt,
            budget=max_body_tokens + 40 * attempt, **kwargs,
        )]
        for i, text in zip(todo, fresh):
            n = body_word_count(text, opening1, opening2)
            if in_band(n) or abs(n - 150) < abs(counts[i] - 150):
                texts[i], counts[i] = text, n
    return texts


def generate_locked_continuation(
    engine,
    tokenizer,
    prompt_tokens: list[int],
    opening1: str,
    opening2: str,
    *,
    max_body_tokens: int = 110,
    temperature: float = 0.6,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    seed: int = 42,
    min_body_words: int = 100,
    max_body_words: int = 220,
    min_para_words: int = 50,
    retries: int = 4,
) -> str:
    return generate_locked_batch(
        engine, tokenizer, prompt_tokens, opening1, opening2,
        num_samples=1,
        max_body_tokens=max_body_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        min_body_words=min_body_words,
        max_body_words=max_body_words,
        min_para_words=min_para_words,
        retries=retries,
    )[0]
