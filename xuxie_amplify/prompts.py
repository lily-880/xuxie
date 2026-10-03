#!/usr/bin/env python3
"""Prompts for stem rewrite + continuation writing."""

from __future__ import annotations

import json
import re


STEM_SYSTEM = """You rewrite Gaokao-style English "continuation writing" reading passages.
Output STRICT JSON only, no markdown fences.
Schema:
{"passage":"...","opening1":"...","opening2":"...","theme":"family|animal|danger|kindness|growth|community"}
Rules:
- Keep similar length to the seed (~280-360 English words). Write a COMPLETE original narrative; do NOT pad with repeated scenic filler sentences.
- NEVER repeat the same sentence or near-duplicate sentence in the passage.
- Change names, places, and the concrete conflict; keep the genre (narrative story).
- opening1 and opening2 MUST use the SAME character names that appear in the passage (no new unexplained names).
- Two paragraph openings must be natural English first sentences for a two-paragraph continuation.
- Do NOT include the continuation essays.
- No Chinese in passage/openings except rare gloss in parentheses if already common in exams.
- theme must be one of the six tags above.
"""


CONT_SYSTEM = """You write Gaokao English continuation writing (读后续写).
Output ONLY the two continuation paragraphs.
Rules:
- Paragraph 1 MUST start with opening1 EXACTLY (copy character-for-character).
- Paragraph 2 MUST start with opening2 EXACTLY.
- Separate the two paragraphs with a blank line.
- CRITICAL length: body words EXCLUDING the two openings must be 130-160 (aim 145). Never exceed 170.
- Prefer shorter sentences; do not pad with adjectives or moral lectures.
- Stay consistent with characters, place, and facts in the passage.
- Positive, coherent ending; written English; no heavy repetition; do not copy long spans from the passage.
- No title, no analysis, no markdown, no word count note.
"""


def stem_user(seed: dict, variant: int) -> str:
    return (
        f"Rewrite variant #{variant} of this seed into a NEW story.\n"
        f"Seed theme hint: {seed.get('theme')}\n"
        f"Seed paper label: {seed.get('paper')}\n\n"
        f"PASSAGE:\n{seed['passage']}\n\n"
        f"OPENING1: {seed['opening1']}\n"
        f"OPENING2: {seed['opening2']}\n\n"
        "Return JSON only."
    )


def cont_user(stem: dict, *, shorten_hint: str | None = None) -> str:
    extra = ""
    if shorten_hint:
        extra = f"\n\nIMPORTANT: Your previous draft was too long ({shorten_hint}). Rewrite SHORTER: body 130-160 words excluding openings."
    return (
        f"PASSAGE:\n{stem['passage']}\n\n"
        f"opening1: {stem['opening1']}\n"
        f"opening2: {stem['opening2']}\n\n"
        "Write the two paragraphs now."
        + extra
    )


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))
