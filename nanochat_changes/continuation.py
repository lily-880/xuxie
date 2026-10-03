"""
Gaokao-style 读后续写 (continuation writing) for SFT.

train: ~/.cache/nanochat/task_data/xuxie/train_v1_clean.jsonl  (messages already present)
val:   repo eval.jsonl converted to the same chat format (held-out; never used as train)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nanochat.common import get_base_dir
from tasks.common import Task

DEFAULT_TRAIN = Path(get_base_dir()) / "task_data" / "xuxie" / "train_v1_clean.jsonl"
DEFAULT_EVAL = (
    Path(__file__).resolve().parents[1]
    / "dev"
    / "d24_core_run"
    / "xuxie_eval"
    / "eval.jsonl"
)

USER_INSTRUCTION = (
    "Write two paragraphs, about 150 words total, not counting the openings."
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _user_content(passage: str, opening1: str, opening2: str) -> str:
    return (
        f"{passage.strip()}\n\n"
        f"Paragraph 1 opening: {opening1.strip()}\n"
        f"Paragraph 2 opening: {opening2.strip()}\n"
        f"{USER_INSTRUCTION}"
    )


def _rows_from_eval(path: Path) -> list[dict]:
    """Convert frozen eval items into train-shaped {messages} rows for val bpb."""
    out = []
    for item in _load_jsonl(path):
        ref = item.get("reference_continuation")
        if not ref or not str(ref).strip():
            continue
        out.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": _user_content(
                            item["passage"], item["opening1"], item["opening2"]
                        ),
                    },
                    {"role": "assistant", "content": str(ref).strip()},
                ]
            }
        )
    return out


class Continuation(Task):
    """读后续写 conversations. split=train|val."""

    def __init__(
        self,
        split: str = "train",
        path: str | os.PathLike | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert split in ("train", "val"), "Continuation split must be train|val"
        self.split = split
        if path is not None:
            self.path = Path(path)
            self.rows = _load_jsonl(self.path)
            # allow raw eval-shaped jsonl via explicit path+val
            if split == "val" and self.rows and "messages" not in self.rows[0]:
                self.rows = _rows_from_eval(self.path)
        elif split == "train":
            self.path = DEFAULT_TRAIN
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"Continuation train data not found: {self.path}\n"
                    "Expected amplify output train_v1_clean.jsonl under "
                    "~/.cache/nanochat/task_data/xuxie/"
                )
            self.rows = _load_jsonl(self.path)
        else:
            self.path = DEFAULT_EVAL
            if not self.path.is_file():
                raise FileNotFoundError(f"Continuation val/eval not found: {self.path}")
            self.rows = _rows_from_eval(self.path)
        self.length = len(self.rows)
        if self.length == 0:
            raise ValueError(f"Continuation({split}) is empty: {self.path}")

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.rows[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking (same spirit as SmolTalk)
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "Continuation messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"Message {i} has role {message['role']} but should be {expected_role}"
            )
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        return {"messages": messages}


if __name__ == "__main__":
    for split in ("train", "val"):
        ds = Continuation(split=split)
        print(f"{split}: len={len(ds)} path={ds.path}")
        ex = ds[0]
        print(f"  roles={[m['role'] for m in ex['messages']]}")
        print(f"  user={ex['messages'][0]['content'][:100]!r}...")
