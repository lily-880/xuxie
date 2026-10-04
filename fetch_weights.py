#!/usr/bin/env python3
"""Join the bf16 weight parts in this repo into the checkpoint nanochat loads.

    python fetch_weights.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PARTS = HERE / "weights" / "parts"
DEST = Path.home() / ".cache" / "nanochat" / "dpo_checkpoints" / "d24_xuxie_dpo_v10_lr3e-6"


def restore(prefix: str) -> dict:
    chunks = sorted(PARTS.glob(prefix + ".*"))
    if not chunks:
        raise SystemExit(f"missing {prefix}.* under {PARTS}")
    blob = PARTS / f"_{prefix}.pt"
    with blob.open("wb") as out:
        for chunk in chunks:
            out.write(chunk.read_bytes())
    state = torch.load(blob, map_location="cpu", weights_only=True)
    blob.unlink()
    return state


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    state = {}
    state.update(restore("core"))
    state.update(restore("value"))
    out = DEST / "model_000050.pt"
    torch.save(state, out)
    shutil.copy(HERE / "weights" / "meta_000050.json", DEST / "meta_000050.json")
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GiB)")
    print("load with: --kind dpo --tag d24_xuxie_dpo_v10_lr3e-6 --step 50")


if __name__ == "__main__":
    main()
