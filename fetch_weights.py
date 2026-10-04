#!/usr/bin/env python3
"""Download the DPO v10 step 50 weights and put them where nanochat loads them.

GitHub rejects a single file over 2GB, so the bf16 checkpoint is two Release
files. This script joins them into model_000050.pt.

    python fetch_weights.py
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import torch

TAG = "v1.0.0"
BASE = f"https://github.com/lily-880/xuxie/releases/download/{TAG}"
PARTS = ("model_core.pt", "model_value_embeds.pt")
HERE = Path(__file__).resolve().parent
DEST = Path.home() / ".cache" / "nanochat" / "dpo_checkpoints" / "d24_xuxie_dpo_v10_lr3e-6"


def download(name: str, dest: Path) -> None:
    url = f"{BASE}/{name}"
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    tmp = DEST / "_parts"
    tmp.mkdir(exist_ok=True)
    state = {}
    for name in PARTS:
        path = tmp / name
        if not path.exists() or path.stat().st_size < 1000:
            download(name, path)
        state.update(torch.load(path, map_location="cpu", weights_only=True))
    out = DEST / "model_000050.pt"
    torch.save(state, out)
    shutil.copy(HERE / "weights" / "meta_000050.json", DEST / "meta_000050.json")
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GiB)", flush=True)
    print("load with: --kind dpo --tag d24_xuxie_dpo_v10_lr3e-6 --step 50")


if __name__ == "__main__":
    main()
