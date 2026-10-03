#!/usr/bin/env python3
"""DeepSeek OpenAI-compatible client. Loads key from local .env only."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ENV = HERE / ".env"
BASE_URL = "https://api.deepseek.com/chat/completions"


def load_env(path: Path = DEFAULT_ENV) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    # process env overrides file
    for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DASHSCOPE_API_KEY"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class DeepSeekClient:
    def __init__(self, env: dict[str, str] | None = None):
        self.env = env or load_env()
        self.api_key = self.env.get("DEEPSEEK_API_KEY") or ""
        self.model = self.env.get("DEEPSEEK_MODEL") or "deepseek-chat"
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY missing in xuxie_amplify/.env")

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 1200,
        retries: int = 4,
    ) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(
                BASE_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                # DeepSeek keeps overloaded connections alive with blank lines, so a
                # socket timeout alone never fires; enforce a total deadline instead.
                deadline = time.time() + float(self.env.get("DEEPSEEK_DEADLINE", 180))
                with urllib.request.urlopen(req, timeout=60) as resp:
                    chunks = []
                    while True:
                        if time.time() > deadline:
                            raise TimeoutError("DeepSeek response exceeded deadline")
                        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    data = json.loads(b"".join(chunks).decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                last_err = e
                raw = e.read().decode("utf-8", errors="ignore")
                # rate limit / busy
                if e.code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"DeepSeek HTTP {e.code}: {raw[:300]}") from e
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek failed after retries: {last_err}")
