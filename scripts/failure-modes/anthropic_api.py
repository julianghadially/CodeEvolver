"""anthropic_api.py -- one direct call to the Anthropic Messages API, for the judge (api_count.py) and the summary fill
(fill_summaries.py).

WHY direct calls instead of an OpenCode session: every pack fits in one prompt (largest about 60k tokens against a 1M
context), so the judge is GIVEN the whole evidence instead of choosing what to read. That removes the reading-coverage gap
of a tool-using session (README "What is NOT guaranteed"), and the judge has no tools at all, so it cannot reach anything
outside the pack (no fence needed).

The key is read from the environment, or from the first .env (repo root, then scripts/macro-counts) that has it, via
judge.load_env_key. Its value is never printed or stored. Only the standard library is used.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "macro-counts"))

MODEL = "claude-sonnet-5-5"          # project rule: never change the model unless the user asks
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
PRICE = {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50}  # USD per million tokens (models.dev 2026-10-02)


def ensure_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    import judge  # macro-counts/judge.py: .env -> os.environ (value never printed)
    judge.load_env_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not found in the environment or a .env")


def cost(usage: dict) -> float:
    u = usage or {}
    return (u.get("input_tokens", 0) * PRICE["input"] + u.get("output_tokens", 0) * PRICE["output"]
            + u.get("cache_read_input_tokens", 0) * PRICE["cache_read"]
            + u.get("cache_creation_input_tokens", 0) * PRICE["cache_write"]) / 1e6


def call(system: str, user: str | None = None, *, messages: list | None = None, max_tokens: int = 32000, effort: str = "high",
         cache_system: bool = True, timeout: int = 900, attempts: int = 6) -> dict:
    """One Messages API call with adaptive thinking: `user` (one user turn) or `messages` (a whole conversation, for a repair
    turn: pass the earlier reply back as {"role": "assistant", "content": <its content>}). Returns {text, thinking, content (the
    raw content blocks, thinking signatures included), usage, stop_reason, model, request_id}.
    Retries transport errors, 429, 5xx and 529 with backoff; raises RuntimeError on anything else."""
    ensure_key()
    sys_block = {"type": "text", "text": system}
    if cache_system:
        sys_block["cache_control"] = {"type": "ephemeral"}
    body = {"model": MODEL, "max_tokens": max_tokens, "system": [sys_block],
            "thinking": {"type": "adaptive"}, "output_config": {"effort": effort},
            "messages": messages if messages is not None else [{"role": "user", "content": user}]}
    data = json.dumps(body).encode("utf-8")
    last = ""
    for k in range(attempts):
        req = urllib.request.Request(API_URL, data=data, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": API_VERSION, "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = json.load(resp)
                rid = resp.headers.get("request-id")
            text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
            thinking = "".join(b.get("thinking", "") for b in r.get("content", []) if b.get("type") == "thinking")
            return {"text": text, "thinking": thinking, "content": r.get("content", []), "usage": r.get("usage") or {},
                    "stop_reason": r.get("stop_reason"), "model": r.get("model"), "request_id": rid}
        except urllib.error.HTTPError as e:
            msg = e.read()[:500].decode("utf-8", "replace")
            last = f"HTTP {e.code}: {msg}"
            if e.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise RuntimeError(last) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(120, (2 ** k) * 5) + random.random() * 3)
    raise RuntimeError(f"API call failed after {attempts} attempts: {last}")
