"""OpenAI over REST (no `openai` package: CORPUS_INDEX_PLAN.md, I3). Public research text only. Every call is logged
with tokens and latency (handoff §9); the key is read from ~/.config/vinfinx/openai.env and never logged."""
from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from . import LOGS

API = "https://api.openai.com/v1"
KEY_FILE = Path(os.path.expanduser("~/.config/vinfinx/openai.env"))
EMBEDDING_MODEL = "text-embedding-3-large"
# Gold-label drafts only. gpt-5-nano (cheapest) labelled no passage as supporting, so the AI engineer chose gpt-5-mini
# (2026-10-22; findings log).
LABEL_MODEL = "gpt-5-mini-2025-08-07"
# USD per 1M tokens (input, cached input, output), standard tier, read from platform.openai.com/docs/pricing.
PRICE_TABLE_VERSION = "openai-pricing-2026-10-22"
PRICES = {"text-embedding-3-large": (0.13, None, None), "text-embedding-3-small": (0.02, None, None),
          "gpt-5-nano": (0.05, 0.005, 0.40), "gpt-5-mini": (0.25, 0.025, 2.00), "gpt-4.1-mini": (0.40, 0.10, 1.60), "gpt-4.1-nano": (0.10, 0.025, 0.40), "gpt-5.4-nano": (0.20, 0.02, 1.25)}


def cost_usd(model: str | None, usage: dict | None) -> float | None:
    """Price of one call from the dated table; None when the model is not in it."""
    if not model or not usage:
        return None
    base = next((m for m in sorted(PRICES, key=len, reverse=True) if model.startswith(m)), None)
    if base is None:
        return None
    inp, cached, out = PRICES[base]
    prompt = usage.get("prompt_tokens", 0)
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if cached else 0
    total = (prompt - cached_tokens) * inp + cached_tokens * (cached or 0) + usage.get("completion_tokens", 0) * (out or 0)
    return round(total / 1e6, 6)
BATCH_INPUTS = 64
BATCH_CHARS = 120_000
TIMEOUT = (15, 180)  # connect, read: a plain-JSON embedding reply once stalled past 60 s (findings, 2026-10-22)
RETRIES = 5


class OpenAIError(Exception):
    pass


def load_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key and KEY_FILE.exists():
        m = re.search(r"OPENAI_API_KEY=['\"]?([^'\"\s]+)", KEY_FILE.read_text())
        key = m.group(1) if m else None
    if not key:
        raise OpenAIError(f"no OpenAI key: set OPENAI_API_KEY or write it to {KEY_FILE}")
    return key


class Client:
    def __init__(self, service: str, key: str | None = None, log_path: Path | None = None,
                 session: requests.Session | None = None, sleep=time.sleep) -> None:
        self.service = service
        self._key = key or load_key()
        self.session = session or requests.Session()
        self.log_path = log_path or LOGS / "openai_calls.jsonl"
        self.sleep = sleep

    def _post(self, path: str, body: dict, n_inputs: int) -> dict:
        delay, last = 2.0, None
        for attempt in range(RETRIES + 1):
            start = time.time()
            try:
                r = self.session.post(f"{API}{path}", json=body, timeout=TIMEOUT,
                                      headers={"Authorization": f"Bearer {self._key}"})
                status = r.status_code
            except requests.RequestException as exc:
                status, r, last = None, None, f"{type(exc).__name__}"
            seconds = time.time() - start
            data = r.json() if r is not None and r.headers.get("content-type", "").startswith("application/json") else {}
            self._log(path, body.get("model"), n_inputs, status, seconds, data.get("usage"), attempt)
            if status == 200:
                return data
            if status is not None and status not in (408, 409, 429) and status < 500:
                raise OpenAIError(f"{path}: HTTP {status}: {data.get('error', {}).get('message', '')[:200]}")
            last = last if status is None else f"HTTP {status}"
            if attempt < RETRIES:
                self.sleep(delay)
                delay = min(delay * 2, 60)
        raise OpenAIError(f"{path}: gave up after {RETRIES + 1} attempts ({last})")

    def _log(self, path, model, n_inputs, status, seconds, usage, attempt) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "trace_id": str(uuid.uuid4()),
                                "service": self.service, "endpoint": path, "model": model, "inputs": n_inputs,
                                "status": status, "attempt": attempt, "seconds": round(seconds, 3),
                                "usage": usage, "cost_usd": cost_usd(model, usage),
                                "price_table": PRICE_TABLE_VERSION}) + "\n")

    # ---------------------------------------------------------------- embeddings

    def embed(self, texts: list[str], model: str = EMBEDDING_MODEL) -> tuple[np.ndarray, str, int]:
        """Unit-length float32 vectors in input order, the model name the API reports, and tokens used."""
        vectors, used_model, tokens = [], model, 0
        for batch in _batches(texts):
            data = self._post("/embeddings", {"model": model, "input": batch, "encoding_format": "base64"},
                              len(batch))
            rows = sorted(data["data"], key=lambda d: d["index"])
            vectors += [np.frombuffer(base64.b64decode(d["embedding"]), dtype="<f4") for d in rows]
            used_model = data.get("model", model)
            tokens += (data.get("usage") or {}).get("prompt_tokens", 0)
        m = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), np.float32)
        if len(m):
            m /= np.linalg.norm(m, axis=1, keepdims=True)
        return m, used_model, tokens


    # ---------------------------------------------------------------- structured chat output

    def chat_json(self, messages: list[dict], schema: dict, model: str = LABEL_MODEL,
                  max_completion_tokens: int = 8000) -> tuple[dict, dict]:
        """A JSON object constrained by `schema` (strict structured output), and the usage."""
        body = {"model": model, "messages": messages, "max_completion_tokens": max_completion_tokens,
                "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True,
                                                                           "schema": schema}}}
        if model.startswith("gpt-5"):
            body["reasoning_effort"] = "minimal"  # reasoning tokens are billed as output
        data = self._post("/chat/completions", body, len(messages))
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise OpenAIError(f"chat stopped early: {choice.get('finish_reason')}")
        return json.loads(choice["message"]["content"]), data.get("usage") or {}


def _batches(texts: list[str]):
    batch, chars = [], 0
    for t in texts:
        if batch and (len(batch) >= BATCH_INPUTS or chars + len(t) > BATCH_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(t)
        chars += len(t)
    if batch:
        yield batch
