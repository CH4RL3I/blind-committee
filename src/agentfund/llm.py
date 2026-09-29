"""LLM backends behind one interface, plus validated structured output with retry."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    cache_dir,
)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMValidationError(LLMError):
    """Raised when the model never produced schema-valid JSON within the retry budget."""


class Backend(Protocol):
    name: str

    def complete(
        self, system: str, prompt: str, schema: dict | None = None, max_tokens: int = 300
    ) -> str: ...


class OllamaBackend:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, url: str = DEFAULT_OLLAMA_URL):
        self.model, self.url = model, url.rstrip("/")
        self.name = f"ollama:{model}"

    def complete(self, system, prompt, schema=None, max_tokens=300) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": schema or "json",  # Ollama constrains decoding to the JSON schema
            "options": {"temperature": 0, "seed": 7, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        r = httpx.post(f"{self.url}/api/chat", json=body, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]


class AnthropicBackend:
    """Optional. Reads ANTHROPIC_API_KEY lazily; never required for the default flow."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL):
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system, prompt, schema=None, max_tokens=300) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


class CachedBackend:
    """Disk cache keyed by the full request. Makes long runs resumable and ablations cheap."""

    def __init__(self, inner: Backend, directory: Path | None = None):
        self.inner, self.name = inner, inner.name
        self.dir = directory or (cache_dir() / "llm")
        self.dir.mkdir(parents=True, exist_ok=True)

    def complete(self, system, prompt, schema=None, max_tokens=300) -> str:
        key = hashlib.sha256(
            json.dumps([self.name, system, prompt, schema, max_tokens]).encode()
        ).hexdigest()
        path = self.dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["out"]
        out = self.inner.complete(system, prompt, schema, max_tokens)
        path.write_text(json.dumps({"out": out}))
        return out


def make_backend(kind: str = "ollama", model: str | None = None, cache: bool = True) -> Backend:
    if kind == "ollama":
        b: Backend = OllamaBackend(model or DEFAULT_OLLAMA_MODEL)
    elif kind == "anthropic":
        b = AnthropicBackend(model or DEFAULT_ANTHROPIC_MODEL)
    else:
        raise ValueError(f"unknown backend {kind!r}")
    return CachedBackend(b) if cache else b


def extract_json(text: str) -> dict:
    """Parse the first JSON object in `text` (tolerates code fences and chatter)."""
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    return obj


def structured(
    backend: Backend,
    model: type[T],
    system: str,
    prompt: str,
    *,
    retries: int = 2,
    max_tokens: int = 300,
) -> tuple[T, int]:
    """Call the model until its output validates against `model`.

    Returns (parsed, attempts). On each failure the validation error is fed back to the model.
    """
    schema = model.model_json_schema()
    keys = ", ".join(schema.get("properties", {}))
    full_system = f"{system}\nRespond with a single JSON object with keys: {keys}. No other text."
    user = prompt
    last_err = ""
    for attempt in range(1, retries + 2):
        raw = backend.complete(full_system, user, schema, max_tokens)
        try:
            return model.model_validate(extract_json(raw)), attempt
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            last_err = str(e)[:400]
            user = (
                f"{prompt}\n\nYour previous reply was invalid: {last_err}\n"
                f"Reply again with valid JSON only."
            )
    raise LLMValidationError(f"no valid output after {retries + 1} attempts: {last_err}")
