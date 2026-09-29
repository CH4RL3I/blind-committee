import json

import numpy as np
import pandas as pd
import pytest


class ScriptedBackend:
    """Fake LLM: returns queued replies (strings), or calls `fn(system, prompt)`."""

    name = "mock"

    def __init__(self, replies=None, fn=None):
        self.replies, self.fn, self.calls = list(replies or []), fn, []

    def complete(self, system, prompt, schema=None, max_tokens=300):
        self.calls.append((system, prompt))
        if self.fn:
            return self.fn(system, prompt)
        return self.replies.pop(0)


@pytest.fixture
def prices():
    """Deterministic synthetic prices: 3 assets, 400 business days."""
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(1)
    data = {t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(idx))) for t in ("A", "B", "C")}
    return pd.DataFrame(data, index=idx)


def view_json(stance="bullish", conf=0.8):
    return json.dumps({"stance": stance, "confidence": conf, "reasoning": "because"})


def make_facts(rows):
    """rows: (concept, start, end, val, filed, form). Builds companyfacts-shaped JSON."""
    us = {}
    for concept, start, end, val, filed, form in rows:
        rec = {"end": end, "val": val, "filed": filed, "form": form}
        if start:
            rec["start"] = start
        us.setdefault(concept, {"units": {"USD": []}})["units"]["USD"].append(rec)
    return {"facts": {"us-gaap": us}}
