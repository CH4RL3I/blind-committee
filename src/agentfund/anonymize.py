"""Anonymisation: neutral ids for assets, relative time, and free-text scrubbing.

The point is to remove the two keys a model can use to recall what happened next: WHO
(ticker, company name, absolute dollar magnitudes) and WHEN (calendar dates).
"""

from __future__ import annotations

import random
import re

EXTRA_ALIASES = {"GOOGL": ["Google"], "META": ["Facebook"], "AMZN": ["AWS"], "AAPL": ["iPhone"]}
_SUFFIX = re.compile(r",?\s+(Inc\.?|Corporation|Corp\.?|Company|& Co\.?|Co\.?)$", re.I)
_GENERIC = {"home", "the", "meta"}  # 'Meta' is kept via the full 'Meta Platforms' alias


def _aliases(ticker: str, name: str) -> list[str]:
    core = re.sub(r"^The\s+", "", _SUFFIX.sub("", name)).strip()
    out = {name, core, ticker, *EXTRA_ALIASES.get(ticker, [])}
    first = re.split(r"[\s.]", core)[0]
    if len(first) >= 4 and first.lower() not in _GENERIC:
        out.add(first)
    if ticker == "META":
        out.add("Meta")
    return sorted(out, key=len, reverse=True)  # longest first so 'Exxon Mobil' beats 'Exxon'


class Anonymizer:
    """Deterministic ticker -> neutral id mapping (`ASSET_07`), shuffled by `seed`."""

    def __init__(self, names: dict[str, str], seed: int = 0):
        tickers = list(names)
        ids = [f"ASSET_{i + 1:02d}" for i in range(len(tickers))]
        random.Random(seed).shuffle(ids)
        self.to_id = dict(zip(tickers, ids, strict=True))
        self.to_ticker = {v: k for k, v in self.to_id.items()}
        self._patterns = [
            (re.compile(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", re.I), self.to_id[t])
            for t, n in names.items()
            for a in _aliases(t, n)
        ]

    def scrub(self, text: str) -> str:
        """Replace any ticker, company name or alias in free text by its neutral id."""
        for pat, rep in self._patterns:
            text = pat.sub(rep, text)
        return text
