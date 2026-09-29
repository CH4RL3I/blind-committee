"""Point-in-time SEC XBRL facts.

The companyfacts API returns every fact as reported in every filing, each with a `filed`
date. Filtering on `filed` (not on the period `end`) is what makes a backtest point-in-time:
a fiscal quarter ends weeks before its 10-Q is public.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date

import httpx

from ..config import cache_dir

BASE = "https://data.sec.gov"
_MIN_INTERVAL = 0.2  # <= 5 requests/second; SEC allows 10
_last_call = 0.0


def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. The SEC requires contact info, e.g. "
            "export SEC_USER_AGENT='yourproject you@example.com'"
        )
    return ua


def _get(url: str) -> dict:
    global _last_call
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    r = httpx.get(url, headers={"User-Agent": user_agent()}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _cached(name: str, url: str) -> dict:
    path = cache_dir() / "sec" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    data = _get(url)
    path.write_text(json.dumps(data))
    return data


def cik_for(ticker: str) -> int:
    tickers = _cached("company_tickers.json", "https://www.sec.gov/files/company_tickers.json")
    for row in tickers.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise KeyError(ticker)


def company_facts(ticker: str) -> dict:
    cik = cik_for(ticker)
    return _cached(f"CIK{cik:010d}.json", f"{BASE}/api/xbrl/companyfacts/CIK{cik:010d}.json")


@dataclass(frozen=True)
class Fact:
    concept: str
    start: date | None  # None for instant (balance sheet) facts
    end: date
    val: float
    filed: date
    form: str

    @property
    def days(self) -> int | None:
        return None if self.start is None else (self.end - self.start).days


def point_in_time(facts: dict, concepts: list[str], asof: date) -> list[Fact]:
    """All facts for any of `concepts` that were public strictly before `asof`.

    For a period reported in several filings (original plus restatements or later
    comparatives) only the latest version filed before `asof` is kept, which is what an
    investor could actually have seen on that date.
    """
    us = facts.get("facts", {}).get("us-gaap", {})
    latest: dict[tuple, Fact] = {}
    for concept in concepts:
        for row in us.get(concept, {}).get("units", {}).get("USD", []):
            filed = date.fromisoformat(row["filed"])
            if filed >= asof:
                continue
            f = Fact(
                concept,
                date.fromisoformat(row["start"]) if row.get("start") else None,
                date.fromisoformat(row["end"]),
                float(row["val"]),
                filed,
                row.get("form", ""),
            )
            key = (concept, f.start, f.end)
            if key not in latest or f.filed > latest[key].filed:
                latest[key] = f
    return sorted(latest.values(), key=lambda f: (f.end, f.filed))
