"""Render the text an analyst (or the leakage probe) actually sees.

Anonymised packets contain a neutral id, ratios and relative time only. Raw packets add the
identifiers a model could use to recall the future: ticker, company name, absolute dates,
price levels and dollar magnitudes.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from .anonymize import Anonymizer
from .config import UNIVERSE
from .features import fundamental_features, technical_features


def _header(ticker: str, asof: date, anon: Anonymizer | None) -> dict:
    if anon:
        return {"asset": anon.to_id[ticker], "decision_date": "T (all timing is relative to T)"}
    return {"company": f"{UNIVERSE[ticker]} ({ticker})", "decision_date": asof.isoformat()}


def technical_packet(
    ticker: str, asof: date, close: pd.Series, anon: Anonymizer | None
) -> tuple[str, dict]:
    ts = pd.Timestamp(asof)
    feats = technical_features(close, ts)
    body = _header(ticker, asof, anon) | {"price_features": feats}
    if not anon:
        c = close.loc[:ts]
        body["last_close_usd"] = round(float(c.iloc[-1]), 2)
        body["high_52w_usd"] = round(float(c.iloc[-252:].max()), 2)
    return json.dumps(body, indent=1), feats


def fundamentals_packet(
    ticker: str, asof: date, facts_json: dict, anon: Anonymizer | None
) -> tuple[str, dict]:
    ratios, levels = fundamental_features(facts_json, asof)
    body = _header(ticker, asof, anon) | {"fundamentals": ratios}
    if not anon:
        body["reported_levels"] = levels
    return json.dumps(body, indent=1), ratios
