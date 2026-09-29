import json
from datetime import date

import pandas as pd

from agentfund.anonymize import Anonymizer
from agentfund.config import UNIVERSE
from agentfund.features import technical_features
from agentfund.packets import fundamentals_packet, technical_packet


def test_mapping_is_bijective_deterministic_and_neutral():
    a, b = Anonymizer(UNIVERSE, seed=3), Anonymizer(UNIVERSE, seed=3)
    assert a.to_id == b.to_id
    assert len(set(a.to_id.values())) == len(UNIVERSE)
    assert {a.to_ticker[i] for i in a.to_id.values()} == set(UNIVERSE)
    assert all(v.startswith("ASSET_") for v in a.to_id.values())
    assert Anonymizer(UNIVERSE, seed=4).to_id != a.to_id  # seed reshuffles


def test_scrub_removes_tickers_names_and_aliases():
    a = Anonymizer(UNIVERSE)
    text = (
        "NVDA beat; NVIDIA Corporation and Apple Inc. rallied, Google (GOOGL) too. "
        "Meta Platforms, JPMorgan Chase & Co. and Chevron fell. My home is nice."
    )
    out = a.scrub(text)
    for banned in ("NVDA", "NVIDIA", "Apple", "Google", "GOOGL", "Meta", "JPMorgan", "Chevron"):
        assert banned.lower() not in out.lower(), banned
    assert a.to_id["NVDA"] in out
    assert "home is nice" in out  # generic words survive


def test_anonymised_packet_has_no_identity_absolute_dates_or_levels(prices):
    a = Anonymizer({"A": "Acme Corporation", "B": "Beta Inc.", "C": "Gamma Co."})
    long = pd.DataFrame({"A": prices["A"]}).rename(columns={"A": "A"})
    # UNIVERSE lookup is only used by raw headers; anonymised path must not need it.
    text, _ = technical_packet("A", prices.index[-1].date(), long["A"], a)
    body = json.loads(text)
    assert body["asset"] == a.to_id["A"]
    assert str(prices.index[-1].year) not in text
    assert "last_close_usd" not in body and "high_52w_usd" not in body
    facts = {"facts": {"us-gaap": {}}}
    ftext, _ = fundamentals_packet("A", date(2023, 6, 1), facts, a)
    assert "reported_levels" not in ftext and "2023" not in ftext


def test_technical_features_ignore_future_rows(prices):
    d = prices.index[300]
    base = technical_features(prices["A"], d)
    mutated = prices["A"].copy()
    mutated.iloc[301:] *= 5.0  # wild future
    assert technical_features(mutated, d) == base
