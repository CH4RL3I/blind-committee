import numpy as np
import pandas as pd
import pytest

from agentfund.risk import RiskLimits, RiskManager


def rets(vols, n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: rng.normal(0, v / np.sqrt(252), n) for k, v in vols.items()})


CALM = {"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.10, "E": 0.10, "F": 0.10, "G": 0.10, "H": 0.10}


def test_position_cap_overrides_pm_all_in():
    rm = RiskManager(RiskLimits(target_vol=1.0))
    d = rm.apply({"A": 1.0}, rets(CALM), drawdown=0.0)
    assert d.weights["A"] == pytest.approx(0.15)
    assert any("cap A" in a for a in d.actions)


def test_gross_never_exceeds_limit_and_no_shorts():
    rm = RiskManager(RiskLimits(target_vol=1.0))
    d = rm.apply({k: 0.4 for k in CALM} | {"A": -0.5}, rets(CALM), drawdown=0.0)
    assert min(d.weights.values()) >= 0
    assert sum(d.weights.values()) <= 1.0 + 1e-9
    assert all(v <= 0.15 + 1e-9 for v in d.weights.values())


def test_high_vol_asset_is_vetoed():
    rm = RiskManager(RiskLimits(target_vol=1.0))
    d = rm.apply({"A": 0.1, "B": 0.1}, rets({"A": 0.10, "B": 1.5}), drawdown=0.0)
    assert d.vetoed == ["B"] and d.weights["B"] == 0.0 and d.weights["A"] > 0


def test_vol_target_scales_exposure_down():
    r = rets({k: 0.40 for k in "ABCDEFGH"})  # uncorrelated 40% vol names
    lim = RiskLimits(max_position=0.125, target_vol=0.10, max_asset_vol=2.0)
    d = RiskManager(lim).apply({k: 0.125 for k in "ABCDEFGH"}, r, drawdown=0.0)
    cov = r.iloc[-60:].cov() * 252
    w = np.array([d.weights[c] for c in cov.columns])
    assert np.sqrt(w @ cov.values @ w) == pytest.approx(0.10, rel=1e-6)


def test_drawdown_stop_derisks_then_halts_then_reenters_half():
    rm = RiskManager(RiskLimits(target_vol=1.0))
    prop = {k: 0.125 for k in CALM}
    r = rets(CALM)
    assert sum(rm.apply(prop, r, 0.0).weights.values()) == pytest.approx(1.0)
    assert sum(rm.apply(prop, r, 0.12).weights.values()) == pytest.approx(0.5)
    halted = rm.apply(prop, r, 0.25)
    assert halted.halted and sum(halted.weights.values()) == 0.0
    # Even though drawdown is still deep, the next decision re-enters at half exposure.
    back = rm.apply(prop, r, 0.25)
    assert not back.halted and sum(back.weights.values()) == pytest.approx(0.5)
