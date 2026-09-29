from datetime import date

from conftest import make_facts

from agentfund.data.sec import point_in_time
from agentfund.features import fundamental_features

REV = "Revenues"
ROWS = [
    # FY2022 annual, filed Feb 2023
    (REV, "2022-01-01", "2022-12-31", 100.0, "2023-02-15", "10-K"),
    # Q1 2023 quarter, period ended Mar 31 but filed May 5
    (REV, "2023-01-01", "2023-03-31", 30.0, "2023-05-05", "10-Q"),
]


def test_fact_filed_after_decision_date_is_excluded():
    facts = make_facts(ROWS)
    # Decision on Apr 20: the Q1 period has ended (Mar 31) but the 10-Q is not public yet.
    got = point_in_time(facts, [REV], date(2023, 4, 20))
    assert [f.val for f in got] == [100.0]
    assert all(f.filed < date(2023, 4, 20) for f in got)


def test_filed_on_decision_date_is_excluded_and_day_after_included():
    facts = make_facts(ROWS)
    assert len(point_in_time(facts, [REV], date(2023, 5, 5))) == 1
    assert len(point_in_time(facts, [REV], date(2023, 5, 6))) == 2


def test_restatement_uses_latest_version_known_at_the_time():
    rows = [
        (REV, "2022-01-01", "2022-12-31", 100.0, "2023-02-15", "10-K"),
        (REV, "2022-01-01", "2022-12-31", 90.0, "2023-08-01", "10-K/A"),
    ]
    facts = make_facts(rows)
    assert point_in_time(facts, [REV], date(2023, 6, 1))[0].val == 100.0
    assert point_in_time(facts, [REV], date(2023, 9, 1))[0].val == 90.0


def test_features_do_not_see_future_filings():
    rows = ROWS + [
        ("NetIncomeLoss", "2022-01-01", "2022-12-31", 10.0, "2023-02-15", "10-K"),
        ("NetIncomeLoss", "2023-01-01", "2023-03-31", 999.0, "2023-05-05", "10-Q"),
    ]
    facts = make_facts(rows)
    before, _ = fundamental_features(facts, date(2023, 4, 20))
    after, levels = fundamental_features(facts, date(2023, 5, 20))
    assert before["net_margin_annual"] == 0.1
    assert before["days_since_latest_quarter_end"] is None  # no quarter public yet
    assert after["days_since_latest_quarter_end"] == 50
    assert levels["latest_filing_date"] == "2023-05-05"
