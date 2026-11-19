import pandas as pd

from src.anomaly import check_duplicate_payout, find_amount_outliers, find_duplicate_payouts


def _df():
    return pd.DataFrame(
        {
            "vendor_canonical": ["a", "a", "b", "a"],
            "amount": [100.0, 100.0, 100.0, 250.0],
            "invoice_date": pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-02", "2025-06-01"]),
            "category": ["x", "x", "x", "x"],
        }
    )


def test_duplicate_flagged_only_for_later_matching_row():
    assert find_duplicate_payouts(_df()).tolist() == [False, True, False, False]


def test_validation_rule_blocks_duplicate():
    new = {"vendor_canonical": "a", "amount": 100.0, "invoice_date": "2025-03-03"}
    assert check_duplicate_payout(new, _df())
    far = {"vendor_canonical": "a", "amount": 100.0, "invoice_date": "2025-09-01"}
    assert not check_duplicate_payout(far, _df())


def test_outlier_detected():
    df = pd.DataFrame({"category": ["x"] * 10, "amount": [10, 11, 9, 10, 12, 10, 11, 9, 10, 500]})
    assert find_amount_outliers(df).iloc[-1]
