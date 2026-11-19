"""Duplicate-payout detection, amount-outlier detection and a validation rule."""
from __future__ import annotations

import pandas as pd


def find_duplicate_payouts(df: pd.DataFrame, window_days: int = 3) -> pd.Series:
    """Flag rows that repeat (canonical vendor, amount) within `window_days` of an earlier row.

    The earlier row is kept; later rows are flagged. Because vendors are mapped to a
    canonical name first, different spellings of one vendor no longer hide duplicates.
    """
    d = df.sort_values("invoice_date")
    flagged = pd.Series(False, index=df.index)
    for _, g in d.groupby(["vendor_canonical", "amount"]):
        if len(g) < 2:
            continue
        dates = g["invoice_date"].tolist()
        idxs = g.index.tolist()
        for i in range(1, len(g)):
            if any((dates[i] - dates[j]).days <= window_days for j in range(i)):
                flagged[idxs[i]] = True
    return flagged


def find_amount_outliers(df: pd.DataFrame, threshold: float = 6.0) -> pd.Series:
    """Robust z-score (median/MAD) within each category; flags unusually large amounts."""
    def score(s: pd.Series) -> pd.Series:
        med = s.median()
        mad = (s - med).abs().median() or 1e-9
        return 0.6745 * (s - med) / mad

    z = df.groupby("category")["amount"].transform(score)
    return z > threshold


def check_duplicate_payout(new_invoice: dict, existing: pd.DataFrame, window_days: int = 3) -> bool:
    """Validation rule: return True if `new_invoice` would duplicate an existing payout.

    Intended to run at entry time, before a payment is recorded.
    """
    date = pd.Timestamp(new_invoice["invoice_date"])
    same = existing[
        (existing["vendor_canonical"] == new_invoice["vendor_canonical"])
        & (existing["amount"] == new_invoice["amount"])
    ]
    return bool(((date - same["invoice_date"]).abs().dt.days <= window_days).any())


def evaluate(flags: pd.Series, truth: pd.Series) -> dict:
    truth = truth.astype(bool)
    tp = int((flags & truth).sum())
    fp = int((flags & ~truth).sum())
    fn = int((~flags & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 3), "recall": round(recall, 3)}
