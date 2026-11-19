"""Generate a synthetic, fully anonymized invoice dataset.

No real client data is used anywhere in this repository. Vendors, amounts and
descriptions are invented. Known problems (messy vendor names, duplicate
payouts, amount outliers) are injected with ground-truth flags so the
detection logic can be evaluated honestly.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# category -> (canonical vendor names, description templates, (mean, sd) of amount)
CATALOG = {
    "Software & Subscriptions": (
        ["Cloudnest Systems", "Pixelforge Software", "Datawave Analytics"],
        ["monthly saas subscription", "annual software license", "cloud hosting fee"],
        (120.0, 40.0),
    ),
    "Office Supplies": (
        ["Paperline Traders", "Deskmate Stationers", "Brightoffice Supplies"],
        ["printer paper and toner", "stationery order", "office desk accessories"],
        (60.0, 20.0),
    ),
    "Travel": (
        ["Skyroute Travels", "Roadrunner Cabs", "Staywell Hotels"],
        ["flight booking", "airport cab fare", "hotel stay client visit"],
        (320.0, 110.0),
    ),
    "Utilities": (
        ["Powergrid Utilities", "Aquaflow Services", "Netlink Broadband"],
        ["electricity bill", "water charges", "internet broadband bill"],
        (90.0, 25.0),
    ),
    "Professional Services": (
        ["Lexa Legal Associates", "Ledgerwise Consultants", "Brandpeak Marketing"],
        ["legal consultation fee", "audit and tax advisory", "marketing retainer"],
        (700.0, 220.0),
    ),
}

SUFFIXES = ["", "", " Inc", " LLC", " Pvt Ltd", " Ltd.", " Co."]


def messy_vendor(name: str, rng: random.Random) -> str:
    """Return a noisy variant of a canonical vendor name."""
    v = name
    choice = rng.random()
    if choice < 0.25:
        v = v.upper()
    elif choice < 0.45:
        v = v.lower()
    elif choice < 0.60:
        v = v.replace(" ", "  ")
    elif choice < 0.72:
        v = v.replace(" ", "-")
    v += rng.choice(SUFFIXES)
    if rng.random() < 0.10 and len(v) > 6:  # single-character typo
        i = rng.randrange(1, len(v) - 1)
        v = v[:i] + v[i + 1:]
    return v


def generate(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01")
    for i in range(n):
        cat = rng.choice(list(CATALOG))
        vendors, descs, (mu, sd) = CATALOG[cat]
        canon = rng.choice(vendors)
        amount = max(5.0, float(nprng.normal(mu, sd)))
        rows.append(
            {
                "invoice_id": f"INV-{100000 + i}",
                "invoice_date": (start + pd.Timedelta(days=int(nprng.integers(0, 330)))).date().isoformat(),
                "vendor_raw": messy_vendor(canon, rng),
                "description": rng.choice(descs),
                "amount": round(amount, 2),
                "category": cat,
                "true_vendor": canon,
                "injected_duplicate": 0,
                "injected_outlier": 0,
            }
        )
    df = pd.DataFrame(rows)

    # Inject duplicate payouts: same vendor and amount, near date, new invoice id, different name format.
    dup_idx = nprng.choice(df.index, size=int(0.04 * n), replace=False)
    dups = []
    for k, idx in enumerate(dup_idx):
        r = df.loc[idx].copy()
        r["invoice_id"] = f"INV-{900000 + k}"
        r["invoice_date"] = (pd.Timestamp(r["invoice_date"]) + pd.Timedelta(days=int(nprng.integers(0, 3)))).date().isoformat()
        r["vendor_raw"] = messy_vendor(r["true_vendor"], rng)
        r["injected_duplicate"] = 1
        dups.append(r)
    df = pd.concat([df, pd.DataFrame(dups)], ignore_index=True)

    # Inject amount outliers (5x-9x the normal amount).
    out_idx = nprng.choice(df.index[df["injected_duplicate"] == 0], size=int(0.03 * n), replace=False)
    df.loc[out_idx, "amount"] = (df.loc[out_idx, "amount"] * nprng.uniform(5, 9, len(out_idx))).round(2)
    df.loc[out_idx, "injected_outlier"] = 1

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/sample_invoices.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    generate(a.rows, a.seed).to_csv(a.out, index=False)
    print(f"Wrote {a.out}")
