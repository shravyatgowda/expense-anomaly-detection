"""End-to-end run: load -> clean -> categorize -> detect anomalies -> write outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.anomaly import evaluate, find_amount_outliers, find_duplicate_payouts
from src.categorize import train_and_evaluate
from src.clean import clean_invoices


def main(inp: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df = clean_invoices(pd.read_csv(inp))

    vendor_metrics = {
        "raw_distinct_vendor_strings": int(df["vendor_raw"].nunique()),
        "canonical_vendors_found": int(df["vendor_canonical"].nunique()),
        "true_vendors": int(df["true_vendor"].nunique()),
    }

    _, report = train_and_evaluate(df)
    cat_metrics = {
        "accuracy": round(report["accuracy"], 3),
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
    }

    df["flag_duplicate"] = find_duplicate_payouts(df)
    df["flag_outlier"] = find_amount_outliers(df)
    metrics = {
        "vendor_mapping": vendor_metrics,
        "categorization_holdout": cat_metrics,
        "duplicate_detection": evaluate(df["flag_duplicate"], df["injected_duplicate"]),
        "outlier_detection": evaluate(df["flag_outlier"], df["injected_outlier"]),
        "note": "All figures are computed on synthetic data; see README.",
    }

    df.to_csv("data/processed/cleaned_invoices.csv", index=False)
    df[df["flag_duplicate"] | df["flag_outlier"]].to_csv(out / "flagged_anomalies.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ax = df.groupby("category")["amount"].median().sort_values().plot.barh(color="#1F3A5F")
    ax.set_xlabel("Median invoice amount")
    ax.set_title("Median amount by category (synthetic data)")
    plt.tight_layout()
    plt.savefig(out / "median_amount_by_category.png", dpi=150)
    plt.close()
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/sample_invoices.csv")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args()
    print(json.dumps(main(a.input, a.outdir), indent=2))
