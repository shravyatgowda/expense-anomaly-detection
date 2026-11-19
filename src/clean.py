"""Vendor-name normalization and canonical mapping using regex + fuzzy matching."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

_SUFFIX = re.compile(r"\b(inc|llc|ltd|pvt|co|corp|company)\b\.?", re.I)
_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize_vendor(name: str) -> str:
    """Lowercase, drop legal suffixes and punctuation, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    s = name.lower().replace("-", " ")
    s = _SUFFIX.sub(" ", s)
    s = _NONALNUM.sub("", s)
    return _SPACES.sub(" ", s).strip()


def build_canonical_map(normalized: pd.Series, threshold: float = 0.88) -> dict[str, str]:
    """Cluster near-identical normalized names (typos) onto the most frequent spelling."""
    counts = normalized.value_counts()
    canon: list[str] = []
    mapping: dict[str, str] = {}
    for name in counts.index:  # most frequent first
        match = next((c for c in canon if SequenceMatcher(None, name, c).ratio() >= threshold), None)
        if match is None:
            canon.append(name)
            mapping[name] = name
        else:
            mapping[name] = match
    return mapping


def clean_invoices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vendor_norm"] = out["vendor_raw"].map(normalize_vendor)
    mapping = build_canonical_map(out["vendor_norm"])
    out["vendor_canonical"] = out["vendor_norm"].map(mapping)
    out["invoice_date"] = pd.to_datetime(out["invoice_date"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    return out
