import pandas as pd

from src.clean import build_canonical_map, normalize_vendor


def test_normalize_removes_suffix_case_and_punctuation():
    assert normalize_vendor("CLOUDNEST  Systems Inc.") == "cloudnest systems"
    assert normalize_vendor("Cloudnest-Systems Pvt Ltd") == "cloudnest systems"


def test_typo_variants_map_to_one_vendor():
    s = pd.Series(["cloudnest systems"] * 5 + ["cloudnest sytems"])
    m = build_canonical_map(s)
    assert m["cloudnest sytems"] == "cloudnest systems"
