"""Expense categorization: TF-IDF + logistic regression, evaluated on a holdout set."""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    text = df["vendor_canonical"].fillna("") + " " + df["description"].fillna("")
    X_tr, X_te, y_tr, y_te = train_test_split(
        text, df["category"], test_size=test_size, random_state=seed, stratify=df["category"]
    )
    model = build_model().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    report = classification_report(y_te, pred, output_dict=True, zero_division=0)
    return model, report
