"""Shared data loading, feature definitions and model pipelines.

Both train.py and predict.py (and the notebook) import from here, so the
cleaning and feature logic lives in exactly one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

TARGET = "Class/ASD"

# A1..A10 are the ten items of the AQ-10 (Adult) screening questionnaire,
# already scored 0/1. A score of 1 means the answer leaned towards
# autistic traits for that item.
AQ_ITEMS = [f"A{i}_Score" for i in range(1, 11)]

# Short description of what each item asks about (not the official wording;
# use the official AQ-10 sheet from the Autism Research Centre for that).
# The second value is which answer direction scores 1.
AQ_ITEM_TOPICS = {
    "A1_Score": ("Notices small sounds others don't", "agree"),
    "A2_Score": ("Focuses on the whole picture rather than small details", "disagree"),
    "A3_Score": ("Finds it easy to do more than one thing at once", "disagree"),
    "A4_Score": ("Gets back to a task quickly after an interruption", "disagree"),
    "A5_Score": ("Finds it easy to read between the lines", "disagree"),
    "A6_Score": ("Can tell when a listener is getting bored", "disagree"),
    "A7_Score": ("Finds it hard to work out characters' intentions in stories", "agree"),
    "A8_Score": ("Likes collecting information about categories of things", "agree"),
    "A9_Score": ("Can work out thoughts/feelings from someone's face", "disagree"),
    "A10_Score": ("Finds it hard to work out people's intentions", "agree"),
}

AQ_CUTOFF = 6  # AQ-10 total >= 6 -> consider referral for full assessment

NUMERIC_FEATURES = AQ_ITEMS + ["age"]
CATEGORICAL_FEATURES = ["gender", "jaundice", "austim", "used_app_before", "relation"]
DEMOGRAPHIC_FEATURES = ["ethnicity", "contry_of_res"]

# Columns never used as model inputs:
#   ID       - row identifier, carries no information (was a feature before)
#   result   - (approximately) the AQ-10 total, redundant with A1..A10
#   age_desc - a single constant value in this dataset
DROPPED_COLUMNS = ["ID", "result", "age_desc"]


def load_data(path: str) -> pd.DataFrame:
    """Read the Kaggle CSV and apply the basic cleaning steps."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # '?' marks missing values in this dataset; keep them as real NaN so the
    # imputers inside the pipeline handle them (fitted on training data only).
    df = df.replace("?", np.nan)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    if "ethnicity" in df:
        df["ethnicity"] = df["ethnicity"].replace({"others": "Others"})
    if "relation" in df:
        df["relation"] = df["relation"].fillna("Others")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if TARGET in df:
        df[TARGET] = df[TARGET].replace({"NO": 0, "YES": 1}).astype(int)

    return df


def feature_columns(with_demographics: bool = False) -> tuple[list[str], list[str]]:
    categorical = CATEGORICAL_FEATURES + (DEMOGRAPHIC_FEATURES if with_demographics else [])
    return list(NUMERIC_FEATURES), categorical


def split_xy(df: pd.DataFrame, with_demographics: bool = False):
    numeric, categorical = feature_columns(with_demographics)
    X = df[numeric + categorical].copy()
    y = df[TARGET] if TARGET in df else None
    return X, y


def aq10_total(X: pd.DataFrame) -> pd.Series:
    return X[AQ_ITEMS].sum(axis=1)


class AQ10Rule(ClassifierMixin, BaseEstimator):
    """Baseline: predict ASD when the AQ-10 total is >= cutoff.

    No learning happens; it exists so the questionnaire's own rule can be
    scored with exactly the same cross-validation as the ML models.
    """

    def __init__(self, cutoff: int = AQ_CUTOFF):
        self.cutoff = cutoff

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        score = aq10_total(X).to_numpy() / len(AQ_ITEMS)
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (aq10_total(X).to_numpy() >= self.cutoff).astype(int)


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )


def build_models(with_demographics: bool = False, random_state: int = 42) -> dict:
    """All candidate models. Every learned step sits inside a Pipeline, so
    imputation, scaling and encoding are fitted on training folds only."""
    numeric, categorical = feature_columns(with_demographics)
    return {
        "AQ-10 rule (total >= 6)": AQ10Rule(),
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", _preprocessor(numeric, categorical)),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }


SEED = 42

SCORING = {
    "recall": "recall",             # share of real ASD cases caught - key for a screener
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",  # better than ROC-AUC on imbalanced data
}


def run_cv(models: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Stratified k-fold CV for every model; returns mean and std per metric."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            row[metric] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
