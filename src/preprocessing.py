"""
preprocessing.py
-----------------
Data cleaning + feature engineering for the cross-sectional wellness
dataset (Level 1 - ML). Produces train/test splits ready for both the
clustering (unsupervised) and supervised (regression/classification)
tasks, and writes a short feature-engineering report.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "employee_wellness_cross_sectional.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "age", "tenure_years", "work_hours_per_week", "sleep_hours",
    "physical_activity_min_week", "diet_score", "screen_time_hours",
    "caffeine_intake_mg", "meditation_min_week", "social_support_score",
    "job_satisfaction_score", "work_life_balance_score", "past_medical_leaves",
]
CATEGORICAL_FEATURES = ["gender", "department"]
TARGET_REGRESSION = "stress_level"
TARGET_CLASSIFICATION = "burnout_risk"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    before = len(df)
    df = df.drop_duplicates(subset="employee_id").reset_index(drop=True)
    after = len(df)

    report = {
        "rows_before_dedup": before,
        "rows_after_dedup": after,
        "duplicates_removed": before - after,
        "missing_values_by_column": df[NUMERIC_FEATURES].isna().sum().to_dict(),
    }
    with open(REPORTS_DIR / "feature_engineering_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Composite wellness indices -- simple, interpretable engineered features
    df["activity_index"] = df["physical_activity_min_week"] / (df["work_hours_per_week"] + 1)
    df["recovery_index"] = df["sleep_hours"] * (df["meditation_min_week"] + 1) / 100
    df["lifestyle_risk_score"] = (
        (df["screen_time_hours"] / 16) * 0.4
        + (df["caffeine_intake_mg"] / 800) * 0.3
        + (1 - df["diet_score"] / 10) * 0.3
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    engineered = ["activity_index", "recovery_index", "lifestyle_risk_score"]

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES + engineered),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_train_test_split(test_size=0.2, random_state=42):
    df = load_raw()
    df = engineer_features(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
        "activity_index", "recovery_index", "lifestyle_risk_score"
    ]
    X = df[feature_cols]
    y_reg = df[TARGET_REGRESSION]
    y_clf = df[TARGET_CLASSIFICATION]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


if __name__ == "__main__":
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = get_train_test_split()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Burnout class balance (train):")
    print(y_clf_train.value_counts(normalize=True))
