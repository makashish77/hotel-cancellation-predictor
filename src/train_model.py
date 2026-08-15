"""
train_model.py

Trains a cancellation-risk classifier on the Hotel Booking Demand dataset
and saves the fitted pipeline (preprocessing + model) to disk.

Usage:
    python src/train_model.py --data data/hotel_bookings.csv --out models/cancellation_model.pkl
"""

from __future__ import annotations
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import (
    build_model_table, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)


def build_pipeline() -> Pipeline:
    """Preprocessing + model pipeline. Uses GradientBoostingClassifier
    (no extra dependency beyond scikit-learn) for portability; swap in
    XGBoost/LightGBM if available in your environment for a speed/accuracy
    boost -- the pipeline interface stays identical.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def train(data_path: str, out_path: str, metrics_path: str) -> None:
    X, y = build_model_table(data_path)
    if y is None:
        raise ValueError("Target column 'is_canceled' not found in dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"\nModel saved to: {out_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to hotel_bookings.csv")
    parser.add_argument("--out", type=str, default="models/cancellation_model.pkl")
    parser.add_argument("--metrics", type=str, default="models/metrics.json")
    args = parser.parse_args()

    train(args.data, args.out, args.metrics)
