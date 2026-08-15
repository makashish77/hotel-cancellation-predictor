"""
predict.py

Loads the trained pipeline and exposes a simple predict_risk() function
used by the Streamlit app and can be reused anywhere else (API, batch job).
"""

from __future__ import annotations
import joblib
import pandas as pd

from data_preprocessing import MODEL_FEATURES


class CancellationPredictor:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)

    def predict_risk(self, booking: dict) -> dict:
        """
        booking: dict with keys matching MODEL_FEATURES
        returns: {"cancel_probability": float, "risk_label": str}
        """
        row = {feat: booking.get(feat) for feat in MODEL_FEATURES}
        X = pd.DataFrame([row])

        proba = self.pipeline.predict_proba(X)[0, 1]

        if proba >= 0.66:
            label = "High Risk"
        elif proba >= 0.33:
            label = "Medium Risk"
        else:
            label = "Low Risk"

        return {"cancel_probability": float(proba), "risk_label": label}
