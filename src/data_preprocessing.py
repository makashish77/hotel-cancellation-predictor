"""
data_preprocessing.py

Loads and cleans the Hotel Booking Demand dataset, and engineers features
used by the cancellation-risk model.

Expected raw columns (standard Kaggle "Hotel Booking Demand" schema):
hotel, is_canceled, lead_time, arrival_date_year, arrival_date_month,
arrival_date_week_number, arrival_date_day_of_month, stays_in_weekend_nights,
stays_in_week_nights, adults, children, babies, meal, country,
market_segment, distribution_channel, is_repeated_guest,
previous_cancellations, previous_bookings_not_canceled, reserved_room_type,
assigned_room_type, booking_changes, deposit_type, agent, company,
days_in_waiting_list, customer_type, adr, required_car_parking_spaces,
total_of_special_requests, reservation_status, reservation_status_date
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# Columns we intentionally engineer or keep for modeling
MODEL_FEATURES = [
    "hotel",
    "lead_time",
    "arrival_month_num",
    "arrival_week_number",
    "stays_total_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "deposit_type",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "room_type_mismatch",
]

TARGET = "is_canceled"

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw hotel bookings CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: handle missing values, drop leakage columns."""
    df = df.copy()

    # Fill common missing-value columns per known dataset quirks
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")
    if "agent" in df.columns:
        df["agent"] = df["agent"].fillna(0)
    if "company" in df.columns:
        df["company"] = df["company"].fillna(0)

    # Drop rows with no guests at all (data quality issue known in this dataset)
    if set(["adults", "children", "babies"]).issubset(df.columns):
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0]

    # Leakage columns: reservation_status / reservation_status_date directly
    # encode the outcome (is_canceled) and must NOT be used as features.
    leakage_cols = ["reservation_status", "reservation_status_date"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from cleaned raw data."""
    df = df.copy()

    # Month name -> numeric
    if "arrival_date_month" in df.columns:
        df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    else:
        df["arrival_month_num"] = np.nan

    if "arrival_date_week_number" in df.columns:
        df["arrival_week_number"] = df["arrival_date_week_number"]
    else:
        df["arrival_week_number"] = np.nan

    # Total nights booked
    weekend = df.get("stays_in_weekend_nights", 0)
    week = df.get("stays_in_week_nights", 0)
    df["stays_total_nights"] = weekend + week

    # Room type mismatch: assigned room differs from what was reserved
    # (a known, mild predictor of dissatisfaction/cancellation behavior)
    if set(["reserved_room_type", "assigned_room_type"]).issubset(df.columns):
        df["room_type_mismatch"] = (
            df["reserved_room_type"] != df["assigned_room_type"]
        ).astype(int)
    else:
        df["room_type_mismatch"] = 0

    return df


def build_model_table(raw_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: load -> clean -> engineer -> split X/y."""
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after processing: {missing}")

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None

    return X, y


CATEGORICAL_FEATURES = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "deposit_type", "customer_type",
]

NUMERIC_FEATURES = [f for f in MODEL_FEATURES if f not in CATEGORICAL_FEATURES]
