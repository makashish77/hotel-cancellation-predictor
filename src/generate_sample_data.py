"""
generate_sample_data.py

IMPORTANT: This creates a small SYNTHETIC dataset that mimics the schema of
the real Kaggle "Hotel Booking Demand" dataset, purely so you can test that
the pipeline (preprocessing -> training -> app) runs end-to-end before you
plug in the real data.

DO NOT use this synthetic data to report real project results/metrics.
For your actual project deliverable, download the real dataset:
    https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
and place it at data/hotel_bookings.csv

Usage:
    python src/generate_sample_data.py --n 2000 --out data/sample_hotel_bookings.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MEALS = ["BB", "HB", "FB", "SC"]
MARKET_SEGMENTS = ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"]
CHANNELS = ["TA/TO", "Direct", "Corporate"]
ROOM_TYPES = list("ABCDEFG")
DEPOSIT_TYPES = ["No Deposit", "Non Refund", "Refundable"]
CUSTOMER_TYPES = ["Transient", "Contract", "Transient-Party", "Group"]
HOTELS = ["City Hotel", "Resort Hotel"]


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    lead_time = rng.exponential(scale=60, size=n).astype(int).clip(0, 500)
    deposit_type = rng.choice(DEPOSIT_TYPES, size=n, p=[0.85, 0.10, 0.05])
    previous_cancellations = rng.poisson(0.2, size=n)
    total_special_requests = rng.integers(0, 4, size=n)

    # Cancellation probability driven by a few realistic-ish signals
    base = 0.05
    cancel_prob = (
        base
        + 0.35 * (lead_time > 100)
        + 0.25 * (deposit_type == "Non Refund")
        + 0.10 * (previous_cancellations > 0)
        - 0.10 * (total_special_requests > 1)
    )
    cancel_prob = np.clip(cancel_prob, 0.02, 0.95)
    is_canceled = rng.binomial(1, cancel_prob)

    reserved_room = rng.choice(ROOM_TYPES, size=n)
    # Small chance assigned room differs from reserved
    assigned_room = reserved_room.copy()
    swap_idx = rng.random(n) < 0.15
    assigned_room[swap_idx] = rng.choice(ROOM_TYPES, size=swap_idx.sum())

    df = pd.DataFrame({
        "hotel": rng.choice(HOTELS, size=n),
        "is_canceled": is_canceled,
        "lead_time": lead_time,
        "arrival_date_year": rng.choice([2023, 2024, 2025], size=n),
        "arrival_date_month": rng.choice(MONTHS, size=n),
        "arrival_date_week_number": rng.integers(1, 53, size=n),
        "arrival_date_day_of_month": rng.integers(1, 29, size=n),
        "stays_in_weekend_nights": rng.integers(0, 3, size=n),
        "stays_in_week_nights": rng.integers(0, 6, size=n),
        "adults": rng.integers(1, 3, size=n),
        "children": rng.choice([0, 0, 0, 1, 2], size=n),
        "babies": rng.choice([0, 0, 0, 0, 1], size=n),
        "meal": rng.choice(MEALS, size=n),
        "country": rng.choice(["USA", "GBR", "IND", "DEU", "FRA", "Unknown"], size=n),
        "market_segment": rng.choice(MARKET_SEGMENTS, size=n),
        "distribution_channel": rng.choice(CHANNELS, size=n),
        "is_repeated_guest": rng.choice([0, 0, 0, 1], size=n),
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": rng.poisson(0.3, size=n),
        "reserved_room_type": reserved_room,
        "assigned_room_type": assigned_room,
        "booking_changes": rng.integers(0, 3, size=n),
        "deposit_type": deposit_type,
        "agent": rng.integers(0, 300, size=n),
        "company": rng.integers(0, 50, size=n),
        "days_in_waiting_list": rng.choice([0, 0, 0, 5, 10], size=n),
        "customer_type": rng.choice(CUSTOMER_TYPES, size=n),
        "adr": rng.normal(100, 30, size=n).clip(0, 400).round(2),
        "required_car_parking_spaces": rng.choice([0, 0, 0, 1], size=n),
        "total_of_special_requests": total_special_requests,
        "reservation_status": np.where(is_canceled == 1, "Canceled", "Check-Out"),
        "reservation_status_date": pd.Timestamp("2024-01-01"),
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=str, default="data/sample_hotel_bookings.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Synthetic sample data ({args.n} rows) written to {args.out}")
    print("Reminder: this is SYNTHETIC data for pipeline testing only.")
