"""
app.py

Streamlit app: Hotel Booking Cancellation Predictor

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then deploy on Streamlit Community Cloud
pointing at app.py as the entrypoint.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import CancellationPredictor  # noqa: E402

MODEL_PATH = os.path.join("models", "cancellation_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    if not os.path.exists(MODEL_PATH):
        return None
    return CancellationPredictor(MODEL_PATH)


def main():
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.caption(
        "AI-powered decision support for hotel revenue managers — "
        "predicts the likelihood a booking will be cancelled, so teams can "
        "make smarter overbooking and follow-up decisions."
    )

    predictor = load_predictor()
    if predictor is None:
        st.error(
            "No trained model found at `models/cancellation_model.pkl`. "
            "Run `python src/train_model.py --data data/hotel_bookings.csv` first."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔮 Predict a Booking", "📊 Model Performance", "ℹ️ About This Project"])

    # ---------------- TAB 1: Prediction ----------------
    with tab1:
        st.subheader("Enter Booking Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
            lead_time = st.slider("Lead Time (days before arrival)", 0, 500, 60)
            arrival_month_num = st.selectbox(
                "Arrival Month", list(range(1, 13)),
                format_func=lambda m: [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December",
                ][m - 1],
            )
            arrival_week_number = st.slider("Arrival Week Number", 1, 53, 25)
            stays_total_nights = st.slider("Total Nights Booked", 0, 20, 3)

        with col2:
            adults = st.number_input("Adults", 1, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            babies = st.number_input("Babies", 0, 5, 0)
            meal = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC"])
            market_segment = st.selectbox(
                "Market Segment",
                ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"],
            )
            distribution_channel = st.selectbox(
                "Distribution Channel", ["TA/TO", "Direct", "Corporate"]
            )

        with col3:
            deposit_type = st.selectbox(
                "Deposit Type", ["No Deposit", "Non Refund", "Refundable"]
            )
            customer_type = st.selectbox(
                "Customer Type",
                ["Transient", "Contract", "Transient-Party", "Group"],
            )
            previous_cancellations = st.number_input("Previous Cancellations", 0, 20, 0)
            previous_bookings_not_canceled = st.number_input(
                "Previous Bookings Not Canceled", 0, 50, 0
            )
            adr = st.number_input("Average Daily Rate (ADR)", 0.0, 1000.0, 100.0)
            total_of_special_requests = st.slider("Special Requests", 0, 5, 0)

        with st.expander("Advanced options"):
            is_repeated_guest = st.checkbox("Repeated Guest?", value=False)
            booking_changes = st.number_input("Booking Changes", 0, 10, 0)
            days_in_waiting_list = st.number_input("Days in Waiting List", 0, 100, 0)
            required_car_parking_spaces = st.number_input("Car Parking Spaces Required", 0, 5, 0)
            room_type_mismatch = st.checkbox("Assigned room differs from reserved room?", value=False)

        if st.button("Predict Cancellation Risk", type="primary"):
            booking = dict(
                hotel=hotel,
                lead_time=lead_time,
                arrival_month_num=arrival_month_num,
                arrival_week_number=arrival_week_number,
                stays_total_nights=stays_total_nights,
                adults=adults,
                children=children,
                babies=babies,
                meal=meal,
                market_segment=market_segment,
                distribution_channel=distribution_channel,
                is_repeated_guest=int(is_repeated_guest),
                previous_cancellations=previous_cancellations,
                previous_bookings_not_canceled=previous_bookings_not_canceled,
                booking_changes=booking_changes,
                deposit_type=deposit_type,
                days_in_waiting_list=days_in_waiting_list,
                customer_type=customer_type,
                adr=adr,
                required_car_parking_spaces=required_car_parking_spaces,
                total_of_special_requests=total_of_special_requests,
                room_type_mismatch=int(room_type_mismatch),
            )

            result = predictor.predict_risk(booking)
            prob = result["cancel_probability"]
            label = result["risk_label"]

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                color = {"High Risk": "🔴", "Medium Risk": "🟠", "Low Risk": "🟢"}[label]
                st.metric("Cancellation Probability", f"{prob:.1%}")
                st.markdown(f"### {color} {label}")

            with c2:
                fig = px.bar(
                    x=[prob, 1 - prob],
                    y=["Cancel", "Show"],
                    orientation="h",
                    color=["Cancel", "Show"],
                    color_discrete_map={"Cancel": "#EF553B", "Show": "#00CC96"},
                    range_x=[0, 1],
                )
                fig.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if label == "High Risk":
                st.warning(
                    "**Recommendation:** Consider this slot for overbooking, "
                    "send a targeted confirmation reminder, or offer flexible rebooking."
                )
            elif label == "Medium Risk":
                st.info("**Recommendation:** Monitor this booking; a standard reminder is likely sufficient.")
            else:
                st.success("**Recommendation:** Low risk — no special action needed.")

    # ---------------- TAB 2: Model Performance ----------------
    with tab2:
        st.subheader("Model Evaluation Metrics")
        if os.path.exists(METRICS_PATH):
            import json
            with open(METRICS_PATH) as f:
                metrics = json.load(f)

            mcols = st.columns(5)
            mcols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            mcols[1].metric("Precision", f"{metrics.get('precision', 0):.1%}")
            mcols[2].metric("Recall", f"{metrics.get('recall', 0):.1%}")
            mcols[3].metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
            mcols[4].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

            st.caption(
                f"Evaluated on a held-out test set of {metrics.get('n_test', 'N/A')} bookings "
                f"(trained on {metrics.get('n_train', 'N/A')} bookings)."
            )
        else:
            st.warning("No metrics file found. Train the model to generate metrics.json.")

    # ---------------- TAB 3: About ----------------
    with tab3:
        st.subheader("About This Project")
        st.markdown(
            """
            **Industry:** Travel / Hospitality — Hotel Revenue Management

            **Problem:** Hotels lose revenue when bookings cancel unpredictably —
            either from empty rooms (under-booking) or guest walk-aways
            (blind overbooking). This tool predicts cancellation risk *per booking*
            so revenue managers can make targeted, data-driven decisions.

            **Data:** [Kaggle — Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

            **Model:** Gradient Boosting Classifier trained on booking, guest,
            and stay-characteristic features (lead time, deposit type, market
            segment, prior cancellation history, etc.)

            **Disclaimer:** This is an educational/demo decision-support tool,
            not a certified production revenue-management system.
            """
        )


if __name__ == "__main__":
    main()
