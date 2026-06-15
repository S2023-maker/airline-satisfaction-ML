import streamlit as st
import pandas as pd
import pickle

# ── Load saved artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("final_model_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("feature_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return pipeline, le, meta

pipeline, le, meta = load_artifacts()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Passenger Satisfaction Predictor")
st.caption(
    f"Model: **{meta['model_name']}** &nbsp;|&nbsp; "
    f"Test Accuracy: **{meta['test_accuracy']:.2%}** &nbsp;|&nbsp; "
    f"F1 Score: **{meta['test_f1']:.2%}**"
)
st.markdown("---")

# ── Sidebar — Passenger & Flight info ─────────────────────────────────────────
st.sidebar.header("🧍 Passenger & Flight Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
age = st.sidebar.slider("Age", 7, 85, 30)
type_of_travel = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.sidebar.selectbox("Class", ["Business", "Eco", "Eco Plus"])
flight_distance = st.sidebar.number_input("Flight Distance (miles)", 50, 5000, 500)
departure_delay = st.sidebar.number_input("Departure Delay (min)", 0, 300, 0)
arrival_delay = st.sidebar.number_input("Arrival Delay (min)", 0, 300, 0)

# ── Main — Service Ratings ────────────────────────────────────────────────────
st.subheader("⭐ Rate Your In-Flight Experience (0 = Not Applicable, 1–5)")

service_labels = {
    "departure_and_arrival_time_convenience": "Departure & Arrival Time Convenience",
    "ease_of_online_booking":                 "Ease of Online Booking",
    "check_in_service":                       "Check-in Service",
    "online_boarding":                        "Online Boarding",
    "gate_location":                          "Gate Location",
    "on_board_service":                       "On-board Service",
    "seat_comfort":                           "Seat Comfort",
    "leg_room_service":                       "Leg Room Service",
    "cleanliness":                            "Cleanliness",
    "food_and_drink":                         "Food & Drink",
    "in_flight_service":                      "In-flight Service",
    "in_flight_wifi_service":                 "In-flight Wi-Fi Service",
    "in_flight_entertainment":                "In-flight Entertainment",
    "baggage_handling":                       "Baggage Handling",
}

ratings = {}
cols = st.columns(2)
for i, (col_name, label) in enumerate(service_labels.items()):
    with cols[i % 2]:
        ratings[col_name] = st.slider(label, 0, 5, 3, key=col_name)

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Satisfaction", use_container_width=True, type="primary"):

    # Build input dataframe — column order must match training
    input_dict = {
        "id":                  0,          # placeholder; model was trained with it
        "gender":              gender,
        "customer_type":       customer_type,
        "age":                 age,
        "type_of_travel":      type_of_travel,
        "class":               travel_class,
        "flight_distance":     flight_distance,
        "departure_delay":     departure_delay,
        "arrival_delay":       arrival_delay,
        **ratings,
    }

    # Reorder to match training column order (cat_cols + num_cols minus 'satisfaction')
    ordered_cols = (
        meta["cat_cols"] +
        [c for c in meta["num_cols"] if c not in meta["cat_cols"]]
    )
    # Add any missing cols (like 'id') in the right spot
    all_cols = ["id"] + meta["cat_cols"] + [c for c in meta["num_cols"] if c not in ["id"] + meta["cat_cols"]]
    input_df = pd.DataFrame([input_dict])[all_cols]

    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    label = le.inverse_transform([prediction])[0]
    confidence = proba[prediction]

    # Display result
    st.markdown("### 🎯 Prediction Result")
    if label == "Satisfied":
        st.success(f"✅ **{label}** — Confidence: {confidence:.1%}")
    else:
        st.warning(f"⚠️ **{label}** — Confidence: {confidence:.1%}")

    col1, col2 = st.columns(2)
    col1.metric("Satisfied", f"{proba[1]:.1%}")
    col2.metric("Neutral or Dissatisfied", f"{proba[0]:.1%}")