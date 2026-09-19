import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Load model and scaler
# -----------------------------

MODEL_DIR = Path(__file__).parent
model = joblib.load(MODEL_DIR / "fraud_detection_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter transaction features to predict whether "
    "the transaction is normal or fraudulent."
)

st.divider()


# -----------------------------
# Time and Amount
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Time",
        value=0.0
    )

with col2:
    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=100.0
    )


# -----------------------------
# V1 - V28
# -----------------------------

st.subheader("Transaction Features")

features = {}

columns = st.columns(4)

for i in range(1, 29):

    with columns[(i - 1) % 4]:

        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )


st.divider()


# -----------------------------
# Prediction button
# -----------------------------

if st.button(
    "🔍 Check Transaction",
    use_container_width=True
):

    # Create input dictionary

    input_dict = {
        "Time": time
    }

    input_dict.update(features)

    input_dict["Amount"] = amount


    # Create dataframe

    input_data = pd.DataFrame(
        [input_dict]
    )


    # -------------------------
    # Scale Time and Amount
    # -------------------------

    input_data[["Time", "Amount"]] = scaler.transform(
        input_data[["Time", "Amount"]]
    )


    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(
        input_data
    )[0]


    probability = model.predict_proba(
        input_data
    )[0][1]


    # -------------------------
    # Display result
    # -------------------------

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "🚨 FRAUDULENT TRANSACTION"
        )

        st.write(
            f"Fraud Probability: **{probability:.2%}**"
        )

    else:

        st.success(
            "✅ NORMAL TRANSACTION"
        )

        st.write(
            f"Fraud Probability: **{probability:.2%}**"
        )