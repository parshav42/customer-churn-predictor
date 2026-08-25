import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)


# ==========================================
# LOAD FILES
# ==========================================

@st.cache_resource
def load_files():
    model = joblib.load("Customer_Churn.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_files()


# ==========================================
# IMPORTANT:
# USE EXACT FEATURES FROM SCALER
# ==========================================

features = list(scaler.feature_names_in_)


# ==========================================
# PAGE HEADER
# ==========================================

st.title("📊 Customer Churn Predictor")

st.subheader("Predict if customer will leave")

st.write(
    "Enter customer details below and click Predict Churn."
)


# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Customer Details")


Age = st.sidebar.slider(
    "Age",
    18,
    100,
    35
)

Tenure = st.sidebar.slider(
    "Tenure in Months",
    0,
    72,
    12
)

Satisfaction = st.sidebar.slider(
    "Satisfaction Score",
    1,
    5,
    3
)

Contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-Month",
        "One Year",
        "Two Year"
    ]
)


# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

# Start with all features expected by scaler
input_df = pd.DataFrame(
    np.zeros((1, len(features))),
    columns=features
)


# Add user values ONLY if those columns exist

if "Age" in input_df.columns:
    input_df["Age"] = Age


if "Tenure in Months" in input_df.columns:
    input_df["Tenure in Months"] = Tenure


if "Satisfaction Score" in input_df.columns:
    input_df["Satisfaction Score"] = Satisfaction


# Contract encoding

if "Contract_One Year" in input_df.columns:
    input_df["Contract_One Year"] = (
        1 if Contract == "One Year" else 0
    )


if "Contract_Two Year" in input_df.columns:
    input_df["Contract_Two Year"] = (
        1 if Contract == "Two Year" else 0
    )


# ==========================================
# DEBUG INFORMATION
# ==========================================

st.caption(
    f"Scaler expects {len(features)} features"
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Churn"):

    try:

        # Scale using exact scaler columns
        scaled_input = scaler.transform(
            input_df
        )


        # IMPORTANT:
        # Model expects the same number of features
        if scaled_input.shape[1] != model.n_features_in_:

            st.error(
                f"Model expects {model.n_features_in_} features "
                f"but scaler produced {scaled_input.shape[1]}."
            )

            st.stop()


        # Predict
        prediction = model.predict(
            scaled_input
        )


        # Probability
        prediction_proba = model.predict_proba(
            scaled_input
        )

        churn_probability = float(
            prediction_proba[0][1]
        )


        # Result

        if prediction[0] == 1:

            st.error(
                "🔴 Customer Will CHURN!"
            )

        else:

            st.success(
                "🟢 Customer Will STAY!"
            )


        # Probability

        st.subheader(
            "Churn Probability"
        )

        st.progress(
            int(churn_probability * 100)
        )

        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )


    except Exception as e:

        st.error("Prediction Error")

        st.exception(e)
