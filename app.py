import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature names
model = joblib.load("Customer_Churn.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")


# Page Header
st.title("Customer Churn Predictor")
st.subheader("Predict if customer will leave")
st.write(
    "Enter customer details in the sidebar to predict whether "
    "the customer is likely to churn."
)


# Sidebar Inputs
st.sidebar.header("Customer Details")

Age = st.sidebar.slider("Age", 18, 100, 35)

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
    ["Month-to-Month", "One Year", "Two Year"]
)

Monthly_Charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0,
    max_value=200,
    value=50
)


# Use the exact feature names expected by the scaler
if hasattr(scaler, "feature_names_in_"):
    expected_features = list(scaler.feature_names_in_)
else:
    expected_features = list(feature_names)


# Create Input DataFrame
input_df = pd.DataFrame(
    np.zeros((1, len(expected_features))),
    columns=expected_features
)


# Fill user inputs only if those columns exist
if "Age" in input_df.columns:
    input_df["Age"] = Age

if "Tenure in Months" in input_df.columns:
    input_df["Tenure in Months"] = Tenure

if "Satisfaction Score" in input_df.columns:
    input_df["Satisfaction Score"] = Satisfaction

if "Monthly Charges" in input_df.columns:
    input_df["Monthly Charges"] = Monthly_Charges


# Encode Contract
if Contract == "One Year":
    if "Contract_One Year" in input_df.columns:
        input_df["Contract_One Year"] = 1

elif Contract == "Two Year":
    if "Contract_Two Year" in input_df.columns:
        input_df["Contract_Two Year"] = 1


# Ensure exact feature order
input_df = input_df[expected_features]


# Prediction Button
if st.button("Predict Churn"):

    try:
        # Scale input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)

        # Get probabilities
        prediction_proba = model.predict_proba(scaled_input)

        # Churn probability
        churn_probability = float(prediction_proba[0][1])

        # Show prediction result
        if prediction[0] == 1:
            st.error("🔴 Customer Will CHURN!")
        else:
            st.success("🟢 Customer Will STAY!")

        # Show probability
        st.subheader("Churn Probability")

        st.progress(churn_probability)

        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    except Exception as e:
        st.error("Prediction Error")
        st.exception(e)


# Debug information
with st.expander("Debug Feature Information"):
    st.write("Features expected by scaler:")
    st.write(expected_features)

    st.write("Features sent to scaler:")
    st.write(list(input_df.columns))
