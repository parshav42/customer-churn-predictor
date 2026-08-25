import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load trained model files
model = joblib.load("Customer_Churn.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")


# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)


# Page Header
st.title("📊 Customer Churn Predictor")
st.subheader("Predict if a customer is likely to leave")
st.write(
    "Enter the customer details in the sidebar and click "
    "**Predict Churn** to see the prediction."
)


# Sidebar
st.sidebar.header("Customer Details")


Age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)


Tenure = st.sidebar.slider(
    "Tenure in Months",
    min_value=0,
    max_value=72,
    value=12
)


Satisfaction = st.sidebar.slider(
    "Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)


Contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-Month",
        "One Year",
        "Two Year"
    ]
)


# Create input DataFrame
input_df = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)


# Add user values
if "Age" in input_df.columns:
    input_df["Age"] = Age


if "Tenure in Months" in input_df.columns:
    input_df["Tenure in Months"] = Tenure


if "Satisfaction Score" in input_df.columns:
    input_df["Satisfaction Score"] = Satisfaction


# Contract encoding
if Contract == "One Year":
    if "Contract_One Year" in input_df.columns:
        input_df["Contract_One Year"] = 1


elif Contract == "Two Year":
    if "Contract_Two Year" in input_df.columns:
        input_df["Contract_Two Year"] = 1


# Make sure column order is correct
input_df = input_df[feature_names]


# Prediction button
if st.button("Predict Churn", use_container_width=True):

    try:
        # Scale data
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)

        # Get churn probability
        prediction_proba = model.predict_proba(scaled_input)

        churn_probability = float(
            prediction_proba[0][1]
        )


        # Show result
        if prediction[0] == 1:

            st.error("🔴 Customer Will CHURN!")

        else:

            st.success("🟢 Customer Will STAY!")


        # Probability section
        st.subheader("Churn Probability")

        st.progress(churn_probability)

        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )


    except Exception as e:

        st.error("Prediction Error")

        st.exception(e)
