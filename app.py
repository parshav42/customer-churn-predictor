import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("Customer_Churn.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")



st.title("Customer Churn Predictor")
st.subheader("Predict if customer will leave")
st.write("Enter customer details in the sidebar to predict whether the customer is likely to churn.")



t.sidebar.header("Customer Details")

Age = st.sidebar.slider("Age", 18, 100, 35)

Tenure = st.sidebar.slider("Tenure in Months", 0, 72, 12)

Satisfaction = st.sidebar.slider("Satisfaction Score", 1, 5, 3)

Contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-Month", "One Year", "Two Year"]
)

Monthly_Charges = st.sidebar.number_input(
    "Monthly Charges",
    0,
    200,
    50
)

# Create Input DataFrame

# Create empty DataFrame with all feature names and fill with zeros
input_df = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

# Fill user inputs into the correct columns
input_df["Age"] = Age
input_df["Tenure in Months"] = Tenure
input_df["Satisfaction Score"] = Satisfaction
input_df["Monthly Charges"] = Monthly_Charges

# Encode Contract type
input_df["Contract_One Year"] = 1 if Contract == "One Year" else 0
input_df["Contract_Two Year"] = 1 if Contract == "Two Year" else 0




# Prediction Button
if st.button("Predict Churn"):

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Get prediction probabilities
    prediction_proba = model.predict_proba(scaled_input)

    # Get churn probability
    churn_probability = prediction_proba[0][1]

    # Show result
    if prediction[0] == 1:
        st.error("🔴 Customer Will CHURN!")
        st.write(f"Churn Probability: {churn_probability * 100:.2f}%")
    else:
        st.success("🟢 Customer Will STAY!")
        st.write(f"Churn Probability: {churn_probability * 100:.2f}%")

    # Show Probability Bar
    st.subheader("Churn Probability")

    st.progress(int(churn_probability * 100))

    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )
