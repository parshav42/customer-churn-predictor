import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_model_files():
    model = joblib.load("Customer_Churn.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")

    return model, scaler, features


try:
    model, scaler, features = load_model_files()

except FileNotFoundError as e:
    st.error("Model files not found!")

    st.write(
        "Make sure these files are uploaded to your GitHub repository:"
    )

    st.code("""
Customer_Churn.pkl
scaler.pkl
features.pkl
""")

    st.stop()


# ==========================================
# TITLE
# ==========================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict "
    "whether the customer is likely to churn."
)


# ==========================================
# CREATE INPUT DATA
# ==========================================

input_data = {}


# ==========================================
# INPUT FIELDS
# ==========================================

st.subheader("Customer Information")

col1, col2 = st.columns(2)


for i, feature in enumerate(features):

    # Alternate between two columns
    current_col = col1 if i % 2 == 0 else col2

    with current_col:

        # Binary / one-hot encoded columns
        if (
            feature.startswith("Gender_")
            or feature.startswith("Married_")
            or feature.startswith("Dependents_")
            or feature.startswith("Phone Service_")
            or feature.startswith("Multiple Lines_")
            or feature.startswith("Internet Type_")
            or feature.startswith("Online Security_")
            or feature.startswith("Online Backup_")
            or feature.startswith("Device Protection_")
            or feature.startswith("Premium Tech Support_")
            or feature.startswith("Streaming TV_")
            or feature.startswith("Streaming Movies_")
            or feature.startswith("Streaming Music_")
            or feature.startswith("Unlimited Data_")
            or feature.startswith("Offer_")
            or feature.startswith("Contract_")
            or feature.startswith("Paperless Billing_")
            or feature.startswith("Payment Method_")
            or feature.startswith("City_")
        ):

            input_data[feature] = st.selectbox(
                feature,
                options=[0, 1],
                key=feature
            )


        # Numeric columns
        else:

            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                key=feature
            )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()


if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    try:

        # ======================================
        # CREATE DATAFRAME
        # ======================================

        input_df = pd.DataFrame([input_data])


        # ======================================
        # FORCE EXACT TRAINING FEATURE ORDER
        # ======================================

        input_df = input_df.reindex(
            columns=features,
            fill_value=0
        )


        # ======================================
        # FEATURE VALIDATION
        # ======================================

        if len(input_df.columns) != model.n_features_in_:

            st.error(
                f"Feature mismatch! "
                f"App has {len(input_df.columns)} features, "
                f"but model expects {model.n_features_in_}."
            )

            st.stop()


        # ======================================
        # SCALE INPUT
        # ======================================

        scaled_input = scaler.transform(input_df)


        # ======================================
        # PREDICT
        # ======================================

        prediction = model.predict(
            scaled_input
        )[0]

        probability = model.predict_proba(
            scaled_input
        )[0][1]


        # ======================================
        # DISPLAY RESULT
        # ======================================

        st.divider()

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if prediction == 1:

                st.error(
                    "⚠️ Customer is likely to churn"
                )

            else:

                st.success(
                    "✅ Customer is likely to stay"
                )


        with result_col2:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        # ======================================
        # PROBABILITY DETAILS
        # ======================================

        st.progress(float(probability))

        st.caption(
            f"Churn risk probability: "
            f"{probability * 100:.2f}%"
        )


    except Exception as e:

        st.error("Prediction Error")

        st.exception(e)
