import streamlit as st
import pandas as pd
import numpy as np
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
# CUSTOM CSS - PREMIUM DESIGN
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #f5f7fb 0%, #e9eef7 100%);
}

/* Main title */
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #1f2937;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    text-align: center;
    color: #6b7280;
    margin-bottom: 35px;
}

/* Cards */
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Section title */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 15px;
}

/* Prediction button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    font-size: 20px;
    font-weight: 700;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


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
# GET EXACT FEATURES
# ==========================================

features = list(scaler.feature_names_in_)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <div class="main-title">
        📊 Customer Churn Predictor
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered customer retention prediction system
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# TOP METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Type",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Input Features",
        len(features)
    )

with col3:
    st.metric(
        "Prediction",
        "AI Powered"
    )


st.write("")


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("## 👤 Customer Details")

st.sidebar.markdown(
    "Enter customer information below to generate a churn prediction."
)


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
    "Contract Type",
    [
        "Month-to-Month",
        "One Year",
        "Two Year"
    ]
)


# ==========================================
# MAIN CONTENT
# ==========================================

left_column, right_column = st.columns([1.2, 1])


# ==========================================
# CUSTOMER SUMMARY
# ==========================================

with left_column:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Customer Profile</div>',
        unsafe_allow_html=True
    )

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:
        st.metric(
            "Age",
            Age
        )

        st.metric(
            "Satisfaction",
            f"{Satisfaction}/5"
        )

    with profile_col2:
        st.metric(
            "Tenure",
            f"{Tenure} Months"
        )

        st.metric(
            "Contract",
            Contract
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================
# MODEL INFORMATION
# ==========================================

with right_column:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Prediction Engine</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This machine learning model analyzes customer information "
        "to estimate the probability that a customer may leave."
    )

    st.write("")

    st.info(
        f"Model is using {len(features)} trained features."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================
# CREATE INPUT DATA
# ==========================================

input_df = pd.DataFrame(
    np.zeros((1, len(features))),
    columns=features
)


# ==========================================
# ADD USER INPUTS
# ==========================================

if "Age" in input_df.columns:
    input_df["Age"] = Age


if "Tenure in Months" in input_df.columns:
    input_df["Tenure in Months"] = Tenure


if "Satisfaction Score" in input_df.columns:
    input_df["Satisfaction Score"] = Satisfaction


# ==========================================
# CONTRACT ENCODING
# ==========================================

if "Contract_One Year" in input_df.columns:

    input_df["Contract_One Year"] = (
        1 if Contract == "One Year" else 0
    )


if "Contract_Two Year" in input_df.columns:

    input_df["Contract_Two Year"] = (
        1 if Contract == "Two Year" else 0
    )


# ==========================================
# PREDICTION SECTION
# ==========================================

st.write("")

st.markdown("### 🔮 Generate Prediction")


if st.button("Predict Customer Churn"):

    try:

        with st.spinner(
            "Analyzing customer data..."
        ):

            # Scale data
            scaled_input = scaler.transform(
                input_df
            )


            # Check model compatibility
            if (
                scaled_input.shape[1]
                != model.n_features_in_
            ):

                st.error(
                    f"""
                    Model feature mismatch.

                    Model expects:
                    {model.n_features_in_}

                    Scaler produced:
                    {scaled_input.shape[1]}
                    """
                )

                st.stop()


            # Prediction
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


        # ======================================
        # RESULT
        # ======================================

        st.write("")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if prediction[0] == 1:

                st.error(
                    "🔴 HIGH CHURN RISK"
                )

                st.write(
                    "This customer may be at risk of leaving."
                )

            else:

                st.success(
                    "🟢 LOW CHURN RISK"
                )

                st.write(
                    "This customer is likely to remain."
                )


        with result_col2:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )


        # ======================================
        # PROBABILITY BAR
        # ======================================

        st.markdown("### 📈 Churn Risk Score")

        st.progress(
            int(churn_probability * 100)
        )


        # ======================================
        # RISK LEVEL
        # ======================================

        if churn_probability < 0.30:

            risk_level = "Low Risk 🟢"

        elif churn_probability < 0.60:

            risk_level = "Medium Risk 🟡"

        else:

            risk_level = "High Risk 🔴"


        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:

            st.metric(
                "Risk Level",
                risk_level
            )


        with risk_col2:

            st.metric(
                "Stay Probability",
                f"{(1 - churn_probability) * 100:.2f}%"
            )


        with risk_col3:

            st.metric(
                "Model Features",
                len(features)
            )


        # ======================================
        # BUSINESS RECOMMENDATION
        # ======================================

        st.write("")

        st.markdown("### 💡 Business Recommendation")

        if churn_probability >= 0.60:

            st.warning(
                """
                High priority customer.

                Consider contacting the customer with a retention offer,
                service improvement, or personalized support.
                """
            )

        elif churn_probability >= 0.30:

            st.info(
                """
                Moderate churn risk.

                Monitor customer satisfaction and engagement.
                Consider targeted retention campaigns.
                """
            )

        else:

            st.success(
                """
                Customer appears stable.

                Continue providing good service and monitor
                satisfaction levels.
                """
            )


    except Exception as e:

        st.error("Prediction Error")

        st.exception(e)


# ==========================================
# FOOTER
# ==========================================

st.write("")

st.divider()

st.caption(
    "Customer Churn Predictor • Machine Learning Project • Streamlit"
)
