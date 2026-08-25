import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

/* IMPORT FONT */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

/* REMOVE STREAMLIT DEFAULTS */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}


/* GLOBAL */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(124, 92, 255, 0.12), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(34, 211, 238, 0.10), transparent 25%),
        #0B0F19;
    color: #F8FAFC;
    font-family: 'DM Sans', sans-serif;
}


/* MAIN CONTAINER */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}


/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827 0%,
        #0B0F19 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}


/* SIDEBAR LABELS */
[data-testid="stSidebar"] label {
    font-weight: 600;
}


/* HERO */
.hero {
    padding: 42px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            rgba(124,92,255,0.95),
            rgba(79,70,229,0.85)
        );
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow:
        0 30px 80px rgba(0,0,0,0.35);
    margin-bottom: 28px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.20);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 52px;
    margin: 18px 0 8px 0;
    font-weight: 800;
}

.hero p {
    font-size: 18px;
    opacity: 0.85;
    max-width: 650px;
}


/* GLASS CARD */
.premium-card {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.03)
        );
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 28px;
    backdrop-filter: blur(20px);
    box-shadow:
        0 20px 50px rgba(0,0,0,0.20);
    min-height: 190px;
}


/* SECTION TITLE */
.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
    color: #FFFFFF;
}

.section-subtitle {
    color: #94A3B8;
    margin-bottom: 22px;
}


/* METRICS */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF;
}


/* BUTTON */
.stButton > button {
    width: 100%;
    min-height: 58px;
    border: none;
    border-radius: 16px;
    font-size: 17px;
    font-weight: 800;
    color: white;
    background:
        linear-gradient(
            135deg,
            #8B5CF6,
            #6366F1
        );
    box-shadow:
        0 15px 35px rgba(99,102,241,0.35);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 20px 45px rgba(99,102,241,0.50);
}


/* PROGRESS */
.stProgress > div > div {
    border-radius: 20px;
}


/* RESULT CARD */
.result-card {
    padding: 32px;
    border-radius: 24px;
    margin-top: 20px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.10),
            rgba(255,255,255,0.04)
        );
    border: 1px solid rgba(255,255,255,0.12);
}


/* FOOTER */
.footer {
    text-align: center;
    color: #64748B;
    padding: 30px;
    font-size: 14px;
}


/* DIVIDER */
hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_files():
    model = joblib.load("Customer_Churn.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_files()


# IMPORTANT: USE EXACT SCALER FEATURES
features = list(scaler.feature_names_in_)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        ✦ AI CUSTOMER INTELLIGENCE
    </div>

    <h1>Predict churn.<br>Protect revenue.</h1>

    <p>
        Turn customer signals into actionable intelligence with
        machine-learning powered churn prediction.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TOP STATS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("AI ENGINE", "Active")

with c2:
    st.metric("MODEL", "Logistic AI")

with c3:
    st.metric("FEATURES", len(features))

with c4:
    st.metric("STATUS", "Ready")


st.write("")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# ✦ ChurnIQ")

st.sidebar.markdown(
    """
    <p style="color:#94A3B8;">
    Customer intelligence dashboard
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

st.sidebar.markdown("### Customer Profile")


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
    "Contract Type",
    [
        "Month-to-Month",
        "One Year",
        "Two Year"
    ]
)


st.sidebar.divider()

st.sidebar.markdown(
    """
    <p style="font-size:12px;color:#64748B;">
    Powered by Machine Learning<br>
    Customer Intelligence Platform
    </p>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CUSTOMER OVERVIEW
# =========================================================

left, right = st.columns([1.2, 1])


with left:

    st.markdown("""
    <div class="premium-card">
        <div class="section-title">
            Customer overview
        </div>
        <div class="section-subtitle">
            Current customer profile and engagement signals
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)

    with m1:
        st.metric("Customer Age", f"{Age} years")
        st.metric("Satisfaction", f"{Satisfaction} / 5")

    with m2:
        st.metric("Customer Tenure", f"{Tenure} months")
        st.metric("Contract", Contract)


with right:

    st.markdown("""
    <div class="premium-card">

        <div class="section-title">
            Intelligence engine
        </div>

        <div class="section-subtitle">
            Machine learning evaluates customer signals to estimate
            the probability of churn.
        </div>

        <p style="color:#CBD5E1; line-height:1.8;">
        The model processes behavioral and customer information
        to identify potential retention risks before they become
        revenue losses.
        </p>

    </div>
    """, unsafe_allow_html=True)


st.write("")
st.divider()


# =========================================================
# CREATE INPUT DATA
# =========================================================

input_df = pd.DataFrame(
    np.zeros((1, len(features))),
    columns=features
)


if "Age" in input_df.columns:
    input_df["Age"] = Age


if "Tenure in Months" in input_df.columns:
    input_df["Tenure in Months"] = Tenure


if "Satisfaction Score" in input_df.columns:
    input_df["Satisfaction Score"] = Satisfaction


if "Contract_One Year" in input_df.columns:
    input_df["Contract_One Year"] = (
        1 if Contract == "One Year" else 0
    )


if "Contract_Two Year" in input_df.columns:
    input_df["Contract_Two Year"] = (
        1 if Contract == "Two Year" else 0
    )


# =========================================================
# PREDICTION AREA
# =========================================================

st.markdown("""
<div style="text-align:center; margin-top:25px;">
    <h2 style="font-family:Manrope;">
        Ready to analyze?
    </h2>

    <p style="color:#94A3B8;">
        Generate an AI-powered churn prediction instantly.
    </p>
</div>
""", unsafe_allow_html=True)


if st.button("✦ Analyze Customer Risk"):

    try:

        with st.spinner("AI is analyzing customer signals..."):

            scaled_input = scaler.transform(
                input_df
            )

            prediction = model.predict(
                scaled_input
            )

            prediction_proba = model.predict_proba(
                scaled_input
            )

            churn_probability = float(
                prediction_proba[0][1]
            )


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )


        if prediction[0] == 1:

            st.markdown("## 🔴 High Churn Risk")

            st.write(
                "This customer shows signals associated with a higher probability of leaving."
            )

        else:

            st.markdown("## 🟢 Customer Likely to Stay")

            st.write(
                "This customer currently shows a relatively stable retention profile."
            )


        st.write("")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Churn Risk",
                f"{churn_probability * 100:.1f}%"
            )

        with r2:
            st.metric(
                "Stay Probability",
                f"{(1 - churn_probability) * 100:.1f}%"
            )

        with r3:

            if churn_probability < 0.30:
                risk = "LOW"

            elif churn_probability < 0.60:
                risk = "MEDIUM"

            else:
                risk = "HIGH"

            st.metric(
                "Risk Level",
                risk
            )


        st.write("")

        st.markdown("### Risk probability")

        st.progress(
            int(churn_probability * 100)
        )


        st.write("")


        # =================================================
        # RECOMMENDATION
        # =================================================

        st.markdown("### ✦ Recommended action")


        if churn_probability >= 0.60:

            st.warning("""
            **Immediate retention action recommended**

            Consider personalized outreach, targeted incentives,
            and proactive customer support.
            """)

        elif churn_probability >= 0.30:

            st.info("""
            **Monitor customer engagement**

            Review satisfaction levels and consider proactive
            retention communication.
            """)

        else:

            st.success("""
            **Customer appears stable**

            Maintain service quality and continue monitoring
            customer satisfaction.
            """)


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error("Prediction Error")

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    ✦ CHURNIQ INTELLIGENCE PLATFORM<br>

    <span style="font-size:12px;">
    Predict • Understand • Retain
    </span>

</div>
""", unsafe_allow_html=True)
