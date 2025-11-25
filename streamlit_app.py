import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ===================== PAGE CONFIG =======================
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
)

# ===================== CUSTOM CSS =========================
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
        }
        .main {
            background-color: #0E1117;
        }
        h1, h2, h3, h4, label, p {
            color: #E4E4E7 !important;
        }
        .metric-card {
            background: #181A20;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL =========================
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ===================== HEADER =============================
st.title("🩺 Diabetes Prediction Dashboard")
st.write("A smart AI-powered tool to estimate diabetes risk using patient medical attributes.")

st.markdown("---")

# ===================== INPUT SECTION ======================
st.header("📋 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    age = st.number_input("Age", 1, 120, 30)

with col2:
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

with col3:
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

# ===================== FEATURE ENGINEERING ======================
age_bmi = age * bmi
glucose_bmi = glucose / (bmi + 1e-6)

input_data = pd.DataFrame({
    "Pregnancies": [preg],
    "Glucose": [glucose],
    "BloodPressure": [bp],
    "SkinThickness": [skin],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age],
    "Age_BMI": [age_bmi],
    "Glucose_per_BMI": [glucose_bmi],
})

# ===================== PREDICTION BUTTON ======================
st.markdown("---")
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] * 100

    # ------------ RESULTS ROW ------------
    st.header("📊 Prediction Results")
    rcol1, rcol2, rcol3 = st.columns(3)

    with rcol1:
        st.markdown(
            f"<div class='metric-card'><h3>Risk Probability</h3>"
            f"<h1>{prob:.2f}%</h1></div>",
            unsafe_allow_html=True
        )

    with rcol2:
        result_text = "High Risk" if prediction == 1 else "Low Risk"
        color = "red" if prediction == 1 else "green"
        st.markdown(
            f"<div class='metric-card'><h3>Prediction</h3>"
            f"<h1 style='color:{color}'>{result_text}</h1></div>",
            unsafe_allow_html=True
        )

    with rcol3:
        st.markdown(
            "<div class='metric-card'><h3>Model Used</h3>"
            "<h1>Random Forest</h1></div>",
            unsafe_allow_html=True
        )

    # ------------ CIRCULAR GAUGE METER (REAL ONE) ------------
    st.subheader("📟 Risk Level Indicator (Gauge Meter)")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 35], 'color': "green"},
                {'range': [35, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ------------ RISK ANALYSIS ------------
    st.subheader("🧠 Risk Factor Analysis")

    if prob > 60:
        st.error("⚠ *High Probability of Diabetes.*")
        st.write("Key contributing factors may include high glucose, insulin resistance, or obesity.")
    elif prob > 35:
        st.warning("🟡 *Moderate Risk.*")
        st.write("Monitor glucose & maintain a healthy lifestyle.")
    else:
        st.success("🟢 *Low Risk.*")
        st.write("Keep maintaining a healthy lifestyle!")

    # ------------ SUMMARY CARD ------------
    st.markdown("### 📄 Patient Summary")
    st.markdown(
        f"""
        - *Age:* {age}  
        - *BMI:* {bmi}  
        - *Glucose:* {glucose}  
        - *Blood Pressure:* {bp}  
        - *Insulin:* {insulin}  
        - *Pregnancies:* {preg}  
        """
    )
