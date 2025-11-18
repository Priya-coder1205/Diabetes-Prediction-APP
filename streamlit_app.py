import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered",
)

# -------------- LOAD MODEL & SCALER ----------------
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ------------------- UI STYLING --------------------
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        h1, h2, h3, label {
            color: white !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            height: 3rem;
            width: 100%;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE -----------------------
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

st.subheader("Enter Patient Details")

# ----------------- INPUT FIELDS ------------------
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# ----------------- FEATURE ENGINEERING -------------------
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

# ------------------- PREDICT -----------------------
if st.button("Predict"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠ High Risk of Diabetes (Probability: {prob:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {prob:.2f}%)")