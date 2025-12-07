🌟 Diabetes Prediction System — ML-Based Web App

A machine-learning powered web application built using Random Forest, deployed with Streamlit, and designed to help users assess their diabetes risk using health parameters.

📌 Project Highlights

🔍 ML Model: Random Forest (Accuracy: 88%)

📊 Feature Engineering: Scaling, outlier handling

🧪 Editable User Inputs: BMI, Glucose, Insulin, Age, Pregnancies, etc.

📈 Gauge Meter Visualization

📥 Downloadable Doctor-Friendly PDF Report

🖼️ Image Upload Feature for Test Reports (OCR)

🌐 Streamlit UI / Online Deployment

🧾 Table of Contents

📂 Project Structure

✨ Abstract

🎯 Objectives

🧬 Features & Feature Scope

🔍 Project Overview

📈 Results & Analysis

✔️ Conclusion

⚙️ Tech Stack

🚀 How to Run the Project

📄 License

📂 Project Structure
├── diabetes_ml_pipeline.py
├── streamlit_app.py
├── requirements.txt
├── model/
│   └── diabetes_random_forest.pkl
└── README.md

✨ Abstract

Diabetes is one of the fastest-growing health concerns globally. Early prediction and preventive care play a major role in reducing long-term complications.
This project aims to create a machine learning–based diagnostic assistant that predicts the likelihood of diabetes using medical features such as glucose level, BMI, insulin, and age. The application provides a user-friendly interface, accepts test report images, and generates doctor-friendly PDF reports, making it practical for academic, clinical, and personal use.

🎯 Objectives

To develop a supervised machine learning model capable of predicting diabetes risk with high accuracy.

To preprocess medical data with techniques like scaling, outlier handling, and feature transformation.

To evaluate and select the best model (Random Forest — 88% accuracy).

To integrate the model with an interactive Streamlit web app.

To provide users with:

📌 Gauge-meter visualization of risk level

📌 Instant prediction results

📌 Downloadable medical-style PDF report

📌 Ability to upload medical reports (OCR)

To build a system that is simple, scalable, and useful for non-technical users and healthcare workers.

🧬 Features & Feature Scope
✔️ Core Features
Feature	Description
🧠 ML Prediction	Predicts diabetes using a trained Random Forest model.
🎛️ Input Form	Users enter health parameters manually.
📈 Gauge Meter	Displays diabetes risk visually.
📄 PDF Report	Downloadable doctor-friendly prediction report.
🖼️ OCR Input (Optional)	Users can upload test reports; values are extracted automatically.
📊 Clean & Modern UI	Well-structured UI for smooth interaction.
🚀 Feature Scope (Future Enhancements)

Cloud-based medical data storage

Model improvement using deep learning

Multi-disease prediction expansion

Multi-language support

Patient history dashboard

🔍 Project Overview

This project follows a complete end-to-end machine learning pipeline:

1️⃣ Data Collection

Dataset containing medical attributes such as:

Glucose

Blood Pressure

Insulin

BMI

Diabetes Pedigree Function

Age

Pregnancies

2️⃣ Preprocessing

Handling missing values

Outlier removal

Feature scaling

Correlation check

3️⃣ Model Training

Multiple algorithms tested:

Logistic Regression

KNN

Random Forest

SVM

Random Forest performed the best (88% accuracy) and was selected.

4️⃣ Model Evaluation

Confusion Matrix

Precision, Recall, F1 Score

Accuracy Score

5️⃣ Streamlit Deployment

A clean UI was created for:

Input form

Gauge meter risk visualization

Prediction display

PDF export

Image upload OCR

📈 Results & Analysis
✔️ Model Performance
Metric	Score
Accuracy	88%
Precision	0.86
Recall	0.82
F1-Score	0.84
✔️ Confusion Matrix
	Predicted: No	Predicted: Yes
Actual: No	TP = 92	FP = 13
Actual: Yes	FN = 18	TN = 49
✔️ Inference

The model performs strongly in detecting diabetic users.

The Random Forest model provides stable performance due to ensemble learning.

A risk meter improves user understanding of prediction results.

✔️ Conclusion

The Diabetes Prediction System successfully demonstrates how machine learning can assist in early detection.
The combination of Random Forest, interactive UI, visual analytics, and PDF generation makes the system highly usable for educational and healthcare purposes.
With further enhancements—such as larger datasets and additional medical features—the system can evolve into a more advanced diagnostic tool.

⚙️ Tech Stack

Python

Pandas, NumPy, Scikit-Learn

Random Forest Classifier

Streamlit

Plotly

Pillow / OCR

FPDF / ReportLab

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Streamlit App
streamlit run streamlit_app.py

3️⃣ View in Browser
http://localhost:8501

📄 License

This project is for academic and research use.
