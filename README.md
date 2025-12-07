⭐ Diabetes Prediction System with Machine Learning
🩺 A Smart Health-Monitoring Application Built Using Random Forest & Streamlit
📌 Overview

This project is a Machine Learning–based Diabetes Prediction System designed to assist users in assessing their likelihood of diabetes using basic medical parameters such as glucose level, BMI, age, blood pressure, etc.
The system processes the user input and predicts the outcome using a Random Forest Classifier, achieving an accuracy of ~88%.

The app also includes:
✔ A modern UI using Streamlit
✔ Gauge Meter Visualization
✔ Doctor-Friendly PDF Report Download
✔ Supports manual input or test-report image upload (OCR)
✔ Feature importance visualization

📊 Demo Preview


🧠 Machine Learning Model

Algorithm Used: Random Forest Classifier

Accuracy Achieved: 88%

Preprocessing Steps:

Handling missing values

Scaling numeric features (MinMaxScaler / StandardScaler)

Outlier handling

Train-test split (80–20)

Dataset: PIMA Diabetes Dataset

🔍 Features
🧮 Machine Learning Features

Random Forest–based classifier

Feature importance visualization

Performance metrics:

Accuracy

Precision

Recall

Confusion Matrix

🖥 Application Features

Clean and responsive Streamlit UI

Input form for all 8 medical parameters

Gauge meter showing diabetes risk

Generate Doctor-Friendly PDF Report

Upload test-report image → extract values using OCR

Light & simple interface for non-technical users

🧾 Input Parameters
Feature	Description
Pregnancies	Number of pregnancies
Glucose	Plasma glucose concentration
Blood Pressure	Diastolic blood pressure
Skin Thickness	Triceps skin fold thickness
Insulin	2-Hour serum insulin
BMI	Body Mass Index
Diabetes Pedigree Function	Genetic influence score
Age	Age in years
📈 Model Performance
Metric	Value
Accuracy	88%
Precision	High (Class wise depends)
Recall	High for diabetic class
Confusion Matrix	Balanced without heavy bias
⚙ Technology Stack
Category	Tech Used
ML Model	Python, Scikit-learn
Frontend/UI	Streamlit
Visualization	Plotly
OCR (Optional)	pytesseract
Deployment	Streamlit Cloud / GitHub

🚀 How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/<your-username>/diabetes-prediction-app.git
cd diabetes-prediction-app

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit
streamlit run streamlit_app.py

📥 Deployment

To deploy on Streamlit Cloud:

Push code to GitHub

Go to share.streamlit.io

Select repo → Select streamlit_app.py → Deploy

📘 Generate PDF Report

User gets a doctor-friendly PDF

Contains input values, ML prediction, gauge meter snapshot

Can be downloaded instantly

👨‍🏫 Use Case

Early diabetes risk screening

Helpful for hospitals, clinics, and health camps

Academic machine learning project

Demonstration of ML deployment skills

📝 License

This project is licensed under the MIT License.

🙌 Acknowledgment

Special appreciation to educators, data providers, open-source libraries, and the PIMA dataset creators.

⭐ If You Like This Project

Please ⭐ star this repository — it motivates further improvements!
