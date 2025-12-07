🩺 Diabetes Prediction ML Web App

A Machine Learning–powered web application that predicts whether a person is diabetic based on medical inputs. The app uses Random Forest Classifier, displays risk level using a gauge meter, and provides an option to download a doctor-friendly PDF report.

🚀 Project Overview

This project provides a user-friendly interface built with Streamlit to predict diabetes from structured input data.
The model is trained on the PIMA Diabetes Dataset, processed through a clean ML pipeline including:

Handling missing values

Label encoding (where needed)

Standard scaling

Model training using Random Forest

Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix

The final deployed model achieves ~88% accuracy.

🎯 Objectives

To build a reliable machine learning model that predicts diabetes with high accuracy.

To create an interactive, visually appealing web interface using Streamlit.

To make predictions easy to understand through a gauge-meter visualization.

To generate a downloadable doctor-friendly PDF report for users.

To allow prediction through manual input or uploaded medical report images (future scope).

To handle real-world data using feature preprocessing and ML pipeline techniques.

To deploy the model on a cloud platform for public accessibility.

🧬 Features

✔ Predict diabetes using trained ML model
✔ Clean and modern UI
✔ Input fields for all required medical parameters
✔ Gauge meter showing diabetes probability
✔ Downloadable PDF report
✔ Fully automated ML pipeline
✔ Trained on PIMA dataset
✔ High-accuracy Random Forest model
✔ Cloud-deployable over Streamlit Cloud or other services

🔧 Technologies Used

Python

NumPy, Pandas, Scikit-learn

Matplotlib / Seaborn (EDA)

Streamlit

Plotly (Gauge meter)

Random Forest Classifier

ReportLab (PDF generation)

📊 Model Performance

After training the Random Forest model on the cleaned dataset:

Metric	Value
Accuracy	~88%
Precision	(example) 0.86
Recall	(example) 0.81
F1-Score	(example) 0.83
Confusion Matrix	Included in report/analysis

Replace the example values with your actual calculated values.

🏗️ Project Workflow

Load and clean dataset

Preprocess features (scaling, encoding, handling outliers)

Split dataset into training and testing sets

Train Random Forest model

Evaluate performance

Build Streamlit UI

Integrate gauge meter + PDF generator

Deploy application

🧪 How to Run Locally
git clone https://github.com/your-repo-name
cd your-repo-name

pip install -r requirements.txt

streamlit run streamlit_app.py

📂 Folder Structure
├── diabetes_ml_pipeline.py
├── streamlit_app.py
├── requirements.txt
├── model.pkl
├── README.md
└── assets/

📥 Downloadable PDF Report

After the prediction, users can click a button to download a:
✔ Doctor-friendly
✔ Professional
✔ Easy-to-understand
PDF containing:

Patient entered values

Model prediction

Diabetes-risk gauge

Additional medical suggestions

👩‍💻 Developer

Your Name
Machine Learning Enthusiast • Python Developer

📜 License

This project is licensed under the MIT License
