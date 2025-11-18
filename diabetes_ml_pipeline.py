# diabetes_ml_pipeline.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)

sns.set(style="whitegrid")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Dataset Loading...")
df = pd.read_csv("diabetes.csv")
print("First 5 rows:\n", df.head(), "\n")

# -------------------------
# Data cleaning: replace 0 with NaN in specific columns and impute by column median
# -------------------------
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("Missing values before filling:\n", df[cols_with_zero].isnull().sum() + (df[cols_with_zero] == 0).sum(), "\n")

for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)
    # teacher-friendly: impute with group median by Outcome to preserve class distribution
    df[col] = df.groupby("Outcome")[col].transform(lambda x: x.fillna(x.median()))

print("Missing values after filling:\n", df[cols_with_zero].isnull().sum(), "\n")

# -------------------------
# Basic EDA - save plots (non-blocking)
# -------------------------
# Outcome distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df)
plt.title("Outcome Distribution (0 = Non-diabetic, 1 = Diabetic)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.savefig(os.path.join(OUT_DIR, "outcome_distribution.png"))
plt.close()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='RdYlBu', fmt=".2f")
plt.title("Feature Correlation")
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
plt.close()

# Feature distributions for diabetics vs non-diabetics (example: Glucose, BMI)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data=df, x="Glucose", hue="Outcome", kde=True, bins=30)
plt.title("Glucose distribution by Outcome")
plt.subplot(1,2,2)
sns.histplot(data=df, x="BMI", hue="Outcome", kde=True, bins=30)
plt.title("BMI distribution by Outcome")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "glucose_bmi_dist.png"))
plt.close()

print("EDA plots saved in 'outputs/' directory.")

# -------------------------
# Feature engineering (small, explainable)
# -------------------------
df['Age_BMI'] = df['Age'] * df['BMI']
df['Glucose_per_BMI'] = df['Glucose'] / (df['BMI'] + 1e-6)

# Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Models to train (only the 3 requested)
# Logistic Regression, SVM (RBF), Random Forest
# -------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM_RBF": SVC(kernel='rbf', probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced')
}

results = {}

# Optional: simple RF hyperparameter tuning (teacher-friendly GridSearch for RF)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8, None]
}

print("Training models...")

for name, model in models.items():
    if name == "RandomForest":
        gs = GridSearchCV(model, rf_param_grid, cv=4, scoring='roc_auc', n_jobs=-1)
        gs.fit(X_train_scaled, y_train)
        best = gs.best_estimator_
        print(f"\nRandomForest best params: {gs.best_params_}")
        clf = best
    else:
        clf = model
        clf.fit(X_train_scaled, y_train)

    if name != "RandomForest":
        # for non-RF we already fit above
        pass
    # if RF, clf is already fitted by GridSearchCV

    # ensure fitted
    if not hasattr(clf, "predict"):
        clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC-AUC: {roc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    # confusion matrix saved
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUT_DIR, f"confusion_{name}.png"))
    plt.close()

# -------------------------
# Compare models: bar chart for Accuracy, Precision, Recall, F1, ROC-AUC
# -------------------------
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
comp_df = pd.DataFrame({name: {m: results[name][m] for m in metrics} for name in results}).T
comp_df_plot = comp_df[metrics] * 100  # convert to %
comp_df_plot.plot(kind='bar', figsize=(10,6))
plt.title("Model Comparison (%)")
plt.ylabel("Score (%)")
plt.ylim(50, 100)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"))
plt.close()

# -------------------------
# Plot ROC curves for each model
# -------------------------
plt.figure(figsize=(8,6))
for name in results:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc='lower right')
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"))
plt.close()

# -------------------------
# Save best model (by accuracy or roc_auc — choose ROC-AUC for medical tasks)
# -------------------------
best_by_roc = max(results.items(), key=lambda kv: kv[1]['roc_auc'])[0]
best_model = results[best_by_roc]['model']
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"\n✅ Best Model Selected: {best_by_roc}")
print("Outputs saved to 'outputs/' and model saved as 'best_model.pkl' and 'scaler.pkl'")