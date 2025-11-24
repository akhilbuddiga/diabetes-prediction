# step7_evaluation.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Step 7: Evaluating saved model...")

# 1) Load dataset
df = pd.read_csv("data/cleaned_diabetes_data.csv")

print("Missing values in each column:\n", df.isna().sum())


X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 2) Load saved scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("diabetes_model.pkl")

# 3) Scale features
X_scaled = scaler.transform(X)

# 4) Predict
y_pred = model.predict(X_scaled)

# 5) Evaluate
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print("\nAccuracy:", round(acc, 4))
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
