import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("diabetes_model.pkl")

# Example patient data: [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]
new_patient = np.array([[2, 120, 70, 20, 79, 25.0, 0.5, 33]])

new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

print("Diabetes Prediction (0 = No, 1 = Yes):", prediction[0])
