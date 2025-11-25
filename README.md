# Diabetes Prediction (Machine Learning Project)

This project predicts whether a person is diabetic based on health-related features using Machine Learning.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## Project Structure
1. Load dataset  
2. Handle missing values  
3. Exploratory Data Analysis (EDA)  
4. Feature scaling  
5. Train/Test split  
6. Model training (RandomForestClassifier)  
7. Model evaluation  
8. Save model  
9. Predict using new data  
10. Build clean project structure  

---


---

## ▶️ How to Run

### 1️⃣ Install dependencies

### 2️⃣ Train model

### 3️⃣ Run prediction

---

## 📌 Author
**Akhil Buddiga**  
ML & AI Enthusiast
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

__pycache__/
*.pyc
.env
.venv
models/*.pkl
.ipynb_checkpoints/

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, scale_features

# Load data
df = load_data("data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale features
X_scaled, scaler = scale_features(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Training complete. Model saved to models/model.pkl")

import joblib
import numpy as np

# Load saved model & scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example patient values (paste any values)
data = np.array([[6,148,72,35,0,33.6,0.627,50]])

# Scale
data_scaled = scaler.transform(data)

# Predict
prediction = model.predict(data_scaled)

print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")



## Screenshots

### Login Page
![Login Page](<images/Screenshot 2025-11-22 at 5.52.53 PM.png>)

### Dashboard
![dashboard](<images/Screenshot 2025-11-22 at 5.55.07 PM.png>)

### Report Page
![Report Page](<images/Screenshot 2025-11-22 at 5.56.10 PM.png>)