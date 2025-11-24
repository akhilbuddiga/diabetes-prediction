print("Step 4 File Loaded!")  # test message

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("Import Done")

# 1. Load data
df = pd.read_csv("data/cleaned_diabetes_data.csv")
print("Data Loaded")
print(df.head())

# 2. Split features (X) and label (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data Split Done")

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model Training Done")

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Accuracy
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# 7. Save model
joblib.dump(model, "diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")
