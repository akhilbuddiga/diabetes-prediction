# step6_prepare_and_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

print("Step 6: Preparing and training model...")

# 1) Load the cleaned dataset

df = pd.read_csv("data/cleaned_diabetes_data.csv")
print("Loaded data shape:", df.shape)

# **Check for missing values**
print("Missing values in each column:\n", df.isnull().sum())


print("Raw data loaded:", df.shape)

# 2) Split into features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3) Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5) Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete!")

# 6) Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# 7) Save model and scaler
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved as diabetes_model.pkl")
print("Scaler saved as scaler.pkl")
