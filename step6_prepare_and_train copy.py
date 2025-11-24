import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

print("Step 6: Preparing and training model...")

# 1️⃣ Load raw data
df = pd.read_csv("diabetes.csv")
print("Raw data loaded:", df.shape)

# 2️⃣ Clean data
df = df.dropna()
df.to_csv("cleaned_diabetes.csv", index=False)
print("Cleaned data saved!")

# 3️⃣ Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Data split done!")

# 4️⃣ Train logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
print("Model training complete!")

# 5️⃣ Show accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 6️⃣ Save model
pickle.dump(model, open("diabetes_model.pkl", "wb"))
print("Model saved as diabetes_model.pkl")
