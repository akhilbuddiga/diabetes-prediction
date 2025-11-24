import pandas as pd

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Step 1: Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Step 2: Basic statistics (to understand the data)
print("\nBasic information:")
print(df.describe())

# Step 3: Replace 0s with NaN in important columns
cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, pd.NA)

# Step 4: Fill missing values with column median
df = df.fillna(df.median(numeric_only=True))

# Step 5: Save cleaned data
df.to_csv("cleaned_diabetes.csv", index=False)

print("\nCleaning complete! Saved as cleaned_diabetes.csv")
