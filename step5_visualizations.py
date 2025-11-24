import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Step 5 running...")

# Load the cleaned dataset
df = pd.read_csv("cleaned_diabetes.csv")
print("Data loaded!")

# Plot 1: Glucose distribution
plt.figure(figsize=(7,5))
sns.histplot(df['Glucose'], kde=True)
plt.title("Glucose Distribution")
plt.savefig("plot_glucose_hist.png")

# Plot 2: BMI Box plot
plt.figure(figsize=(7,5))
sns.boxplot(x=df['BMI'])
plt.title("BMI Boxplot")
plt.savefig("plot_bmi_box.png")

# Plot 3: Outcome counts
plt.figure(figsize=(7,5))
sns.countplot(x=df['Outcome'])
plt.title("Outcome Count")
plt.savefig("plot_outcome_counts.png")

print("Plots saved successfully!")
