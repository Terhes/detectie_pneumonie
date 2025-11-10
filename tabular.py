# visualize_data.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load your data
df = pd.read_csv("Data_Entry_2017.csv")

# Clean columns
df = df.drop(columns=["Unnamed: 11", "Image Index"], errors="ignore")
df["Finding Labels"] = df["Finding Labels"].fillna("No Finding")

# Create binary pneumonia target (1 = pneumonia, 0 = not)
df["Pneumonia"] = df["Finding Labels"].apply(lambda x: 1 if "Pneumonia" in x else 0)

# Extract features and target
X = df[["Patient Age"]].values
y = df["Pneumonia"].values

# Scale ages (helps logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression (for probability curve)
model = LogisticRegression()
model.fit(X_scaled, y)

# Predict probabilities over age range
age_range = np.linspace(df["Patient Age"].min(), df["Patient Age"].max(), 200).reshape(-1, 1)
age_scaled = scaler.transform(age_range)
pneumonia_prob = model.predict_proba(age_scaled)[:, 1]

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df["Patient Age"], df["Pneumonia"], alpha=0.1, label="Patients (0=No, 1=Yes)")
plt.plot(age_range, pneumonia_prob, color="red", linewidth=2, label="Logistic Regression Fit")
plt.title("Probability of Pneumonia vs Patient Age")
plt.xlabel("Patient Age (years)")
plt.ylabel("Predicted Probability of Pneumonia")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Print sample probabilities
for a, p in zip(age_range[::40].ravel(), pneumonia_prob[::40]):
    print(f"Age {int(a):>3d}: {p*100:.2f}% chance of pneumonia")