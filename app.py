import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("Customer_Churn.csv")


# Target: 1 = churned, 0 = not churned
y = df["Churn Label_Yes"]


# Features
# Remove target and columns that leak customer status/churn information
X = df.drop(
    columns=[
        "Churn Label_Yes",
        "Customer Status_Joined",
        "Customer Status_Stayed",
        "Churn Score",
        "CLTV",
        "Churn Category_Competitor",
        "Churn Category_Dissatisfaction",
        "Churn Category_Other",
        "Churn Category_Price",
        "Churn Category_Unknown"
    ],
    errors="ignore"
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create scaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Test accuracy
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# Check features
print("\nModel features:", model.n_features_in_)
print("Scaler features:", scaler.n_features_in_)
print("Saved features:", len(X.columns))


# Save everything
joblib.dump(model, "Customer_Churn.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "features.pkl")

print("\nFiles saved successfully!")
print("Customer_Churn.pkl")
print("scaler.pkl")
print("features.pkl")
