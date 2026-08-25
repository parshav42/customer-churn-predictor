# ==========================================
# CUSTOMER CHURN MODEL - FULL TRAINING CODE
# ==========================================

# 1. Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================
# 2. Load dataset
# ==========================================

df = pd.read_csv("Customer_Churn.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 3. Remove unnecessary columns
# ==========================================

# Remove customer ID if it exists
if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)


# ==========================================
# 4. Convert Churn target to numbers
# ==========================================

# Change these values if your dataset uses
# different target labels

if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })


# ==========================================
# 5. Separate features and target
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==========================================
# 6. Convert categorical columns to numbers
# ==========================================

X = pd.get_dummies(X, drop_first=True)


# ==========================================
# 7. Split data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. Create and fit scaler
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 9. Train Logistic Regression model
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# ==========================================
# 10. Test model accuracy
# ==========================================

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)


# ==========================================
# 11. Check number of features
# ==========================================

print("\n================================")
print("FEATURE CHECK")
print("================================")

print("Model features:", model.n_features_in_)
print("Scaler features:", scaler.n_features_in_)
print("Saved features:", len(X.columns))

print("\nFeature names:")
print(X.columns.tolist())


# ==========================================
# 12. Save model files
# ==========================================

joblib.dump(model, "Customer_Churn.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "features.pkl")

print("\n================================")
print("FILES SAVED SUCCESSFULLY!")
print("================================")

print("Customer_Churn.pkl")
print("scaler.pkl")
print("features.pkl")
