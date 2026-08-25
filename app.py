import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Customer_Churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. CREATE TARGET
# ==========================================

# 1 = Customer churned
# 0 = Customer did not churn

y = df["Churn Label_Yes"]


# ==========================================
# 3. CREATE FEATURES
# ==========================================

# Remove target column and columns that give
# information directly about the final outcome

remove_columns = [
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
]

X = df.drop(
    columns=remove_columns,
    errors="ignore"
)


# ==========================================
# 4. CHECK FEATURES
# ==========================================

print("\nNumber of features:")
print(len(X.columns))

print("\nFeature names:")
print(list(X.columns))


# ==========================================
# 5. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. CREATE SCALER
# ==========================================

# IMPORTANT:
# Fit the scaler using the SAME training
# columns that the model will use

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 8. TEST MODEL
# ==========================================

predictions = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:")
print(accuracy)


# ==========================================
# 9. CHECK MODEL AND SCALER FEATURES
# ==========================================

print("\n==============================")
print("FEATURE CHECK")
print("==============================")

print(
    "X feature count:",
    len(X.columns)
)

print(
    "Model feature count:",
    model.n_features_in_
)

print(
    "Scaler feature count:",
    scaler.n_features_in_
)


# Check that feature names match exactly

print("\nFeature names used for training:")
print(list(X.columns))

print("\nFeature names expected by scaler:")
print(list(scaler.feature_names_in_))


# Final comparison

features_match = (
    list(X.columns)
    ==
    list(scaler.feature_names_in_)
)

print("\nDo features match exactly?")
print(features_match)


# ==========================================
# 10. SAVE FILES
# ==========================================

# Save model
joblib.dump(
    model,
    "Customer_Churn.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "scaler.pkl"
)

# Save feature names
joblib.dump(
    list(X.columns),
    "features.pkl"
)


# ==========================================
# 11. LOAD FILES AGAIN TO VERIFY
# ==========================================

saved_model = joblib.load(
    "Customer_Churn.pkl"
)

saved_scaler = joblib.load(
    "scaler.pkl"
)

saved_features = joblib.load(
    "features.pkl"
)


print("\n==============================")
print("SAVED FILE CHECK")
print("==============================")

print(
    "Saved model features:",
    saved_model.n_features_in_
)

print(
    "Saved scaler features:",
    saved_scaler.n_features_in_
)

print(
    "Saved feature names:",
    len(saved_features)
)


# Check exact feature match

saved_features_match = (
    saved_features
    ==
    list(saved_scaler.feature_names_in_)
)

print(
    "\nSaved features match scaler:"
)

print(
    saved_features_match
)


# ==========================================
# 12. FINAL RESULT
# ==========================================

print("\n==============================")
print("FINAL CHECK")
print("==============================")

if (
    len(X.columns)
    == model.n_features_in_
    == scaler.n_features_in_
    == len(saved_features)
    and features_match
    and saved_features_match
):

    print("SUCCESS! All model files match correctly.")

else:

    print("ERROR! Feature files do not match.")


print("\nFiles created:")
print("Customer_Churn.pkl")
print("scaler.pkl")
print("features.pkl")
