import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from src.preprocess import feature_engineering

# Load data
train_data = pd.read_csv(
    "../data/bank-full_train.csv"
)

# Apply feature engineering
train_data = feature_engineering(train_data)

X = train_data.drop("y", axis=1)
y = train_data["y"].map({"no": 0, "yes": 1})

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Load model
model = joblib.load(
    "../models/xgboost_subscription_model.pkl"
)

# Predict
preds = model.predict(X_test)

# Metrics
print(classification_report(y_test, preds))

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        preds
    )
)