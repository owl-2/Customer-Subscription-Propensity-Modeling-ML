import pandas as pd
import joblib

from src.preprocess import feature_engineering

# Load model
model = joblib.load(
    "../models/xgboost_subscription_model.pkl"
)

# Example customer
new_customer = pd.DataFrame({
    'age':[65],
    'job':['retired'],
    'marital':['married'],
    'education':['secondary'],
    'default':['no'],
    'balance':[5000],
    'housing':['no'],
    'loan':['no'],
    'contact':['cellular'],
    'day':[15],
    'month':['mar'],
    'duration':[400],
    'campaign':[1],
    'pdays':[999],
    'previous':[0],
    'poutcome':['unknown']
})

# Apply feature engineering
new_customer = feature_engineering(new_customer)

# Predict
prediction = model.predict(new_customer)[0]

probability = model.predict_proba(new_customer)[0][1]

print(f"Prediction: {prediction}")
print(f"Subscription Probability: {probability:.2%}")