import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from src.preprocess import feature_engineering


# Load data
train_data = pd.read_csv("../data/bank-full_train.csv")

# Feature engineering
train_data = feature_engineering(train_data)

# Split features and target
X = train_data.drop("y", axis=1)
y = train_data["y"].map({"no": 0, "yes": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Identify columns
categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

# Pipeline
pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        )
    )
])

# Train model
pipeline.fit(X_train, y_train)

# Predictions
preds = pipeline.predict(X_test)

# Evaluation
print(classification_report(y_test, preds))

# Save model
joblib.dump(
    pipeline,
    "../models/xgboost_subscription_model.pkl"
)

print("\nModel saved successfully!")

