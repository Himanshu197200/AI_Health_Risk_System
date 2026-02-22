import pandas as pd
from sklearn.model_selection import train_test_split


SELECTED_FEATURES = [
    "Age",
    "bmi",
    "systolic_bp",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "glucose_fasting",
    "hba1c",
    "smoking_status",
    "physical_activity_minutes_per_week",
    "family_history_diabetes",
]

CATEGORICAL_COLUMNS = ["smoking_status"]

TARGET_HEART = "heart_disease_risk_score"
TARGET_DIABETES = "diabetes_risk_score"


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df


def select_features(df):
    keep_cols = SELECTED_FEATURES + [TARGET_HEART, TARGET_DIABETES]
    return df[keep_cols].copy()


def encode_categoricals(df):
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return df


def separate_features_targets(df):
    target_cols = [TARGET_HEART, TARGET_DIABETES]
    X = df.drop(columns=target_cols, errors="ignore").copy()
    y_heart = df[TARGET_HEART].copy()
    y_diabetes = df[TARGET_DIABETES].copy()
    return X, y_heart, y_diabetes


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
