import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "Age",
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
]

TARGET_DIABETES = "diabetes_risk_score"
TARGET_HYPERTENSION = "hypertension_risk_score"
TARGET_HEART = "heart_disease_risk_score"
TARGET_OBESITY = "obesity_risk_score"
TARGET_CHOLESTEROL = "cholesterol_risk_score"

TARGET_COLUMNS = [
    TARGET_DIABETES,
    TARGET_HEART,
    TARGET_HYPERTENSION,
    TARGET_OBESITY,
    TARGET_CHOLESTEROL,
]


def load_dataset(filepath):
    return pd.read_csv(filepath)


def select_features(df):
    keep_cols = FEATURE_COLUMNS + TARGET_COLUMNS
    return df[keep_cols].copy()


def encode_categoricals(df):
    return pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)


def separate_features_targets(df):
    X = df.drop(columns=TARGET_COLUMNS, errors="ignore").copy()
    targets = {target: df[target].copy() for target in TARGET_COLUMNS}
    return X, targets


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
