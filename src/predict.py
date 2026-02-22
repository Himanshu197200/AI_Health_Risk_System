import os
import numpy as np
import pandas as pd
import joblib


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
HEART_MODEL_PATH = os.path.join(MODELS_DIR, "best_heart_model.pkl")
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, "best_diabetes_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def _score_to_risk_level(score):
    score = max(0, min(100, score))
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Moderate"
    else:
        return "High"


def load_model(path):
    return joblib.load(path)


def load_scaler(path=SCALER_PATH):
    return joblib.load(path)


def load_feature_columns(path=COLUMNS_PATH):
    return joblib.load(path)


def predict_risk(input_df, model_path, scaler_path=SCALER_PATH, columns_path=COLUMNS_PATH):
    scaler = load_scaler(scaler_path)
    model = load_model(model_path)
    feature_columns = load_feature_columns(columns_path)

    from src.preprocess import CATEGORICAL_COLUMNS
    input_encoded = pd.get_dummies(input_df, columns=[
        c for c in CATEGORICAL_COLUMNS if c in input_df.columns
    ], drop_first=True)

    input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled_input = scaler.transform(input_aligned.values)
    score = float(model.predict(scaled_input)[0])
    score = round(max(0, min(100, score)), 2)

    return {
        "risk_score": score,
        "risk_level": _score_to_risk_level(score),
    }


def predict_heart_risk(input_df):
    return predict_risk(input_df, HEART_MODEL_PATH)


def predict_diabetes_risk(input_df):
    return predict_risk(input_df, DIABETES_MODEL_PATH)
