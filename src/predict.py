import os
import json
import pandas as pd
import joblib


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
HEART_MODEL_PATH = os.path.join(MODELS_DIR, "best_heart_model.pkl")
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, "best_diabetes_model.pkl")
HYPERTENSION_MODEL_PATH = os.path.join(MODELS_DIR, "best_hypertension_model.pkl")
OBESITY_MODEL_PATH = os.path.join(MODELS_DIR, "best_obesity_model.pkl")
CHOLESTEROL_MODEL_PATH = os.path.join(MODELS_DIR, "best_cholesterol_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

MODEL_PATHS = {
    "diabetes": DIABETES_MODEL_PATH,
    "heart_disease": HEART_MODEL_PATH,
    "hypertension": HYPERTENSION_MODEL_PATH,
    "obesity": OBESITY_MODEL_PATH,
    "cholesterol": CHOLESTEROL_MODEL_PATH,
}


def _score_to_risk_level(score):
    score = max(0, min(100, score))
    if score <= 33:
        return "Low"
    elif score <= 66:
        return "Medium"
    else:
        return "High"


def load_model(path):
    return joblib.load(path)


def load_scaler(path=SCALER_PATH):
    return joblib.load(path)


def load_feature_columns(path=COLUMNS_PATH):
    return joblib.load(path)


def load_metrics(path=METRICS_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_risk(input_df, model_path, scaler_path=SCALER_PATH, columns_path=COLUMNS_PATH):
    scaler = load_scaler(scaler_path)
    model = load_model(model_path)
    feature_columns = load_feature_columns(columns_path)
    model_metrics = _lookup_model_metrics(model_path)

    from src.preprocess import CATEGORICAL_COLUMNS
    input_encoded = pd.get_dummies(input_df, columns=[
        c for c in CATEGORICAL_COLUMNS if c in input_df.columns
    ], drop_first=True)

    input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled_input = scaler.transform(input_aligned)
    scaled_row = pd.Series(scaled_input[0], index=feature_columns)
    score = float(model.predict(scaled_input)[0])
    score = round(max(0, min(100, score)), 2)
    feature_impacts = _get_feature_impacts(model, scaled_row, feature_columns)

    return {
        "risk_score": score,
        "risk_level": _score_to_risk_level(score),
        "risk_category": _score_to_risk_level(score),
        "confidence": _estimate_confidence(model_metrics),
        "contributing_factors": feature_impacts,
        "model_quality": model_metrics,
    }


def _lookup_model_metrics(model_path):
    metrics = load_metrics()
    disease_key = _metrics_key_from_model_path(model_path)
    disease_metrics = metrics.get(disease_key, {})
    best_model_name = disease_metrics.get("best_model")
    if not best_model_name:
        return {}
    model_metrics = disease_metrics.get(best_model_name, {}).copy()
    model_metrics["best_model"] = best_model_name
    return model_metrics


def _metrics_key_from_model_path(model_path):
    filename = os.path.basename(model_path)
    mapping = {
        "best_diabetes_model.pkl": "diabetes",
        "best_heart_model.pkl": "heart",
        "best_hypertension_model.pkl": "hypertension",
        "best_obesity_model.pkl": "obesity",
        "best_cholesterol_model.pkl": "cholesterol",
    }
    return mapping.get(filename, "")


def _estimate_confidence(model_metrics):
    r2_value = model_metrics.get("R2")
    if r2_value is None:
        return 0.0
    return round(max(0.0, min(1.0, float(r2_value))), 2)


def _get_feature_impacts(model, scaled_row, feature_columns, top_n=5):
    if hasattr(model, "coef_"):
        raw_impacts = {
            column: float(abs(coef) * abs(scaled_row[column]))
            for column, coef in zip(feature_columns, model.coef_)
        }
    elif hasattr(model, "feature_importances_"):
        raw_impacts = {
            column: float(importance * abs(scaled_row[column]))
            for column, importance in zip(feature_columns, model.feature_importances_)
        }
    else:
        return {}

    top_items = sorted(raw_impacts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return {_humanize_feature_name(feature): round(value, 4) for feature, value in top_items}


def _humanize_feature_name(feature):
    cleaned = feature.replace("_", " ").replace("Age", "Age")
    return cleaned.title()


def predict_heart_risk(input_df):
    return predict_risk(input_df, HEART_MODEL_PATH)


def predict_diabetes_risk(input_df):
    return predict_risk(input_df, DIABETES_MODEL_PATH)


def predict_hypertension_risk(input_df):
    return predict_risk(input_df, HYPERTENSION_MODEL_PATH)


def predict_obesity_risk(input_df):
    return predict_risk(input_df, OBESITY_MODEL_PATH)


def predict_cholesterol_risk(input_df):
    return predict_risk(input_df, CHOLESTEROL_MODEL_PATH)


def predict_all_risks(input_df):
    return {
        disease: predict_risk(input_df, model_path)
        for disease, model_path in MODEL_PATHS.items()
    }
