import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import (
    TARGET_COLUMNS,
    load_dataset,
    select_features,
    encode_categoricals,
    separate_features_targets,
    split_data,
)
from src.features import fit_scaler, transform_features, save_scaler, save_feature_columns
from src.train import train_candidate_models, save_model
from src.evaluate import calculate_metrics, compare_models, print_metrics


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_data.csv")


def run_pipeline():

    print("\nLoading dataset ...")
    df = load_dataset(DATA_PATH)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns.")

    print("Selecting important features ...")
    df = select_features(df)
    print(f"   Kept {len(df.columns)} columns (incl. targets).")

    print("Encoding categorical features ...")
    df = encode_categoricals(df)
    print(f"   Columns after encoding: {len(df.columns)}")

    X, targets = separate_features_targets(df)
    target_df = pd.DataFrame(targets)
    X_train, X_test, y_train_df, y_test_df = split_data(X, target_df)

    save_feature_columns(list(X_train.columns))
    print("   Feature columns saved -> models/feature_columns.pkl")

    print("Fitting scaler on training data ...")
    scaler = fit_scaler(X_train)
    save_scaler(scaler)
    print("   Scaler saved -> models/scaler.pkl")

    X_train_scaled = transform_features(scaler, X_train)
    X_test_scaled = transform_features(scaler, X_test)
    model_filename_map = {
        "diabetes_risk_score": "best_diabetes_model.pkl",
        "heart_disease_risk_score": "best_heart_model.pkl",
        "hypertension_risk_score": "best_hypertension_model.pkl",
        "obesity_risk_score": "best_obesity_model.pkl",
        "cholesterol_risk_score": "best_cholesterol_model.pkl",
    }
    metrics_key_map = {
        "diabetes_risk_score": "diabetes",
        "heart_disease_risk_score": "heart",
        "hypertension_risk_score": "hypertension",
        "obesity_risk_score": "obesity",
        "cholesterol_risk_score": "cholesterol",
    }
    display_name_map = {
        "diabetes_risk_score": "Diabetes",
        "heart_disease_risk_score": "Heart Disease",
        "hypertension_risk_score": "Hypertension",
        "obesity_risk_score": "Obesity",
        "cholesterol_risk_score": "Cholesterol",
    }

    metrics_report = {}

    for target in TARGET_COLUMNS:
        label = display_name_map[target]
        print(f"\nTraining models for {target} ...")

        y_train = y_train_df[target]
        y_test = y_test_df[target]

        candidate_models = train_candidate_models(X_train_scaled, y_train)
        candidate_metrics = {}

        for model_name, model in candidate_models.items():
            predictions = model.predict(X_test_scaled)
            metrics = calculate_metrics(y_test, predictions)
            candidate_metrics[model_name] = metrics
            print_metrics(metrics, f"{model_name} - {label}")

        best_model_name = compare_models(
            candidate_metrics["LinearRegression"],
            candidate_metrics["DecisionTree"],
            name_a="LinearRegression",
            name_b="DecisionTree",
        )
        best_model = candidate_models[best_model_name]
        filename = model_filename_map[target]
        save_model(best_model, filename)
        print(f"Best {label.lower()} model: {best_model_name} -> models/{filename}")

        metrics_report[metrics_key_map[target]] = {
            "LinearRegression": candidate_metrics["LinearRegression"],
            "DecisionTree": candidate_metrics["DecisionTree"],
            "best_model": best_model_name,
        }

    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print("Metrics saved -> models/metrics.json")

    print("\n" + "=" * 50)
    print("  Pipeline complete!")
    for key, details in metrics_report.items():
        print(f"     {key.title()} -> {details['best_model']}")
    print("  Artifacts saved in models/")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_pipeline()
