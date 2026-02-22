import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import load_dataset, select_features, encode_categoricals, separate_features_targets, split_data
from src.features import fit_scaler, transform_features, save_scaler, save_feature_columns
from src.train import train_linear_regression, train_decision_tree, save_model
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

    X, y_heart, y_diabetes = separate_features_targets(df)

    X_train, X_test, y_heart_train, y_heart_test = split_data(X, y_heart)
    _, _, y_diab_train, y_diab_test = split_data(X, y_diabetes)

    save_feature_columns(list(X_train.columns))
    print("   Feature columns saved -> models/feature_columns.pkl")

    print("Fitting scaler on training data ...")
    scaler = fit_scaler(X_train)
    save_scaler(scaler)
    print("   Scaler saved -> models/scaler.pkl")

    X_train_scaled = transform_features(scaler, X_train)
    X_test_scaled = transform_features(scaler, X_test)

    print("\nTraining models for heart_disease_risk_score ...")

    lr_heart = train_linear_regression(X_train_scaled, y_heart_train)
    dt_heart = train_decision_tree(X_train_scaled, y_heart_train)

    lr_heart_preds = lr_heart.predict(X_test_scaled)
    dt_heart_preds = dt_heart.predict(X_test_scaled)

    lr_heart_metrics = calculate_metrics(y_heart_test, lr_heart_preds)
    dt_heart_metrics = calculate_metrics(y_heart_test, dt_heart_preds)

    print_metrics(lr_heart_metrics, "LinearRegression - Heart")
    print_metrics(dt_heart_metrics, "DecisionTree - Heart")

    best_heart_name = compare_models(
        lr_heart_metrics, dt_heart_metrics,
        name_a="LinearRegression", name_b="DecisionTree",
    )
    best_heart_model = lr_heart if best_heart_name == "LinearRegression" else dt_heart
    save_model(best_heart_model, "best_heart_model.pkl")
    print(f"Best heart model: {best_heart_name} -> models/best_heart_model.pkl")

    print("\nTraining models for diabetes_risk_score ...")

    lr_diab = train_linear_regression(X_train_scaled, y_diab_train)
    dt_diab = train_decision_tree(X_train_scaled, y_diab_train)

    lr_diab_preds = lr_diab.predict(X_test_scaled)
    dt_diab_preds = dt_diab.predict(X_test_scaled)

    lr_diab_metrics = calculate_metrics(y_diab_test, lr_diab_preds)
    dt_diab_metrics = calculate_metrics(y_diab_test, dt_diab_preds)

    print_metrics(lr_diab_metrics, "LinearRegression - Diabetes")
    print_metrics(dt_diab_metrics, "DecisionTree - Diabetes")

    best_diab_name = compare_models(
        lr_diab_metrics, dt_diab_metrics,
        name_a="LinearRegression", name_b="DecisionTree",
    )
    best_diab_model = lr_diab if best_diab_name == "LinearRegression" else dt_diab
    save_model(best_diab_model, "best_diabetes_model.pkl")
    print(f"Best diabetes model: {best_diab_name} -> models/best_diabetes_model.pkl")

    metrics_report = {
        "heart": {
            "LinearRegression": lr_heart_metrics,
            "DecisionTree": dt_heart_metrics,
            "best_model": best_heart_name,
        },
        "diabetes": {
            "LinearRegression": lr_diab_metrics,
            "DecisionTree": dt_diab_metrics,
            "best_model": best_diab_name,
        },
    }
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print("Metrics saved -> models/metrics.json")

    print("\n" + "=" * 50)
    print("  Pipeline complete!")
    print(f"     Heart  -> {best_heart_name}")
    print(f"     Diabetes -> {best_diab_name}")
    print("  Artifacts saved in models/")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_pipeline()
