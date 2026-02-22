import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def compare_models(metrics_a, metrics_b, name_a="Model A", name_b="Model B"):
    if metrics_a["RMSE"] < metrics_b["RMSE"]:
        return name_a
    elif metrics_b["RMSE"] < metrics_a["RMSE"]:
        return name_b
    else:
        return name_a if metrics_a["R2"] >= metrics_b["R2"] else name_b


def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*40}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*40}")
    for key, value in metrics.items():
        print(f"  {key:>5}: {value:.4f}")
    print(f"{'='*40}\n")
