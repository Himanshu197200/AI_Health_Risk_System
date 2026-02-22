import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(scaler, X):
    return scaler.transform(X)


def save_scaler(scaler, path=SCALER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path=SCALER_PATH):
    return joblib.load(path)


def save_feature_columns(columns, path=COLUMNS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(columns, path)


def load_feature_columns(path=COLUMNS_PATH):
    return joblib.load(path)
