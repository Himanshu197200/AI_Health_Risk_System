import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    return path
