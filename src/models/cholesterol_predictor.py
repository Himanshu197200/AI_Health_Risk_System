from src.predict import CHOLESTEROL_MODEL_PATH
from src.models.base_predictor import BaseRiskPredictor


class CholesterolPredictor(BaseRiskPredictor):
    model_path = CHOLESTEROL_MODEL_PATH
