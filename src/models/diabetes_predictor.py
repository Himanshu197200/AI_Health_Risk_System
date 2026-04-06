from src.predict import DIABETES_MODEL_PATH
from src.models.base_predictor import BaseRiskPredictor


class DiabetesPredictor(BaseRiskPredictor):
    model_path = DIABETES_MODEL_PATH
