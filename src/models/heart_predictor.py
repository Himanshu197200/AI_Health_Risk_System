from src.predict import HEART_MODEL_PATH
from src.models.base_predictor import BaseRiskPredictor


class HeartDiseasePredictor(BaseRiskPredictor):
    model_path = HEART_MODEL_PATH
