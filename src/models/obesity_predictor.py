from src.predict import OBESITY_MODEL_PATH
from src.models.base_predictor import BaseRiskPredictor


class ObesityPredictor(BaseRiskPredictor):
    model_path = OBESITY_MODEL_PATH
