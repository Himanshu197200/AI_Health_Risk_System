from src.predict import HYPERTENSION_MODEL_PATH
from src.models.base_predictor import BaseRiskPredictor


class HypertensionPredictor(BaseRiskPredictor):
    model_path = HYPERTENSION_MODEL_PATH
