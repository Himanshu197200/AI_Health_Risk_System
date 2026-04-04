import pandas as pd

from src.predict import predict_risk


class BaseRiskPredictor:
    model_path = None

    def predict(self, patient_data):
        if self.model_path is None:
            raise ValueError("model_path must be configured in subclasses.")
        input_df = pd.DataFrame([patient_data])
        return predict_risk(input_df, self.model_path)
