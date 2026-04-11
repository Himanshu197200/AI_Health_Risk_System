import unittest

import pandas as pd

from src.agent.health_agent import HealthAgent
from src.agent.knowledge_base import MedicalKnowledgeBase
from src.preprocess import FEATURE_COLUMNS, TARGET_COLUMNS, encode_categoricals, select_features
from src.predict import MODEL_PATHS, predict_all_risks


SAMPLE_PATIENT = {
    "Age": 58,
    "gender": "Male",
    "ethnicity": "Asian",
    "education_level": "Highschool",
    "income_level": "Lower-Middle",
    "employment_status": "Employed",
    "smoking_status": "Never",
    "alcohol_consumption_per_week": 0,
    "physical_activity_minutes_per_week": 215,
    "diet_score": 5.7,
    "sleep_hours_per_day": 7.9,
    "screen_time_hours_per_day": 7.9,
    "family_history_diabetes": 0,
    "hypertension_history": 0,
    "cardiovascular_history": 0,
    "bmi": 30.5,
    "waist_to_hip_ratio": 0.89,
    "systolic_bp": 134,
    "diastolic_bp": 78,
    "heart_rate": 68,
    "cholesterol_total": 239,
    "hdl_cholesterol": 41,
    "ldl_cholesterol": 160,
    "triglycerides": 145,
    "glucose_fasting": 136,
    "glucose_postprandial": 236,
    "insulin_level": 6.36,
    "hba1c": 8.18,
}


class HealthSystemTests(unittest.TestCase):
    def test_feature_and_target_constants_cover_expected_schema(self):
        self.assertEqual(len(FEATURE_COLUMNS), 28)
        self.assertEqual(len(TARGET_COLUMNS), 5)

    def test_categorical_encoding_expands_smoking_and_demographic_columns(self):
        df = pd.DataFrame(
            [
                {
                    **SAMPLE_PATIENT,
                    "diabetes_risk_score": 50,
                    "hypertension_risk_score": 52,
                    "heart_disease_risk_score": 49,
                    "obesity_risk_score": 56,
                    "cholesterol_risk_score": 51,
                },
                {
                    **{
                        **SAMPLE_PATIENT,
                        "gender": "Female",
                        "smoking_status": "Former",
                        "income_level": "Middle",
                    },
                    "diabetes_risk_score": 45,
                    "hypertension_risk_score": 48,
                    "heart_disease_risk_score": 41,
                    "obesity_risk_score": 47,
                    "cholesterol_risk_score": 43,
                },
            ]
        )
        selected = select_features(df)
        encoded = encode_categoricals(selected)
        self.assertIn("gender_Male", encoded.columns)
        self.assertIn("smoking_status_Never", encoded.columns)
        self.assertIn("income_level_Middle", encoded.columns)

    def test_all_model_paths_are_defined(self):
        self.assertEqual(set(MODEL_PATHS.keys()), {"diabetes", "heart_disease", "hypertension", "obesity", "cholesterol"})

    def test_prediction_pipeline_returns_five_scores(self):
        result = predict_all_risks(pd.DataFrame([SAMPLE_PATIENT]))
        self.assertEqual(len(result), 5)
        for payload in result.values():
            self.assertIn("risk_score", payload)
            self.assertIn("risk_category", payload)

    def test_health_agent_rule_based_report_is_generated_without_api_key(self):
        predictions = {
            "diabetes": {"risk_score": 62.0, "risk_category": "Medium", "contributing_factors": {"hba1c": 1.2}},
            "heart_disease": {"risk_score": 48.0, "risk_category": "Medium", "contributing_factors": {"ldl_cholesterol": 1.1}},
            "hypertension": {"risk_score": 54.0, "risk_category": "Medium", "contributing_factors": {"systolic_bp": 1.0}},
            "obesity": {"risk_score": 67.0, "risk_category": "High", "contributing_factors": {"bmi": 1.5}},
            "cholesterol": {"risk_score": 58.0, "risk_category": "Medium", "contributing_factors": {"cholesterol_total": 0.8}},
        }
        agent = HealthAgent()
        report = agent.analyze_patient(predictions, SAMPLE_PATIENT, "How can I lower my risk?")
        self.assertIn("Medical Disclaimer", report["report"])
        self.assertTrue(report["generation_mode"] in {"rule_based", "llm:groq", "llm:gemini"})

    def test_knowledge_base_retrieval_changes_with_query(self):
        kb = MedicalKnowledgeBase()
        diet_docs = kb.retrieve(diseases=["diabetes"], user_query="What diet changes lower diabetes risk?", top_k=2)
        pressure_docs = kb.retrieve(diseases=["hypertension"], user_query="How do I reduce blood pressure?", top_k=2)
        self.assertNotEqual(diet_docs[0]["id"], pressure_docs[0]["id"])

    def test_rule_based_report_reflects_user_query(self):
        predictions = {
            "diabetes": {"risk_score": 62.0, "risk_category": "Medium", "contributing_factors": {"hba1c": 1.2}, "model_quality": {"R2": 0.95, "MAE": 1.9, "best_model": "LinearRegression"}},
            "heart_disease": {"risk_score": 48.0, "risk_category": "Medium", "contributing_factors": {"ldl_cholesterol": 1.1}, "model_quality": {"R2": 0.88, "MAE": 4.2, "best_model": "LinearRegression"}},
            "hypertension": {"risk_score": 54.0, "risk_category": "Medium", "contributing_factors": {"systolic_bp": 1.0}, "model_quality": {"R2": 0.96, "MAE": 2.0, "best_model": "LinearRegression"}},
            "obesity": {"risk_score": 67.0, "risk_category": "High", "contributing_factors": {"bmi": 1.5}, "model_quality": {"R2": 0.96, "MAE": 1.5, "best_model": "LinearRegression"}},
            "cholesterol": {"risk_score": 58.0, "risk_category": "Medium", "contributing_factors": {"cholesterol_total": 0.8}, "model_quality": {"R2": 0.98, "MAE": 1.3, "best_model": "LinearRegression"}},
        }
        agent = HealthAgent()
        report_diet = agent.analyze_patient(predictions, SAMPLE_PATIENT, "What diet changes should I make?")
        report_bp = agent.analyze_patient(predictions, SAMPLE_PATIENT, "How can I lower my blood pressure?")
        self.assertNotEqual(report_diet["report"], report_bp["report"])
        self.assertIn("What diet changes should I make?", report_diet["report"])
        self.assertIn("How can I lower my blood pressure?", report_bp["report"])


if __name__ == "__main__":
    unittest.main()
