import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(_file_), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_heart_risk, predict_diabetes_risk


st.set_page_config(
    page_title="Patient Risk Assessment",
    page_icon="+",
    layout="centered",
)


def main():

    st.title("Intelligent Patient Risk Assessment System")
    st.markdown(
        "Enter clinical parameters below and click *Predict* to assess "
        "heart-disease and diabetes risk scores."
    )

    _show_model_metrics()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=20, max_value=300, value=130)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=10, max_value=120, value=50)

    with col2:
        glucose_fasting = st.number_input("Fasting Glucose (mg/dL)", min_value=40.0, max_value=500.0, value=100.0, step=0.1)
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        smoking_status = st.selectbox("Smoking Status", options=["Never", "Former", "Current"])
        activity = st.number_input("Physical Activity (min/week)", min_value=0, max_value=600, value=150)
        fam_diabetes = st.selectbox("Family History - Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    st.divider()

    if st.button("Predict Risk", use_container_width=True):
        input_data = {
            "Age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "ldl_cholesterol": ldl,
            "hdl_cholesterol": hdl,
            "glucose_fasting": glucose_fasting,
            "hba1c": hba1c,
            "smoking_status": smoking_status,
            "physical_activity_minutes_per_week": activity,
            "family_history_diabetes": fam_diabetes,
        }
        input_df = pd.DataFrame([input_data])

        try:
            heart_result = predict_heart_risk(input_df)
            diabetes_result = predict_diabetes_risk(input_df)

            st.success("Prediction complete!")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.subheader("Heart Disease Risk")
                st.metric("Risk Score", f"{heart_result['risk_score']:.2f} / 100")
                _show_risk_badge(heart_result["risk_level"])

            with res_col2:
                st.subheader("Diabetes Risk")
                st.metric("Risk Score", f"{diabetes_result['risk_score']:.2f} / 100")
                _show_risk_badge(diabetes_result["risk_level"])

        except FileNotFoundError:
            st.error(
                "Model files not found! Please run the training pipeline "
                "first (python main.py) to generate the model artifacts."
            )


def _show_risk_badge(level):
    st.markdown(f"### Risk Level: *{level}*")


def _show_model_metrics():
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")

    if not os.path.exists(metrics_path):
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    with st.sidebar:
        st.header("Model Performance")

        for disease, label in [("heart", "Heart Disease"), ("diabetes", "Diabetes")]:
            data = metrics[disease]
            best = data["best_model"]

            st.subheader(label)

            lr_r2 = data["LinearRegression"]["R2"]
            dt_r2 = data["DecisionTree"]["R2"]

            acc_col1, acc_col2 = st.columns(2)
            with acc_col1:
                tag = " (Active)" if best == "LinearRegression" else ""
                st.metric(f"Linear Reg{tag}", f"{lr_r2 * 100:.2f}%")
            with acc_col2:
                tag = " (Active)" if best == "DecisionTree" else ""
                st.metric(f"Decision Tree{tag}", f"{dt_r2 * 100:.2f}%")

            st.caption(f"Accuracy = R2 Score | Best: {best}")

            rows = []
            for model_name in ["LinearRegression", "DecisionTree"]:
                m = data[model_name]
                prefix = "> " if model_name == best else "  "
                rows.append({
                    "Model": f"{prefix}{model_name}",
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "R2": m["R2"],
                })

            st.dataframe(
                pd.DataFrame(rows).set_index("Model"),
                use_container_width=True,
            )
            st.divider()

if _name_ == "_main_":
    main()
