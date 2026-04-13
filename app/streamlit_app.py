import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.agent import HealthAgent
from src.models import (
    CholesterolPredictor,
    DiabetesPredictor,
    HeartDiseasePredictor,
    HypertensionPredictor,
    ObesityPredictor,
)
from src.utils import (
    create_feature_importance_chart,
    create_risk_comparison_bar,
    create_risk_gauge,
    generate_pdf_report,
)


st.set_page_config(
    page_title="AI Health Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {font-size: 2.7rem; font-weight: 700; color: #1a365d;}
        .subtitle {color: #4a5568; margin-bottom: 1rem;}
        .section-card {
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            border: 1px solid #d8e2ef;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .badge-low, .badge-medium, .badge-high {
            padding: 0.55rem 0.9rem; border-radius: 999px; display: inline-block;
            font-weight: 600; margin-top: 0.4rem;
        }
        .badge-low {background: #e6fffa; color: #22543d;}
        .badge-medium {background: #feebc8; color: #9c4221;}
        .badge-high {background: #fed7d7; color: #742a2a;}
        .metric-panel {
            background: linear-gradient(135deg, #edf2f7, #f7fafc);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .report-hero {
            background: linear-gradient(135deg, #12355b, #2f6ea8);
            color: white;
            border-radius: 24px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(18, 53, 91, 0.22);
        }
        .report-metric {
            background: #ffffff;
            border: 1px solid #dce5f0;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            min-height: 112px;
        }
        .report-metric-label {
            color: #66788a;
            font-size: 0.88rem;
            margin-bottom: 0.3rem;
        }
        .report-metric-value {
            color: #17324d;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .report-pill {
            display: inline-block;
            padding: 0.35rem 0.72rem;
            margin: 0.2rem 0.35rem 0 0;
            background: #edf2f7;
            border: 1px solid #d7e0ea;
            border-radius: 999px;
            color: #23405c;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .report-section {
            background: white;
            border: 1px solid #dde6f0;
            border-radius: 20px;
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .report-section h2 {
            color: #17324d;
            margin-bottom: 0.65rem;
            font-size: 1.15rem;
        }
        .report-section p, .report-section li {
            color: #334e68;
            line-height: 1.65;
            font-size: 0.98rem;
        }
        .answer-card {
            background: linear-gradient(135deg, #fff8e1, #fff3c4);
            border: 1px solid #f0d98a;
            border-radius: 22px;
            padding: 1.05rem 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 30px rgba(171, 132, 32, 0.12);
        }
        .answer-card h2 {
            color: #7b4f00;
            margin-bottom: 0.5rem;
        }
        .answer-card p, .answer-card li {
            color: #5c4700;
            line-height: 1.7;
            font-size: 1rem;
        }
        .evidence-card {
            background: #f8fbff;
            border: 1px solid #d9e7f7;
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin: 0.75rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DISEASE_LABELS = {
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "hypertension": "Hypertension",
    "obesity": "Obesity",
    "cholesterol": "Cholesterol",
}

EXAMPLE_PATIENT = {
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


@st.cache_resource
def load_models():
    return {
        "diabetes": DiabetesPredictor(),
        "heart_disease": HeartDiseasePredictor(),
        "hypertension": HypertensionPredictor(),
        "obesity": ObesityPredictor(),
        "cholesterol": CholesterolPredictor(),
    }


@st.cache_resource
def load_agent():
    return HealthAgent()


def load_metrics():
    path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_patient_data():
    with st.sidebar:
        st.header("Patient Profile")
        if st.button("Load Example Patient", use_container_width=True):
            for key, value in EXAMPLE_PATIENT.items():
                st.session_state[key] = value
            st.rerun()

        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.get("Age", 45))
        gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=["Female", "Male", "Other"].index(st.session_state.get("gender", "Male")))
        ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "Other", "White"], index=["Asian", "Black", "Hispanic", "Other", "White"].index(st.session_state.get("ethnicity", "Asian")))
        education = st.selectbox("Education Level", ["Graduate", "Highschool", "No formal", "Postgraduate"], index=["Graduate", "Highschool", "No formal", "Postgraduate"].index(st.session_state.get("education_level", "Graduate")))
        income = st.selectbox("Income Level", ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"], index=["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"].index(st.session_state.get("income_level", "Middle")))
        employment = st.selectbox("Employment Status", ["Employed", "Retired", "Student", "Unemployed"], index=["Employed", "Retired", "Student", "Unemployed"].index(st.session_state.get("employment_status", "Employed")))

        st.subheader("Lifestyle")
        smoking = st.selectbox("Smoking Status", ["Current", "Former", "Never"], index=["Current", "Former", "Never"].index(st.session_state.get("smoking_status", "Never")))
        alcohol = st.number_input("Alcohol Consumption / Week", min_value=0, max_value=15, value=int(st.session_state.get("alcohol_consumption_per_week", 2)))
        activity = st.number_input("Physical Activity (min/week)", min_value=0, max_value=900, value=int(st.session_state.get("physical_activity_minutes_per_week", 150)))
        diet_score = st.slider("Diet Score", min_value=0.0, max_value=10.0, value=float(st.session_state.get("diet_score", 6.0)), step=0.1)
        sleep = st.slider("Sleep Hours / Day", min_value=3.0, max_value=10.0, value=float(st.session_state.get("sleep_hours_per_day", 7.0)), step=0.1)
        screen_time = st.slider("Screen Time Hours / Day", min_value=0.5, max_value=16.8, value=float(st.session_state.get("screen_time_hours_per_day", 6.0)), step=0.1)

        st.subheader("History")
        family_history = st.selectbox("Family History Of Diabetes", [0, 1], format_func=lambda value: "Yes" if value else "No", index=int(st.session_state.get("family_history_diabetes", 0)))
        hypertension_history = st.selectbox("Hypertension History", [0, 1], format_func=lambda value: "Yes" if value else "No", index=int(st.session_state.get("hypertension_history", 0)))
        cardiovascular_history = st.selectbox("Cardiovascular History", [0, 1], format_func=lambda value: "Yes" if value else "No", index=int(st.session_state.get("cardiovascular_history", 0)))

        st.subheader("Vitals & Labs")
        bmi = st.number_input("BMI", min_value=15.0, max_value=45.0, value=float(st.session_state.get("bmi", 25.0)), step=0.1)
        waist_ratio = st.number_input("Waist-to-Hip Ratio", min_value=0.67, max_value=1.10, value=float(st.session_state.get("waist_to_hip_ratio", 0.86)), step=0.01)
        systolic = st.number_input("Systolic BP", min_value=90, max_value=200, value=int(st.session_state.get("systolic_bp", 120)))
        diastolic = st.number_input("Diastolic BP", min_value=50, max_value=130, value=int(st.session_state.get("diastolic_bp", 80)))
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=120, value=int(st.session_state.get("heart_rate", 70)))
        total_chol = st.number_input("Total Cholesterol", min_value=100, max_value=350, value=int(st.session_state.get("cholesterol_total", 185)))
        hdl = st.number_input("HDL Cholesterol", min_value=20, max_value=100, value=int(st.session_state.get("hdl_cholesterol", 54)))
        ldl = st.number_input("LDL Cholesterol", min_value=50, max_value=280, value=int(st.session_state.get("ldl_cholesterol", 103)))
        triglycerides = st.number_input("Triglycerides", min_value=30, max_value=400, value=int(st.session_state.get("triglycerides", 121)))
        fasting_glucose = st.number_input("Fasting Glucose", min_value=60.0, max_value=220.0, value=float(st.session_state.get("glucose_fasting", 111.0)), step=0.1)
        postprandial = st.number_input("Postprandial Glucose", min_value=70.0, max_value=320.0, value=float(st.session_state.get("glucose_postprandial", 160.0)), step=0.1)
        insulin = st.number_input("Insulin Level", min_value=2.0, max_value=35.0, value=float(st.session_state.get("insulin_level", 9.06)), step=0.01)
        hba1c = st.number_input("HbA1c", min_value=4.0, max_value=12.0, value=float(st.session_state.get("hba1c", 6.5)), step=0.01)

        analyze = st.button("Analyze Health Risks", type="primary", use_container_width=True)

    patient_data = {
        "Age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education,
        "income_level": income,
        "employment_status": employment,
        "smoking_status": smoking,
        "alcohol_consumption_per_week": alcohol,
        "physical_activity_minutes_per_week": activity,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep,
        "screen_time_hours_per_day": screen_time,
        "family_history_diabetes": family_history,
        "hypertension_history": hypertension_history,
        "cardiovascular_history": cardiovascular_history,
        "bmi": bmi,
        "waist_to_hip_ratio": waist_ratio,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "heart_rate": heart_rate,
        "cholesterol_total": total_chol,
        "hdl_cholesterol": hdl,
        "ldl_cholesterol": ldl,
        "triglycerides": triglycerides,
        "glucose_fasting": fasting_glucose,
        "glucose_postprandial": postprandial,
        "insulin_level": insulin,
        "hba1c": hba1c,
    }
    return patient_data, analyze


def render_metrics_sidebar(metrics):
    with st.sidebar:
        st.markdown("---")
        st.header("Model Performance")
        if not metrics:
            st.caption("Run `python main.py` to generate updated model metrics.")
            return

        for key, label in [
            ("diabetes", "Diabetes"),
            ("heart", "Heart Disease"),
            ("hypertension", "Hypertension"),
            ("obesity", "Obesity"),
            ("cholesterol", "Cholesterol"),
        ]:
            if key not in metrics:
                continue
            item = metrics[key]
            best = item["best_model"]
            st.subheader(label)
            st.caption(f"Best model: {best}")
            metrics_rows = pd.DataFrame.from_dict(
                {name: values for name, values in item.items() if isinstance(values, dict)},
                orient="index",
            )
            st.dataframe(metrics_rows, use_container_width=True)


def run_predictions(models, patient_data):
    predictions = {}
    for disease, model in models.items():
        try:
            predictions[disease] = model.predict(patient_data)
        except Exception as exc:
            predictions[disease] = {"error": str(exc), "risk_category": "Unknown"}
    return predictions


def render_dashboard(predictions):
    st.subheader("Risk Dashboard")
    chart_columns = st.columns(3)
    for index, (disease, result) in enumerate(predictions.items()):
        if "error" in result:
            continue
        with chart_columns[index % 3]:
            st.plotly_chart(
                create_risk_gauge(DISEASE_LABELS[disease], result["risk_score"], result["risk_category"]),
                use_container_width=True,
            )
            badge_class = {
                "Low": "badge-low",
                "Medium": "badge-medium",
                "High": "badge-high",
            }.get(result["risk_category"], "badge-medium")
            st.markdown(
                f"<span class='{badge_class}'>{result['risk_category']} Risk</span>",
                unsafe_allow_html=True,
            )

    st.plotly_chart(create_risk_comparison_bar(predictions), use_container_width=True)


def render_detailed_analysis(predictions):
    st.subheader("Contributing Factors")
    for disease, result in predictions.items():
        st.markdown(f"### {DISEASE_LABELS[disease]}")
        if "error" in result:
            st.error(result["error"])
            continue
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown("<div class='metric-panel'>", unsafe_allow_html=True)
            st.metric("Risk Score", f"{result['risk_score']:.2f} / 100")
            st.metric("Category", result["risk_category"])
            st.metric("Confidence", f"{result['confidence']:.0%}")
            quality = result.get("model_quality", {})
            if quality:
                st.caption(
                    f"Validated with {quality.get('best_model', 'model')} | "
                    f"R²: {quality.get('R2', 0):.3f} | MAE: {quality.get('MAE', 0):.2f}"
                )
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            factors = result.get("contributing_factors", {})
            if factors:
                st.plotly_chart(create_feature_importance_chart(factors), use_container_width=True)
            else:
                st.info("No feature attribution available for this model.")


def render_export_tab(predictions, patient_data):
    st.subheader("Export Results")
    agent_report = st.session_state.get("agent_report", "Generate the AI report first.")
    try:
        pdf_bytes = generate_pdf_report(predictions, agent_report, patient_data)
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "fpdf":
            st.warning("PDF export is unavailable because the `fpdf` package is not installed. Run `pip install -r requirements.txt` to enable it.")
        else:
            raise

    export_payload = {
        "timestamp": datetime.now().isoformat(),
        "patient_data": patient_data,
        "predictions": predictions,
        "agent_report": agent_report,
    }

    st.markdown("""
        <div style="background: linear-gradient(145deg, #FDFDF8, #F4F3EA); border: 1px solid #E8E5D5; border-top: 4px solid #C49A45; border-radius: 10px; padding: 24px; margin-top: 32px; margin-bottom: 20px; box-shadow: 0 8px 16px rgba(112, 128, 144, 0.08);">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="background-color: rgba(196, 154, 69, 0.15); border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                    <span style="color: #C49A45; font-size: 16px;">✦</span>
                </div>
                <h4 style="color: #4A5560; margin: 0; font-size: 1.2rem; font-weight: 600;">System Payload</h4>
            </div>
            <p style="color: #7A8694; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                Export the native, machine-readable JSON document. This comprehensive payload includes raw model analytics, serialized patient inputs, and programmatic AI inferences metadata.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.download_button(
        "Download JSON Payload",
        data=json.dumps(export_payload, indent=2),
        file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )


def render_model_validation_summary(predictions):
    st.subheader("Model Validation Summary")
    rows = []
    for disease, result in predictions.items():
        quality = result.get("model_quality", {})
        rows.append(
            {
                "Risk Area": DISEASE_LABELS[disease],
                "Score": result.get("risk_score"),
                "Category": result.get("risk_category"),
                "Best Model": quality.get("best_model", "N/A"),
                "R2": quality.get("R2", 0),
                "MAE": quality.get("MAE", 0),
                "RMSE": quality.get("RMSE", 0),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def update_session_memory(user_query, result):
    memory = st.session_state.setdefault("report_memory", [])
    memory.append(
        {
            "timestamp": result.get("timestamp"),
            "question": user_query or "General report",
            "priorities": [DISEASE_LABELS[item] for item in result.get("high_risk_areas", [])],
            "mode": result.get("generation_mode"),
        }
    )
    st.session_state["report_memory"] = memory[-5:]


def render_report_extras():
    metadata = st.session_state.get("agent_metadata", {})
    state = metadata.get("agent_state", {})
    follow_up_questions = metadata.get("follow_up_questions", [])

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Agent State")
        st.caption(f"Generation mode: {metadata.get('generation_mode', 'unknown')}")
        st.write("Priority risks:", ", ".join(DISEASE_LABELS[item] for item in metadata.get("high_risk_areas", [])) or "None")
        st.write("Retrieved topics:", ", ".join(state.get("knowledge_topics", [])) or "None")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Suggested Follow-up Questions")
        for item in follow_up_questions:
            st.write(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("report_memory"):
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Session Memory")
        for entry in reversed(st.session_state["report_memory"]):
            priorities = ", ".join(entry.get("priorities", [])) or "None"
            st.caption(f"{entry.get('timestamp', '')} | {entry.get('mode', '')}")
            st.write(f"Question: {entry.get('question', '')}")
            st.write(f"Priorities: {priorities}")
            st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)


def _extract_report_sections(report_text):
    sections = []
    current_title = None
    current_lines = []

    for line in str(report_text).splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.replace("## ", "", 1).strip()
            current_lines = []
        elif line.startswith("# "):
            continue
        else:
            current_lines.append(line)

    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    elif report_text:
        sections.append(("Report", str(report_text)))

    return sections


def _split_report_sections(report_text, user_question):
    sections = _extract_report_sections(report_text)
    answer_section = None
    report_sections = []

    for title, body in sections:
        normalized = title.strip().lower()
        if normalized == "direct answer to your question":
            answer_section = (user_question or title, body)
        elif user_question and normalized == user_question.lower():
            answer_section = (title, body)
        else:
            report_sections.append((title, body))

    return answer_section, report_sections


def render_answer_card():
    metadata = st.session_state.get("agent_metadata", {})
    state = metadata.get("agent_state", {})
    report_text = st.session_state.get("agent_report", "")
    user_question = state.get("query_used", "").strip()
    answer_section, _ = _split_report_sections(report_text, user_question)

    if not answer_section:
        return

    title, body = answer_section
    st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown(body if body else "_No answer is available yet._")
    st.markdown("</div>", unsafe_allow_html=True)


def render_quick_answer_card():
    answer_data = st.session_state.get("quick_answer")
    if not answer_data:
        return

    st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
    st.markdown(f"## {answer_data.get('question', 'Answer')}")
    st.markdown(answer_data.get("answer", "_No answer generated._"))
    st.markdown("</div>", unsafe_allow_html=True)

    evidence = answer_data.get("evidence", [])
    if evidence:
        st.markdown("#### Evidence")
        for item in evidence:
            st.markdown("<div class='evidence-card'>", unsafe_allow_html=True)
            st.markdown(f"**{item.get('topic', 'Evidence')}**")
            st.markdown(item.get("summary", ""))
            for source in item.get("sources", []):
                st.markdown(f"- {source}")
            st.markdown("</div>", unsafe_allow_html=True)


def render_visual_report(predictions):
    metadata = st.session_state.get("agent_metadata", {})
    state = metadata.get("agent_state", {})
    report_text = st.session_state.get("agent_report", "")
    user_question = state.get("query_used", "").strip()
    _, other_sections = _split_report_sections(report_text, user_question)

    top_disease = None
    top_result = {}
    valid_items = [(k, v) for k, v in predictions.items() if "error" not in v]
    if valid_items:
        top_disease, top_result = max(valid_items, key=lambda item: item[1].get("risk_score", 0))

    avg_score = 0.0
    valid_scores = [item.get("risk_score", 0) for item in predictions.values() if "error" not in item]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)

    metric_cols = st.columns(4)
    metric_data = [
        ("Average Risk", f"{avg_score:.1f}/100"),
        ("Generation Mode", metadata.get("generation_mode", "unknown")),
        ("Priority Areas", str(len(metadata.get("high_risk_areas", [])))),
        ("Evidence Topics", str(len(state.get("knowledge_topics", [])))),
    ]
    for col, (label, value) in zip(metric_cols, metric_data):
        with col:
            st.markdown(
                f"<div class='report-metric'><div class='report-metric-label'>{label}</div><div class='report-metric-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='report-hero'>", unsafe_allow_html=True)
    st.markdown("### Detailed Health Report")
    if top_disease:
        st.markdown(
            f"Highest current focus: **{DISEASE_LABELS[top_disease]}** with **{top_result.get('risk_score', 0):.2f}/100**. "
            "The sections below highlight the main drivers, practical next steps, and evidence used."
        )
    else:
        st.markdown("The sections below summarize the current risk profile and recommended next steps.")
    st.markdown("</div>", unsafe_allow_html=True)

    if metadata.get("high_risk_areas"):
        st.markdown("#### Priority Focus")
        st.markdown(
            "".join(
                f"<span class='report-pill'>{DISEASE_LABELS[item]}</span>"
                for item in metadata.get("high_risk_areas", [])
            ),
            unsafe_allow_html=True,
        )

    if state.get("knowledge_topics"):
        st.markdown("#### Evidence Topics Used")
        st.markdown(
            "".join(f"<span class='report-pill'>{topic}</span>" for topic in state.get("knowledge_topics", [])),
            unsafe_allow_html=True,
        )

    render_report_extras()

    for title, body in other_sections:
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.markdown(f"## {title}")
        st.markdown(body if body else "_Nothing to show in this section yet._")
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.markdown("<div class='main-title'>Intelligent Patient Risk Assessment & Agentic Health Support System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Five ML risk models, one patient dashboard, and an AI health-support report powered by free-tier LLM options.</div>",
        unsafe_allow_html=True,
    )

    metrics = load_metrics()
    render_metrics_sidebar(metrics)
    models = load_models()
    agent = load_agent()
    patient_data, analyze = collect_patient_data()

    overview_cols = st.columns(4)
    overview_cols[0].metric("Risk Models", "5")
    overview_cols[1].metric("Dataset Features", "28")
    overview_cols[2].metric("Agent Mode", "LLM" if agent.llm.available else "Rule-based")
    overview_cols[3].metric("Deployment", "Streamlit Ready")

    if analyze:
        with st.spinner("Running five risk models and preparing your dashboard..."):
            predictions = run_predictions(models, patient_data)
            st.session_state["predictions"] = predictions
            st.session_state["patient_data"] = patient_data

    predictions = st.session_state.get("predictions")
    current_patient_data = st.session_state.get("patient_data", patient_data)

    if not predictions:
        st.info("Fill in the patient profile from the sidebar and click `Analyze Health Risks` to generate the full assessment.")
        return

    tab_dashboard, tab_report, tab_analysis, tab_export = st.tabs(
        ["Risk Dashboard", "AI Report", "Detailed Analysis", "Export"]
    )

    with tab_dashboard:
        render_dashboard(predictions)
        render_model_validation_summary(predictions)

    with tab_report:
        st.subheader("AI Health Support")
        user_query = st.text_area(
            "Ask a follow-up health question",
            placeholder="What can I do to reduce my diabetes and cholesterol risk?",
            height=100,
        )
        action_col1, action_col2 = st.columns([1, 1.35])
        with action_col1:
            get_answer_clicked = st.button("Get Answer", use_container_width=True)
        with action_col2:
            generate_report_clicked = st.button("Generate Personalized Report", type="primary", use_container_width=True)

        if get_answer_clicked:
            with st.spinner("Preparing your answer..."):
                answer_result = agent.answer_question(
                    predictions,
                    current_patient_data,
                    user_query,
                    memory=st.session_state.get("report_memory", []),
                )
                st.session_state["quick_answer"] = answer_result

        if generate_report_clicked:
            with st.spinner("Generating report..."):
                result = agent.analyze_patient(
                    predictions,
                    current_patient_data,
                    user_query,
                    memory=st.session_state.get("report_memory", []),
                )
                st.session_state["agent_report"] = result["report"]
                st.session_state["agent_metadata"] = result
                update_session_memory(user_query, result)

        if "quick_answer" in st.session_state:
            render_quick_answer_card()

        if "agent_report" in st.session_state:
            render_answer_card()
            render_visual_report(predictions)
        else:
            if "quick_answer" not in st.session_state:
                st.info("Use `Get Answer` for a quick response, or generate the full report for a complete overview.")

    with tab_analysis:
        render_detailed_analysis(predictions)

    with tab_export:
        render_export_tab(predictions, current_patient_data)


if __name__ == "__main__":
    main()
