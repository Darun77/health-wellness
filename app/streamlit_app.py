"""
streamlit_app.py
-----------------
Interactive demo for the Level 1 (ML) models: enter an employee's
wellness profile and get back their segment, predicted stress level,
and burnout risk. Run from the project root:

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="Corporate Wellness Coach", layout="centered")
st.title("Personalized Health & Wellness Coaching")
st.caption("Level 1 (ML) demo -- employee segmentation, stress prediction, burnout risk")


@st.cache_resource
def load_models():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    stress_model = joblib.load(MODELS_DIR / "stress_regressor.joblib")
    burnout_model = joblib.load(MODELS_DIR / "burnout_classifier_ml_baseline.joblib")
    kmeans_model = joblib.load(MODELS_DIR / "kmeans_segmentation.joblib")
    return preprocessor, stress_model, burnout_model, kmeans_model


try:
    preprocessor, stress_model, burnout_model, kmeans_model = load_models()
except FileNotFoundError:
    st.error("Models not found. Run `python main.py --stage ml` first to train and save them.")
    st.stop()

submitted = False

with st.form("employee_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 22, 60, 35)
        tenure_years = st.slider("Tenure (years)", 0.0, 25.0, 4.0)
        work_hours_per_week = st.slider("Work hours / week", 30.0, 75.0, 45.0)
        sleep_hours = st.slider("Sleep hours / night", 3.0, 9.0, 6.5)
        physical_activity_min_week = st.slider("Physical activity (min/week)", 0.0, 500.0, 120.0)
        diet_score = st.slider("Diet quality score (1-10)", 1.0, 10.0, 6.0)
        screen_time_hours = st.slider("Screen time (hrs/day)", 1.0, 16.0, 7.0)
    with col2:
        caffeine_intake_mg = st.slider("Caffeine intake (mg/day)", 0.0, 800.0, 250.0)
        meditation_min_week = st.slider("Meditation (min/week)", 0.0, 300.0, 20.0)
        social_support_score = st.slider("Social support score (1-10)", 1.0, 10.0, 6.0)
        job_satisfaction_score = st.slider("Job satisfaction score (1-10)", 1.0, 10.0, 6.0)
        work_life_balance_score = st.slider("Work-life balance score (1-10)", 1.0, 10.0, 6.0)
        past_medical_leaves = st.slider("Past medical leaves (count)", 0, 10, 1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        department = st.selectbox(
            "Department", ["Engineering", "Sales", "Operations", "HR", "Finance", "Support"]
        )

    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    row = pd.DataFrame([{
        "age": age, "tenure_years": tenure_years, "work_hours_per_week": work_hours_per_week,
        "sleep_hours": sleep_hours, "physical_activity_min_week": physical_activity_min_week,
        "diet_score": diet_score, "screen_time_hours": screen_time_hours,
        "caffeine_intake_mg": caffeine_intake_mg, "meditation_min_week": meditation_min_week,
        "social_support_score": social_support_score, "job_satisfaction_score": job_satisfaction_score,
        "work_life_balance_score": work_life_balance_score, "past_medical_leaves": past_medical_leaves,
        "gender": gender, "department": department,
    }])
    row["activity_index"] = row["physical_activity_min_week"] / (row["work_hours_per_week"] + 1)
    row["recovery_index"] = row["sleep_hours"] * (row["meditation_min_week"] + 1) / 100
    row["lifestyle_risk_score"] = (
        (row["screen_time_hours"] / 16) * 0.4
        + (row["caffeine_intake_mg"] / 800) * 0.3
        + (1 - row["diet_score"] / 10) * 0.3
    )

    X = preprocessor.transform(row)
    predicted_stress = stress_model.predict(X)[0]
    burnout_prob = burnout_model.predict_proba(X)[0, 1]
    cluster = kmeans_model.predict(X)[0]

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Stress Level", f"{predicted_stress:.1f} / 100")
    c2.metric("Burnout Risk Probability", f"{burnout_prob * 100:.1f}%")
    c3.metric("Wellness Segment", f"Cluster {cluster}")

    if burnout_prob > 0.6:
        st.warning("High burnout risk -- recommend a check-in with an HR wellness coach and workload review.")
    elif burnout_prob > 0.35:
        st.info("Moderate risk -- suggest stress-reduction resources (meditation, flexible hours).")
    else:
        st.success("Low risk -- maintain current wellness routine.")
