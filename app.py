import streamlit as st
import pandas as pd
import joblib


model = joblib.load("mental_health_model.pkl")

st.set_page_config(page_title="Mental Health Risk Monitoring System")

st.title("🧠 Mental Health Risk Monitoring System")

with st.form("form"):
    age = st.slider("Age", 18, 60, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-Employed"])
    sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 6.0)
    work_hours = st.slider("Work Hours", 2.0, 14.0, 8.0)
    physical_activity = st.slider("Physical Activity", 0, 120, 30)
    screen_time = st.slider("Screen Time", 1.0, 14.0, 6.0)
    stress_level = st.slider("Stress Level", 1, 10, 5)
    anxiety_level = st.slider("Anxiety Level", 1, 10, 5)
    depression_score = st.slider("Depression Score", 0, 30, 10)
    social_support = st.slider("Social Support", 1, 5, 3)
    family_history = st.radio("Family History", [0, 1])
    substance_use = st.radio("Substance Use", [0, 1])
    submit = st.form_submit_button("Predict")

if submit:
    df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "sleep_hours": sleep_hours,
        "work_hours": work_hours,
        "physical_activity": physical_activity,
        "screen_time": screen_time,
        "stress_level": stress_level,
        "anxiety_level": anxiety_level,
        "depression_score": depression_score,
        "social_support": social_support,
        "family_history": family_history,
        "substance_use": substance_use
    }])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    if pred == 1:
        st.error("⚠️ Mental Health Risk Detected")
    else:
        st.success("✅ No Mental Health Risk Detected")

    st.write("Risk Probability:", round(prob, 2))
