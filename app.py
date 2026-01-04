import streamlit as st
import pandas as pd
import joblib
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Mental Health Risk Monitoring System",
    page_icon="🧠",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
MODEL_PATH = "mental_health_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found: mental_health_model.pkl")
    st.stop()

model = joblib.load(MODEL_PATH)

# ------------------ TITLE ------------------
st.title("🧠 Mental Health Risk Monitoring System")
st.write(
    "This system evaluates mental health risk based on lifestyle and psychological factors."
)

st.divider()

# ------------------ INPUT FORM ------------------
with st.form("mental_health_form"):
    st.subheader("👤 Personal Details")
    age = st.slider("Age", 18, 60, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-Employed"])

    st.subheader("🕒 Daily Routine")
    sleep_hours = st.slider("Sleep Hours (per day)", 3.0, 10.0, 6.0)
    work_hours = st.slider("Work Hours (per day)", 2.0, 14.0, 8.0)
    screen_time = st.slider("Screen Time (hours)", 1.0, 14.0, 6.0)
    physical_activity = st.slider("Physical Activity (minutes)", 0, 120, 30)

    st.subheader("🧠 Mental Health Indicators")
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
    anxiety_level = st.slider("Anxiety Level (1–10)", 1, 10, 5)
    depression_score = st.slider("Depression Score (0–30)", 0, 30, 10)
    social_support = st.slider("Social Support (1–5)", 1, 5, 3)

    st.subheader("🧬 Background")
    family_history = st.radio("Family History of Mental Illness", [0, 1])
    substance_use = st.radio("Substance Use", [0, 1])

    submit = st.form_submit_button("🔍 Predict Risk")

# ------------------ PREDICTION ------------------
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

    # Probability-based prediction
    prob = model.predict_proba(df)[0][1]

    st.divider()
    st.subheader("📊 Result")

    if prob >= 0.3:
        st.error("⚠️ Mental Health Risk Detected")

        st.subheader("🛠️ What You Can Do Now")
        st.markdown("""
**Immediate Steps**
- Take regular breaks and reduce screen time
- Talk to friends or family members

**Lifestyle Improvements**
- Maintain 7–8 hours of sleep
- Exercise or walk daily
- Practice meditation or breathing exercises

**Professional Support**
- Consider consulting a mental health professional
- Online counseling can also be helpful
        """)
    else:
        st.success("✅ No Mental Health Risk Detected")

        st.markdown("""
**Healthy Habits to Continue**
- Maintain good sleep and routine
- Stay physically active
- Manage stress regularly
        """)

    st.write("**Risk Probability:**", round(prob, 2))

# ------------------ FOOTER ------------------
st.divider()
st.caption("© 2026 | Mental Health Risk Monitoring System")
