import streamlit as st
import pickle
import numpy as np
import os

# ── Path Setup ───────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load Model and Scaler ────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = pickle.load(open(os.path.join(MODEL_DIR,
                                "best_model.pkl"),  "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR,
                                "scaler.pkl"),      "rb"))
    return model, scaler

model, scaler = load_artifacts()

# ── Encoding Maps ────────────────────────────────────────────
# These must match exactly what LabelEncoder learned in Phase 2
gender_map = {
    "Female" : 0,
    "Male"   : 1
}
race_map = {
    "Group A" : 0,
    "Group B" : 1,
    "Group C" : 2,
    "Group D" : 3,
    "Group E" : 4
}
parent_edu_map = {
    "Associate's Degree" : 0,
    "Bachelor's Degree"  : 1,
    "High School"        : 2,
    "Master's Degree"    : 3,
    "Some College"       : 4,
    "Some High School"   : 5
}
lunch_map = {
    "Free / Reduced" : 0,
    "Standard"       : 1
}
test_prep_map = {
    "Completed" : 0,
    "None"      : 1
}

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Student Score Predictor",
    page_icon  = "🎓",
    layout     = "centered"
)

# ── Header ───────────────────────────────────────────────────
st.title("🎓 Student Score Predictor")
st.markdown("#### Built by Bramha Gulavani | VIT Pune | AI & ML")
st.markdown("---")

# ── Model Info Cards ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Best Model",  "Linear Regression")
col2.metric("R2 Score",    "0.8838")
col3.metric("Avg Error",   "±4.13 marks")

st.markdown("---")

# ── Input Section ────────────────────────────────────────────
st.subheader("📋 Enter Student Details")
st.caption("Fill in the details below and click Predict")

# Row 1
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", list(gender_map.keys()))
with col2:
    race = st.selectbox("Race / Ethnicity", list(race_map.keys()))

# Row 2
col3, col4 = st.columns(2)
with col3:
    parent_edu = st.selectbox("Parental Education Level",
                               list(parent_edu_map.keys()))
with col4:
    lunch = st.selectbox("Lunch Type", list(lunch_map.keys()))

# Row 3
test_prep = st.selectbox("Test Preparation Course",
                          list(test_prep_map.keys()))

# Row 4 — Score Sliders
st.markdown("#### 📊 Known Scores")
col5, col6 = st.columns(2)
with col5:
    reading_score = st.slider("Reading Score", 0, 100, 70)
with col6:
    writing_score = st.slider("Writing Score", 0, 100, 70)

# ── Predict Button ───────────────────────────────────────────
st.markdown("---")
if st.button("🎯 Predict Math Score", use_container_width=True):

    # Build input array in same order as Phase 2 features
    input_data = np.array([[
        gender_map[gender],
        race_map[race],
        parent_edu_map[parent_edu],
        lunch_map[lunch],
        test_prep_map[test_prep],
        reading_score,
        writing_score
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    predicted_score = model.predict(input_scaled)[0]
    predicted_score = round(float(predicted_score), 1)
    predicted_score = max(0, min(100, predicted_score))

    # ── Result Display ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    # Big score display
    if predicted_score >= 70:
        st.success(f"🎉 Predicted Math Score: **{predicted_score} / 100**")
    elif predicted_score >= 50:
        st.warning(f"📘 Predicted Math Score: **{predicted_score} / 100**")
    else:
        st.error(f"📉 Predicted Math Score: **{predicted_score} / 100**")

    # Progress bar
    st.markdown(f"**Score Progress:**")
    st.progress(int(predicted_score))

    # Grade
    if predicted_score >= 90:
        grade = "A+"
    elif predicted_score >= 80:
        grade = "A"
    elif predicted_score >= 70:
        grade = "B"
    elif predicted_score >= 60:
        grade = "C"
    elif predicted_score >= 50:
        grade = "D"
    else:
        grade = "F"

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Score", f"{predicted_score}")
    m2.metric("Grade",           grade)
    m3.metric("Model Error",     "± 4.13 marks")

    st.markdown("---")
    st.caption("⚠️ This prediction is based on historical"
               " student data. Use it as a guide only.")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this Project")
    st.markdown("""
    This app uses a **Linear Regression** model
    trained on 1,000 student records to predict
    math exam scores.

    **How it works:**
    1. You fill in student details
    2. Categorical values get encoded to numbers
    3. Values get scaled to the same range
    4. Model predicts the math score

    **Features used:**
    - Gender
    - Race / Ethnicity
    - Parental Education
    - Lunch Type
    - Test Preparation
    - Reading Score
    - Writing Score
    """)

    st.markdown("---")
    st.markdown("**Model Comparison**")
    st.markdown("- Linear Regression : R2 = 0.8838 🏆")
    st.markdown("- Random Forest     : R2 = 0.8491")
    st.markdown("- XGBoost           : R2 = 0.8249")

    st.markdown("---")
    st.caption("Built by Bramha Gulavani")
    st.caption("VIT Pune | AI & ML | 2nd Year")