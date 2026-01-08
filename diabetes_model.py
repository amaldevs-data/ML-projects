import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load Saved Model and Objects
# -----------------------------
model = joblib.load("diabetes.pkl")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("le0.pkl")
le_smoking = joblib.load("le1.pkl")


# Streamlit UI

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes.")


# User Input Fields

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
smoking_history = st.selectbox(
    "Smoking History",
    ["never", "No Info", "former", "current", "ever", "not current"]
)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)


# Preprocessing + Prediction

if st.button("Predict"):

    # Convert yes/no to 1/0
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_val = 1 if heart_disease == "Yes" else 0

    # Encode categorical
    gender_val = le_gender.transform([gender])[0]
    smoking_val = le_smoking.transform([smoking_history])[0]

    # Create DataFrame with correct column names
    input_df = pd.DataFrame([{
        'gender': gender_val,
        'age': age,
        'hypertension': hypertension_val,
        'heart_disease': heart_val,
        'smoking_history': smoking_val,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }])

    # Scale data
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Result Display
    if prediction == 1:
        st.error("🔴 Patient is likely to have Diabetes.")
    else:
        st.success("🟢 Patient is NOT likely to have Diabetes.")

# ------------------------------------------------------
#  BACKGROUND IMAGE (Added Only at the END as requested)
# ------------------------------------------------------
import base64

def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("bg.jpg")  # <-- Your image name

