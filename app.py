import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load KNN Model
# -------------------------------
with open("knn_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Healthcare Prediction", page_icon="🏥", layout="wide")

st.title("🏥 Healthcare Prediction System")
st.markdown("Enter the patient details and click **Predict**.")

# -------------------------------
# Input Fields
# -------------------------------

name = st.text_input("Patient Name")

age = st.number_input("Age", min_value=0, max_value=120, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

blood_type = st.selectbox(
    "Blood Type",
    ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
)

medical_condition = st.text_input("Medical Condition")

date_of_admission = st.date_input("Date of Admission")

doctor = st.text_input("Doctor")

hospital = st.text_input("Hospital")

insurance_provider = st.text_input("Insurance Provider")

billing_amount = st.number_input(
    "Billing Amount",
    min_value=0.0,
    value=1000.0
)

room_number = st.number_input(
    "Room Number",
    min_value=1,
    value=101
)

admission_type = st.selectbox(
    "Admission Type",
    ["Emergency", "Urgent", "Elective"]
)

discharge_date = st.date_input("Discharge Date")

medication = st.text_input("Medication")

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Blood Type": [blood_type],
        "Medical Condition": [medical_condition],
        "Date of Admission": [str(date_of_admission)],
        "Doctor": [doctor],
        "Hospital": [hospital],
        "Insurance Provider": [insurance_provider],
        "Billing Amount": [billing_amount],
        "Room Number": [room_number],
        "Admission Type": [admission_type],
        "Discharge Date": [str(discharge_date)],
        "Medication": [medication]
    })

    try:
        prediction = model.predict(input_df)

        st.success(f"Prediction : {prediction[0]}")

    except Exception as e:
        st.error(e)