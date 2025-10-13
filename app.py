import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fitted_random_forest_pipeline.joblib")
df = pd.read_csv("my_data.csv")

college = st.selectbox("Select College", sorted(df["name"].unique()))
degree = st.selectbox("Select your degree", sorted(df["degree_type"].unique()))

row = df[df["name"] == college].iloc[0]

input_data = pd.DataFrame({
    "faculty_monthly_salary": [row["faculty_monthly_salary"]],
    "public_school_average_cost": [row["public_school_average_cost"]],
    "private_school_average_cost": [row["private_school_average_cost"]],
    "pell_grant_rate": [row["pell_grant_rate"]],
    "student_loan_principal": [row["student_loan_principal"]],
    "median_student_loan_debt": [row["median_student_loan_debt"]],
    "ownership_labels": [row["ownership"]],
    "degree_type_labels": [degree],
    "school_region_labels": [row["school_region"]],
    "school_locale_labels": [row["school_locale"]],
    "log_student_size": [row["log_student_size"]],
    "log_cost_of_attendance": [row["log_cost_of_attendance"]],
    "log_in_state_tuition": [row["log_in_state_tuition"]],
    "log_out_of_state_tuition": [row["log_out_of_state_tuition"]],
})

if st.button("Predict Earnings"):
    prediction = model.predict(input_data)
    st.success(f"Estimated earnings: ${prediction[0]:,.2f} per year")
