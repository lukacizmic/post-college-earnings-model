import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fitted_random_forest_pipeline.joblib")

df = pd.read_csv("my_data.csv")

college = st.selectbox("Select College", sorted(df["name"].unique()))
degree = st.selectbox("Select your degree", sorted(df["degree_type_labels"].unique()))


if st.button("Predict"):
    input_data = pd.DataFrame({
        "college": [college],
        "degree": [degree]
    })
    prediction = model.predict(input_data)
    st.success(f" Estimated earnings: ${prediction[0]:,.2f} per year")
