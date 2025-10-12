import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fitted_random_forest_pipeline.joblib")

df = pd.read_csv("my_data.csv")

college = st.text_input("College")
region = st.text_input("Region")

if st.button("Predict"):
    input_data = pd.DataFrame({
        "college": [college],
        "region": [region]
    })
    prediction = model.predict(input_data)
    st.success(f" Estimated earnings: ${prediction[0]:,.2f} per year")
