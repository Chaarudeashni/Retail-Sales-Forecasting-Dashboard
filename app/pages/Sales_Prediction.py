import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load('C:\\Users\\chaar\\OneDrive\\Desktop\\Sales_Forecasting_Project\\models\\sales_model.pkl')

st.title("AI Sales Prediction")

st.markdown("""
Predict future retail sales using Machine Learning.
""")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", value=2026)

with col2:
    month = st.number_input("Month", value=5)

with col3:
    day = st.number_input("Day", value=15)

st.write("")

if st.button("Predict Sales"):

    prediction = model.predict([
        [year, month, day]
    ])

    predicted_value = prediction[0]

    st.success(
        f"Predicted Sales: {predicted_value:.2f}"
    )

    # Business Insight
    if predicted_value > 500:
        st.info(
            "High sales expected. Increase inventory levels."
        )

    else:
        st.warning(
            "Moderate sales expected. Consider promotions."
        )