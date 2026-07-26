import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("iris_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🌸 Iris Flower Classification")
st.write("Enter the flower measurements and click Predict.")

# User input
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

# Predict button
if st.button("Predict"):

    sample = pd.DataFrame({
        "SepalLengthCm": [sepal_length],
        "SepalWidthCm": [sepal_width],
        "PetalLengthCm": [petal_length],
        "PetalWidthCm": [petal_width]
    })

    prediction = model.predict(sample)
    flower = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Flower: {flower}")