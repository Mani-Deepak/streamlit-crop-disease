import streamlit as st
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
MODEL_PATH = "crop_disease_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# OpenWeatherMap API Key
WEATHER_API_KEY = "ee16d417a2247d41e7bb72ab630b3f28"

# Function to predict disease
def predict_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return "Diseased" if prediction[0][0] > 0.5 else "Healthy"
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("🌾 Smart Farm Dashboard")

# Weather Forecast
st.header("🌤 Weather Forecast")
city = st.text_input("Enter city name")

if st.button("Get Weather"):
    if city:
        api_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(api_url)

        if response.status_code == 200:
            weather_data = response.json()
            st.write(f"🌡 **Temperature:** {weather_data['main']['temp']}°C")
            st.write(f"🌦 **Condition:** {weather_data['weather'][0]['description'].capitalize()}")
        else:
            st.error("Failed to fetch weather data. Check city name or API key.")
    else:
        st.warning("Please enter a city name.")

# Disease Detection
st.header("🍂 Crop Disease Detection")
uploaded_file = st.file_uploader("test_leaf", type=["jpg", "png"])

if uploaded_file:
    img_path = f"temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = predict_disease(img_path)
    st.image(uploaded_file, caption=f"Prediction: {result}")

    # Remove temporary file
    os.remove(img_path)

# Inventory Management
st.header("📊 Inventory Management")
inventory_file = "inventory.csv"

if os.path.exists(inventory_file):
    inventory_data = pd.read_csv(inventory_file)
    st.dataframe(inventory_data)
else:
    st.warning("Inventory file not found. Upload `inventory.csv` to display inventory data.")

