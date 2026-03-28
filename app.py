import numpy as np
import streamlit as st
import pickle

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mc = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
mx = pickle.load(open(os.path.join(BASE_DIR, 'minmaxscaler.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(BASE_DIR, 'scandscaler.pkl'), 'rb'))

def recommendation(n, p, k, temperature, humidity, ph, rainfall):
    crop_dict = {
        0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea',
        4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes',
        8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize',
        12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon',
        16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate',
        20: 'rice', 21: 'watermelon'
    }
    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_mx_features = sc.transform(mx_features)
    prediction = randclf.predict(sc_mx_features)
    crop_name = crop_dict.get(prediction[0], "Unknown Crop")
    return crop_name

st.title("🌾 Crop Recommendation System")

n = st.number_input("Nitrogen (N)", value=90)
p = st.number_input("Phosphorus (P)", value=42)
k = st.number_input("Potassium (K)", value=43)
temperature = st.number_input("Temperature", value=20.87)
humidity = st.number_input("Humidity", value=91.00)
ph = st.number_input("pH", value=6.50)
rainfall = st.number_input("Rainfall", value=202.0)

if st.button("Predict"):
    result = recommendation(n, p, k, temperature, humidity, ph, rainfall)
    st.success(f"🌱 Recommended Crop: {result}")

    
