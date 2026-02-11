import streamlit as st
import torch
import numpy as np
from PIL import Image

st.title("Jharkhand Forest Fire AI Predictor")


st.sidebar.header("Current Conditions")
temp = st.sidebar.slider("Temperature (°C)", 20, 50, 35)
humidity = st.sidebar.slider("Humidity (%)", 5, 100, 20)


if st.button('Predict Fire Spread'):
    with st.spinner('AI is analyzing forest fuel and weather...'):
        st.image('outputs/animations/spread_prediction.gif', caption='Predicted 12-hour Fire Spread')
        st.success('Analysis Complete!')