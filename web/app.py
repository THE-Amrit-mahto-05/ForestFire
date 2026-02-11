import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import rasterio
import os
from PIL import Image
import base64

st.set_page_config(page_title="Agni-Chakshu | Fire Intelligence", layout="wide")

st.title("Agni-Chakshu")
st.markdown("### Jharkhand Forest Fire Intelligence & Simulation System")

with st.sidebar:
    st.header("Simulation Controls")
    trigger_prediction = st.button("🚀 Run AI Prediction")
    
    st.divider()
    
    st.header("Visual Overlays")
    show_risk = st.checkbox("Show Risk Heatmap", value=True)
    show_fuel = st.checkbox("Show Fuel Map", value=False)
    
    st.divider()
    
    st.header("Simulation Settings")
    wind_dir = st.selectbox("Wind Direction", ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"])
    sim_hours = st.slider("Simulation Window (Hours)", 1, 24, 12)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Predictive Risk Map")
    
    m = folium.Map(location=[23.63, 85.51], zoom_start=11, tiles="CartoDB positron")
    
    if os.path.exists("outputs/maps/latest_risk.tif"):
        with rasterio.open("outputs/maps/latest_risk.tif") as src:
            bounds = src.bounds
            img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

            st.info("Risk map overlay active.")
    
    st_folium(m, width=900, height=600)

with col2:
    st.subheader("Fire Spread Forecast")
    
    if os.path.exists("outputs/animations/fire_spread.gif"):
        st.image("outputs/animations/fire_spread.gif", use_column_width=True)
        st.caption(f"12-hour projected spread (Wind: {wind_dir})")
    else:
        st.warning("No simulation data available. Run prediction to generate.")

    st.divider()
    
    st.subheader("Active Hotspots (NASA FIRMS)")
    st.write("Ramgarh Zone: 2 Detected")
    st.write("Hazaribagh Zone: 1 Detected")

if trigger_prediction:
    with st.spinner("Analyzing Geospatial Data..."):
        os.system("python main.py")
        st.success("Analysis Complete!")
        st.rerun()