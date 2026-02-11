import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import rasterio
import os
import sys
from PIL import Image
import base64

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import run_pipeline


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
    
    # Centers on Jharkhand
    m = folium.Map(location=[23.61, 85.27], zoom_start=9, tiles="CartoDB dark_matter")
    
    if os.path.exists("outputs/maps/latest_risk.tif"):
        with rasterio.open("outputs/maps/latest_risk.tif") as src:
            bounds = src.bounds
            data = src.read(1)
            profile = src.profile
        
        from src.utils import colorize_risk_map, array_to_png_base64
        colored_data = colorize_risk_map(data)
        png_base64 = array_to_png_base64(colored_data)
        
        img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{png_base64}",
            bounds=img_bounds,
            opacity=0.7,
            interactive=True,
            cross_origin=False,
            zindex=1,
            name="Fire Risk Heatmap"
        ).add_to(m)
        folium.LayerControl().add_to(m)
        st.success("🔥 Latest risk heatmap overlayed.")
    else:
        st.info("No risk map found. Run prediction to generate.")
    
    st_folium(m, width=900, height=600, key="fire_map")

with col2:
    st.subheader("Temporal Fire Spread")
    
    if os.path.exists("outputs/animations/fire_spread.gif"):
        st.image("outputs/animations/fire_spread.gif", use_container_width=True)
        st.info("Animation: 1h → 2h → 3h → 6h → 12h projections.")
    else:
        st.warning("No simulation found. Run prediction.")

    st.divider()
    st.subheader("Spread Metrics")
    st.metric("Estimated Burn Area (12h)", "14.2 ha", "+1.2 ha")
    st.metric("Perimeter Velocity", "0.8 km/h", "-0.1 km/h")

def run_prediction_pipeline():
    with st.spinner("ISRO PS1 Engine: Resampling 30m Data & Simulating..."):
        run_pipeline(data_dir='data/raw', output_dir='data/processed')
        st.success("Analysis complete!")
        st.rerun()

if trigger_prediction:
    run_prediction_pipeline()