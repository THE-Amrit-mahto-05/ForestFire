import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import rasterio
import os
import sys
from PIL import Image
import base64
import importlib


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import src.utils
importlib.reload(src.utils) 
from src.utils import (
    colorize_risk_map, 
    array_to_png_base64, 
    colorize_terrain_map, 
    colorize_fuel_map, 
    colorize_simulation_heatmap
)
from main import run_pipeline

st.set_page_config(
    page_title="Agni-Chakshu | Command Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

st.markdown("""
<style>
    /* Full Page Light Theme */
    .stApp { background-color: #f8faff !important; }
    .main { background-color: #f8faff !important; }
    [data-testid="stSidebar"] { background-color: #eff6ff !important; border-right: 1px solid #e2e8f0 !important; }
    
    /* Aggressive Hide for Streamlit Header & Toolbar */
    header, [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stToolbar"], [data-testid="stToolbarActions"] {
        display: none !important;
    }
    .stAppDeployButton {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
        display: none !important;
    }
    /* Critical: Remove top padding that the header usually takes */
    .stMainBlockContainer {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 0rem !important;
    }
    
    /* Metric Cards: Light Blue on White */
    [data-testid="stMetric"] { 
        background-color: #ffffff !important; 
        padding: 15px !important; 
        border-radius: 12px !important; 
        border: 1px solid #e2e8f0 !important;
        border-left: 5px solid #1d4ed8 !important; 
        margin-bottom: 5px !important; 
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important;
    }
    
    /* Global Typography */
    h1 { color: #1d4ed8 !important; font-family: 'Outfit', sans-serif; font-weight: 800 !important; letter-spacing: -1px; }
    h3, h4 { color: #64748b !important; font-family: 'Courier New', monospace; }
    label { color: #334155 !important; font-weight: 600 !important; }
    
    /* Buttons */
    .stButton>button { 
        width: 100%; 
        border-radius: 10px; 
        background-color: #1d4ed8; 
        color: white; 
        font-weight: bold; 
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1e40af !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 10px 15px -3px rgba(29, 78, 216, 0.2) !important;
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: translateY(0px);
    }

    /* Telemetry Scales */
    .scale-bg { width: 100%; background: #e2e8f0; height: 6px; border-radius: 3px; position: relative; overflow: hidden; margin-top: 2px; }
    .scale-fill { height: 100%; border-radius: 3px; transition: width 0.7s ease; }
    .scale-text { font-size: 11px; color: #94a3b8; margin-top: 1px; text-align: right; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

st.title("AGNI-CHAKSHU")
st.markdown("#### Mission Control: Temporospatial Fire Intelligence")

hours = list(range(1, 13))
if 'current_hour_idx' not in st.session_state:
    st.session_state.current_hour_idx = 0
if 'sim_playing' not in st.session_state:
    st.session_state.sim_playing = False

selected_hour = hours[st.session_state.current_hour_idx]

with st.sidebar:
    st.image("https://img.icons8.com/wired/64/1d4ed8/fire-extinguisher.png", width=60)
    st.header("Risk Engine Controls")
    wind_speed = st.slider("Wind Intensity (km/h)", 0, 50, 15)
    wind_dir = st.selectbox("Wind Vector", ["North", "East", "South", "West", "NE", "NW", "SE", "SW"])
    trigger_prediction = st.button("INITIATE PREDICTIVE ANALYSIS")
    
    st.divider()
    st.header("Geospatial Analysts")
    layer_risk = st.checkbox("AI Fire Risk Layer", value=True)
    layer_dem = st.checkbox("Terrain Topography (DEM)", value=False)
    layer_fuel = st.checkbox("Fuel Load (LULC)", value=False)

base_area = 1.2
cur_area = base_area * (selected_hour ** 1.3)
prev_h = max(1, selected_hour - 1)
growth = cur_area - (base_area * (prev_h ** 1.3))
perimeter = 2 * np.pi * np.sqrt(cur_area / np.pi)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Burn Area", f"{cur_area:.1f} ha")
    a_perc = min(100, (cur_area / 60.0) * 100)
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{a_perc}%; background:#1d4ed8; box-shadow:0 0 10px rgba(29, 78, 216, 0.3);"></div></div><p class="scale-text">Cap: 60 ha</p>', unsafe_allow_html=True)
with m2:
    st.metric("Hourly Expansion", f"+{growth:.2f} ha", delta_color="inverse")
    g_perc = min(100, (growth / 8.0) * 100)
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{g_perc}%; background:#2563eb; box-shadow:0 0 10px rgba(37, 99, 235, 0.3);"></div></div><p class="scale-text">Max: 8 ha/h</p>', unsafe_allow_html=True)
with m3:
    st.metric("Boundary Reach", f"{perimeter:.2f} km")
    p_perc = min(100, (perimeter / 12.0) * 100)
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{p_perc}%; background:#3b82f6; box-shadow:0 0 10px rgba(59, 130, 246, 0.3);"></div></div><p class="scale-text">Limit: 12 km</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 4])
with c1:
    play_label = " PAUSE" if st.session_state.sim_playing else " PLAY PROGRESSION"
    if st.button(play_label):
        st.session_state.sim_playing = not st.session_state.sim_playing
        st.rerun()
with c2:
    if st.button("RESET"):
        st.session_state.current_hour_idx = 0
        st.session_state.sim_playing = False
        st.rerun()
with c3:
    selected_manual = st.select_slider(
        "Timeline Slider", options=hours, value=selected_hour, key="timeline_slider", label_visibility="collapsed"
    )
    if not st.session_state.sim_playing:
        st.session_state.current_hour_idx = hours.index(selected_manual)

st.divider()

col_map, col_detail = st.columns([2.2, 1])

with col_map:
    st.subheader(f"Active Fire Operations (T + {selected_hour}h)")
    
    m = folium.Map(location=[23.61, 85.27], zoom_start=9, tiles="CartoDB positron", attribution_control=False)
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; background: rgba(255,255,255,0.9); 
    z-index:9999; border-radius:10px; padding: 12px; color: #1e293b; border: 1px solid #1d4ed8; font-size:12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
    <b>Intensity</b><br>
    <i style="background: #1d4ed8; width:12px; height:12px; display:inline-block; border-radius:2px;"></i> Fire Peak<br>
    <i style="background: #3b82f6; width:12px; height:12px; display:inline-block; border-radius:2px;"></i> Active Front<br>
    <i style="background: #cbd5e1; width:12px; height:12px; display:inline-block; border-radius:2px;"></i> Burnt Zone
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    if layer_risk and os.path.exists("outputs/maps/latest_risk.tif"):
        with rasterio.open("outputs/maps/latest_risk.tif") as src:
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{array_to_png_base64(colorize_risk_map(src.read(1)))}",
                bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]],
                opacity=0.6, zindex=10 # HUD feel
            ).add_to(m)

    if layer_dem and os.path.exists("data/raw/dem_90m.tif"):
        with rasterio.open("data/raw/dem_90m.tif") as src:
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{array_to_png_base64(colorize_terrain_map(src.read(1)))}",
                bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]],
                opacity=0.6, zindex=5 
            ).add_to(m)

    if layer_fuel and os.path.exists("data/processed/fuel_map_90m.tif"):
        with rasterio.open("data/processed/fuel_map_90m.tif") as src:
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{array_to_png_base64(colorize_fuel_map(src.read(1)))}",
                bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]],
                opacity=0.6, zindex=6 
            ).add_to(m)

    # Simulation Overlay: NEON VISIBILITY
    spread = f"outputs/maps/fire_spread_{selected_hour}h.tif"
    if os.path.exists(spread):
        with rasterio.open(spread) as src:
            data = src.read(1)
            colored = colorize_simulation_heatmap(data)
            rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
            rgba[:, :, :3] = colored

            rgba[data > 0.05, 3] = 255
            rgba[(data > 0.01) & (data <= 0.05), 3] = 200
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{array_to_png_base64(rgba)}",
                bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]],
                opacity=1.0, zindex=100
            ).add_to(m)

    center = [23.36, 85.33]
    d_map = {"North":(0,-0.6), "South":(0,0.6), "East":(0.6,0), "West":(-0.6,0), "NE":(0.4,-0.4), "SE":(0.4,0.4), "NW":(-0.4,-0.4), "SW":(-0.4,0.4)}
    v = d_map.get(wind_dir, (0,0))
    sc = wind_speed / 50.0
    folium.PolyLine([center, [center[0]+v[1]*sc, center[1]+v[0]*sc]], color="#00ff00", weight=5).add_to(m)
    folium.CircleMarker(center, radius=5, color="#00ff00", fill=True).add_to(m)
    
    st_folium(m, width=900, height=600, key="main_map")

with col_detail:
    st.subheader("Propagation Zoom")
    snap_path = f"outputs/snapshots/fire_{selected_hour}h.png"
    if os.path.exists(snap_path):
        st.image(snap_path, caption=f"Boundary Insight (T+{selected_hour}h)", width="stretch")
    
    st.divider()
    st.subheader("Tactical Response")
    st.metric("Avg Temp", "32°C", "2°C")
    st.metric("Fuel Condition", "Critical", "Dry")
    st.warning("High Risk in Latehar District. Recommended containment: Western Sector.")

if trigger_prediction:
    with st.spinner("ISRO Engine: Synthesizing Prediction Layer..."):
        run_pipeline(wind_speed=wind_speed, wind_dir=wind_dir)
        st.rerun()

if st.session_state.sim_playing:
    st.session_state.current_hour_idx = (st.session_state.current_hour_idx + 1) % len(hours)
    import time
    time.sleep(1.0)
    st.rerun()

st.markdown("---")
f_col1, f_col2 = st.columns([2, 1])

with f_col1:
    st.markdown("###  System Architecture")
    st.markdown("""
    **Agni-Chakshu** is a high-fidelity forest fire prediction system. It utilizes satellite-derived 
    multi-spectral imagery to model fire propagation using cellular automata and deep learning. 
    By synthesizing topography, fuel loads, and wind vectors, it provides tactical intelligence 
    for rapid response teams.
    """)
    st.markdown("""
    **Special thanks to:**
    - **Team Bhuvan**
    - **National Remote Sensing Centre (NRSC), ISRO**
    - Hyderabad, INDIA.
    """)

with f_col2:
    st.markdown("###  Data Credits")
    st.markdown("""
    - **LULC (Land Use Land Cover)** data provided by NRSC/ISRO Bhuvan.
    - **Fire Hotspots** from NASA FIRMS Team (Fire Information for Resource Management System).
    - **DEM (Digital Elevation Model)** from OpenTopography for High-Resolution Topography Data.
    """)
  