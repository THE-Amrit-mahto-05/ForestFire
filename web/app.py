from dotenv import load_dotenv
load_dotenv()
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
import requests
import json
import wave
import io
import streamlit.components.v1 as components

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

st.set_page_config(page_title="Agni-Chakshu | Command Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #f8faff !important; }
    body { background-color: #f8faff !important; }
    [data-testid="stSidebar"] { background-color: #eff6ff !important; border-right: 1px solid #e2e8f0 !important; }
    .stMainBlockContainer { padding-top: 0.5rem !important; padding-bottom: 0rem !important; }
    [data-testid="stMetric"] { 
        background-color: #ffffff !important; padding: 15px !important; border-radius: 12px !important; 
        border: 1px solid #e2e8f0 !important; border-left: 5px solid #1d4ed8 !important; 
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important;
    }
    h1 { color: #1d4ed8 !important; font-family: 'Outfit', sans-serif; font-weight: 800 !important; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #1d4ed8 !important; color: white !important; font-weight: bold; }
    .stButton>button:hover, .stButton>button:active, .stButton>button:focus { 
        background-color: #1d4ed8 !important; color: white !important; 
    }
    .scale-bg { width: 100%; background: #e2e8f0; height: 6px; border-radius: 3px; margin-top: 2px; }
    .scale-fill { height: 100%; border-radius: 3px; transition: width 0.7s ease; }
    [data-testid="stStatusWidget"] { display: none !important; }

    .countdown-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(15, 23, 42, 0.9);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        z-index: 10000; color: white; transition: all 0.5s ease;
    }
    .countdown-timer { font-size: 15rem; font-weight: 900; color: #3b82f6; text-shadow: 0 0 50px rgba(59, 130, 246, 0.5); }
    .countdown-msg { font-size: 2.5rem; font-weight: 700; margin-top: -2rem; text-transform: uppercase; letter-spacing: 0.2rem; }
    .vol-msg { font-size: 1.5rem; color: #94a3b8; margin-bottom: 2rem; display: flex; align-items: center; gap: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("AGNI-CHAKSHU")
st.markdown("#### Mission Control: Temporospatial Fire Intelligence")

hours = list(range(1, 13))
if 'current_hour_idx' not in st.session_state: st.session_state.current_hour_idx = 0
if 'sim_playing' not in st.session_state: st.session_state.sim_playing = False
if 'voice_mute' not in st.session_state: st.session_state.voice_mute = False
if 'voice_audio' not in st.session_state: st.session_state.voice_audio = None
if 'last_sim_hour' not in st.session_state: st.session_state.last_sim_hour = -1
if 'countdown' not in st.session_state: st.session_state.countdown = -1

def get_secret(key):
    return os.environ.get(key) or (st.secrets.get(key) if hasattr(st, "secrets") else None)

def get_deepgram_audio(text, hour=0):
    voices = ["aura-asteria-en", "aura-luna-en", "aura-stella-en", "aura-hera-en", "aura-vesta-en"]
    voice = voices[hour % len(voices)]
    api_url = f"https://api.deepgram.com/v1/speak?model={voice}&encoding=linear16&sample_rate=24000"
    api_key = get_secret("DEEPGRAM_API_KEY")
    if not api_key: return None
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, json={"text": text}, timeout=5)
        if response.status_code == 200:
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, 'wb') as wav_file:
                    wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(24000)
                    wav_file.writeframes(response.content)
                return base64.b64encode(wav_io.getvalue()).decode("utf-8")
    except: pass
    return None

def get_narration(hour, area):
    return f"T plus {hour} hours. Total burn area {area:.1f} hectares."

selected_hour = hours[st.session_state.current_hour_idx]
cur_area = 1.2 * (selected_hour ** 1.3)
growth = cur_area - (1.2 * (max(1, selected_hour - 1) ** 1.3))
perimeter = 2 * np.pi * np.sqrt(cur_area / np.pi)

if st.session_state.sim_playing and not st.session_state.voice_mute:
    if st.session_state.last_sim_hour != selected_hour:
        st.session_state.voice_audio = None
        txt = get_narration(selected_hour, cur_area)
        audio = get_deepgram_audio(txt, hour=selected_hour)
        if audio: st.session_state.voice_audio = audio
        st.session_state.last_sim_hour = selected_hour

with st.sidebar:
    st.image("https://img.icons8.com/wired/64/1d4ed8/fire-extinguisher.png", width=60)
    st.header("Risk Engine Controls")
    wind_speed = st.slider("Wind Intensity km/h", 0, 50, 15)
    wind_dir = st.selectbox("Wind Vector", ["North", "East", "South", "West", "NE", "NW", "SE", "SW"])
    if st.button("INITIATE PREDICTIVE ANALYSIS"):
        with st.spinner("Synthesizing Prediction Layer"): run_pipeline(wind_speed=wind_speed, wind_dir=wind_dir); st.rerun()
    
    st.divider()
    st.header("Geospatial Analysts")
    layer_risk = st.checkbox("AI Fire Risk Layer", value=True)
    layer_dem = st.checkbox("Terrain Topography DEM", value=False)
    layer_fuel = st.checkbox("Fuel Load LULC", value=False)
    
    st.divider()
    st.subheader("Mission Communications")
    st.session_state.voice_mute = not st.toggle("Enable Agni Mission Voice", value=not st.session_state.voice_mute)
    if get_secret("DEEPGRAM_API_KEY"): st.success("Premium Vesta Active")
    
    if st.session_state.sim_playing and not st.session_state.voice_mute:
        st.caption(f"Generating T plus {selected_hour}h tactical report")
    
    if not st.session_state.voice_mute:
        if st.session_state.voice_audio:
            st.audio(base64.b64decode(st.session_state.voice_audio), format="audio/wav", autoplay=True)
            st.session_state.voice_audio = None
    
    st.info("Commands: Play, Status, Analyze")

if st.session_state.countdown >= 0:
    st.markdown(f"""
        <div class="countdown-overlay">
            <div class="vol-msg">INCREASE DEVICE VOLUME</div>
            <div class="countdown-timer">{st.session_state.countdown if st.session_state.countdown > 0 else "GO"}</div>
            <div class="countdown-msg">Starting Tactical Narration</div>
        </div>
    """, unsafe_allow_html=True)
    import time
    time.sleep(1.0)
    st.session_state.countdown -= 1
    if st.session_state.countdown < 0:
        st.session_state.sim_playing = True
        st.session_state.last_sim_hour = -1
    st.rerun()

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Burn Area", f"{cur_area:.1f} ha")
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{min(100, (cur_area/60.0)*100)}%; background:#1d4ed8;"></div></div>', unsafe_allow_html=True)
with m2:
    st.metric("Hourly Expansion", f"+{growth:.2f} ha")
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{min(100, (growth/8.0)*100)}%; background:#2563eb;"></div></div>', unsafe_allow_html=True)
with m3:
    st.metric("Boundary Reach", f"{perimeter:.2f} km")
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{min(100, (perimeter/12.0)*100)}%; background:#3b82f6;"></div></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 4])
with c1:
    if st.button("PAUSE" if st.session_state.sim_playing else "PLAY PROGRESSION"):
        if st.session_state.sim_playing:
            st.session_state.sim_playing = False
        else:
            st.session_state.countdown = 3
        st.rerun()
with c2:
    if st.button("RESET"):
        st.session_state.current_hour_idx = 0; st.session_state.sim_playing = False; st.session_state.last_sim_hour = -1; st.rerun()
with c3:
    selected_manual = st.select_slider("Timeline", options=hours, value=selected_hour, label_visibility="collapsed")
    if not st.session_state.sim_playing: st.session_state.current_hour_idx = hours.index(selected_manual)

st.divider()

if not st.session_state.sim_playing:
    col_map, col_detail = st.columns([2.2, 1])
else:
    col_map = st.container()
    col_detail = None

with col_map:
    st.subheader(f"Active Fire Operations T plus {selected_hour}h")
    m = folium.Map(location=[23.61, 85.27], zoom_start=9, tiles="OpenStreetMap", attribution_control=False)
    
    if layer_risk and os.path.exists("outputs/maps/latest_risk.tif"):
        with rasterio.open("outputs/maps/latest_risk.tif") as src:
            folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{array_to_png_base64(colorize_risk_map(src.read(1)))}", bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]], opacity=0.4, zindex=10).add_to(m)
    if layer_dem and os.path.exists("data/raw/dem_90m.tif"):
        with rasterio.open("data/raw/dem_90m.tif") as src:
            folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{array_to_png_base64(colorize_terrain_map(src.read(1)))}", bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]], opacity=0.5, zindex=5).add_to(m)
    if layer_fuel and os.path.exists("data/processed/fuel_map_90m.tif"):
        with rasterio.open("data/processed/fuel_map_90m.tif") as src:
            folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{array_to_png_base64(colorize_fuel_map(src.read(1)))}", bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]], opacity=0.5, zindex=6).add_to(m)
    
    spread_path = f"outputs/maps/fire_spread_{selected_hour}h.tif"
    if os.path.exists(spread_path):
        with rasterio.open(spread_path) as src:
            data = src.read(1); colored = colorize_simulation_heatmap(data)
            rgba = np.zeros((*data.shape, 4), dtype=np.uint8); rgba[:, :, :3] = colored; rgba[data > 0.01, 3] = 255
            folium.raster_layers.ImageOverlay(image=f"data:image/png;base64,{array_to_png_base64(rgba)}", bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]], opacity=0.9, zindex=100).add_to(m)
    
    st_folium(m, width=900, height=600, key=f"main_map_{st.session_state.current_hour_idx}", returned_objects=[])

if col_detail:
    with col_detail:
        st.subheader("Propagation Zoom")
        snap_path = f"outputs/snapshots/fire_{selected_hour}h.png"
        if os.path.exists(snap_path): st.image(snap_path, caption=f"Boundary Insight T plus {selected_hour}h", use_container_width=True)
        st.divider()
        st.metric("Avg Temp", "32C", "2C"); st.metric("Fuel Condition", "Critical", "Dry"); st.warning("High Risk in Latehar District")

if st.session_state.sim_playing:
    import time
    time.sleep(1.0)
    st.session_state.current_hour_idx = (st.session_state.current_hour_idx + 1) % len(hours)
    st.rerun()

if "voice_cmd" in st.query_params:
    v_cmd = st.query_params["voice_cmd"]
    st.query_params.clear() 
    if v_cmd == "play": st.session_state.sim_playing = True; st.session_state.last_sim_hour = -1
    elif v_cmd == "pause": st.session_state.sim_playing = False
    elif v_cmd == "reset": st.session_state.current_hour_idx = 0; st.session_state.sim_playing = False
    elif v_cmd == "status":
        txt = get_narration(selected_hour, cur_area); st.session_state.voice_audio = get_deepgram_audio(txt, hour=selected_hour)
    st.rerun()

components.html("""
<script>
    const rec = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    rec.continuous = true; 
    rec.interimResults = true; 
    rec.lang = 'en-US'; 
    
    rec.onresult = (e) => {
        const results = e.results[e.results.length - 1];
        if (!results.isFinal) return; 
        
        const cmd = results[0].transcript.toLowerCase();
        let action = null;
        if (cmd.includes("play") || cmd.includes("start")) action = "play";
        else if (cmd.includes("pause") || cmd.includes("stop") || cmd.includes("wait")) action = "pause";
        else if (cmd.includes("reset") || cmd.includes("restart")) action = "reset";
        else if (cmd.includes("status") || cmd.includes("report")) action = "status";
        
        if (action) {
            const url = new URL(window.parent.location.href);
            url.searchParams.set("voice_cmd", action);
            window.parent.location.replace(url.href); 
        }
    };
    rec.onend = () => { try { rec.start(); } catch(e) {} };
    rec.onerror = (e) => { console.error("Voice Error", e.error); };
    try { rec.start(); } catch(e) {}
</script>
""", height=0)