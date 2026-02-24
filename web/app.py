import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import rasterio
import os
import sys
from PIL import Image
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import base64
import importlib
import requests
import json
import wave
import io
import streamlit.components.v1 as components

# Setup paths
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

# Page Config
st.set_page_config(page_title="Agni-Chakshu | Command Dashboard", layout="wide", initial_sidebar_state="expanded")

# Styles
st.markdown("""
<style>
    .stApp { background-color: #f8faff !important; }
    /* Ensure the app stays solid during reloads */
    body { background-color: #f8faff !important; }
    [data-testid="stSidebar"] { background-color: #eff6ff !important; border-right: 1px solid #e2e8f0 !important; }
    /* header, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; } */
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
    /* Remove the default Streamlit loading dimming if possible */
    [data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("AGNI-CHAKSHU")
st.markdown("#### Mission Control: Temporospatial Fire Intelligence")

# --- SESSION STATE ---
hours = list(range(1, 13))
if 'current_hour_idx' not in st.session_state: st.session_state.current_hour_idx = 0
if 'sim_playing' not in st.session_state: st.session_state.sim_playing = False
if 'voice_mute' not in st.session_state: st.session_state.voice_mute = False
if 'voice_audio' not in st.session_state: st.session_state.voice_audio = None
if 'voice_reply' not in st.session_state: st.session_state.voice_reply = None
if 'last_sim_hour' not in st.session_state: st.session_state.last_sim_hour = -1

# --- UTILS ---
def query_hf_model(prompt):
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    token = st.secrets.get('HF_TOKEN', os.environ.get('HF_TOKEN', ''))
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"inputs": f"<s>[INST] You are Agni, an AI Fire Intelligence Officer. {prompt} [/INST]", "parameters": {"max_new_tokens": 80}}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=5)
        return response.json()[0]['generated_text'].split("[/INST]")[-1].strip()
    except: return None

def get_deepgram_audio(text):
    api_url = "https://api.deepgram.com/v1/speak?model=aura-2-vesta-en&encoding=linear16&sample_rate=24000"
    api_key = st.secrets.get("DEEPGRAM_API_KEY", os.environ.get("DEEPGRAM_API_KEY", ""))
    if not api_key: return None
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, json={"text": text}, timeout=8)
        if response.status_code == 200:
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, 'wb') as wav_file:
                    wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(24000)
                    wav_file.writeframes(response.content)
                return base64.b64encode(wav_io.getvalue()).decode("utf-8")
    except: pass
    return None

def get_narration(hour, area, growth, perimeter):
    """Generates a consistent, high-fidelity Mission Control tactical report"""
    return (
        f"Mission Update. ... T-plus {hour} hours. ... "
        f"The total burn area is now {area:.1f} hectares. ... "
        f"The hourly expansion rate is currently {growth:.2f} hectares. ... "
        f"The fire boundary spans {perimeter:.2f} kilometers. ... "
        "Continuing tactical surveillance."
    )

# --- SHARED STATE & METRICS ---
selected_hour = hours[st.session_state.current_hour_idx]
cur_area = 1.2 * (selected_hour ** 1.3)
growth = cur_area - (1.2 * (max(1, selected_hour - 1) ** 1.3))
perimeter = 2 * np.pi * np.sqrt(cur_area / np.pi)

# --- AUTO-NARRATION TRIGGER (CRITICAL FOR SYNC) ---
if st.session_state.sim_playing and not st.session_state.voice_mute:
    if st.session_state.last_sim_hour != selected_hour:
        # REMOVED st.spinner to prevent brightness flashes
        txt = get_narration(selected_hour, cur_area, growth, perimeter)
        audio = get_deepgram_audio(txt)
        if audio: st.session_state.voice_audio = audio
        else: st.session_state.voice_reply = txt
        st.session_state.last_sim_hour = selected_hour

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/wired/64/1d4ed8/fire-extinguisher.png", width=60)
    st.header("Risk Engine Controls")
    wind_speed = st.slider("Wind Intensity (km/h)", 0, 50, 15)
    wind_dir = st.selectbox("Wind Vector", ["North", "East", "South", "West", "NE", "NW", "SE", "SW"])
    if st.button("INITIATE PREDICTIVE ANALYSIS"):
        with st.spinner("Synthesizing Prediction Layer..."): run_pipeline(wind_speed=wind_speed, wind_dir=wind_dir); st.rerun()
    
    st.divider()
    st.header("Geospatial Analysts")
    layer_risk = st.checkbox("AI Fire Risk Layer", value=True)
    layer_dem = st.checkbox("Terrain Topography (DEM)", value=False)
    layer_fuel = st.checkbox("Fuel Load (LULC)", value=False)
    
    st.divider()
    st.subheader("Mission Communications")
    st.session_state.voice_mute = not st.toggle("Enable Agni Mission Voice", value=not st.session_state.voice_mute)
    if st.secrets.get("DEEPGRAM_API_KEY", os.environ.get("DEEPGRAM_API_KEY", "")): st.success("**Premium Vesta Active**")
    else: st.warning("🎙️ **Browser Voice Fallback**")
    
    # Subtle status indicator instead of st.spinner
    if st.session_state.sim_playing and not st.session_state.voice_mute:
        st.caption(f"📡 Generating T+{selected_hour}h tactical report...")
    
    # --- VOICE PLAYBACK (Moved to Sidebar to prevent main layout shifts) ---
    if not st.session_state.voice_mute:
        if st.session_state.voice_audio:
            st.audio(base64.b64decode(st.session_state.voice_audio), format="audio/wav", autoplay=True)
            st.session_state.voice_audio = None
        elif st.session_state.voice_reply:
            components.html(f"<script>const u=new SpeechSynthesisUtterance({json.dumps(st.session_state.voice_reply)}); u.lang='en-IN'; window.speechSynthesis.speak(u);</script>", height=0)
            st.session_state.voice_reply = None
    
    st.info("Commands: 'Play', 'Status', 'Analyze'")

# --- MAIN UI ---
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Burn Area", f"{cur_area:.1f} ha")
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{min(100, (cur_area/60.0)*100)}%; background:#1d4ed8;"></div></div>', unsafe_allow_html=True)
with m2:
    st.metric("Hourly Expansion", f"+{growth:.2f} ha", delta_color="inverse")
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{min(100, (growth/8.0)*100)}%; background:#2563eb;"></div></div>', unsafe_allow_html=True)
with m3:
    st.metric("Boundary Reach", f"{perimeter:.2f} km")
    st.markdown(f'<div class="scale-bg"><div class="scale-fill" style="width:{min(100, (perimeter/12.0)*100)}%; background:#3b82f6;"></div></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 4])
with c1:
    if st.button(" PAUSE" if st.session_state.sim_playing else " PLAY PROGRESSION"):
        st.session_state.sim_playing = not st.session_state.sim_playing
        if st.session_state.sim_playing: st.session_state.last_sim_hour = -1 # Reset sync
        st.rerun()
with c2:
    if st.button("RESET"):
        st.session_state.current_hour_idx = 0; st.session_state.sim_playing = False; st.session_state.last_sim_hour = -1; st.rerun()
with c3:
    selected_manual = st.select_slider("Timeline", options=hours, value=selected_hour, label_visibility="collapsed")
    if not st.session_state.sim_playing: st.session_state.current_hour_idx = hours.index(selected_manual)

st.divider()

# --- MAP DISPLAY ---
# Use columns to manage "Two Maps" issue. Only show details when NOT playing to reduce clutter.
if not st.session_state.sim_playing:
    col_map, col_detail = st.columns([2.2, 1])
else:
    col_map = st.container()
    col_detail = None

with col_map:
    st.subheader(f"Active Fire Operations (T + {selected_hour}h)")
    # CHANGED: Use OpenStreetMap for better contrast/visibility
    # Create main tactical map
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
        if os.path.exists(snap_path): st.image(snap_path, caption=f"Boundary Insight (T+{selected_hour}h)", use_container_width=True)
        st.divider()
        st.metric("Avg Temp", "32°C", "2°C"); st.metric("Fuel Condition", "Critical", "Dry"); st.warning("High Risk in Latehar District.")

# --- SIM ENGINE LOOP ---
if st.session_state.sim_playing:
    import time
    # Reduced wait time for smoother experience, narration handled in sidebar
    time.sleep(1.0)
    st.session_state.current_hour_idx = (st.session_state.current_hour_idx + 1) % len(hours)
    st.rerun()

# --- VOICE COMMAND LOGIC ---
if "voice_cmd" in st.query_params:
    v_cmd = st.query_params["voice_cmd"]
    # FIX: Use the new dictionary-like access and clearing for modern Streamlit
    st.query_params.clear() 
    if v_cmd == "play": st.session_state.sim_playing = True; st.session_state.last_sim_hour = -1
    elif v_cmd == "pause": st.session_state.sim_playing = False
    elif v_cmd == "reset": st.session_state.current_hour_idx = 0; st.session_state.sim_playing = False
    elif v_cmd == "status":
        txt = get_narration(selected_hour, cur_area, growth, perimeter); st.session_state.voice_audio = get_deepgram_audio(txt)
        if not st.session_state.voice_audio: st.session_state.voice_reply = txt
    st.rerun()

components.html("""
<script>
    const rec = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    rec.continuous = true; 
    rec.interimResults = true; // Faster response
    rec.lang = 'en-US'; // Standardized
    
    rec.onresult = (e) => {
        const results = e.results[e.results.length - 1];
        if (!results.isFinal) return; // Wait for clear command
        
        const cmd = results[0].transcript.toLowerCase();
        let action = null;
        if (cmd.includes("play") || cmd.includes("start")) action = "play";
        else if (cmd.includes("pause") || cmd.includes("stop") || cmd.includes("wait")) action = "pause";
        else if (cmd.includes("reset") || cmd.includes("restart")) action = "reset";
        else if (cmd.includes("status") || cmd.includes("report")) action = "status";
        
        if (action) {
            const url = new URL(window.parent.location.href);
            url.searchParams.set("voice_cmd", action);
            window.parent.location.replace(url.href); // Slightly cleaner than assignment
        }
    };
    rec.onend = () => { try { rec.start(); } catch(e) {} };
    rec.onerror = (e) => { console.error("Voice Error:", e.error); };
    try { rec.start(); } catch(e) {}
</script>
""", height=0)