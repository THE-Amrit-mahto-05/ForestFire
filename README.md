# Agni-Chakshu (Jharkhand Forest Fire Intelligence System)

**Agni-Chakshu** is a high-resolution Geospatial AI system designed to predict and simulate forest fire spread across Jharkhand, India.

##  Overview
The system fuses multi-source satellite and atmospheric data to generate 12-hour predictive risk maps and real-time fire spread simulations.

## Technical Architecture
- **Model**: Custom U-Net Deep Learning architecture optimized for **Mac M-series (MPS)**.
- **Data Pipeline**: Fuses COP90 (Elevation), Bhuvan LULC (Fuel), NASA FIRMS (Fire History), OSM (Human Activity), and NetCDF (Weather).
- **Simulation**: Cellular Automata (CA) engine for temporal fire spread forecasting.
- **Frontend**: Streamlit dashboard with interactive Folium maps and animated spread visualizations.

##  Project Structure
```text
ForestFire/
├── data/
│   ├── raw/                 # Original satellite/GIS data
│   ├── processed/           # AI-ready feature stacks
├── src/                     # Core Python engines
│   ├── model.py            # U-Net Architecture
│   ├── preprocess.py       # GIS Data Fusion
│   ├── simulation.py       # Fire Spread Engine
│   └── utils.py            # Visualization & GIS Tools
├── web/                     # Dashboard & API
│   ├── app.py              # Streamlit Interface
│   └── api_server.py       # FastAPI Backend
└── main.py                  # Pipeline Orchestrator
```

## Setup & Usage
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Pipeline**:
   ```bash
   python main.py
   ```
3. **Launch Dashboard**:
   ```bash
   streamlit run web/app.py
   ```

## CI/CD
The project includes GitHub Actions workflows for:
- Automated Jupyter Notebook testing.
- Automated package building and publishing validation.
- GIS system dependency management on Ubuntu runners.

---
*Developed for Forest Fire Intelligence in Jharkhand.*
