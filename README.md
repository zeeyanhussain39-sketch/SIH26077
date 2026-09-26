# ⚡ AI-Driven Hyper-Local Severe Weather Nowcasting System
### Smart India Hackathon (SIH) — Problem Statement 26077

> **Objective:** Real-time, hyper-local prediction and early warning of severe thunderstorms, cloudbursts, and flash floods **2 to 6 hours ahead** of onset.
> **Constraint:** Built exclusively using **100% free and open-source tools** (zero paid cloud, zero paid weather or maps APIs).

---

## 🏗️ Architecture & Technology Stack

| Layer | Technology | Free / Open Source Rationale |
| :--- | :--- | :--- |
| **Interactive Dashboard** | [Streamlit](https://streamlit.io/) | Pure Python rapid reactive UI, zero frontend build step |
| **Spatial Mapping** | [Folium](https://python-visualization.github.io/folium/) & OpenStreetMap | Free worldwide tiles, interactive leaflet maps, zero map API keys |
| **Alert & Inference API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Lightweight asynchronous REST API producing CAP-v1.2 compliant alerts |
| **Spatial & Gridded Data** | `xarray`, `netCDF4`, `rasterio` | Open multidimensional array handling for satellite & radar data |
| **Deep Learning Engine** | [PyTorch](https://pytorch.org/) (CPU/GPU) | Spatio-temporal ConvLSTM / U-Net sequence-to-sequence nowcaster |
| **Explainable AI (XAI)** | [SHAP](https://shap.readthedocs.io/) | Transparent feature attribution (CAPE, radar dBZ, cloud top temp) |
| **Meteorological Data** | MOSDAC (ISRO), IMD DWR, Open-Meteo | Public scientific open data & keyless open NWP APIs |

---

## 📁 Repository Structure

```text
SIH26077/
├── data/
│   ├── raw/                  # Downloaded NetCDF/HDF5/GeoTIFF radar & satellite files
│   └── processed/            # Normalized multi-channel spatio-temporal tensors
├── notebooks/
│   └── exploration/
│       └── 01_satellite_radar_eda.ipynb  # Jupyter notebook for satellite/radar EDA
├── src/
│   ├── __init__.py
│   ├── data_ingestion/       # Ingestion for INSAT-3D/3DR, Doppler Radar, Open-Meteo
│   │   ├── __init__.py
│   │   ├── satellite_loader.py
│   │   ├── radar_loader.py
│   │   └── open_meteo_client.py
│   ├── feature_engineering/  # Atmospheric instability & convective severity indices
│   │   ├── __init__.py
│   │   └── atmospheric_indices.py  # CAPE, CIN, Cloudburst & Flash Flood risk formulas
│   ├── model/                # PyTorch Spatio-Temporal Nowcasting Network
│   │   ├── __init__.py
│   │   ├── nowcast_net.py    # Multi-horizon ConvNet/ConvLSTM architecture
│   │   └── inference.py      # Lead-time inference (2 to 6 hours ahead)
│   ├── xai/                  # Model explainability for disaster managers (NDMA/SDMA)
│   │   ├── __init__.py
│   │   └── explainability.py # SHAP attribution & hazard driver breakdowns
│   └── alerts/               # Warning generation engine
│       ├── __init__.py
│       └── alert_engine.py   # Common Alerting Protocol (CAP-v1.2) & GeoJSON output
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI backend with /api/v1/nowcast and /api/v1/alerts
├── app/
│   ├── __init__.py
│   └── main.py               # Streamlit interactive dashboard with Folium & OSM
├── requirements.txt          # 100% free and open-source library dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Set Up Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Or activate on Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501` to view the interactive nowcasting control center.

### 4. (Optional) Run the FastAPI Alert Backend

```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive Swagger API docs: `http://localhost:8000/docs`
- Nowcast endpoint: `http://localhost:8000/api/v1/nowcast?latitude=30.3165&longitude=78.0322&lead_hours=3`
- Active CAP Alerts: `http://localhost:8000/api/v1/alerts?latitude=30.3165&longitude=78.0322&lead_hours=3`

---

## 🛰️ Open Scientific Data Sources

1. **INSAT-3D / 3DR Imager & Sounder (ISRO / MOSDAC):**
   - Public open portal: [MOSDAC Data Portal](https://www.mosdac.gov.in/)
   - Channels: Thermal Infrared (TIR1: 10.8 µm), Water Vapor (WV: 6.8 µm), Cloud Motion Vectors (CMV).
2. **Doppler Weather Radar (IMD):**
   - Max-Z Reflectivity grids (dBZ), radial velocity, and storm structure.
3. **Open-Meteo High-Resolution NWP:**
   - Free, keyless API providing hourly CAPE, surface pressure, precipitation rates, and wind gusts.
4. **SRTM / CartoDEM Topography:**
   - 30m digital elevation model for steep catchment slope and flash flood funneling calculation.

---

## 📄 License
Open-source under the Apache 2.0 / MIT License.
Built for the Smart India Hackathon (SIH 26077).
