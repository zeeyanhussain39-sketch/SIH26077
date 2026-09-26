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

## 🌪️ Benchmark Historical Severe Weather Case Studies

We have selected **4 rigorously documented historical severe weather events across India** for model development, evaluation, and nowcasting validation. Detailed reports, official IMD citations, and registration guides are in [docs/HISTORICAL_CASE_STUDIES.md](file:///c:/Users/zeeya/Desktop/SIH26077/docs/HISTORICAL_CASE_STUDIES.md).

| Case ID | Event Name | Hazard Type | Target Window (UTC) | Key Official Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **`case_01_amarnath_cloudburst_2022`** | Amarnath Cave Cloudburst | Cloudburst & Flash Flood | 2022-07-08 (10:00–15:00 UTC) | IMD Srinagar Special Report on Amarnath Cloudburst |
| **`case_02_north_india_squall_2018`** | North India Squall / Derecho | Severe Squall & Thunderstorm | 2018-05-02 (11:00–19:00 UTC) | IMD New Delhi Report on 02 May 2018 Squall Outbreak |
| **`case_03_himachal_flash_flood_2023`** | Himachal Beas Deluge | Extreme Orographic Flash Flood | 2023-07-09 to 10 (48h storm) | IMD Extreme Weather Bulletin on Northwest India |
| **`case_04_wayanad_deluge_2024`** | Wayanad Orographic Deluge | Extreme Deluge & Debris Flow | 2024-07-29 to 30 (24h burst) | IMD Thiruvananthapuram Report on Wayanad Deluge |

### Running the Data Acquisition Pipeline:

```bash
# 1. Download/prepare all 4 historical case studies (Satellite, Reanalysis, 30m DEM):
python -m src.data_ingestion.case_studies_manager --case all

# 2. Or acquire an individual case study:
python -m src.data_ingestion.case_studies_manager --case case_01_amarnath_cloudburst_2022

# 3. Individual channel / dataset acquisition:
python -m src.data_ingestion.mosdac_downloader --case case_01_amarnath_cloudburst_2022
python -m src.data_ingestion.era5_imdaa_downloader --case case_01_amarnath_cloudburst_2022
python -m src.data_ingestion.srtm_dem_downloader --case case_01_amarnath_cloudburst_2022
```

---

## 🛰️ Open Scientific Data Sources

1. **INSAT-3D / 3DR Imager & Sounder (ISRO / MOSDAC):**
   - Public open portal: [MOSDAC Data Portal](https://www.mosdac.gov.in/) (Free registration required)
   - Channels: Thermal Infrared (TIR1: 10.8 µm), Split-window (TIR2: 12.0 µm), Water Vapor (WV: 6.8 µm).
2. **Atmospheric Reanalysis:**
   - **Primary:** IMDAA 12 km Regional Reanalysis (NCMRWF/MoES - academic request).
   - **Fallback:** ERA5 Hourly Reanalysis on Pressure Levels (Copernicus CDS - Free automated API).
3. **Doppler Weather Radar (IMD):**
   - Max-Z Reflectivity grids (dBZ), radial velocity, and storm structure.
4. **Topography (SRTM / Copernicus 30m Global DEM):**
   - 30m digital elevation model for steep catchment slope and flash flood funneling calculation (**100% free, zero registration**).

---

## 📄 License
Open-source under the Apache 2.0 / MIT License.
Built for the Smart India Hackathon (SIH 26077).
