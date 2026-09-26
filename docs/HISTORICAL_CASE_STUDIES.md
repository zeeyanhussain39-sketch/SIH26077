# 🌪️ Historical Severe Weather Case Studies (SIH Problem Statement 26077)
### Benchmark Indian Events for Model Training, Evaluation, and Nowcasting Validation

This document outlines **four rigorously documented severe weather events** across India representing:
1. **A Severe Cloudburst & Himalayan Flash Flood**
2. **A Catastrophic Thunderstorm / Derecho & Squall Outbreak**
3. **An Extreme Monsoon Deluge & Riverine Flash Flood**
4. **An Orographic Torrential Burst & Debris Flow**

Each case is accompanied by official post-event documentation from the **India Meteorological Department (IMD)** and the **National Centre for Medium Range Weather Forecasting (NCMRWF)**, along with precise geographic bounding boxes, timestamps, and open scientific data acquisition protocols.

---

## 📌 Summary Matrix of Selected Case Studies

| Case ID | Event Name | Hazard Type | Target Date & Time (UTC) | Coordinates (Centroid & Bounding Box) | Confirmed Impact & Key Features | Official Document Citation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`case_01_amarnath_cloudburst_2022`** | **Amarnath Cave Cloudburst** | Cloudburst & Micro-catchment Flash Flood | **2022-07-08**<br>`10:00 - 15:00 UTC`<br>*(Peak: 12:00 UTC / 17:30 IST)* | **34.215°N, 75.503°E**<br>Bounding Box:<br>`[33.8°N, 75.0°E, 34.6°N, 76.0°E]` | >31 mm in ~1 hour over high-altitude rocky micro-basin (~3,880m MSL); triggered violent flash flood & mudflows sweeping pilgrim tents; 16 fatalities. | IMD MC Srinagar: *"Special Weather Report on Cloudburst Incident near Holy Amarnath Cave on 08 July 2022"*; NCMRWF NCUM convective simulation. |
| **`case_02_north_india_squall_2018`** | **North India Squall & Thunderstorm Outbreak** | Severe Convective Storm / Squall Line / Derecho | **2018-05-02**<br>`11:00 - 19:00 UTC`<br>*(16:30 - 00:30 IST)* | **27.180°N, 78.010°E** (Agra/Bharatpur)<br>Bounding Box:<br>`[26.5°N, 76.0°E, 28.5°N, 79.0°E]` | Gale-force downbursts exceeding 126–130 km/h, intense squall line with high CAPE (>3500 J/kg), deep convective dust storm; >110 fatalities across Western UP & Rajasthan. | IMD New Delhi: *"Report on Severe Thunderstorms/Squall over Northwest India on 02 May 2018"*; IMD Met Monograph on Thunderstorms. |
| **`case_03_himachal_flash_flood_2023`** | **Himachal Pradesh Beas Deluge** | Extreme Monsoon Precipitation & Basin Flash Flood | **2023-07-09 to 2023-07-10**<br>`00:00 UTC Jul 9 - 18:00 UTC Jul 10` | **31.710°N, 76.930°E** (Mandi/Kullu/Beas)<br>Bounding Box:<br>`[31.0°N, 76.2°E, 32.8°N, 77.8°E]` | Unprecedented 24-48h rainfall (>250-300 mm) caused by interaction of Western Disturbance with active Monsoon surge; Beas river submerged towns & highways; >70 fatalities. | IMD Climate Diagnostics Bulletin: *"Very Severe Rainfall Event over Himachal Pradesh and Northwest India (July 8-10, 2023)"*. |
| **`case_04_wayanad_deluge_2024`** | **Wayanad Orographic Deluge** | Extreme Localized Deluge, Flash Flood & Debris Flow | **2024-07-29 to 2024-07-30**<br>`12:00 UTC Jul 29 - 06:00 UTC Jul 30` | **11.530°N, 76.180°E** (Meppadi/Chooralmala)<br>Bounding Box:<br>`[11.2°N, 75.8°E, 11.9°N, 76.5°E]` | >372 mm rainfall in 24 hours onto steep, saturated Western Ghats hill-slopes; triggered dual catastrophic debris flows destroying Chooralmala & Mundakkai; >400 casualties. | IMD MC Thiruvananthapuram: *"Special Report on Extremely Heavy Rainfall over Wayanad on 29-30 July 2024"*; NDMA Geotechnical Report. |

---

## 🛰️ 1. INSAT-3D / INSAT-3DR Data Acquisition from MOSDAC

ISRO's **Meteorological and Oceanographic Satellite Data Archival Centre (MOSDAC)** archives geostationary observations from **INSAT-3D** (at 82°E) and **INSAT-3DR** (at 74°E), providing rapid-scan multispectral observations every **15 to 30 minutes**.

### Key Meteorological Channels for Convective Nowcasting:
1. **Water Vapor (`WV`, 6.5 – 7.1 µm):** Detects mid-to-upper tropospheric moisture dynamics, dry slot intrusions, and jet streaks that trigger convective instability.
2. **Thermal Infrared 1 (`TIR1`, 10.3 – 11.3 µm):** Measures cloud top brightness temperature (BT). Temperatures dropping below **210 K (-63°C)** indicate violent vertical updrafts and severe convective cloud tops.
3. **Thermal Infrared 2 (`TIR2`, 11.5 – 12.5 µm):** Split-window channel used in tandem with TIR1 for atmospheric moisture attenuation and microphysics.
4. **Hydro-Estimator Rainfall (`HEM` / `IMS` L2B products):** Satellite-derived instantaneous precipitation rates (mm/hr).

### 📝 Step-by-Step MOSDAC Registration Guide
Because MOSDAC requires user authentication under ISRO security policies, registration must be completed manually once:
1. Navigate to the official portal: **[https://www.mosdac.gov.in](https://www.mosdac.gov.in)**.
2. Click **"Register"** in the top navigation bar.
3. Select your user category: **"Academic / Indian Citizen"**.
4. Fill in:
   - Full Name, Email Address, Mobile Number
   - Affiliation (Institute / University / Hackathon Team Name)
   - Purpose of Data Use: *Select "Academic Research / Disaster Early Warning Nowcasting (SIH 26077)"*
5. Submit the form and verify your email address via the automated confirmation link.
6. Once logged in:
   - Go to **Data Access -> Search & Download**.
   - Select Satellite: `INSAT-3D` or `INSAT-3DR`.
   - Select Sensor: `IMAGER`.
   - Select Product: `3D_IMG_L1B_STD` (Standard Level-1B radiances) or `3R_IMG_L1B_STD`.
   - Enter the target date & bounding box coordinates from the matrix above.
   - Add files to your cart and download the `.h5` (HDF5) or NetCDF files directly into `/data/raw/<case_id>/insat3d/`.

> **Automated Script Integration:**
> Run `python -m src.data_ingestion.mosdac_downloader --case <case_id>` to download via MOSDAC API/FTP or generate realistic NetCDF grids for local pipeline testing while awaiting registration approval.

---

## 🌐 2. IMDAA Reanalysis Request Guide & ERA5 Reanalysis Fallback

### What is IMDAA?
**IMDAA (Indian Monsoon Data Assimilation and Analysis)** is a high-resolution (**12 km**, ~0.12° horizontal grid) regional atmospheric reanalysis for 1979–present, generated by **NCMRWF / MoES** in partnership with the UK Met Office. It is the premier localized reanalysis dataset for Indian extreme weather.

### 📋 Requesting IMDAA Data (Non-Institutional / Student Teams):
Because IMDAA is hosted on government servers at NCMRWF, non-institutional teams must submit a formal access request:
1. Visit the NCMRWF Research Data Service portal: **[https://rds.ncmrwf.gov.in](https://rds.ncmrwf.gov.in)** or **[https://www.ncmrwf.gov.in](https://www.ncmrwf.gov.in)**.
2. Complete the online data request form specifying:
   - **Project Title:** *AI-Driven Hyper-Local Severe Weather Nowcasting System (SIH Problem Statement 26077)*
   - **Dataset:** IMDAA Regional Atmospheric Reanalysis (Hourly / 3-hourly pressure level and surface products).
   - **Target Dates:** Specific event windows listed in the matrix above.
   - **Variables:**
     - Geopotential height (`z`), Temperature (`t`), Specific Humidity (`q`), U-wind (`u`), V-wind (`v`) at 1000, 925, 850, 700, 500, 300, 200 hPa.
     - Surface: Convective Available Potential Energy (`CAPE`), Total Precipitation (`tp`), Surface Pressure (`sp`).
3. Attach a letter of affiliation or hackathon participant declaration.
4. *Approval Note:* Approval typically takes **5 to 15 working days**.

---

### 🔄 The Seamless Fallback: Free ERA5 Reanalysis via Copernicus (CDS)
If IMDAA approval is delayed, the **ECMWF ERA5 Reanalysis** (hosted on the European Union's Copernicus Climate Data Store) provides equivalent hourly atmospheric fields at **0.25° (~25–30 km)** resolution **100% free with immediate automated API access**.

#### 🔑 Setting Up Free ERA5 Access (Copernicus CDS):
1. Register for free at **[https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)**.
2. Accept the standard terms of use for ERA5 data on the dataset page.
3. Copy your Personal Access Token (API Key) from your user profile.
4. Create a file named `.cdsapirc` in your home directory (`C:\Users\<username>\.cdsapirc` on Windows or `~/.cdsapirc` on Linux):
   ```text
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-PERSONAL-ACCESS-TOKEN>
   ```
5. Run the provided script:
   ```bash
   python -m src.data_ingestion.era5_imdaa_downloader --case case_01_amarnath_cloudburst_2022
   ```
This downloads NetCDF files directly into `/data/raw/<case_id>/reanalysis_era5/` containing 3D pressure levels (`t`, `r`, `z`, `u`, `v`) and surface convective diagnostics (`cape`, `total_precipitation`).

---

## 🏔️ 3. SRTM 30m Digital Elevation Model (DEM) (Free & Zero Registration)

Topography is a critical physical forcing mechanism for cloudbursts and flash floods in India (orographic lifting against Himalayan ridges and Western Ghats escarpments).

We provide automated 30m (1-arcsecond) DEM acquisition from public **AWS Open Data (Copernicus GLO-30 / SRTM)** requiring **zero registration, zero API keys, and zero payment**:
- For each case study, the script calculates the covering DEM tiles (e.g., `N34E075` for Amarnath, `N11E076` for Wayanad).
- Downloads the Cloud-Optimized GeoTIFF (COG) or elevation grid.
- Crops the DEM precisely to the case's catchment bounding box and stores it in `/data/raw/<case_id>/dem_srtm/`.

Run the automated acquisition:
```bash
python -m src.data_ingestion.srtm_dem_downloader --case case_01_amarnath_cloudburst_2022
```

---

## 📂 Raw Data Folder Hierarchy

```text
data/raw/
├── case_01_amarnath_cloudburst_2022/
│   ├── metadata.json
│   ├── insat3d/
│   │   └── insat3d_wv_tir_20220708_1200utc.nc
│   ├── reanalysis_era5/
│   │   ├── era5_pressure_levels_20220708.nc
│   │   └── era5_surface_cape_precip_20220708.nc
│   └── dem_srtm/
│       └── srtm_30m_amarnath_catchment.tif
│
├── case_02_north_india_squall_2018/
│   ├── metadata.json
│   ├── insat3d/
│   ├── reanalysis_era5/
│   └── dem_srtm/
│
├── case_03_himachal_flash_flood_2023/
│   ├── metadata.json
│   ├── insat3d/
│   ├── reanalysis_era5/
│   └── dem_srtm/
│
└── case_04_wayanad_deluge_2024/
    ├── metadata.json
    ├── insat3d/
    ├── reanalysis_era5/
    └── dem_srtm/
```
