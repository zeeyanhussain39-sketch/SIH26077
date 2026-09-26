# 🔬 Severe Weather Feature Engineering Pipeline
### SIH Problem Statement 26077 — Hyper-Local Severe Weather Nowcasting (2–6 Hours Ahead)

This module (`src/feature_engineering/feature_extractor.py`) computes 5 essential meteorological, kinematic, and hydrological feature layers from the aligned multi-sensor dataset. Each feature is designed to isolate the physical precursors of **severe thunderstorms**, **cloudbursts (>100 mm/h)**, and **catchment flash floods** 2 to 6 hours before peak ground impact.

---

## 📑 Feature Layers & Physical Formulation

### 1. Integrated Water Vapor (IWV) & Rate of Change ($d(\text{IWV})/dt$)
- **Physical Meaning:** Quantifies the total column precipitable water available for convective condensation and cloudburst deluge.
- **Formulation:**
  $$\text{Specific Humidity: } q(p) = \frac{0.622 \cdot e(T, r)}{p - 0.378 \cdot e(T, r)} \quad [\text{kg/kg}]$$
  $$\text{IWV} = \frac{1}{g} \int_{200\text{ hPa}}^{1000\text{ hPa}} q(p) \, dp \approx \frac{1}{g_0} \sum_{k} \frac{q_k + q_{k+1}}{2} \cdot \Delta p_k \times 100 \quad [\text{mm or kg/m}^2]$$
  $$\text{Rate of Change: } \frac{d(\text{IWV})}{dt} = \frac{\text{IWV}(t) - \text{IWV}(t - \Delta t)}{\Delta t} \quad [\text{mm/hr}]$$
- **Nowcasting Significance:** A rapid positive surge ($d(\text{IWV})/dt > 0$) indicates strong low-level moisture convergence feeding into an active storm corridor 2–4 hours ahead of cloudburst initiation.

---

### 2. Convective Available Potential Energy (CAPE) & Inhibition (CIN)
- **Physical Meaning:** Quantifies the buoyant energy available to accelerate vertical updrafts through the tropospheric column.
- **Engine:** Evaluated using **MetPy (`metpy.calc.surface_based_cape_cin`)** and **`metpy.calc.dewpoint_from_relative_humidity`**:
  $$\text{Dewpoint: } T_d = \text{DewpointFromRH}(T, r)$$
  $$\text{CAPE} = \int_{z_{\text{LFC}}}^{z_{\text{EL}}} g \left( \frac{T_{v,\text{parcel}} - T_{v,\text{env}}}{T_{v,\text{env}}} \right) dz \quad [\text{J/kg}]$$
  $$\text{CIN} = \int_{z_{\text{sfc}}}^{z_{\text{LFC}}} g \left( \frac{T_{v,\text{parcel}} - T_{v,\text{env}}}{T_{v,\text{env}}} \right) dz \quad [\text{J/kg}]$$
- **Severity Thresholds:**
  - $\text{CAPE} < 1000 \text{ J/kg}$: Low / Marginal Instability
  - $\text{CAPE} = 1500\text{--}2500 \text{ J/kg}$: Moderate Instability (Thunderstorms probable)
  - $\text{CAPE} > 3000\text{--}4000 \text{ J/kg}$: Extreme Instability (Cloudburst / Supercell updrafts)
  - $\text{CIN} \to 0 \text{ J/kg}$: Convective "cap" broken; explosive initiation imminent.

---

### 3. Low-Level Wind Convergence & Vertical Wind Shear
- **Low-Level Horizontal Wind Convergence ($-\nabla_H \cdot \vec{V}$ at 850 hPa):**
  $$\text{Convergence}_{850} = -\left( \frac{\partial u_{850}}{\partial x} + \frac{\partial v_{850}}{\partial y} \right) \quad [\text{s}^{-1}]$$
  *Significance:* Directly forces mechanical ascent of boundary-layer moisture into the free troposphere.
- **Bulk Vertical Wind Shear (850 to 500 hPa & 850 to 200 hPa):**
  $$\text{Shear}_{850\text{--}500} = \sqrt{(u_{500} - u_{850})^2 + (v_{500} - v_{850})^2} \quad [\text{m/s}]$$
  $$\text{Shear}_{850\text{--}200} = \sqrt{(u_{200} - u_{850})^2 + (v_{200} - v_{850})^2} \quad [\text{m/s}]$$
  *Significance:* Moderate-to-high shear ($> 15\text{--}25 \text{ m/s}$) tilts convective updrafts away from downdrafts, preventing self-destruction and organizing multicellular squall lines and supercells.

---

### 4. Cloud Top Temperature (CTT) & Rapid Cooling Drop Rate
- **Cloud Top Temperature:** Extracted directly from INSAT-3D Thermal Infrared (`TIR1`: 10.8 µm):
  $$\text{CTT} = \text{TIR1} \quad [\text{K}]$$
- **CTT Cooling / Drop Rate:**
  $$\text{Cooling Rate} = -\frac{d(\text{CTT})}{dt} = \frac{\text{CTT}(t - \Delta t) - \text{CTT}(t)}{\Delta t} \quad [\text{K/hr}]$$
- **Nowcasting Significance:**
  - CTT cooling rates exceeding **$+15\text{ to } +25 \text{ K/hr}$** indicate rapid cloud vertical ascent ($w > 15 \text{ m/s}$).
  - CTT dropping below **$215 \text{ K (-58°C)}$** signals convective overshoot penetrating the tropopause.

---

### 5. Topography: Elevation, Slope & D8 Flow Accumulation
- **Elevation ($h$):** Above Mean Sea Level (MSL) in meters from 30m DEM.
- **Topographic Slope ($\theta$):**
  $$\theta = \arctan \sqrt{\left(\frac{\partial h}{\partial x}\right)^2 + \left(\frac{\partial h}{\partial y}\right)^2} \quad [\text{degrees}]$$
- **D8 Hydrological Routing:**
  - Computes flow direction codes ($1\text{--}8$) directing water into the steepest descent neighbor.
  - Computes **Flow Accumulation**: total upstream drainage area draining through each grid cell.
- **Flash Flood Significance:** Identifies high-risk natural torrent channels, nullahs, and valley bottoms where flash flood debris concentrates.

---

## 📦 Output Files & Manifest

For each historical case study, outputs are organized in `/data/processed/<case_id>/`:

| File | Format | Purpose & Description |
| :--- | :--- | :--- |
| `feature_cube_<case_id>.nc` | NetCDF4 (`xarray`) | Multi-channel 4D tensor grid `(time, lat, lon)` ready for PyTorch ConvLSTM / U-Net nowcasting models. |
| `feature_table_<case_id>.parquet` | Apache Parquet | High-speed columnar tabular matrix ready for gradient-boosted trees (XGBoost, LightGBM, CatBoost) and scikit-learn. |
| `feature_table_<case_id>.csv` | Plain CSV | Human-readable inspection and validation matrix with coordinates, timestamps, and severity flags. |

---

## 💻 Python Usage Examples

### 1. Run Feature Extraction via CLI

```bash
# Extract features for all 4 historical case studies:
python -m src.feature_engineering.feature_extractor --case all

# Or extract for a single case study:
python -m src.feature_engineering.feature_extractor --case case_01_amarnath_cloudburst_2022
```

### 2. Loading into PyTorch as a 4D Convective Feature Tensor

```python
import xarray as xr
import torch

# Open engineered feature cube
ds_features = xr.open_dataset("data/processed/case_01_amarnath_cloudburst_2022/feature_cube_case_01_amarnath_cloudburst_2022.nc")

# Select channels for spatio-temporal nowcasting
channel_names = [
    "layer_cloud_top_temp_k",
    "layer_ctt_cooling_rate_k_hr",
    "layer_metpy_cape_j_kg",
    "layer_low_level_convergence_s1",
    "layer_vertical_wind_shear_mps",
    "layer_iwv_rate_of_change_mm_hr",
    "layer_terrain_slope_deg",
    "layer_flow_accumulation"
]

# Stack into numpy array: (Time, Channels, Height, Width)
tensor_data = np.stack([ds_features[ch].values for ch in channel_names], axis=1)
tensor_torch = torch.from_numpy(tensor_data).float()

print("PyTorch Spatio-Temporal Tensor Shape:", tensor_torch.shape)
# Output: torch.Size([6, 8, 41, 51]) -> (Timesteps, Channels, Lat, Lon)
```

### 3. Loading Tabular Matrix for Tree-Based Models (XGBoost / LightGBM)

```python
import pandas as pd

df = pd.read_parquet("data/processed/case_01_amarnath_cloudburst_2022/feature_table_case_01_amarnath_cloudburst_2022.parquet")

features = [
    "layer_iwv_mm", "layer_iwv_rate_of_change_mm_hr",
    "layer_metpy_cape_j_kg", "layer_metpy_cin_j_kg",
    "layer_low_level_convergence_s1", "layer_vertical_wind_shear_mps",
    "layer_cloud_top_temp_k", "layer_ctt_cooling_rate_k_hr",
    "layer_elevation_m", "layer_terrain_slope_deg", "layer_flow_accumulation"
]

X = df[features]
y_convective = df["flag_convective_initiation"]
y_flood = df["flag_flash_flood_susceptibility"]

print(f"Tabular dataset loaded: {X.shape[0]} samples with {X.shape[1]} physical features.")
```
