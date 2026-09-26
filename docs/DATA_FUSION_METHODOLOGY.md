# 📐 Spatiotemporal Data Fusion & Resampling Methodology
### SIH Problem Statement 26077 — Hyper-Local Severe Weather Nowcasting (2–6 Hours Ahead)

---

## 🎯 1. Executive Summary & Objective

Severe convective weather phenomena — particularly **cloudbursts (>100 mm/h localized)**, **squall lines / derechos**, and **mountain flash floods** — are governed by nonlinear interactions spanning multiple physical domains:
1. **Cloud microphysics & radiative emission:** Rapid vertical cloud expansion observed in geostationary satellite infrared/water vapor radiances.
2. **Mesoscale & synoptic thermodynamic instability:** Convective Available Potential Energy (CAPE), vertical wind shear, and moisture transport captured in numerical atmospheric reanalysis.
3. **Orographic forcing & catchment hydrology:** Steep terrain slope, ridge funneling, and catchment elevation captured in digital elevation models (DEM).

However, raw meteorological datasets exhibit **extreme spatiotemporal asymmetry**:

| Modality | Source | Native Spatial Resolution | Native Temporal Frequency | Coordinate / Projection |
| :--- | :--- | :--- | :--- | :--- |
| **Satellite Imagery** | ISRO MOSDAC (INSAT-3D/3DR) | **~4.0 km** (0.04° at nadir) | **15 – 30 minutes** | Geostationary fixed grid / WGS84 |
| **Atmospheric Reanalysis** | ECMWF CDS (ERA5) / NCMRWF (IMDAA) | **~28 km** (0.25°) or **~12 km** (0.12°) | **1 – 3 hourly** | Regular lat/lon grid |
| **Digital Elevation Model** | SRTM / Copernicus GLO-30 | **~30 meters** (1 arc-second) | **Static in time** | EPSG:4326 (WGS84) GeoTIFF |

The objective of our **Data Fusion Pipeline** (`src/data_ingestion/data_fusion.py`) is to reconcile these disparate sources into a single, physically consistent, multi-dimensional **`xarray.Dataset`** with dimensions `(time, lat, lon, level)`.

---

## 🧭 2. Common Reference Grid Selection

### Spatial Grid: $0.02^\circ \times 0.02^\circ$ (~2.2 km × 2.2 km)
- **Scale Matching:** Meso-$\gamma$ atmospheric scale (2–20 km), which is the characteristic horizontal dimension of individual convective updrafts and cloudburst thunderstorm cells.
- **Sensor MTF Alignment:** A $0.02^\circ$ grid oversamples the native ~4 km satellite infrared footprint just enough to enable smooth convolutional feature extraction in downstream deep learning models (ConvLSTM/U-Net) without fabricating sub-pixel artifacts.

### Temporal Grid: 30-Minute Uniform Intervals
- Matches the standard geostationary scanning cadence of INSAT-3D (30 min) and INSAT-3DR (interleaved 15 min rapid scan mode).
- Matches the required **2 to 6 hours operational nowcasting lead-time window** specified in SIH Problem Statement 26077 (yielding $N = 4 \text{ to } 12$ forward prediction frames).

---

## 🔬 3. Resampling Choices & Mathematical Formulation

### A. Continuous Thermodynamic & Mass Fields (Bilinear Interpolation)
*Variables:* Temperature ($T$), Geopotential ($z$), Relative Humidity ($r$), Horizontal Winds ($u, v$), Satellite Radiances ($\text{TIR1}, \text{TIR2}, \text{WV}$).

#### Methodological Choice: **2D Bilinear Interpolation**
Given a target coordinate $(x, y)$ bounded by four grid vertices $Q_{11}=(x_1, y_1)$, $Q_{12}=(x_1, y_2)$, $Q_{21}=(x_2, y_1)$, and $Q_{22}=(x_2, y_2)$:
$$f(x, y) \approx \frac{1}{(x_2 - x_1)(y_2 - y_1)} \begin{bmatrix} x_2 - x & x - x_1 \end{bmatrix} \begin{bmatrix} f(Q_{11}) & f(Q_{12}) \\ f(Q_{21}) & f(Q_{22}) \end{bmatrix} \begin{bmatrix} y_2 - y \\ y - y_1 \end{bmatrix}$$

#### Why Not Nearest Neighbor?
Nearest-neighbor regridding creates artificial block step-discontinuities across grid borders. When calculating spatial derivatives (such as horizontal temperature gradients $\nabla T$ or vorticity $\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$), nearest-neighbor produces infinite mathematical singularities and spurious ringing in convolutional kernels.

#### Why Not High-Order Bicubic or Splines?
Higher-order polynomial interpolations suffer from **Runge’s phenomenon** and overshoot near sharp boundaries (e.g., producing physically impossible negative CAPE or relative humidity $>100\%$). Bilinear interpolation guarantees that interpolated values are strictly bounded by their local convex hull:
$$\min(f(Q_{ij})) \le f(x, y) \le \max(f(Q_{ij}))$$

---

### B. High-Resolution Topography (Slope-Preserving Downsampling)
*Variables:* Elevation ($h$), Topographic Slope ($\theta$).

#### Methodological Problem:
If one downsamples a 30m DEM directly to a 2 km grid using simple spatial averaging, steep mountain ravines, cliff faces, and gorge walls are smoothed out. For example, a 40° Himalayan cliff would be smoothed to an apparent 8° hill, severely underestimating gravitational catchment runoff and flash flood propagation velocity!

#### Our Solution: **Native Gradient Calculation Prior to Regridding**
1. We first compute the topographic slope $\theta$ on the **native high-resolution 30m grid** using central differences in Cartesian coordinates:
   $$\nabla h = \left( \frac{\partial h}{\partial x}, \frac{\partial h}{\partial y} \right)$$
   $$\theta_{\text{native}} = \arctan \sqrt{\left(\frac{\partial h}{\partial x}\right)^2 + \left(\frac{\partial h}{\partial y}\right)^2}$$
2. We then interpolate both the elevation $h$ and the true topographic slope $\theta_{\text{native}}$ onto the common $0.02^\circ$ target grid using bilinear interpolation.
3. This preserves the acute hydraulic energy of mountain corridors while ensuring computational alignment with atmospheric tensors.

---

### C. Temporal Resampling of Synoptic Reanalysis Fields
*Variables:* ERA5 / IMDAA reanalysis available at 1-hour or 3-hour timesteps; satellite available at 30-minute timesteps.

#### Methodological Choice: **Piecewise Linear Temporal Interpolation**
Synoptic and mesoscale mass variables (CAPE, 500 hPa geopotential, 850 hPa moisture flux) evolve continuously on 3–6 hour advection timescales. Linearly interpolating reanalysis states along the temporal axis to match intermediate satellite scan epochs mirrors standard 4D continuous data assimilation (analogous to 4D-Var trajectory nudging) and prevents artificial step jumps at hourly boundaries.

#### Precipitation Rate Normalization:
Accumulated reanalysis precipitation $\text{tp}$ [m] is converted to an instantaneous rate [mm/hr]:
$$\text{Rain Rate } [\text{mm/hr}] = \frac{\text{tp} \times 1000.0}{\Delta t_{\text{hours}}}$$

---

## 📊 4. Unit Normalization & Dimensional Consistency Matrix

To prevent numerical gradient instability during model training, all raw units are converted to standard meteorological SI conventions:

| Field Name | Raw Source | Raw Unit | Unified Target Unit | Conversion Formulation | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tir1_brightness_temp` | MOSDAC L1B | Kelvin | **Kelvin [K]** | Direct | Cloud top radiative temperature |
| `tir2_brightness_temp` | MOSDAC L1B | Kelvin | **Kelvin [K]** | Direct | Split-window thermal channel |
| `water_vapor_brightness_temp` | MOSDAC L1B | Kelvin | **Kelvin [K]** | Direct | Upper-tropospheric moisture |
| `surface_pressure_hpa` | ERA5 `sp` | Pascals [Pa] | **hPa** | `sp / 100.0` | Surface atmospheric pressure |
| `precip_rate_mm_hr` | ERA5 `tp` | meters [m] | **mm/hr** | `(tp * 1000) / 3.0` | Instantaneous rainfall intensity |
| `geopotential_height_gpm`| ERA5 `z` | $\text{m}^2/\text{s}^2$ | **gpm** | `z / 9.80665` | Height of pressure surfaces |
| `cape_j_kg` | ERA5 `cape` | J/kg | **J/kg** | Direct | Convective Available Potential Energy |
| `t`, `r`, `u`, `v` | ERA5 PL | K, %, m/s, m/s | **K, %, m/s, m/s** | Direct | 3D atmospheric thermodynamic state |
| `elevation` | SRTM | meters | **m (MSL)** | Direct | Altitude above sea level |
| `topographic_slope` | SRTM | degrees | **degrees [°]** | $\arctan(\|\nabla h\|)$ | Terrain steepness |

---

## ⚡ 5. Derived Severe Storm Diagnostic Features

Our fusion pipeline automatically computes 5 domain-specific meteorological indicators ready for machine learning feature extraction:

1. **Split-Window BTD (`split_window_btd_k`):**
   $$\text{BTD} = \text{TIR1} - \text{TIR2} \quad [\text{K}]$$
   *Purpose:* Differentiates optically thick cumulonimbus cloud shields ($\text{BTD} \approx 0\text{ K}$) from thin semitransparent cirrus ($\text{BTD} > 2\text{--}4\text{ K}$).

2. **Convective Overshoot Index (`convective_overshoot_index`):**
   $$\text{Overshoot} = \max(0, 215.0 - \text{TIR1}) \quad [\text{K}]$$
   *Purpose:* Quantifies severe storm updraft penetration above the tropical equilibrium level (colder than 215 K / -58°C).

3. **Deep-Layer Bulk Wind Shear (850 to 500 hPa & 850 to 200 hPa):**
   $$\text{Shear}_{850\text{--}500} = \sqrt{(u_{500} - u_{850})^2 + (v_{500} - v_{850})^2} \quad [\text{m/s}]$$
   *Purpose:* Key kinematic driver for multicell storm organization, supercells, and squall line maintenance.

4. **Horizontal Wind Speed at Each Level:**
   $$\text{WindSpeed}_p = \sqrt{u_p^2 + v_p^2} \quad [\text{m/s}]$$

5. **Orographic Upslope Index (`orographic_upslope_index`):**
   $$\text{Upslope} = \|\vec{V}_{850}\| \cdot \sin(\theta_{\text{slope}})$$
   *Purpose:* Approximates forced mechanical lift of warm, moist monsoon air along mountain escarpments (the primary trigger for Himalayan cloudbursts and Western Ghats torrential bursts).

---

## ❓ 6. Hackathon / SIH Defense Q&A Guide

#### Q1: "Why did you interpolate the reanalysis data instead of just keeping it on its native 25 km grid?"
> **Answer:** *"Severe thunderstorms and cloudbursts are hyper-local phenomena (typically 5 to 15 km wide). If reanalysis fields like CAPE and moisture remained at 25 km, a single pixel would cover both the cloudburst catchment and unaffected neighboring valleys. Interpolating continuous thermodynamic mass fields bilinearly onto our 2 km common grid provides a smoothly differentiable background environment that allows our spatial ConvNet to combine large-scale instability with fine-scale satellite convective cloud tops and topography."*

#### Q2: "Doesn't temporal interpolation of reanalysis introduce artificial data between 3-hour timesteps?"
> **Answer:** *"Synoptic thermodynamic features like geopotential height, CAPE, and deep-layer wind shear evolve over multi-hour scales. Piecewise linear interpolation between verified reanalysis steps is standard practice in meteorological 4D data assimilation. Crucially, the **high-frequency temporal dynamics** are provided directly by the 30-minute INSAT-3D satellite scans, which capture real-time convective initiation."*

#### Q3: "Why did you compute terrain slope on the 30m DEM before downsampling?"
> **Answer:** *"Because slope is a derivative operator. If you average elevation over a 2 km pixel first, mountain peaks are pulled down and valley floors are pushed up, artificially reducing slope angles by over 60%. By computing slope at native 30m resolution first, we preserve the true hydrological steepness of narrow gullies where flash flood runoff originates."*

---

## 💻 7. Execution

```bash
# Fuse all 4 historical case studies into processed NetCDF files:
python -m src.data_ingestion.data_fusion --case all

# Or fuse a specific case study (e.g. Amarnath Cloudburst):
python -m src.data_ingestion.data_fusion --case case_01_amarnath_cloudburst_2022 --res 0.02
```

Outputs are saved to:
`/data/processed/<case_id>/aligned_features_<case_id>.nc`
