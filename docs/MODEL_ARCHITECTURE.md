# 🧠 Multi-Task Severe Weather Nowcasting Architecture
### SIH Problem Statement 26077 — Hyper-Local Severe Weather Nowcasting (2–6 Hours Ahead)

---

## 🎯 1. Problem Formulation & Multi-Task Design

Operationally predicting severe convective weather events across India requires simultaneously forecasting three distinct but physically interlinked hazard types:
1. **Severe Thunderstorms / Squall Lines:** High convective energy, strong vertical wind shear, and gale-force downbursts ($>80\text{--}130 \text{ km/h}$).
2. **Cloudbursts:** Extreme localized precipitation bursts ($\ge 100 \text{ mm/hr}$ over a micro-catchment).
3. **Catchment Flash Floods:** Surface inundation and violent debris flows caused by torrential rain funneling through steep mountain drainage gullies.

Instead of three isolated models, we utilize a **Multi-Task Architecture** with a shared physical feature representation and three specialized hazard prediction heads:

```
                            Shared Input Features (14 Physical Channels)
                 [ IWV, d(IWV)/dt, MetPy CAPE, CIN, Convergence, Bulk Shear,
                   Deep Shear, CTT, CTT Cooling Rate, Precip, Elevation, Slope, Acc, Lead ]
                                               │
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │   Shared Feature Representation Engine    │
                         │   Nonlinear Interaction & Spatial Fusion  │
                         └─────────────────────┬─────────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
     ┌───────────────────┐           ┌───────────────────┐           ┌───────────────────┐
     │ Thunderstorm Head │           │  Cloudburst Head  │           │ Flash Flood Head  │
     │   (Squall/Gale)   │           │ (Extreme Deluge)  │           │   (Catchment/DEM) │
     └─────────┬─────────┘           └─────────┬─────────┘           └─────────┬─────────┘
               │                               │                               │
               ▼                               ▼                               ▼
       P(Thunderstorm)                   P(Cloudburst)                   P(Flash Flood)
      at T+2h to T+6h                   at T+2h to T+6h                 at T+2h to T+6h
```

---

## ⚙️ 2. Architectural Rationale: Operational Proxy vs Production Transformer

> **Important Methodological Note (Judges Reference):**
> As specified in our hackathon engineering design, our current operational implementation in `src/model/multitask_model.py` uses a **multi-task gradient-boosted feature extractor (`HistGradientBoostingClassifier`)** with calibrated probability heads as a compute-efficient proxy for the full **Spatio-Temporal Transformer (e.g. Earthformer / MetNet-3)** described in SIH Problem Statement 26077.

### Why HistGradientBoosting for the Hackathon Prototype?
1. **Sample Efficiency:** Operates robustly on 156,000+ spatiotemporal cell samples without overfitting or vanishing gradients.
2. **Zero GPU Barrier:** Trains in under 30 seconds on standard CPU hardware, ensuring complete reproducibility without expensive cloud compute.
3. **Tabular & Physical Mixed Inputs:** Naturally handles mixed scales (elevation in meters, CAPE in thousands of J/kg, convergence in $10^{-4}\text{ s}^{-1}$) without requiring complex learned positional encodings.
4. **Production Roadmap (Future Work):** Full 4D Spatio-Temporal Cuboid Attention (Earthformer / ConvLSTM backbones) is implemented in `src/model/nowcast_net.py` and scheduled for GPU cluster fine-tuning during national deployment.

---

## 🏷️ 3. Physically-Grounded Weak Supervision & Label Generation

Because real-world labeled ground truth for extreme micro-scale events is scarce, we generated training targets using **domain physics rules from IMD and NCMRWF meteorological literature**, anchored against verified historical disasters:

| Hazard Target | Physical Domain Rule | Anchored Ground Truth Disaster Events |
| :--- | :--- | :--- |
| **Severe Thunderstorm** ($Y_{\text{ts}}$) | $\text{CAPE} \ge 1800 \text{ J/kg}$ **AND** $\text{Vertical Shear} \ge 8\text{ m/s}$ **AND** $\text{Convergence} > 0$<br>*OR* $\text{CTT Cooling Rate} \ge 10 \text{ K/hr}$ with $\text{CTT} \le 245 \text{ K}$ | Case 2: North India May 2018 Squall Outbreak (>126 km/h downbursts, 110+ fatalities) |
| **Cloudburst** ($Y_{\text{cb}}$) | $\frac{d(\text{IWV})}{dt} > 0.8 \text{ mm/hr}$ **AND** $\text{CAPE} \ge 2200 \text{ J/kg}$ **AND** $\text{CTT} \le 220 \text{ K}$ (overshooting top)<br>*OR* $\text{Precip Rate} \ge 25 \text{ mm/hr}$ with $\text{Cooling} \ge 12 \text{ K/hr}$ | Case 1: Amarnath Cave Cloudburst (July 2022)<br>Case 4: Wayanad Torrential Burst (July 2024) |
| **Flash Flood** ($Y_{\text{ff}}$) | $(\text{Cloudburst} = 1 \text{ or Precip} \ge 15\text{ mm/hr})$ **AND** $\text{Slope} \ge 10^\circ$ **AND** $\text{Flow Accumulation} \ge 65\text{th percentile}$ | Case 1: Amarnath Baltal Catchment<br>Case 3: Himachal Beas River Deluge<br>Case 4: Wayanad Debris Flow |

---

## ⏱️ 4. Lead-Time Modulation (T+2h to T+6h)

Atmospheric predictability limits (Lorenz chaos theory) dictate that forecast uncertainty grows with increasing projection horizons. Our model incorporates an explicit predictability decay function:
$$\text{Decay Factor: } \gamma(\tau) = \exp(-(\tau - 2) \cdot 0.09) \quad \text{for } \tau \in [2, 6]\text{ hours}$$
$$P_{\text{calibrated}}(\text{Hazard} \mid \tau) = P_{\text{raw}} \cdot \gamma(\tau)$$
- At **$T+2\text{h}$ (Immediate Nowcast)**: $\gamma(2) = 1.0$ (Peak physical confidence).
- At **$T+6\text{h}$ (Horizon Boundary)**: $\gamma(6) \approx 0.69$ (Probabilities reflect broader synoptic uncertainty).
- Flash flood risk includes a hydrological catchment concentration lag ($\text{Lag Factor} = 1.0 + (\tau - 2) \cdot 0.04$) to account for upstream-to-downstream water transit times.

---

## 📊 5. Validation Results on Held-Out Test Set (31,211 Samples)

Trained on 124,843 samples and evaluated on 31,211 held-out spatiotemporal grid cells:

| Hazard Head | ROC-AUC | F1-Score | Precision | Recall | Brier Loss Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Severe Thunderstorm** | **1.0000** | **0.9991** | **0.9992** | **0.9990** | **0.0008** |
| **Cloudburst Deluge** | **0.9990** | **0.9809** | **0.9716** | **0.9903** | **0.0006** |
| **Flash Flood** | **0.9997** | **0.9994** | **0.9997** | **0.9990** | **0.0003** |

*Artifacts saved to: [`models/multitask_nowcast_model.joblib`](file:///c:/Users/zeeya/Desktop/SIH26077/models/multitask_nowcast_model.joblib) and [`models/model_metrics.json`](file:///c:/Users/zeeya/Desktop/SIH26077/models/model_metrics.json)*

---

## 🗺️ 6. Spatial Risk Grid Output (GeoTIFF & NumPy)

Inference produces GIS-ready spatial surfaces via [`src/model/predict_hazard_grid.py`](file:///c:/Users/zeeya/Desktop/SIH26077/src/model/predict_hazard_grid.py):

### Multi-Band GeoTIFF Specifications:
- **Projection:** Standard WGS84 (`EPSG:4326`)
- **Band 1:** Severe Thunderstorm Probability ($0.0\text{ to }1.0$)
- **Band 2:** Cloudburst Probability ($0.0\text{ to }1.0$)
- **Band 3:** Flash Flood Probability ($0.0\text{ to }1.0$)
- **Compatibility:** Directly viewable in QGIS, ArcGIS, Mapbox, or Leaflet/Folium.

```bash
# Generate 2-6 hour nowcast risk grids for Amarnath Cloudburst at T+3h:
python -m src.model.predict_hazard_grid --case case_01_amarnath_cloudburst_2022 --lead-hours 3

# Generate nowcast risk grids for Wayanad Deluge at T+4h:
python -m src.model.predict_hazard_grid --case case_04_wayanad_deluge_2024 --lead-hours 4
```

Outputs are stored in:
- GeoTIFF: `/data/processed/<case_id>/hazard_risk_grid_<case_id>_T+{lead_hours}h.tif`
- Compressed NumPy: `/data/processed/<case_id>/hazard_risk_grid_<case_id>_T+{lead_hours}h.npz`
- Alert Summary: `/data/processed/<case_id>/hazard_summary_<case_id>_T+{lead_hours}h.json`
