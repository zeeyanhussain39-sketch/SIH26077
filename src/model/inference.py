"""
Inference pipeline for severe weather nowcasting across 2 to 6 hours lead time.
Predicts risk scores for:
1. Severe Thunderstorms / Squall lines
2. Cloudbursts (Localized extreme precipitation)
3. Flash Floods (Catchment inundation risk)
"""

from typing import Dict, Any, Optional
import numpy as np

from src.feature_engineering.atmospheric_indices import (
    classify_cape_instability,
    calculate_cloudburst_risk,
    calculate_flash_flood_risk,
)


def run_nowcast_inference(
    lead_hours: int = 2,
    latitude: float = 30.3165,
    longitude: float = 78.0322,
    met_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes nowcasting inference for a specific lead time between 2 and 6 hours.
    Lead time factor modulates uncertainty and advection displacement.
    """
    if lead_hours < 2 or lead_hours > 6:
        raise ValueError("Lead hours must be within the 2 to 6 hours nowcasting window.")

    if met_data is None:
        met_data = {
            "cape_j_kg": 2650.0,
            "precipitation_mm": 24.0,
            "radar_dbz": 48.5,
            "wind_gusts_kmh": 62.0,
            "soil_saturation": 0.78,
            "terrain_slope_deg": 28.0,
        }

    cape = float(met_data.get("cape_j_kg", 2000.0))
    precip = float(met_data.get("precipitation_mm", 15.0))
    radar_dbz = float(met_data.get("radar_dbz", 40.0))
    soil_sat = float(met_data.get("soil_saturation", 0.7))
    slope = float(met_data.get("terrain_slope_deg", 25.0))

    # Lead-time decay factor: confidence decreases with increasing lead time (2h to 6h)
    lead_uncertainty_factor = 1.0 - (lead_hours - 2) * 0.08
    decay = np.exp(-(lead_hours - 2) * 0.12)

    # 1. Severe Thunderstorm Calculation
    cape_info = classify_cape_instability(cape)
    ts_base = 0.5 * cape_info["instability_score"] + 0.35 * min(radar_dbz / 60.0, 1.0) + 0.15 * min(precip / 50.0, 1.0)
    thunderstorm_prob = float(np.clip(ts_base * decay, 0.05, 0.98))

    # 2. Cloudburst Calculation
    cloudburst_info = calculate_cloudburst_risk(
        precip_rate_mm_hr=precip * (1.2 if lead_hours <= 3 else 0.9),
        radar_dbz=radar_dbz,
        cape_value=cape
    )
    cloudburst_prob = float(np.clip(cloudburst_info["probability"] * decay, 0.02, 0.95))

    # 3. Flash Flood Calculation
    flood_info = calculate_flash_flood_risk(
        cloudburst_prob=cloudburst_prob,
        terrain_slope_deg=slope,
        soil_saturation_index=soil_sat
    )
    flood_prob = float(np.clip(flood_info["probability"] * (1.0 + 0.05 * (lead_hours - 2)), 0.02, 0.95))

    def get_tier(prob: float) -> str:
        if prob >= 0.70:
            return "RED ALERT (High Danger)"
        elif prob >= 0.45:
            return "ORANGE ALERT (Severe Risk)"
        elif prob >= 0.25:
            return "YELLOW ALERT (Advisory)"
        return "GREEN (Normal)"

    return {
        "lead_hours": lead_hours,
        "lead_time_target": f"T+{lead_hours}h",
        "coordinates": {"latitude": latitude, "longitude": longitude},
        "model_confidence": round(float(lead_uncertainty_factor * 0.92), 2),
        "hazards": {
            "severe_thunderstorm": {
                "name": "Severe Thunderstorm / Squall",
                "probability": round(thunderstorm_prob, 3),
                "alert_level": get_tier(thunderstorm_prob),
            },
            "cloudburst": {
                "name": "Cloudburst (Extreme Local Deluge)",
                "probability": round(cloudburst_prob, 3),
                "alert_level": get_tier(cloudburst_prob),
            },
            "flash_flood": {
                "name": "Flash Flood & Debris Flow",
                "probability": round(flood_prob, 3),
                "alert_level": get_tier(flood_prob),
            }
        },
        "key_meteorological_drivers": {
            "cape_j_kg": cape,
            "radar_reflectivity_dbz": radar_dbz,
            "precip_mm_hr": precip,
            "soil_saturation": soil_sat
        }
    }
