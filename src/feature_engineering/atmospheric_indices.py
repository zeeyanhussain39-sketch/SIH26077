"""
Atmospheric & Severe Convective Weather Feature Calculations.
Domain logic aligned with IMD (India Meteorological Department) severe weather definitions.
"""

from typing import Dict, Any
import numpy as np


def classify_cape_instability(cape_value: float) -> Dict[str, Any]:
    """
    Classifies atmospheric instability based on Convective Available Potential Energy (J/kg).
    - < 1000 J/kg: Marginal Instability
    - 1000 - 2500 J/kg: Moderate Instability (Thunderstorms possible)
    - 2500 - 4000 J/kg: Very High Instability (Severe thunderstorms & squall lines)
    - > 4000 J/kg: Extreme Instability (Cloudburst / Supercell potential)
    """
    if cape_value < 1000:
        level = "Low"
        score = 0.2
    elif cape_value < 2500:
        level = "Moderate"
        score = 0.55
    elif cape_value < 4000:
        level = "High"
        score = 0.85
    else:
        level = "Extreme"
        score = 1.0

    return {
        "cape_value": cape_value,
        "instability_level": level,
        "instability_score": score
    }


def cloud_top_convective_score(brightness_temp_k: float) -> float:
    """
    Calculates convective depth from thermal infrared brightness temperature.
    Overshooting convective tops with BT < 210K indicate intense vertical updrafts.
    """
    if brightness_temp_k >= 260.0:
        return 0.1
    elif brightness_temp_k >= 230.0:
        return 0.4
    elif brightness_temp_k >= 210.0:
        return 0.75
    else:
        # Deep convective tower
        return 0.98


def calculate_cloudburst_risk(
    precip_rate_mm_hr: float,
    radar_dbz: float,
    cape_value: float
) -> Dict[str, Any]:
    """
    Assesses cloudburst potential (IMD definition: rainfall rate ~ 100mm/hr localized).
    Combines radar reflectivity (dBZ > 50), precipitation rate, and thermodynamic energy.
    """
    # Normalized weights
    r_score = min(precip_rate_mm_hr / 100.0, 1.0)
    dbz_score = min(max((radar_dbz - 30.0) / 35.0, 0.0), 1.0)
    cape_score = classify_cape_instability(cape_value)["instability_score"]

    composite_risk = 0.45 * r_score + 0.35 * dbz_score + 0.20 * cape_score
    probability = float(np.clip(composite_risk, 0.0, 1.0))

    return {
        "hazard": "Cloudburst",
        "probability": round(probability, 3),
        "risk_tier": "CRITICAL" if probability > 0.7 else ("WARNING" if probability > 0.4 else "ADVISORY" if probability > 0.2 else "LOW"),
        "metrics": {
            "rain_rate_mm_hr": precip_rate_mm_hr,
            "radar_dbz": radar_dbz,
            "cape_j_kg": cape_value
        }
    }


def calculate_flash_flood_risk(
    cloudburst_prob: float,
    terrain_slope_deg: float = 25.0,
    soil_saturation_index: float = 0.8
) -> Dict[str, Any]:
    """
    Evaluates flash flood susceptibility, particularly in hilly or riverine catchment basins.
    """
    slope_factor = min(terrain_slope_deg / 45.0, 1.0)
    flood_prob = float(np.clip(0.6 * cloudburst_prob + 0.25 * soil_saturation_index + 0.15 * slope_factor, 0.0, 1.0))

    return {
        "hazard": "Flash Flood",
        "probability": round(flood_prob, 3),
        "risk_tier": "CRITICAL" if flood_prob > 0.65 else ("WARNING" if flood_prob > 0.4 else "LOW"),
        "soil_saturation": soil_saturation_index,
        "slope_deg": terrain_slope_deg
    }
