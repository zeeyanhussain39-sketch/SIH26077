"""
Free, Open-Source Weather Data Ingestion using Open-Meteo API.
Requires NO API key and provides high-resolution NWP atmospheric variables:
- Convective Available Potential Energy (CAPE) [J/kg]
- Precipitation & Rain rate [mm/h]
- Relative Humidity [%]
- Wind Gusts [km/h]
- Surface Pressure [hPa]
"""

from typing import Dict, Any
import requests


def fetch_nowcast_atmospheric_features(latitude: float = 30.3165, longitude: float = 78.0322) -> Dict[str, Any]:
    """
    Fetches real-time hourly meteorological parameters for severe weather assessment.
    Default coords: Dehradun / Uttarakhand (high-risk Himalayan cloudburst zone).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,weather_code,"
        f"surface_pressure,wind_gusts_10m,cape&"
        f"forecast_days=1&timezone=auto"
    )

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        hourly = data.get("hourly", {})

        # Extract current / earliest timestep
        idx = 0
        return {
            "source": "Open-Meteo (Free Open API)",
            "latitude": latitude,
            "longitude": longitude,
            "temperature_2m_c": hourly.get("temperature_2m", [25.0])[idx],
            "relative_humidity_pct": hourly.get("relative_humidity_2m", [80.0])[idx],
            "precipitation_mm": hourly.get("precipitation", [0.0])[idx],
            "cape_j_kg": hourly.get("cape", [1850.0])[idx] if hourly.get("cape") else 1850.0,
            "wind_gusts_kmh": hourly.get("wind_gusts_10m", [35.0])[idx],
            "surface_pressure_hpa": hourly.get("surface_pressure", [1005.0])[idx],
            "status": "success"
        }
    except Exception as exc:
        # Robust fallback when offline
        return {
            "source": "Offline Fallback Simulation",
            "latitude": latitude,
            "longitude": longitude,
            "temperature_2m_c": 28.5,
            "relative_humidity_pct": 88.0,
            "precipitation_mm": 18.4,
            "cape_j_kg": 2400.0,  # High instability
            "wind_gusts_kmh": 65.0,
            "surface_pressure_hpa": 998.2,
            "status": f"fallback_simulated ({str(exc)})"
        }
