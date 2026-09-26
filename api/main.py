"""
FastAPI Server for Severe Weather Nowcasting & Alert Dissemination.
Zero paid cloud dependencies. Run with: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from src.model.inference import run_nowcast_inference
from src.alerts.alert_engine import generate_cap_alert, generate_hazard_geojson
from src.data_ingestion.open_meteo_client import fetch_nowcast_atmospheric_features
from src.xai.explainability import compute_hazard_attributions, explain_prediction_summary

app = FastAPI(
    title="SIH 26077: Severe Weather Nowcasting & Alert API",
    description="Hyper-local 2-6 hour predictive engine for thunderstorms, cloudbursts, and flash floods using open-source scientific tools.",
    version="1.0.0"
)


class SimulationRequest(BaseModel):
    latitude: float = Field(default=30.3165, description="Target Latitude (e.g. Dehradun)")
    longitude: float = Field(default=78.0322, description="Target Longitude")
    lead_hours: int = Field(default=3, ge=2, le=6, description="Nowcasting lead time (2 to 6 hours)")
    cape_override: Optional[float] = Field(default=None, description="Optional CAPE override (J/kg)")
    precip_override: Optional[float] = Field(default=None, description="Optional precipitation rate override (mm/hr)")
    radar_dbz_override: Optional[float] = Field(default=None, description="Optional radar reflectivity override (dBZ)")


@app.get("/")
def root():
    return {
        "project": "SIH 26077 - AI-Driven Hyper-Local Severe Weather Nowcasting System",
        "status": "online",
        "documentation": "/docs",
        "lead_time_window_hours": "2 to 6 hours",
        "hazards": ["severe_thunderstorm", "cloudburst", "flash_flood"]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "nowcast-backend"}


@app.get("/api/v1/nowcast")
def get_nowcast(
    latitude: float = Query(30.3165, description="Latitude"),
    longitude: float = Query(78.0322, description="Longitude"),
    lead_hours: int = Query(2, ge=2, le=6, description="Lead hours (2, 3, 4, 5, or 6)")
):
    """
    Runs automated nowcast inference fetching real-time atmospheric features
    and returning 2-6h hazard probabilities.
    """
    # Ingest open atmospheric data
    met_data = fetch_nowcast_atmospheric_features(latitude=latitude, longitude=longitude)

    # Execute spatio-temporal inference
    inference_result = run_nowcast_inference(
        lead_hours=lead_hours,
        latitude=latitude,
        longitude=longitude,
        met_data=met_data
    )

    # Compute XAI explanation for top hazard
    cb_prob = inference_result["hazards"]["cloudburst"]["probability"]
    attributions = compute_hazard_attributions("cloudburst", inference_result["key_meteorological_drivers"])
    explanation = explain_prediction_summary("Cloudburst", cb_prob, attributions)

    return {
        "nowcast": inference_result,
        "xai_attributions": attributions,
        "plain_explanation": explanation
    }


@app.get("/api/v1/alerts")
def get_active_alerts(
    latitude: float = Query(30.3165, description="Latitude"),
    longitude: float = Query(78.0322, description="Longitude"),
    lead_hours: int = Query(2, ge=2, le=6, description="Lead hours ahead")
):
    """
    Returns Common Alerting Protocol (CAP) and GeoJSON representations for emergency responders.
    """
    nowcast = run_nowcast_inference(lead_hours=lead_hours, latitude=latitude, longitude=longitude)
    hazards = nowcast["hazards"]

    # Filter critical or warning alerts
    cap_alerts = []
    for key, data in hazards.items():
        if "RED" in data["alert_level"] or "ORANGE" in data["alert_level"]:
            severity = "Extreme" if "RED" in data["alert_level"] else "Severe"
            cap_alerts.append(
                generate_cap_alert(
                    hazard_type=key,
                    severity=severity,
                    lead_hours=lead_hours,
                    latitude=latitude,
                    longitude=longitude,
                    probability=data["probability"]
                )
            )

    geojson_layer = generate_hazard_geojson(latitude, longitude, hazards)

    return {
        "lead_hours": lead_hours,
        "cap_alerts": cap_alerts,
        "geojson_layer": geojson_layer
    }


@app.post("/api/v1/simulate")
def simulate_hazard_scenario(payload: SimulationRequest):
    """
    Simulates high-impact convective scenarios with user-defined overrides.
    """
    met_data = {
        "cape_j_kg": payload.cape_override if payload.cape_override is not None else 3200.0,
        "precipitation_mm": payload.precip_override if payload.precip_override is not None else 75.0,
        "radar_dbz": payload.radar_dbz_override if payload.radar_dbz_override is not None else 58.0,
        "soil_saturation": 0.85,
        "terrain_slope_deg": 30.0
    }

    result = run_nowcast_inference(
        lead_hours=payload.lead_hours,
        latitude=payload.latitude,
        longitude=payload.longitude,
        met_data=met_data
    )
    return result
