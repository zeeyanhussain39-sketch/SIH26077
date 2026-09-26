"""
FastAPI Server for Severe Weather Nowcasting & Alert Dissemination.
Zero paid cloud dependencies. Run with: uvicorn api.main:app --reload --port 8000
"""

import uuid
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from src.model.inference import run_nowcast_inference
from src.alerts.alert_engine import (
    generate_cap_alert,
    generate_hazard_geojson,
    get_default_alert_engine,
    get_default_feed_manager,
    get_default_smtp_dispatcher,
    SmtpAlertDispatcher,
    CategorizedAlert
)
from src.data_ingestion.open_meteo_client import fetch_nowcast_atmospheric_features
from src.xai.explainability import compute_hazard_attributions, explain_prediction_summary

app = FastAPI(
    title="SIH 26077: Severe Weather Nowcasting & Alert API",
    description="Hyper-local 2-6 hour predictive engine for thunderstorms, cloudbursts, and flash floods with CAP-v1.2, Live Feed, and smtplib email alerts.",
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


# -----------------------------------------------------------------------------
# Categorized Alert Evaluation, Live Feed, & smtplib Email Dispatch Endpoints
# -----------------------------------------------------------------------------

class AlertEvaluationRequest(BaseModel):
    hazard_type: str = Field(default="cloudburst", description="Hazard type (cloudburst, flash_flood, severe_thunderstorm)")
    probability: float = Field(default=0.85, ge=0.0, le=1.0, description="Predicted hazard probability (0.0 to 1.0)")
    latitude: float = Field(default=34.215, description="Target Latitude")
    longitude: float = Field(default=75.503, description="Target Longitude")
    lead_hours: int = Field(default=3, ge=2, le=6, description="Nowcasting lead time (2 to 6 hours)")
    location_name: Optional[str] = Field(default="Amarnath Holy Cave Corridor", description="Location / catchment descriptor")
    recipient_email: Optional[str] = Field(default=None, description="Optional recipient email for automated notification")
    send_email: bool = Field(default=False, description="Whether to trigger email dispatch upon threshold crossing")


class EmailDispatchRequest(BaseModel):
    recipient_email: str = Field(..., description="Target email recipient (e.g. emergency-operations@state.gov.in)")
    alert_id: Optional[str] = Field(default=None, description="Optional existing CAP alert ID")
    hazard_type: str = Field(default="cloudburst", description="Hazard type")
    severity: str = Field(default="Extreme", description="Severity (Extreme, Severe, Moderate)")
    location_name: str = Field(default="Target Catchment Basin", description="Location name")
    probability: float = Field(default=0.88, description="Hazard probability")
    lead_hours: int = Field(default=3, description="Lead hours (2 to 6)")
    smtp_host: Optional[str] = Field(default=None, description="Optional custom SMTP server host")
    smtp_port: Optional[int] = Field(default=None, description="Optional custom SMTP port (default 587)")
    smtp_user: Optional[str] = Field(default=None, description="Optional SMTP username / email")
    smtp_password: Optional[str] = Field(default=None, description="Optional SMTP app password")


class AlertAcknowledgeRequest(BaseModel):
    alert_id: str = Field(..., description="Identifier of alert to acknowledge")


@app.get("/api/v1/alerts/feed")
def get_alerts_feed(
    min_tier: Optional[str] = Query(None, description="Filter minimum tier (YELLOW, ORANGE, RED)"),
    hazard: Optional[str] = Query(None, description="Filter by hazard substring (cloudburst, flood, thunderstorm)")
):
    """
    Returns the live stream of categorized active alerts for the in-dashboard feed and external subscribers.
    """
    feed_manager = get_default_feed_manager()
    feed_items = feed_manager.get_feed(min_tier=min_tier, hazard=hazard)
    return {
        "status": "online",
        "total_active_alerts": len(feed_items),
        "feed": feed_items
    }


@app.post("/api/v1/alerts/evaluate")
def evaluate_cell_alert(payload: AlertEvaluationRequest):
    """
    Evaluates a risk grid cell against warning thresholds.
    If threshold is crossed, generates a categorized alert, logs to live feed,
    and optionally dispatches an email alert via smtplib.
    """
    alert_engine = get_default_alert_engine()
    feed_manager = get_default_feed_manager()

    alert = alert_engine.evaluate_cell_risk(
        hazard_type=payload.hazard_type,
        probability=payload.probability,
        latitude=payload.latitude,
        longitude=payload.longitude,
        lead_hours=payload.lead_hours,
        location_name=payload.location_name
    )

    if alert is None:
        return {
            "status": "BELOW_THRESHOLD",
            "message": f"Hazard probability {payload.probability*100:.1f}% is below warning threshold (20%). No alert triggered.",
            "alert": None
        }

    # Add to in-memory feed
    alert_dict = feed_manager.add_alert(alert)

    # Optional email dispatch
    email_result = None
    if payload.send_email and payload.recipient_email:
        dispatcher = get_default_smtp_dispatcher()
        email_result = dispatcher.dispatch_alert(alert_dict, recipient_email=payload.recipient_email)

    return {
        "status": "ALERT_TRIGGERED",
        "alert": alert_dict,
        "email_dispatch": email_result
    }


@app.post("/api/v1/alerts/dispatch-email")
def dispatch_email_alert(payload: EmailDispatchRequest):
    """
    Dispatches an email alert using Python's standard `smtplib`.
    Supports free SMTP providers (e.g. Gmail / Outlook with App Password).
    When credentials are not provided, operates in graceful Simulated Demo Dispatch mode.
    """
    dispatcher = SmtpAlertDispatcher(
        smtp_host=payload.smtp_host,
        smtp_port=payload.smtp_port,
        smtp_user=payload.smtp_user,
        smtp_password=payload.smtp_password
    )

    # Build or fetch alert dict
    feed_manager = get_default_feed_manager()
    alert_dict = None
    if payload.alert_id:
        for a in feed_manager.get_feed():
            if a.get("alert_id") == payload.alert_id:
                alert_dict = a
                break

    if alert_dict is None:
        # Synthesize from request
        engine = get_default_alert_engine()
        temp_alert = engine.evaluate_cell_risk(
            hazard_type=payload.hazard_type,
            probability=payload.probability,
            latitude=34.215,
            longitude=75.503,
            lead_hours=payload.lead_hours,
            location_name=payload.location_name
        )
        alert_dict = temp_alert.to_dict() if temp_alert else {
            "alert_id": payload.alert_id or f"SIH26077-{uuid.uuid4().hex[:8].upper()}",
            "hazard_title": payload.hazard_type.replace("_", " ").title(),
            "severity": payload.severity,
            "alert_tier": "RED" if payload.severity == "Extreme" else "ORANGE",
            "probability_pct": round(payload.probability * 100.0, 1),
            "location_name": payload.location_name,
            "latitude": 34.215,
            "longitude": 75.503,
            "lead_hours": payload.lead_hours,
            "time_window": {"window_summary": f"Next {payload.lead_hours} Hours"},
            "trigger_reason": "Manual dispatch request from emergency control console.",
            "recommended_action": "Evacuate low-lying riverbeds and drainage paths."
        }

    dispatch_result = dispatcher.dispatch_alert(alert_dict, recipient_email=payload.recipient_email)
    return {
        "status": dispatch_result.get("status"),
        "message": dispatch_result.get("message", "Dispatched successfully"),
        "recipient": payload.recipient_email,
        "is_live_smtp": dispatcher.is_configured(),
        "dispatched_at": dispatch_result.get("timestamp"),
        "html_preview_length": len(dispatch_result.get("html_preview", ""))
    }


@app.post("/api/v1/alerts/acknowledge")
def acknowledge_alert_notification(payload: AlertAcknowledgeRequest):
    """
    Marks an alert in the feed as acknowledged by an operator.
    """
    feed_manager = get_default_feed_manager()
    success = feed_manager.acknowledge_alert(payload.alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert ID {payload.alert_id} not found in active feed.")
    return {"status": "ACKNOWLEDGED", "alert_id": payload.alert_id}

