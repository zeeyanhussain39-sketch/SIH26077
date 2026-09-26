"""
Alert Generation Engine conforming to Common Alerting Protocol (CAP-v1.2).
Integrates with open-source dissemination pipelines (webhooks, SMS gateways, sirens).
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
import uuid


def generate_cap_alert(
    hazard_type: str,
    severity: str,
    lead_hours: int,
    latitude: float,
    longitude: float,
    probability: float,
    instructions: str = "Seek sturdy indoor shelter away from rivers, drainage channels, and low-lying valleys."
) -> Dict[str, Any]:
    """
    Constructs a CAP-compatible alert message dictionary.
    Severity: 'Extreme', 'Severe', 'Moderate', 'Minor'
    """
    alert_id = f"SIH26077-{uuid.uuid4().hex[:8].upper()}"
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "identifier": alert_id,
        "sender": "SIH26077-HyperLocal-Nowcaster@open-earth.in",
        "sent": now_iso,
        "status": "Actual",
        "msgType": "Alert",
        "scope": "Public",
        "info": {
            "category": "Met",
            "event": hazard_type.replace("_", " ").title(),
            "urgency": "Immediate",
            "severity": severity,
            "certainty": "Observed" if probability > 0.8 else "Likely",
            "headline": f"Severe Weather Nowcast (Lead Time T+{lead_hours}h): High Risk of {hazard_type.replace('_', ' ').title()}",
            "description": (
                f"Hyper-local atmospheric nowcasting indicates an estimated {probability*100:.1f}% probability "
                f"of {hazard_type} near coordinate ({latitude:.4f}, {longitude:.4f}) within the next {lead_hours} hours."
            ),
            "instruction": instructions,
            "area": {
                "areaDesc": f"Catchment buffer around ({latitude:.4f}, {longitude:.4f})",
                "circle": f"{latitude},{longitude},15.0"  # 15km radius circle
            }
        }
    }


def generate_hazard_geojson(latitude: float, longitude: float, hazards: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a GeoJSON FeatureCollection representing the pinpointed alert zones.
    """
    features: List[Dict[str, Any]] = []

    for key, data in hazards.items():
        prob = data.get("probability", 0.0)
        level = data.get("alert_level", "NORMAL")
        # Generate an alert point
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [longitude, latitude]
            },
            "properties": {
                "hazard_id": key,
                "hazard_name": data.get("name", key),
                "probability": prob,
                "alert_level": level,
                "marker_color": "#d90429" if "RED" in level else ("#f77f00" if "ORANGE" in level else "#fcbf49")
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }
