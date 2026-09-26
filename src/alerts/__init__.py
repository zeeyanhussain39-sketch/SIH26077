"""
Automated Early Warning & Alert Engine.
Constructs Common Alerting Protocol (CAP) compliant JSON payloads and GeoJSON hazard zones.
"""

from .alert_engine import generate_cap_alert, generate_hazard_geojson

__all__ = [
    "generate_cap_alert",
    "generate_hazard_geojson",
]
