"""
Automated Early Warning & Alert Engine.
Constructs Common Alerting Protocol (CAP) compliant JSON payloads and GeoJSON hazard zones.
"""

from .alert_engine import (
    generate_cap_alert,
    generate_hazard_geojson,
    CategorizedAlert,
    AlertEngine,
    SmtpAlertDispatcher,
    AlertFeedManager,
    get_default_alert_engine,
    get_default_feed_manager,
    get_default_smtp_dispatcher,
    ALERT_THRESHOLDS,
    NDMA_INSTRUCTIONS,
)

__all__ = [
    "generate_cap_alert",
    "generate_hazard_geojson",
    "CategorizedAlert",
    "AlertEngine",
    "SmtpAlertDispatcher",
    "AlertFeedManager",
    "get_default_alert_engine",
    "get_default_feed_manager",
    "get_default_smtp_dispatcher",
    "ALERT_THRESHOLDS",
    "NDMA_INSTRUCTIONS",
]
