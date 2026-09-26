"""
Alert Generation & Dissemination Engine (SIH Problem Statement 26077).
======================================================================
Generates categorized early warnings when risk grid cells exceed defined thresholds.
Conforms to the international Common Alerting Protocol (CAP-v1.2).

Delivery Channels (100% Free & Open-Source Stack):
1. Live In-Dashboard Notification Feed (Active in Streamlit and FastAPI)
2. Optional Free-Tier Email Alerting via Python's standard `smtplib`
   (e.g., Gmail / Outlook SMTP with an App Password), with graceful fallback
   to formatted simulated email logging when credentials are not configured.

DEMO NOTICE:
This email/feed module serves as a zero-cost operational prototype demonstrating
how automated triggers translate model predictions into machine-readable CAP alerts
and incident response dispatches, substituting for multi-channel national infrastructure
(such as NDMA SACHET or Telecom Cell Broadcast systems).
"""

import os
import json
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALERTS_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "alerts"
ALERTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Default Warning Thresholds & Standard NDMA Action Directives
# -----------------------------------------------------------------------------
ALERT_THRESHOLDS = {
    "extreme_red": 0.70,   # >= 70%: Imminent danger to life / infrastructure
    "severe_orange": 0.40, # 40-70%: High hazard probability; prepare response
    "moderate_yellow": 0.20 # 20-40%: Convective watch / advisory
}

NDMA_INSTRUCTIONS = {
    "cloudburst": (
        "Immediate evacuation of riverbeds, dry nullahs, talus slopes, and temporary camping zones. "
        "Suspend pilgrim transit and tourism on mountain causeways. Relocate livestock and personnel to pre-designated high-ground pucca shelters."
    ),
    "flash_flood": (
        "Immediately move away from natural drainage ravines, streams, and low-lying culverts. "
        "Do not attempt to walk, drive, or ford across flowing watercourses. Mountain ridges are safe from water pooling; seek stable elevation."
    ),
    "severe_thunderstorm": (
        "Seek immediate indoor shelter in sturdy pucca structures. Disconnect sensitive electronic equipment. "
        "Avoid tall isolated trees, tin sheds, wire fences, and open exposed ridgelines."
    )
}


class CategorizedAlert:
    """
    Structured categorized alert representing a threshold-crossing severe weather event.
    """

    def __init__(
        self,
        alert_id: str,
        hazard_type: str,
        severity: str,
        alert_tier: str,
        probability: float,
        latitude: float,
        longitude: float,
        lead_hours: int,
        location_name: str,
        time_window: Dict[str, str],
        trigger_reason: str,
        recommended_action: str,
        cap_payload: Dict[str, Any],
        created_at: Optional[str] = None
    ):
        self.alert_id = alert_id
        self.hazard_type = hazard_type
        self.severity = severity
        self.alert_tier = alert_tier
        self.probability = probability
        self.latitude = latitude
        self.longitude = longitude
        self.lead_hours = lead_hours
        self.location_name = location_name
        self.time_window = time_window
        self.trigger_reason = trigger_reason
        self.recommended_action = recommended_action
        self.cap_payload = cap_payload
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.acknowledged = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "hazard_type": self.hazard_type,
            "hazard_title": self.hazard_type.replace("_", " ").title(),
            "severity": self.severity,
            "alert_tier": self.alert_tier,
            "probability": round(self.probability, 4),
            "probability_pct": round(self.probability * 100.0, 1),
            "latitude": round(self.latitude, 4),
            "longitude": round(self.longitude, 4),
            "lead_hours": self.lead_hours,
            "location_name": self.location_name,
            "time_window": self.time_window,
            "trigger_reason": self.trigger_reason,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
            "acknowledged": self.acknowledged,
            "cap_payload": self.cap_payload
        }


# -----------------------------------------------------------------------------
# Common Alerting Protocol (CAP-v1.2) Generator
# -----------------------------------------------------------------------------
def generate_cap_alert(
    hazard_type: str,
    severity: str,
    lead_hours: int,
    latitude: float,
    longitude: float,
    probability: float,
    instructions: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs a CAP-v1.2 compatible alert message dictionary.
    Severity: 'Extreme', 'Severe', 'Moderate', 'Minor'
    """
    alert_id = f"SIH26077-{uuid.uuid4().hex[:8].upper()}"
    now_dt = datetime.now(timezone.utc)
    onset_dt = now_dt + timedelta(hours=lead_hours)
    expires_dt = onset_dt + timedelta(hours=2)

    h_key = hazard_type.lower().replace(" ", "_")
    if instructions is None:
        if "cloudburst" in h_key:
            instructions = NDMA_INSTRUCTIONS["cloudburst"]
        elif "flood" in h_key:
            instructions = NDMA_INSTRUCTIONS["flash_flood"]
        else:
            instructions = NDMA_INSTRUCTIONS["severe_thunderstorm"]

    h_title = hazard_type.replace("_", " ").title()

    return {
        "identifier": alert_id,
        "sender": "SIH26077-HyperLocal-Nowcaster@open-earth.in",
        "sent": now_dt.isoformat(),
        "status": "Actual",
        "msgType": "Alert",
        "scope": "Public",
        "info": {
            "category": "Met",
            "event": h_title,
            "urgency": "Immediate" if lead_hours <= 3 else "Expected",
            "severity": severity,
            "certainty": "Observed" if probability >= 0.80 else ("Likely" if probability >= 0.50 else "Possible"),
            "effective": now_dt.isoformat(),
            "onset": onset_dt.isoformat(),
            "expires": expires_dt.isoformat(),
            "headline": f"Severe Weather Nowcast (Lead Time T+{lead_hours}h): High Risk of {h_title}",
            "description": (
                f"Hyper-local atmospheric nowcasting indicates an estimated {probability*100:.1f}% probability "
                f"of {h_title} near coordinate ({latitude:.4f}°N, {longitude:.4f}°E) within the next {lead_hours} hours."
            ),
            "instruction": instructions,
            "area": {
                "areaDesc": f"Catchment buffer around ({latitude:.4f}°N, {longitude:.4f}°E)",
                "circle": f"{latitude:.4f},{longitude:.4f},15.0"  # 15km perimeter circle
            }
        }
    }


def generate_hazard_geojson(latitude: float, longitude: float, hazards: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a GeoJSON FeatureCollection representing pinpointed hazard zones.
    """
    features: List[Dict[str, Any]] = []

    for key, data in hazards.items():
        prob = data.get("probability", 0.0)
        level = data.get("alert_level", "NORMAL")
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
                "marker_color": "#ef4444" if "RED" in level else ("#f97316" if "ORANGE" in level else "#eab308")
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }


# -----------------------------------------------------------------------------
# Alert Engine: Threshold Evaluation & Alert Generation
# -----------------------------------------------------------------------------
class AlertEngine:
    """
    Evaluates grid cell hazard probabilities against multi-tier thresholds
    and triggers categorized alerts with actionable instructions.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or ALERT_THRESHOLDS

    def evaluate_cell_risk(
        self,
        hazard_type: str,
        probability: float,
        latitude: float,
        longitude: float,
        lead_hours: int = 3,
        location_name: Optional[str] = None,
        physical_drivers: Optional[Dict[str, Any]] = None
    ) -> Optional[CategorizedAlert]:
        """
        Evaluates whether a risk grid cell crosses defined warning thresholds.
        If crossed, constructs a CategorizedAlert.
        """
        prob = float(probability)
        if prob < self.thresholds["moderate_yellow"]:
            return None  # Below warning threshold

        if prob >= self.thresholds["extreme_red"]:
            severity = "Extreme"
            alert_tier = "RED"
        elif prob >= self.thresholds["severe_orange"]:
            severity = "Severe"
            alert_tier = "ORANGE"
        else:
            severity = "Moderate"
            alert_tier = "YELLOW"

        h_key = hazard_type.lower().replace(" ", "_")
        if "cloudburst" in h_key:
            rec_action = NDMA_INSTRUCTIONS["cloudburst"]
            h_clean = "cloudburst"
        elif "flood" in h_key:
            rec_action = NDMA_INSTRUCTIONS["flash_flood"]
            h_clean = "flash_flood"
        else:
            rec_action = NDMA_INSTRUCTIONS["severe_thunderstorm"]
            h_clean = "severe_thunderstorm"

        # Estimated time window
        now_dt = datetime.now(timezone.utc)
        start_impact = now_dt + timedelta(hours=max(1, lead_hours - 1))
        peak_impact = now_dt + timedelta(hours=lead_hours)
        time_to_impact_str = f"{lead_hours}h 00m"

        time_window = {
            "start_utc": start_impact.strftime("%H:%M UTC"),
            "peak_utc": peak_impact.strftime("%H:%M UTC"),
            "time_to_impact": time_to_impact_str,
            "window_summary": f"{start_impact.strftime('%H:%M')} - {peak_impact.strftime('%H:%M')} UTC (T+{lead_hours}h)"
        }

        # Location name default
        loc_str = location_name or f"Sector ({latitude:.3f}°N, {longitude:.3f}°E)"

        # Physical trigger reason
        if physical_drivers:
            top_reasons = []
            if "cape_j_kg" in physical_drivers:
                top_reasons.append(f"Extreme CAPE ({physical_drivers['cape_j_kg']:.0f} J/kg)")
            if "precip_mm_hr" in physical_drivers:
                top_reasons.append(f"High Rain Rate ({physical_drivers['precip_mm_hr']:.1f} mm/hr)")
            if "radar_reflectivity_dbz" in physical_drivers:
                top_reasons.append(f"Radar Reflectivity ({physical_drivers['radar_reflectivity_dbz']:.1f} dBZ)")
            trigger_reason = f"Threshold exceeded due to {', '.join(top_reasons)}."
        else:
            trigger_reason = (
                f"Nowcasting model predicts elevated {h_clean.replace('_', ' ')} probability ({prob*100:.1f}%) "
                f"exceeding the {severity.lower()} hazard threshold."
            )

        # Build full CAP-v1.2 payload
        cap_payload = generate_cap_alert(
            hazard_type=h_clean,
            severity=severity,
            lead_hours=lead_hours,
            latitude=latitude,
            longitude=longitude,
            probability=prob,
            instructions=rec_action
        )

        alert_id = cap_payload["identifier"]

        return CategorizedAlert(
            alert_id=alert_id,
            hazard_type=h_clean,
            severity=severity,
            alert_tier=alert_tier,
            probability=prob,
            latitude=latitude,
            longitude=longitude,
            lead_hours=lead_hours,
            location_name=loc_str,
            time_window=time_window,
            trigger_reason=trigger_reason,
            recommended_action=rec_action,
            cap_payload=cap_payload
        )


# -----------------------------------------------------------------------------
# Optional Free-Tier Email Alert Dispatcher (smtplib)
# -----------------------------------------------------------------------------
class SmtpAlertDispatcher:
    """
    Dispatches formatted email alerts using Python's standard `smtplib`.
    Supports free SMTP servers (e.g. Gmail SMTP `smtp.gmail.com:587` with App Password).
    When SMTP credentials are not configured, operates in a graceful Simulated
    Delivery Log mode so users and judges can preview the full email without entering credentials.
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        sender_email: Optional[str] = None
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.sender_email = sender_email or os.getenv("SMTP_SENDER", self.smtp_user or "nowcast-alerts@open-earth.in")

    def is_configured(self) -> bool:
        """Checks if real SMTP credentials are provided."""
        return bool(self.smtp_user and self.smtp_password)

    def build_email_html(self, alert_dict: Dict[str, Any]) -> str:
        """Constructs a responsive HTML email message with severe weather styling."""
        tier = alert_dict.get("alert_tier", "RED")
        banner_bg = "#dc2626" if tier == "RED" else ("#ea580c" if tier == "ORANGE" else "#ca8a04")
        h_title = alert_dict.get("hazard_title", "Severe Weather Hazard")
        prob_pct = alert_dict.get("probability_pct", 85.0)
        loc_name = alert_dict.get("location_name", "Target Sector")
        lat = alert_dict.get("latitude", 0.0)
        lon = alert_dict.get("longitude", 0.0)
        time_win = alert_dict.get("time_window", {})
        impact_window = time_win.get("window_summary", "Within 2-3 hours")
        action = alert_dict.get("recommended_action", "")
        reason = alert_dict.get("trigger_reason", "")
        alert_id = alert_dict.get("alert_id", "SIH26077-ALERT")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 20px; color: #1e293b; }}
                .container {{ max-width: 620px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
                .header {{ background-color: {banner_bg}; color: #ffffff; padding: 24px; text-align: left; }}
                .header h1 {{ margin: 0; font-size: 22px; font-weight: 800; letter-spacing: 0.02em; }}
                .header p {{ margin: 6px 0 0 0; font-size: 14px; opacity: 0.9; }}
                .badge {{ display: inline-block; background: rgba(255,255,255,0.25); color: #ffffff; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-bottom: 8px; }}
                .content {{ padding: 24px; }}
                .info-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                .info-table td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
                .info-table td.label {{ color: #64748b; font-weight: 600; width: 38%; }}
                .info-table td.val {{ color: #0f172a; font-weight: 700; }}
                .action-box {{ background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 14px 16px; border-radius: 4px; margin-bottom: 20px; font-size: 13px; line-height: 1.5; color: #991b1b; }}
                .footer {{ background-color: #f8fafc; padding: 16px 24px; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; line-height: 1.4; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <span class="badge">SIH 26077 • IMMEDIATE EMERGENCY NOWCAST</span>
                    <h1>⚠️ {alert_dict.get('severity', 'Severe').upper()} {h_title.upper()} ALERT</h1>
                    <p>Hyper-Local Early Warning Dissemination • Estimated Probability: <strong>{prob_pct}%</strong></p>
                </div>
                <div class="content">
                    <table class="info-table">
                        <tr>
                            <td class="label">Hazard Event:</td>
                            <td class="val" style="color: {banner_bg};">{h_title}</td>
                        </tr>
                        <tr>
                            <td class="label">Impact Location:</td>
                            <td class="val">{loc_name} ({lat:.4f}°N, {lon:.4f}°E)</td>
                        </tr>
                        <tr>
                            <td class="label">Estimated Impact Window:</td>
                            <td class="val">{impact_window}</td>
                        </tr>
                        <tr>
                            <td class="label">Forecast Lead Horizon:</td>
                            <td class="val">T + {alert_dict.get('lead_hours', 3)} Hours Ahead</td>
                        </tr>
                        <tr>
                            <td class="label">Physical Trigger Drivers:</td>
                            <td class="val" style="font-weight: 500; font-size: 13px;">{reason}</td>
                        </tr>
                        <tr>
                            <td class="label">CAP Alert ID:</td>
                            <td class="val" style="font-family: monospace; font-size: 12px;">{alert_id}</td>
                        </tr>
                    </table>

                    <div class="action-box">
                        <strong style="display:block; margin-bottom: 4px; font-size: 14px;">🚨 NDMA Recommended Directive:</strong>
                        {action}
                    </div>
                </div>
                <div class="footer">
                    <strong>Demo Alert Substitute:</strong> This email was generated using Python standard library <code>smtplib</code> as a zero-cost hackathon demonstration for SIH Problem Statement 26077. In full operational deployment, alerts are disseminated via national multi-channel gateways (NDMA SACHET, IMD DWR warnings, and CAP-v1.2 sirens).
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def dispatch_alert(
        self,
        alert_dict: Dict[str, Any],
        recipient_email: str
    ) -> Dict[str, Any]:
        """
        Dispatches an email alert to the specified recipient.
        If SMTP credentials are provided, attempts live delivery; otherwise,
        records a simulated delivery log for demonstration review.
        """
        subject = f"[{alert_dict.get('severity', 'Severe').upper()} ALERT] {alert_dict.get('hazard_title', 'Weather Alert')} - {alert_dict.get('location_name', 'Sector')}"
        html_body = self.build_email_html(alert_dict)

        result = {
            "alert_id": alert_dict.get("alert_id"),
            "recipient": recipient_email,
            "subject": subject,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "html_preview": html_body,
            "status": "PENDING"
        }

        # Check if live SMTP credentials are configured
        if self.is_configured():
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = self.sender_email
                msg["To"] = recipient_email
                msg.attach(MIMEText(html_body, "html", "utf-8"))

                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.sender_email, [recipient_email], msg.as_string())

                result["status"] = "SENT_LIVE_SMTP"
                result["message"] = f"Successfully delivered live email to {recipient_email} via {self.smtp_host}:{self.smtp_port}"
                print(f"[AlertEngine] Live email alert dispatched to: {recipient_email}")
            except Exception as e:
                result["status"] = "SMTP_FAILED_FALLBACK_SIMULATED"
                result["error"] = str(e)
                print(f"[AlertEngine] SMTP connection failed: {e}. Falling back to simulated log.")
        else:
            result["status"] = "SIMULATED_DEMO_DISPATCH"
            result["message"] = (
                f"Simulated email delivery to {recipient_email}. (To enable live transmission, "
                f"provide SMTP_USER and SMTP_PASSWORD or enter credentials in the dashboard)."
            )

        # Log dispatch payload
        log_file = ALERTS_DATA_DIR / "email_dispatch_log.json"
        try:
            logs = []
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            logs.insert(0, {k: v for k, v in result.items() if k != "html_preview"})
            logs = logs[:50]  # Keep last 50 logs
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception:
            pass

        return result


# -----------------------------------------------------------------------------
# Live In-Dashboard Notification Feed Manager
# -----------------------------------------------------------------------------
class AlertFeedManager:
    """
    Manages active categorized alerts for the live in-dashboard feed
    and REST API consumers.
    """

    def __init__(self, max_feed_size: int = 40):
        self.max_feed_size = max_feed_size
        self._alerts: List[Dict[str, Any]] = []
        self._load_feed_from_disk()

    def _load_feed_from_disk(self) -> None:
        feed_file = ALERTS_DATA_DIR / "live_alerts_feed.json"
        if feed_file.exists():
            try:
                with open(feed_file, "r", encoding="utf-8") as f:
                    self._alerts = json.load(f)
            except Exception:
                self._alerts = []

    def _persist_feed(self) -> None:
        feed_file = ALERTS_DATA_DIR / "live_alerts_feed.json"
        try:
            with open(feed_file, "w", encoding="utf-8") as f:
                json.dump(self._alerts[:self.max_feed_size], f, indent=2)
        except Exception:
            pass

    def add_alert(self, alert: Union[CategorizedAlert, Dict[str, Any]]) -> Dict[str, Any]:
        """Adds a new alert to the head of the live feed."""
        alert_dict = alert.to_dict() if isinstance(alert, CategorizedAlert) else alert
        # Check if already present
        self._alerts = [a for a in self._alerts if a.get("alert_id") != alert_dict.get("alert_id")]
        self._alerts.insert(0, alert_dict)
        self._alerts = self._alerts[:self.max_feed_size]
        self._persist_feed()
        return alert_dict

    def get_feed(
        self,
        min_tier: Optional[str] = None,
        hazard: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns the active list of alerts, optionally filtered by tier or hazard."""
        out = self._alerts
        if min_tier:
            tier_order = {"YELLOW": 1, "ORANGE": 2, "RED": 3}
            min_val = tier_order.get(min_tier.upper(), 1)
            out = [a for a in out if tier_order.get(a.get("alert_tier", "YELLOW"), 1) >= min_val]
        if hazard:
            out = [a for a in out if hazard.lower() in a.get("hazard_type", "").lower()]
        return out

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Marks an alert as acknowledged by an operator."""
        for a in self._alerts:
            if a.get("alert_id") == alert_id:
                a["acknowledged"] = True
                self._persist_feed()
                return True
        return False

    def clear(self) -> None:
        self._alerts = []
        self._persist_feed()


# -----------------------------------------------------------------------------
# Singleton Accessors
# -----------------------------------------------------------------------------
_ALERT_ENGINE_INSTANCE: Optional[AlertEngine] = None
_FEED_MANAGER_INSTANCE: Optional[AlertFeedManager] = None
_SMTP_DISPATCHER_INSTANCE: Optional[SmtpAlertDispatcher] = None

def get_default_alert_engine() -> AlertEngine:
    global _ALERT_ENGINE_INSTANCE
    if _ALERT_ENGINE_INSTANCE is None:
        _ALERT_ENGINE_INSTANCE = AlertEngine()
    return _ALERT_ENGINE_INSTANCE

def get_default_feed_manager() -> AlertFeedManager:
    global _FEED_MANAGER_INSTANCE
    if _FEED_MANAGER_INSTANCE is None:
        _FEED_MANAGER_INSTANCE = AlertFeedManager()
    return _FEED_MANAGER_INSTANCE

def get_default_smtp_dispatcher() -> SmtpAlertDispatcher:
    global _SMTP_DISPATCHER_INSTANCE
    if _SMTP_DISPATCHER_INSTANCE is None:
        _SMTP_DISPATCHER_INSTANCE = SmtpAlertDispatcher()
    return _SMTP_DISPATCHER_INSTANCE
