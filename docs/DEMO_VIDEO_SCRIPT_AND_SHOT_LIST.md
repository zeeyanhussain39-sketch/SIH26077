# 🎬 4.5-Minute Video Demo Script & Shot List
### AI-Driven Hyper-Local Severe Weather Nowcasting (SIH Problem Statement 26077)
**Target Duration:** 4 minutes 30 seconds (270 seconds)  
**Tone:** Authoritative, scientific, energetic, presentation-ready  
**Format:** Dual-column (Visual Shot List + Verbatim Spoken Dialogue with Timestamps)

---

## ⏱️ Master Timing Schedule

| Segment | Timestamp | Duration | Focus Area |
| :--- | :---: | :---: | :--- |
| **Part 1: The Opening Hook** | `0:00 – 0:45` | 45s | India's severe weather blindspot & NWP latency gap |
| **Part 2: Live Dashboard Walkthrough** | `0:45 – 2:20` | 95s | Amarnath Cloudburst historical replay ($T-4\text{h} \rightarrow T-2\text{h} \rightarrow T=0$) |
| **Part 3: Explainable AI (SHAP)** | `2:20 – 3:25` | 65s | Hotspot inspection & physical feature attribution |
| **Part 4: Automated Alert Dissemination** | `3:25 – 4:05` | 40s | CAP-v1.2 alerts, live dashboard feed & `smtplib` email |
| **Part 5: Production Roadmap & Close** | `4:05 – 4:30` | 25s | Free-stack compliance & real-time scaling path |

---

## 📽️ Detailed Script & Shot List

### PART 1: THE OPENING HOOK (0:00 – 0:45)
**Theme:** The Mesoscale Warning Gap & Latency Bottleneck

| Timestamp | Visual Shot List (What appears on screen) | Speaker Narration (Verbatim live dialogue) |
| :--- | :--- | :--- |
| **0:00 – 0:15** | **Shot 1.1 — Title Graphic & Disaster Collage:**<br>• Title screen: *AI-Driven Hyper-Local Severe Weather Nowcasting (SIH 26077)*.<br>• Split montage: IMD headlines from Amarnath 2022 (16 fatalities), North India 2018 squalls (110+ deaths), and 2023 Himachal flash floods. | *"Across India, mesoscale convective weather events—violent cloudbursts, severe squall lines, and flash floods—strike with catastrophic speed. In mountainous terrain like the Himalayas and Western Ghats, entire communities and pilgrimage routes can be devastated in under an hour."* |
| **0:15 – 0:30** | **Shot 1.2 — The NWP Latency Diagram:**<br>• Visual showing 6–12 hour compute cycles of standard NWP models (WRF / GFS / NCUM) vs. a fast 60–90 minute convective storm cycle.<br>• Highlight Doppler radar blindspots behind mountain ridges. | *"Yet our operational forecasting systems face a critical blindspot: traditional Numerical Weather Prediction models update on 6 to 12 hour cycles. Severe convective storms form, burst, and dissipate entirely in between those cycles. Meanwhile, mountain radar beams suffer severe blockage."* |
| **0:30 – 0:45** | **Shot 1.3 — Solution Introduction:**<br>• Camera switches to speaker or high-res system banner.<br>• Text overlay: *2 to 6 Hours Actionable Lead Time • 100% Free & Open-Source Stack*. | *"To bridge this golden warning window, we built an AI-driven, hyper-local nowcasting platform that fuses geostationary INSAT-3D satellite radiances, atmospheric reanalysis, and 30-meter DEM topography to deliver 2 to 6 hours of verified advance warning."* |

---

### PART 2: LIVE DASHBOARD WALKTHROUGH — AMARNATH HISTORICAL REPLAY (0:45 – 2:20)
**Theme:** Multi-Stage Physical Buildup Leading to Documented Disaster

| Timestamp | Visual Shot List (What appears on screen) | Speaker Narration (Verbatim live dialogue) |
| :--- | :--- | :--- |
| **0:45 – 1:05** | **Shot 2.1 — Dashboard Overview & Historical Transparency:**<br>• Switch to full-screen Streamlit dashboard (`localhost:8501`).<br>• Cursor navigates to sidebar: Selects **"🎬 Scripted Replay Walkthrough"**.<br>• Scenario selected: `Amarnath Cave Cloudburst (July 8, 2022)`.<br>• Highlight the **Scientific Transparency Banner**. | *"Let's see the system in action. We are demonstrating our model using a verified historical replay of the tragic Amarnath Cave cloudburst of July 8, 2022. We state clearly: this is an empirical historical reconstruction using real INSAT-3D and IMD records, not an unpredictable live demo. Validating against documented disaster ground truth is essential for safety-critical systems."* |
| **1:05 – 1:25** | **Shot 2.2 — Phase 1: Baseline Watch ($T-4\text{h}$ / 10:00 UTC):**<br>• Dashboard shows Stage 1 ($T-4\text{h}$).<br>• Map displays low baseline risk; stats card shows CAPE: 1450 J/kg, CIN: -38 J/kg.<br>• Teleprompter card active. | *"At T minus 4 hours, INSAT-3D detects moisture advecting into the Pir Panjal mountains. MetPy thermodynamic calculations indicate moderate CAPE of 1,450 Joules per kilogram, but convective inhibition of -38 Joules still caps the atmosphere. Our multi-task model correctly maintains a calm Yellow Watch with only 18% cloudburst probability, preventing false alarms."* |
| **1:25 – 1:45** | **Shot 2.3 — Phase 2: Updraft Explosion ($T-3\text{h}$ / 11:00 UTC):**<br>• Cursor clicks `⏭️ Next Stage` button to advance to Stage 2 ($T-3\text{h}$).<br>• Risk card turns **ORANGE** (58% Cloudburst).<br>• Highlight metric: CTT Cooling Rate: **-16.5 K/hr**, CAPE: 2650 J/kg. | *"Now, watch T minus 3 hours. INSAT thermal infrared detects cloud-top temperatures plunging at 16.5 Kelvin per hour—a classic signature of an explosive convective updraft piercing the tropopause. The capping inversion shatters. Our model immediately escalates to an Orange Warning at 58% cloudburst risk. In an operational setting, this provides a vital 3-hour advance window to begin clearing low-lying nullahs."* |
| **1:45 – 2:20** | **Shot 2.4 — Phase 3: DEM Routing & Critical Red Alert ($T-2\text{h}$ / 12:00 UTC):**<br>• Advance to Stage 3 ($T-2\text{h}$). Card flashes **RED ALERT** (88% Cloudburst, 94% Flash Flood).<br>• Cursor toggles **"🌊 Flash Flood (D8 DEM Routed Risk)"** on the Folium map.<br>• Zoom into the narrow valley corridor: show how the flood layer funnels along the drainage nullah while ridges stay blue/green. | *"At T minus 2 hours, the convective core hits peak severity with radar exceeding 54 dBZ. Here is our system's decisive breakthrough: while the atmospheric cloudburst footprint covers 15 kilometers, our D8 hydrologic routing engine uses the 30-meter DEM to calculate that the 34-degree glaciated slopes funnel 94% of flash flood surge directly into the narrow Amarnath campsite ravine. The system issues an automated Critical Red Alert: Evacuate nullahs immediately."* |

---

### PART 3: EXPLAINABLE AI (XAI) WITH SHAP (2:20 – 3:25)
**Theme:** Opening the Black Box for Disaster Decision-Makers

| Timestamp | Visual Shot List (What appears on screen) | Speaker Narration (Verbatim live dialogue) |
| :--- | :--- | :--- |
| **2:20 – 2:40** | **Shot 3.1 — Hotspot Selection & SHAP Panel:**<br>• Cursor clicks on the peak high-risk grid cell on the interactive map.<br>• Dashboard scrolls to the **"🔍 Explainable AI (SHAP) Diagnostics"** panel.<br>• Dynamic horizontal bar chart animates on screen with dark theme styling. | *"Disaster authorities like the NDMA cannot act on an opaque probability number alone. They need to know why the model flagged this coordinate. When an operator clicks any flagged zone, our SHAP TreeExplainer computes the exact Shapley feature attributions for all 13 physical parameters."* |
| **2:40 – 3:05** | **Shot 3.2 — Feature Breakdown & Physical Attribution:**<br>• Cursor highlights top bars:<br>  1. `ctt_cooling_rate`: +38%<br>  2. `cape_j_kg`: +28%<br>  3. `d8_flow_accumulation`: +22%<br>  4. `low_level_convergence`: +12%. | *"Here, the system reveals that 38% of the risk spike is driven by rapid cloud-top cooling, 28% by extreme thermodynamic CAPE buoyancy, and 22% by steep terrain flow accumulation. This directly confirms to meteorologists that we have a severe convective updraft with extreme gravitational runoff potential, not mere stratiform drizzle."* |
| **3:05 – 3:25** | **Shot 3.3 — Plain-Language Emergency Brief:**<br>• Highlight the auto-generated textual explanation box below the chart: *"Severe Convective Alert: Rapid vertical cloud-top cooling (-22 K/hr) combined with high buoyant energy (3,450 J/kg) indicates severe cloudburst risk..."* | *"The module automatically translates these mathematical Shapley vectors into a plain-language operational brief, allowing district collectors to brief field rescue teams in seconds with full confidence in the system's reasoning."* |

---

### PART 4: AUTOMATED ALERT DISSEMINATION (3:25 – 4:05)
**Theme:** Standards-Compliant, Zero-Cost Multi-Channel Alerting

| Timestamp | Visual Shot List (What appears on screen) | Speaker Narration (Verbatim live dialogue) |
| :--- | :--- | :--- |
| **3:25 – 3:45** | **Shot 4.1 — Live In-Dashboard Alert Feed:**<br>• Switch to Tab 1: **"🚨 Categorized Alerts & Dissemination"**.<br>• Show color-coded Red/Orange warning cards with countdown timer: *Estimated Impact: 2h 15m*.<br>• Expand the **CAP-v1.2 JSON Payload** viewer. | *"When thresholds cross critical limits, the automated alert engine kicks in. Built upon international Common Alerting Protocol standards (CAP-v1.2), alerts include pinpoint coordinates, severity tiers, time-to-impact estimates, and official NDMA safety directives in standardized JSON format."* |
| **3:45 – 4:05** | **Shot 4.2 — Free-Tier Email Dispatch (`smtplib`):**<br>• Cursor shows the Email Alert Dispatcher card.<br>• Click **"📧 Dispatch Emergency Alert Email"**.<br>• Show success badge and pop up the responsive HTML email preview showing formatted table and evacuation instructions. | *"Adhering strictly to hackathon zero-cost constraints, we implemented alert delivery via an in-dashboard reactive feed and Python's standard library `smtplib` with TLS encryption—compatible with free Gmail or our simulated demo dispatcher. This serves as a working prototype for national integration with NDMA's SACHET cell-broadcast sirens."* |

---

### PART 5: PRODUCTION ROADMAP & CLOSING STATEMENT (4:05 – 4:30)
**Theme:** Compliance, Scientific Integrity, and National Scale

| Timestamp | Visual Shot List (What appears on screen) | Speaker Narration (Verbatim live dialogue) |
| :--- | :--- | :--- |
| **4:05 – 4:20** | **Shot 5.1 — Architecture & Dual-Track Model:**<br>• Display the system architecture diagram or slide showing the dual track: lightweight gradient-boosted engine (<50ms CPU) and PyTorch `SpatioTemporalNowcastNet`.<br>• Text badges: *100% Free • Zero Paid APIs • Open Data*. | *"Our architecture is 100% free and open-source: zero paid cloud bills, zero paid map APIs. For hackathon execution, our multi-task model delivers sub-50-millisecond CPU inference, while our repository includes a complete PyTorch ConvLSTM spatiotemporal sequence-to-sequence backbone ready for multi-GPU scaling."* |
| **4:20 – 4:30** | **Shot 5.2 — Final Team Close & Call to Action:**<br>• Team names, GitHub repository link, and SIH Problem Statement 26077 sign-off screen.<br>• Fade to black. | *"By combining satellite thermodynamics, digital elevation routing, and explainable AI, our system transforms the critical 2-to-6 hour window into actionable, life-saving lead time. Thank you, judges!"* |

---

## 🎙️ Rehearsal Tips for the Team

1. **Pacing:** Speak at a steady ~130–140 words per minute. Do not rush through the numbers; let the key figures (*"plunging at 16.5 Kelvin per hour"*, *"94% flash flood surge"*) land with impact.
2. **Cursor Synchronization:** Make sure the mouse cursor smoothly highlights the exact metric or button the speaker is describing. Avoid frantic clicking or erratic cursor movement.
3. **Historical Transparency Emphasis:** Deliver the line at `0:45 – 1:05` with pride. Evaluators respect teams that openly state why historical validation is an indispensable scientific requirement over fake "live" feeds.
4. **Resolution Settings:** Record at 1080p (1920×1080) at 30 or 60 fps. Zoom the browser window to 90% or 100% so the map, cards, and teleprompter fit comfortably within the frame.
