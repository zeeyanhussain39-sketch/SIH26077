"""
Master Historical Case Studies Manager & Data Orchestrator.
==========================================================
Coordinates raw data acquisition for the 4 benchmark Indian severe weather events:
1. case_01_amarnath_cloudburst_2022 (Cloudburst & Himalayan Flash Flood)
2. case_02_north_india_squall_2018 (Severe Squall Line & Convective Storm Outbreak)
3. case_03_himachal_flash_flood_2023 (Monsoon Deluge & Beas River Flash Flood)
4. case_04_wayanad_deluge_2024 (Western Ghats Orographic Torrential Burst & Debris Flow)

Orchestrates:
- INSAT-3D / INSAT-3DR Multispectral Satellite Imagery (MOSDAC)
- Atmospheric Reanalysis (ERA5 Fallback / IMDAA Specifications)
- SRTM 30m Digital Elevation Model (Zero Registration)

Generates:
- /data/raw/<case_id>/metadata.json
- /data/raw/<case_id>/insat3d/
- /data/raw/<case_id>/reanalysis_era5/
- /data/raw/<case_id>/dem_srtm/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

from src.data_ingestion.mosdac_downloader import download_mosdac_data
from src.data_ingestion.era5_imdaa_downloader import download_era5_data
from src.data_ingestion.srtm_dem_downloader import download_srtm_dem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

CASE_STUDIES_REGISTRY: Dict[str, Dict[str, Any]] = {
    "case_01_amarnath_cloudburst_2022": {
        "case_id": "case_01_amarnath_cloudburst_2022",
        "title": "Amarnath Cave Cloudburst & Micro-Catchment Flash Flood",
        "hazard_type": "Cloudburst & Flash Flood / Mudflow",
        "date": "2022-07-08",
        "peak_time_utc": "12:00 UTC",
        "peak_time_ist": "17:30 IST",
        "time_window_utc": ["10:00", "15:00"],
        "location": {
            "name": "Baltal / Amarnath Holy Cave, Ganderbal / Anantnag District, Jammu & Kashmir",
            "centroid_latitude": 34.215,
            "centroid_longitude": 75.503,
            "elevation_msl_m": 3888,
            "bounding_box": [33.8, 75.0, 34.6, 76.0]
        },
        "meteorological_impact": {
            "peak_rainfall": ">31 mm in ~1 hour over a high-altitude rocky basin",
            "observed_damage": "Devastating flash flood and debris flow swept pilgrim campsite near cave lower reaches; 16 fatalities, extensive camp damage.",
            "synoptic_trigger": "Localized severe convective updraft aided by orographic funneling and high middle-tropospheric moisture transport."
        },
        "official_reports": [
            "IMD Meteorological Centre Srinagar: 'Special Weather Report on Cloudburst Incident near Holy Amarnath Cave on 08 July 2022'",
            "NCMRWF MoES NCUM Convective Simulation Report (July 2022)"
        ]
    },
    "case_02_north_india_squall_2018": {
        "case_id": "case_02_north_india_squall_2018",
        "title": "North India Severe Convective Squall Line & Thunderstorm Outbreak",
        "hazard_type": "Severe Thunderstorm / Squall / Derecho",
        "date": "2018-05-02",
        "peak_time_utc": "14:00 UTC",
        "peak_time_ist": "19:30 IST",
        "time_window_utc": ["11:00", "19:00"],
        "location": {
            "name": "Western Uttar Pradesh (Agra) & Eastern Rajasthan (Bharatpur, Alwar, Dholpur)",
            "centroid_latitude": 27.180,
            "centroid_longitude": 78.010,
            "elevation_msl_m": 170,
            "bounding_box": [26.5, 76.0, 28.5, 79.0]
        },
        "meteorological_impact": {
            "peak_wind_gusts": ">126-130 km/h squall downdrafts with severe lightning and dust storm",
            "observed_damage": ">110 casualties, uprooted electrical pylons, flattened residential structures across Agra and Bharatpur.",
            "synoptic_trigger": "Upper-level cyclonic vorticity combined with intense pre-monsoon heat, high CAPE (>3500 J/kg), and a strong dry line boundary."
        },
        "official_reports": [
            "IMD New Delhi: 'Report on Severe Thunderstorms/Squall over Northwest India on 02 May 2018'",
            "IMD Meteorological Monograph: 'Thunderstorms over India - Climatology, Dynamics and Nowcasting'"
        ]
    },
    "case_03_himachal_flash_flood_2023": {
        "case_id": "case_03_himachal_flash_flood_2023",
        "title": "Himachal Pradesh Monsoon Deluge & Beas River Flash Floods",
        "hazard_type": "Extreme Orographic Rainfall & Basin Flash Flood",
        "date": "2023-07-09",
        "peak_time_utc": "06:00 UTC",
        "peak_time_ist": "11:30 IST",
        "time_window_utc": ["00:00", "23:59"],
        "location": {
            "name": "Beas River Basin (Mandi, Kullu, Manali, Pandoh), Himachal Pradesh",
            "centroid_latitude": 31.710,
            "centroid_longitude": 76.930,
            "elevation_msl_m": 1250,
            "bounding_box": [31.0, 76.2, 32.8, 77.8]
        },
        "meteorological_impact": {
            "peak_rainfall": "24-48 hr precipitation totals exceeding 250-320 mm across multiple valley stations",
            "observed_damage": "Historic flooding of the Beas River, bridges washed away, national highways breached; >70 casualties.",
            "synoptic_trigger": "Rare and intense interaction between an active Western Disturbance and monsoon low-level jet carrying deep Arabian Sea moisture."
        },
        "official_reports": [
            "IMD Climate Diagnostics & Extreme Weather Bulletin: 'Very Severe Rainfall Event over Himachal Pradesh and Northwest India (July 8-10, 2023)'",
            "NCMRWF Technical Advisory on Monsoon-WD Coupling"
        ]
    },
    "case_04_wayanad_deluge_2024": {
        "case_id": "case_04_wayanad_deluge_2024",
        "title": "Wayanad Extreme Orographic Deluge & Dual Debris Flow",
        "hazard_type": "Torrential Convective Rainfall & Massive Landslide / Flash Flood",
        "date": "2024-07-29",
        "peak_time_utc": "21:00 UTC",
        "peak_time_ist": "02:30 IST (July 30)",
        "time_window_utc": ["12:00", "23:59"],
        "location": {
            "name": "Meppadi, Chooralmala & Mundakkai, Wayanad District, Western Ghats, Kerala",
            "centroid_latitude": 11.530,
            "centroid_longitude": 76.180,
            "elevation_msl_m": 900,
            "bounding_box": [11.2, 75.8, 11.9, 76.5]
        },
        "meteorological_impact": {
            "peak_rainfall": ">372 mm in 24 hours, with intense localized bursts onto saturated slopes (>140 mm in 4 hours)",
            "observed_damage": "Catastrophic debris flows and surging mud inundation wiped out Chooralmala and Mundakkai villages; >400 casualties.",
            "synoptic_trigger": "Vigorous offshore monsoon trough pumping saturated equatorial maritime air against the steep Western Ghats scarp."
        },
        "official_reports": [
            "IMD Meteorological Centre Thiruvananthapuram: 'Special Report on Extremely Heavy Rainfall over Wayanad on 29-30 July 2024'",
            "Geological Survey of India (GSI) & NDMA Rapid Post-Disaster Geotechnical Assessment (August 2024)"
        ]
    }
}


def initialize_case_directories(case_id: str) -> Path:
    """Creates directory structure for a specific case study."""
    case_dir = RAW_DATA_DIR / case_id
    (case_dir / "insat3d").mkdir(parents=True, exist_ok=True)
    (case_dir / "reanalysis_era5").mkdir(parents=True, exist_ok=True)
    (case_dir / "dem_srtm").mkdir(parents=True, exist_ok=True)
    return case_dir


def write_case_metadata_file(case_id: str, acquired_files: Dict[str, Any]) -> str:
    """Writes detailed metadata.json file into the raw case study folder."""
    case_info = CASE_STUDIES_REGISTRY[case_id]
    case_dir = RAW_DATA_DIR / case_id

    metadata_payload = {
        "case_metadata": case_info,
        "raw_datasets_manifest": acquired_files,
        "data_sources": {
            "satellite": "ISRO MOSDAC INSAT-3D/3DR (Free registration required)",
            "reanalysis": "ECMWF Copernicus CDS ERA5 (Free fallback) / NCMRWF IMDAA (Manual approval)",
            "dem_topography": "SRTM / Copernicus 30m Global DEM (Free, no registration)"
        },
        "sih_problem_statement": {
            "id": "26077",
            "theme": "AI-Driven Hyper-Local Severe Weather Nowcasting (2-6h Ahead)"
        }
    }

    meta_file = case_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return str(meta_file)


def process_case_study(case_id: str) -> Dict[str, Any]:
    """Runs data acquisition pipeline for a given case."""
    print(f"\n{'='*75}")
    print(f"Executing Data Acquisition for: {case_id}")
    print(f"Title: {CASE_STUDIES_REGISTRY[case_id]['title']}")
    print(f"{'='*75}")

    case_dir = initialize_case_directories(case_id)

    # 1. Download / prepare INSAT-3D/3DR Satellite Data
    sat_file = download_mosdac_data(case_id=case_id, simulate_if_offline=True)

    # 2. Download / prepare Atmospheric Reanalysis (ERA5 CDS)
    reanalysis_files = download_era5_data(case_id=case_id, simulate=True)

    # 3. Download / generate SRTM 30m DEM GeoTIFF
    dem_info = download_srtm_dem(case_id=case_id)

    manifest = {
        "satellite_insat3d": sat_file,
        "reanalysis_era5": reanalysis_files,
        "dem_srtm_30m": dem_info
    }

    meta_file = write_case_metadata_file(case_id, manifest)
    print(f"[Done] Case {case_id} datasets and metadata successfully saved to: {case_dir}")
    return manifest


def process_all_cases() -> Dict[str, Any]:
    """Runs data acquisition for all 4 benchmark cases."""
    results = {}
    for cid in CASE_STUDIES_REGISTRY.keys():
        results[cid] = process_case_study(cid)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate raw data acquisition for SIH 26077 historical severe weather cases.")
    parser.add_argument("--case", type=str, default="all", choices=list(CASE_STUDIES_REGISTRY.keys()) + ["all"],
                        help="Specific case study ID or 'all'")
    args = parser.parse_args()

    if args.case == "all":
        process_all_cases()
    else:
        process_case_study(args.case)
