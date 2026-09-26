"""
Data ingestion module for satellite imagery (MOSDAC/INSAT-3D/3DR),
Doppler Weather Radar (DWR) datasets, and free open-source meteorological APIs.
"""

from .satellite_loader import load_satellite_netcdf
from .radar_loader import load_radar_reflectivity
from .open_meteo_client import fetch_nowcast_atmospheric_features
from .mosdac_downloader import download_mosdac_data
from .era5_imdaa_downloader import download_era5_data
from .srtm_dem_downloader import download_srtm_dem
from .case_studies_manager import process_case_study, process_all_cases, CASE_STUDIES_REGISTRY

__all__ = [
    "load_satellite_netcdf",
    "load_radar_reflectivity",
    "fetch_nowcast_atmospheric_features",
    "download_mosdac_data",
    "download_era5_data",
    "download_srtm_dem",
    "process_case_study",
    "process_all_cases",
    "CASE_STUDIES_REGISTRY",
]
