"""
Data ingestion module for satellite imagery (MOSDAC/INSAT-3D/3DR),
Doppler Weather Radar (DWR) datasets, and free open-source meteorological APIs.
"""

from .satellite_loader import load_satellite_netcdf
from .radar_loader import load_radar_reflectivity
from .open_meteo_client import fetch_nowcast_atmospheric_features

__all__ = [
    "load_satellite_netcdf",
    "load_radar_reflectivity",
    "fetch_nowcast_atmospheric_features",
]
