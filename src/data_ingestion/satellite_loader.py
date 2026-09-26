"""
Satellite Data Ingestion for INSAT-3D / 3DR (MOSDAC / ISRO Open Data).
Handles NetCDF4 / HDF5 / GeoTIFF multidimensional raster files.
"""

from typing import Dict, Any, Optional
import numpy as np


def load_satellite_netcdf(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads INSAT-3D/3DR satellite products (TIR1, TIR2, WV, MIR) using xarray.
    If no file path is provided or the file is missing, generates a mock spatial grid
    to allow end-to-end testing without external downloads.

    Returns:
        Dict containing:
            - 'tir1_bt': Thermal Infrared Brightness Temperature (Kelvin)
            - 'water_vapor': Water Vapor band
            - 'latitude': 1D or 2D coordinate array
            - 'longitude': 1D or 2D coordinate array
    """
    if file_path:
        try:
            import xarray as xr
            ds = xr.open_dataset(file_path)
            return {
                "dataset": ds,
                "status": "loaded_from_disk",
                "source": file_path
            }
        except Exception as exc:
            # Fall back to simulation if file is unreadable or xarray not yet configured
            print(f"[satellite_loader] Warning: Could not open {file_path}: {exc}. Using mock data.")

    # Synthetic grid over the Indian Subcontinent (~8°N to 37°N, 68°E to 97°E)
    lat = np.linspace(8.0, 37.0, 64)
    lon = np.linspace(68.0, 97.0, 64)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Simulated Brightness Temperature in Kelvin (Colder tops < 220K indicate deep convective storms)
    simulated_bt = 285.0 - 65.0 * np.exp(-((lat_grid - 29.5)**2 / 4.0 + (lon_grid - 79.5)**2 / 4.0))

    return {
        "tir1_bt": simulated_bt,
        "latitude": lat,
        "longitude": lon,
        "status": "simulated_insat3d_grid",
        "description": "Simulated convective cloud top brightness temperature for testing"
    }
