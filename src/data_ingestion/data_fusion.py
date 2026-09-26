"""
Spatio-Temporal Multi-Sensor Data Fusion Pipeline (SIH Problem Statement 26077).
================================================================================
Aligns multi-modal, multi-resolution severe weather datasets onto a unified
spatiotemporal xarray.Dataset (lat, lon, time, pressure_level):
1. Satellite: INSAT-3D / INSAT-3DR (MOSDAC) - High temporal (30-min), medium spatial (~4km).
2. Atmospheric Reanalysis: ERA5 / IMDAA (CDS / NCMRWF) - Coarse spatial (12-28km), 1-3 hourly.
3. Topography: SRTM 30m DEM - Static, high spatial resolution (~30m).

Methodological Resampling & Alignment Strategy:
-----------------------------------------------
- Common Spatial Grid:
  Standardized 0.02° to 0.04° regular grid (matching the convective meso-gamma scale ~2-4 km).
  * Continuous atmospheric fields (Temperature, Geopotential, Humidity, Satellite BT):
    Interpolated using 2D Bilinear Interpolation. Continuous thermodynamic mass fields vary
    smoothly; bilinear interpolation preserves spatial gradients without creating spurious extrema.
  * Topography (DEM):
    Topographic slope is computed on the native high-resolution DEM grid via central differences
    (preserving localized gully steepness) before bilinear regridding to the target grid.
- Common Temporal Grid:
  Standardized 30-minute interval matching the satellite observation epochs.
  * Reanalysis fields (CAPE, geopotential height, upper-level winds) evolve on synoptic
    timescales; piecewise linear temporal interpolation between reanalysis timesteps provides
    a physically realistic continuous trajectory of atmospheric instability.
- Unit Normalization:
  * Brightness Temperatures: Kelvin [K]
  * Surface Pressure: hPa (converted from Pa: sp / 100)
  * Total Precipitation: Converted to instantaneous rain rate [mm/hr]
  * Geopotential: Converted to Geopotential Height [gpm] via division by standard gravity g0 (9.80665 m/s^2)
  * Wind speeds & shears: m/s

Outputs:
- Single aligned CF-compliant NetCDF dataset stored in /data/processed/<case_id>/aligned_features_<case_id>.nc
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from scipy.interpolate import griddata

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Standard Earth gravity constant for geopotential height conversion
STANDARD_GRAVITY_G0 = 9.80665


class SpatioTemporalDataFusion:
    """
    Orchestrates ingestion, unit normalization, spatial regridding,
    temporal interpolation, and physical feature engineering.
    """

    def __init__(
        self,
        case_id: str,
        target_spatial_res_deg: float = 0.02,
        target_time_freq: str = "30min",
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        self.case_id = case_id
        self.target_res = target_spatial_res_deg
        self.time_freq = target_time_freq
        self.raw_case_dir = (raw_dir or RAW_DATA_DIR) / case_id
        self.processed_case_dir = (processed_dir or PROCESSED_DATA_DIR) / case_id
        self.processed_case_dir.mkdir(parents=True, exist_ok=True)

    def load_satellite_dataset(self) -> xr.Dataset:
        """Loads and standardizes coordinate names for INSAT-3D/3DR observations."""
        sat_dir = self.raw_case_dir / "insat3d"
        sat_files = list(sat_dir.glob("*.nc"))
        if not sat_files:
            raise FileNotFoundError(f"No satellite NetCDF files found in {sat_dir}")

        ds_sat = xr.open_dataset(sat_files[0])
        # Standardize coordinate naming to 'lat' and 'lon'
        rename_map = {}
        if "latitude" in ds_sat.coords or "latitude" in ds_sat.dims:
            rename_map["latitude"] = "lat"
        if "longitude" in ds_sat.coords or "longitude" in ds_sat.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            ds_sat = ds_sat.rename(rename_map)

        # Sort coordinates monotonically ascending
        if float(ds_sat.lat[0]) > float(ds_sat.lat[-1]):
            ds_sat = ds_sat.reindex(lat=ds_sat.lat[::-1])
        if float(ds_sat.lon[0]) > float(ds_sat.lon[-1]):
            ds_sat = ds_sat.reindex(lon=ds_sat.lon[::-1])

        return ds_sat

    def load_reanalysis_dataset(self) -> xr.Dataset:
        """
        Loads pressure-level and surface-level reanalysis files,
        unifies them into a single dataset, and normalizes units.
        """
        reanalysis_dir = self.raw_case_dir / "reanalysis_era5"
        pl_files = list(reanalysis_dir.glob("*pressure_levels*.nc"))
        sfc_files = list(reanalysis_dir.glob("*surface*.nc"))

        if not pl_files or not sfc_files:
            raise FileNotFoundError(f"Missing reanalysis NetCDF files in {reanalysis_dir}")

        ds_pl = xr.open_dataset(pl_files[0])
        ds_sfc = xr.open_dataset(sfc_files[0])

        # Standardize coordinate naming
        for ds in (ds_pl, ds_sfc):
            rename_map = {}
            if "latitude" in ds.coords or "latitude" in ds.dims:
                rename_map["latitude"] = "lat"
            if "longitude" in ds.coords or "longitude" in ds.dims:
                rename_map["longitude"] = "lon"
            if rename_map:
                ds = ds.rename(rename_map)

        # Re-assign if renamed
        if "latitude" in ds_pl.dims:
            ds_pl = ds_pl.rename({"latitude": "lat", "longitude": "lon"})
        if "latitude" in ds_sfc.dims:
            ds_sfc = ds_sfc.rename({"latitude": "lat", "longitude": "lon"})

        # Ensure lat/lon ascending
        if float(ds_pl.lat[0]) > float(ds_pl.lat[-1]):
            ds_pl = ds_pl.reindex(lat=ds_pl.lat[::-1])
        if float(ds_sfc.lat[0]) > float(ds_sfc.lat[-1]):
            ds_sfc = ds_sfc.reindex(lat=ds_sfc.lat[::-1])

        # Unit Normalization:
        # 1. Surface Pressure: Pa -> hPa
        if "sp" in ds_sfc:
            if float(ds_sfc["sp"].mean()) > 2000.0:
                ds_sfc["surface_pressure_hpa"] = ds_sfc["sp"] / 100.0
            else:
                ds_sfc["surface_pressure_hpa"] = ds_sfc["sp"]
            ds_sfc["surface_pressure_hpa"].attrs = {
                "units": "hPa",
                "long_name": "Surface Pressure in hectopascals",
                "normalization": "sp_pascals / 100.0"
            }

        # 2. Total Precipitation: Cumulative m -> Instantaneous mm/hr rate
        if "tp" in ds_sfc:
            # Assuming 3-hourly accumulated reanalysis intervals
            interval_hours = 3.0
            ds_sfc["precip_rate_mm_hr"] = (ds_sfc["tp"] * 1000.0) / interval_hours
            ds_sfc["precip_rate_mm_hr"].attrs = {
                "units": "mm/hr",
                "long_name": "Precipitation Rate in mm per hour",
                "normalization": "(tp_meters * 1000.0) / 3.0"
            }

        # 3. Geopotential: m^2/s^2 -> Geopotential Height (gpm)
        if "z" in ds_pl:
            ds_pl["geopotential_height_gpm"] = ds_pl["z"] / STANDARD_GRAVITY_G0
            ds_pl["geopotential_height_gpm"].attrs = {
                "units": "gpm",
                "long_name": "Geopotential Height in geopotential meters",
                "normalization": "z / 9.80665"
            }

        # 4. CAPE
        if "cape" in ds_sfc:
            ds_sfc["cape_j_kg"] = ds_sfc["cape"]
            ds_sfc["cape_j_kg"].attrs = {"units": "J/kg", "long_name": "Convective Available Potential Energy"}

        # Merge surface and 3D pressure fields
        ds_reanalysis = xr.merge([ds_pl, ds_sfc])
        return ds_reanalysis

    def load_dem_dataset(self) -> xr.Dataset:
        """
        Loads SRTM 30m DEM GeoTIFF, calculates local topographic slope (degrees),
        and wraps into an xarray.Dataset with lat/lon coordinates.
        """
        dem_dir = self.raw_case_dir / "dem_srtm"
        dem_files = list(dem_dir.glob("*.tif"))
        if not dem_files:
            raise FileNotFoundError(f"No DEM GeoTIFF files found in {dem_dir}")

        with rasterio.open(dem_files[0]) as src:
            elev_data = src.read(1).astype(np.float32)
            elev_data[elev_data == src.nodata] = np.nan
            bounds = src.bounds  # left, bottom, right, top

            lats = np.linspace(bounds.bottom, bounds.top, src.height)
            lons = np.linspace(bounds.left, bounds.right, src.width)

            # Topographic Slope calculation on native high-res grid (degrees)
            # dy, dx in approximate meters
            dy_meters = 111000.0 / src.height
            dx_meters = 111000.0 / src.width
            gy, gx = np.gradient(elev_data, dy_meters, dx_meters)
            slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))).astype(np.float32)

        ds_dem = xr.Dataset(
            data_vars={
                "elevation": (["lat", "lon"], elev_data, {"units": "m", "long_name": "Topographic Elevation MSL"}),
                "topographic_slope": (["lat", "lon"], slope_deg, {"units": "degrees", "long_name": "Terrain Slope Angle"}),
            },
            coords={"lat": lats, "lon": lons},
            attrs={"crs": "EPSG:4326", "source": "SRTM / Copernicus 30m DEM"}
        )
        return ds_dem

    def build_common_grid(self, ds_sat: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Derives the common spatial bounding box and temporal epochs.
        The spatial grid is based on the target bounding box of the satellite observations,
        resampled to target_res (e.g. 0.02° ~ 2km).
        """
        min_lat = float(ds_sat.lat.min())
        max_lat = float(ds_sat.lat.max())
        min_lon = float(ds_sat.lon.min())
        max_lon = float(ds_sat.lon.max())

        common_lats = np.arange(min_lat, max_lat + (self.target_res / 2.0), self.target_res)
        common_lons = np.arange(min_lon, max_lon + (self.target_res / 2.0), self.target_res)

        # Common time grid matching satellite timestamps
        common_times = pd.to_datetime(ds_sat.time.values)
        return common_lats, common_lons, common_times

    def regrid_dem_to_target(
        self,
        ds_dem: xr.Dataset,
        target_lats: np.ndarray,
        target_lons: np.ndarray
    ) -> xr.Dataset:
        """
        Interpolates the high-resolution DEM onto the common spatial grid.
        Uses bilinear interpolation with nearest-neighbor extrapolation along boundaries.
        """
        # Interpolate DEM using xarray bilinear interp
        dem_interp = ds_dem.interp(lat=target_lats, lon=target_lons, method="linear")

        # If boundary NaN exists due to slight extent mismatch, fill with nearest neighbor
        if np.isnan(dem_interp["elevation"].values).any():
            orig_lon_mesh, orig_lat_mesh = np.meshgrid(ds_dem.lon.values, ds_dem.lat.values)
            points = np.column_stack([orig_lat_mesh.ravel(), orig_lon_mesh.ravel()])
            elev_values = ds_dem["elevation"].values.ravel()
            slope_values = ds_dem["topographic_slope"].values.ravel()

            tgt_lon_mesh, tgt_lat_mesh = np.meshgrid(target_lons, target_lats)
            tgt_points = np.column_stack([tgt_lat_mesh.ravel(), tgt_lon_mesh.ravel()])

            filled_elev = griddata(points, elev_values, tgt_points, method="nearest").reshape(tgt_lat_mesh.shape)
            filled_slope = griddata(points, slope_values, tgt_points, method="nearest").reshape(tgt_lat_mesh.shape)

            dem_interp["elevation"].values = filled_elev.astype(np.float32)
            dem_interp["topographic_slope"].values = filled_slope.astype(np.float32)

        return dem_interp

    def fuse(self) -> xr.Dataset:
        """
        Executes end-to-end spatiotemporal data fusion.
        Returns a single unified xarray.Dataset.
        """
        print(f"\n[DataFusion] Starting fusion pipeline for case: {self.case_id}...")

        # 1. Ingest all modalities
        ds_sat = self.load_satellite_dataset()
        ds_reanalysis = self.load_reanalysis_dataset()
        ds_dem = self.load_dem_dataset()

        # 2. Build common spatiotemporal reference grid
        target_lats, target_lons, target_times = self.build_common_grid(ds_sat)
        print(f"[DataFusion] Target Grid Dimensions: Lat: {len(target_lats)} (step: {self.target_res}°), "
              f"Lon: {len(target_lons)}, Time steps: {len(target_times)}")

        # 3. Regrid Satellite (Spatial 2D Bilinear Interpolation)
        print("[DataFusion] Regridding satellite brightness temperatures (Bilinear)...")
        sat_aligned = ds_sat.interp(lat=target_lats, lon=target_lons, method="linear")

        # 4. Regrid Reanalysis (Temporal Interpolation -> Spatial 2D Bilinear Interpolation)
        print("[DataFusion] Resampling reanalysis fields in time & space...")
        # Temporal interpolation of reanalysis to match satellite epochs
        reanalysis_t_interp = ds_reanalysis.interp(time=target_times, method="linear")
        # Spatial interpolation of reanalysis onto target high-resolution grid
        reanalysis_aligned = reanalysis_t_interp.interp(lat=target_lats, lon=target_lons, method="linear")

        # 5. Regrid DEM (Bilinear spatial downsampling with slope preservation)
        print("[DataFusion] Regridding high-resolution DEM & slope...")
        dem_aligned = self.regrid_dem_to_target(ds_dem, target_lats, target_lons)

        # 6. Merge all aligned modalities
        fused = xr.Dataset()

        # Add Satellite Channels
        for v in ["tir1_brightness_temp", "tir2_brightness_temp", "water_vapor_brightness_temp"]:
            if v in sat_aligned:
                fused[v] = sat_aligned[v].astype(np.float32)

        # Add Surface Reanalysis Indicators
        for v in ["cape_j_kg", "surface_pressure_hpa", "precip_rate_mm_hr"]:
            if v in reanalysis_aligned:
                fused[v] = reanalysis_aligned[v].astype(np.float32)

        # Add Multilevel Atmospheric Profiles
        for v in ["t", "r", "u", "v", "geopotential_height_gpm"]:
            if v in reanalysis_aligned:
                fused[v] = reanalysis_aligned[v].astype(np.float32)

        # Add Static Topography (Elevation & Slope)
        fused["elevation"] = dem_aligned["elevation"].astype(np.float32)
        fused["topographic_slope"] = dem_aligned["topographic_slope"].astype(np.float32)

        # 7. Compute Derived Physical Nowcasting Diagnostic Variables
        print("[DataFusion] Computing derived thermodynamic and kinematic features...")
        fused = self.compute_derived_features(fused)

        # 8. Attach CF-compliant metadata and methodological notes
        fused.attrs = {
            "title": f"Unified Spatiotemporally Aligned Severe Weather Dataset - {self.case_id}",
            "case_id": self.case_id,
            "spatial_resolution_deg": self.target_res,
            "temporal_frequency": self.time_freq,
            "crs": "EPSG:4326 (WGS84)",
            "conventions": "CF-1.8",
            "sih_problem_statement": "26077 (Hyper-Local Severe Weather Nowcasting 2-6h Ahead)",
            "spatial_interpolation_method": "2D Bilinear Interpolation (scipy/xarray)",
            "temporal_interpolation_method": "Piecewise Linear Temporal Spline matching satellite scans",
            "topography_processing": "Gradient slope calculation on native 30m grid prior to spatial regridding",
            "creator": "SIH26077 AI Nowcasting Team",
            "source_datasets": "ISRO MOSDAC (INSAT-3D/3DR), ECMWF Copernicus CDS (ERA5), SRTM 30m DEM"
        }

        # Save to disk
        out_file = self.processed_case_dir / f"aligned_features_{self.case_id}.nc"
        print(f"[DataFusion] Exporting unified NetCDF dataset to: {out_file}...")
        fused.to_netcdf(out_file)
        print(f"[DataFusion] Successfully created: {out_file} (Size: {out_file.stat().st_size / 1024:.1f} KB)")

        return fused

    @staticmethod
    def compute_derived_features(ds: xr.Dataset) -> xr.Dataset:
        """
        Derives key convective and kinematic features used directly by the ML nowcaster:
        1. Split-window Brightness Temperature Difference (TIR1 - TIR2)
        2. Convective Cloud Top Depression / Overshoot Index (215K - TIR1)
        3. Deep-layer Wind Shear (850 hPa to 500 hPa and 850 hPa to 200 hPa)
        4. Bulk Wind Speed at each pressure level
        5. Orographic Upslope Wind Proxy (Wind speed * Topographic Slope)
        """
        # 1. Split-Window BTD (Cloud Optical Thickness / Microphysics)
        if "tir1_brightness_temp" in ds and "tir2_brightness_temp" in ds:
            ds["split_window_btd_k"] = ds["tir1_brightness_temp"] - ds["tir2_brightness_temp"]
            ds["split_window_btd_k"].attrs = {
                "units": "Kelvin",
                "long_name": "Split-Window Brightness Temperature Difference (TIR1 - TIR2)",
                "description": "Distinguishes deep convective cumulonimbus anvils from thin cirrus"
            }

        # 2. Convective Overshoot Index (Tropopause penetration)
        if "tir1_brightness_temp" in ds:
            overshoot = np.clip(215.0 - ds["tir1_brightness_temp"], 0.0, None)
            ds["convective_overshoot_index"] = overshoot
            ds["convective_overshoot_index"].attrs = {
                "units": "Kelvin",
                "long_name": "Convective Cloud Top Overshoot Intensity",
                "description": "Positive values (>0K) indicate overshooting tops penetrating the equilibrium level"
            }

        # 3. Wind Speeds & Deep-Layer Bulk Shear
        if "u" in ds and "v" in ds and "level" in ds.coords:
            wind_speed = np.sqrt(ds["u"]**2 + ds["v"]**2)
            ds["wind_speed_level"] = wind_speed
            ds["wind_speed_level"].attrs = {"units": "m/s", "long_name": "Horizontal Wind Speed at Pressure Level"}

            levels_list = [int(lvl) for lvl in ds.level.values]
            # Deep Layer Shear between 850 hPa and 500 hPa
            if 850 in levels_list and 500 in levels_list:
                u850 = ds["u"].sel(level=850)
                v850 = ds["v"].sel(level=850)
                u500 = ds["u"].sel(level=500)
                v500 = ds["v"].sel(level=500)
                shear_500 = np.sqrt((u500 - u850)**2 + (v500 - v850)**2)
                ds["shear_850_500_mps"] = shear_500
                ds["shear_850_500_mps"].attrs = {
                    "units": "m/s",
                    "long_name": "Bulk Wind Shear (850 to 500 hPa)",
                    "description": "Key thermodynamic driver for multicellular storm organization"
                }

            # Deep Layer Shear between 850 hPa and 200 hPa
            if 850 in levels_list and 200 in levels_list:
                u200 = ds["u"].sel(level=200)
                v200 = ds["v"].sel(level=200)
                shear_200 = np.sqrt((u200 - u850)**2 + (v200 - v850)**2)
                ds["shear_850_200_mps"] = shear_200
                ds["shear_850_200_mps"].attrs = {
                    "units": "m/s",
                    "long_name": "Deep-Layer Bulk Wind Shear (850 to 200 hPa)",
                    "description": "Critical discriminator for supercell and squall line severity"
                }

            # 4. Orographic Upslope Forcing Proxy (Low-level wind * Topographic slope)
            if 850 in levels_list and "topographic_slope" in ds:
                w850 = ds["wind_speed_level"].sel(level=850)
                slope_rad = np.radians(ds["topographic_slope"])
                upslope = w850 * np.sin(slope_rad)
                ds["orographic_upslope_index"] = upslope
                ds["orographic_upslope_index"].attrs = {
                    "units": "m/s",
                    "long_name": "Orographic Moisture Lift Proxy",
                    "description": "Represents forced vertical ascent of low-level moisture along mountain ridges"
                }

        return ds


def fuse_case_study_data(
    case_id: str,
    target_res_deg: float = 0.02,
    time_freq: str = "30min",
    output_dir: Optional[Path] = None
) -> xr.Dataset:
    """Convenience function to run data fusion on a case study."""
    fusion = SpatioTemporalDataFusion(
        case_id=case_id,
        target_spatial_res_deg=target_res_deg,
        target_time_freq=time_freq,
        processed_dir=output_dir
    )
    return fusion.fuse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-sensor data fusion onto common xarray grid for SIH 26077.")
    parser.add_argument("--case", type=str, default="case_01_amarnath_cloudburst_2022",
                        help="Case study ID (e.g. case_01_amarnath_cloudburst_2022 or 'all')")
    parser.add_argument("--res", type=float, default=0.02, help="Target spatial grid resolution in degrees (default: 0.02 ~ 2km)")
    args = parser.parse_args()

    if args.case == "all":
        case_ids = [
            "case_01_amarnath_cloudburst_2022",
            "case_02_north_india_squall_2018",
            "case_03_himachal_flash_flood_2023",
            "case_04_wayanad_deluge_2024"
        ]
        for cid in case_ids:
            fuse_case_study_data(cid, target_res_deg=args.res)
    else:
        fuse_case_study_data(args.case, target_res_deg=args.res)
