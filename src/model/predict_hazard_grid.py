"""
Inference & Spatial Risk Grid Generation Engine (SIH Problem Statement 26077).
=============================================================================
Runs multi-task inference on an input spatiotemporal feature cube or table,
predicting 2 to 6 hours ahead nowcasts, and generating:
1. Multi-Band GeoTIFF Risk Surface (.tif) with EPSG:4326 geospatial metadata:
   - Band 1: Severe Thunderstorm Probability
   - Band 2: Cloudburst Deluge Probability
   - Band 3: Flash Flood / Debris Flow Probability
2. Compressed NumPy Risk Archive (.npz)
3. Emergency Alert Summary Report (.json)

Usage:
    python -m src.model.predict_hazard_grid --case case_01_amarnath_cloudburst_2022 --lead-hours 3
    python -m src.model.predict_hazard_grid --case case_04_wayanad_deluge_2024 --lead-hours 4
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import xarray as xr
try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except Exception as _rasterio_err:
    rasterio = None
    from_bounds = None
    HAS_RASTERIO = False

from src.model.multitask_model import MultiTaskSevereWeatherModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_model(model_path: Optional[Path] = None) -> MultiTaskSevereWeatherModel:
    """Loads the trained multi-task model artifact."""
    if model_path is None:
        model_path = MODELS_DIR / "multitask_nowcast_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run 'python -m src.model.train_multitask_model' first."
        )
    return MultiTaskSevereWeatherModel.load(model_path)


def run_hazard_grid_inference(
    case_id: str,
    lead_hours: int = 2,
    time_idx: int = -1,
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes nowcasting inference on the specified case study for lead_hours ahead.
    Reconstructs 2D spatial risk grids and exports GeoTIFF and NumPy artifacts.
    """
    lead_hours = int(np.clip(lead_hours, 2, 6))
    case_dir = PROCESSED_DATA_DIR / case_id

    # 1. Locate feature cube or parquet table
    cube_file = case_dir / f"feature_cube_{case_id}.nc"
    parquet_file = case_dir / f"feature_table_{case_id}.parquet"

    if not cube_file.exists() or not parquet_file.exists():
        raise FileNotFoundError(f"Feature datasets missing for {case_id}. Run feature_extractor first.")

    if output_dir is None:
        output_dir = case_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"[InferenceEngine] Running Multi-Task Nowcasting Inference")
    print(f"Case Study: {case_id} | Forecast Horizon: T+{lead_hours} Hours")
    print(f"{'='*75}")

    # 2. Load model & data
    model = load_model(model_path)
    ds_cube = xr.open_dataset(cube_file)
    df_features = pd.read_parquet(parquet_file)

    lats = ds_cube.lat.values
    lons = ds_cube.lon.values
    times = pd.to_datetime(ds_cube.time.values)
    target_time = times[time_idx]

    print(f"[InferenceEngine] Selected Target Timestep: {target_time.isoformat()} (Index: {time_idx})")
    print(f"[InferenceEngine] Spatial Grid Size: {len(lats)} Latitudes x {len(lons)} Longitudes")

    # Filter dataframe for selected timestep
    df_step = df_features[df_features["time"] == target_time].copy()
    if df_step.empty:
        df_step = df_features.iloc[: len(lats) * len(lons)].copy()

    # 3. Model Prediction
    print(f"[InferenceEngine] Predicting multi-hazard probabilities for T+{lead_hours}h...")
    preds = model.predict_proba(df_step, lead_hours=lead_hours)

    p_ts = preds["thunderstorm_prob"]
    p_cb = preds["cloudburst_prob"]
    p_ff = preds["flash_flood_prob"]

    # 4. Reshape predictions back into 2D Spatial Grids (Height x Width)
    nrows = len(lats)
    ncols = len(lons)

    # Ensure shape alignment
    grid_ts = p_ts[: nrows * ncols].reshape(nrows, ncols).astype(np.float32)
    grid_cb = p_cb[: nrows * ncols].reshape(nrows, ncols).astype(np.float32)
    grid_ff = p_ff[: nrows * ncols].reshape(nrows, ncols).astype(np.float32)

    # Invert row order if latitudes descending to maintain standard North-up GeoTIFF convention
    if lats[0] < lats[-1]:
        # Latitudes are ascending (South to North). Flip vertically for top-down GeoTIFF
        grid_ts_tif = np.flipud(grid_ts)
        grid_cb_tif = np.flipud(grid_cb)
        grid_ff_tif = np.flipud(grid_ff)
    else:
        grid_ts_tif = grid_ts
        grid_cb_tif = grid_cb
        grid_ff_tif = grid_ff

    # 5. Export Multi-Band GeoTIFF
    west = float(np.min(lons))
    east = float(np.max(lons))
    south = float(np.min(lats))
    north = float(np.max(lats))

    # Affine transform for rasterio (bounds: west, south, east, north)
    transform = from_bounds(west, south, east, north, ncols, nrows)

    tif_filename = f"hazard_risk_grid_{case_id}_T+{lead_hours}h.tif"
    tif_path = output_dir / tif_filename

    print(f"[InferenceEngine] Exporting 3-Band GeoTIFF: {tif_filename}...")
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=3,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(grid_ts_tif, 1)
        dst.write(grid_cb_tif, 2)
        dst.write(grid_ff_tif, 3)

        dst.set_band_description(1, "Severe Thunderstorm Probability (0-1)")
        dst.set_band_description(2, "Cloudburst Probability (0-1)")
        dst.set_band_description(3, "Flash Flood Probability (0-1)")

        dst.update_tags(
            case_id=case_id,
            forecast_lead_hours=f"{lead_hours}",
            timestamp_utc=str(target_time),
            model_architecture="Multi-Task HistGradientBoosting Proxy",
            sih_problem_statement="26077"
        )

    # 6. Export Compressed NumPy Archive
    npz_filename = f"hazard_risk_grid_{case_id}_T+{lead_hours}h.npz"
    npz_path = output_dir / npz_filename
    print(f"[InferenceEngine] Exporting NumPy Archive: {npz_filename}...")
    np.savez_compressed(
        npz_path,
        thunderstorm_risk=grid_ts,
        cloudburst_risk=grid_cb,
        flash_flood_risk=grid_ff,
        lats=lats,
        lons=lons,
        lead_hours=lead_hours,
        timestamp=str(target_time)
    )

    # 7. Generate Emergency Alert Summary JSON Report
    max_ts = float(np.max(grid_ts))
    max_cb = float(np.max(grid_cb))
    max_ff = float(np.max(grid_ff))

    # Coordinates of maximum danger point
    max_cb_idx = np.unravel_index(np.argmax(grid_cb), grid_cb.shape)
    peak_lat = float(lats[max_cb_idx[0]])
    peak_lon = float(lons[max_cb_idx[1]])

    def get_tier(prob: float) -> str:
        if prob >= 0.70:
            return "RED (Extreme Danger)"
        elif prob >= 0.45:
            return "ORANGE (Severe Alert)"
        elif prob >= 0.25:
            return "YELLOW (Advisory)"
        return "GREEN (Normal)"

    summary = {
        "case_id": case_id,
        "lead_hours": lead_hours,
        "lead_time_target": f"T+{lead_hours}h",
        "reference_time_utc": str(target_time),
        "peak_hazard_location": {
            "latitude": round(peak_lat, 4),
            "longitude": round(peak_lon, 4)
        },
        "max_risk_scores": {
            "severe_thunderstorm": {"probability": round(max_ts, 3), "alert_level": get_tier(max_ts)},
            "cloudburst": {"probability": round(max_cb, 3), "alert_level": get_tier(max_cb)},
            "flash_flood": {"probability": round(max_ff, 3), "alert_level": get_tier(max_ff)}
        },
        "high_risk_area_coverage_pct": {
            "thunderstorm": round(float(np.mean(grid_ts >= 0.50) * 100), 1),
            "cloudburst": round(float(np.mean(grid_cb >= 0.50) * 100), 1),
            "flash_flood": round(float(np.mean(grid_ff >= 0.50) * 100), 1)
        },
        "artifacts_generated": {
            "geotiff_path": str(tif_path),
            "numpy_archive_path": str(npz_path)
        }
    }

    json_filename = f"hazard_summary_{case_id}_T+{lead_hours}h.json"
    json_path = output_dir / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[InferenceEngine] Nowcasting Complete for T+{lead_hours}h!")
    print(f"  * Peak Cloudburst Risk:   {max_cb*100:.1f}% -> {summary['max_risk_scores']['cloudburst']['alert_level']}")
    print(f"  * Peak Flash Flood Risk:  {max_ff*100:.1f}% -> {summary['max_risk_scores']['flash_flood']['alert_level']}")
    print(f"  * Peak Thunderstorm Risk: {max_ts*100:.1f}% -> {summary['max_risk_scores']['severe_thunderstorm']['alert_level']}")
    print(f"  * Output GeoTIFF: {tif_path}")
    print(f"  * Output NPZ:     {npz_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spatial nowcast risk grid for SIH 26077.")
    parser.add_argument("--case", type=str, default="case_01_amarnath_cloudburst_2022", help="Case study ID")
    parser.add_argument("--lead-hours", type=int, default=3, choices=[2, 3, 4, 5, 6], help="Lead time in hours (2 to 6)")
    parser.add_argument("--time-idx", type=int, default=-1, help="Time index in feature cube (default: -1 for latest)")
    args = parser.parse_args()

    run_hazard_grid_inference(case_id=args.case, lead_hours=args.lead_hours, time_idx=args.time_idx)
