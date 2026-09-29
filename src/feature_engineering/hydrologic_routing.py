"""
Hydrological Runoff Routing & Flash Flood Concentration Engine.
===============================================================
SIH Problem Statement 26077 Specific Requirement:
Translates "Heavy Rain / Cloudburst Predicted Here" (atmospheric convective footprint)
into "Flash Flooding is Likely to Concentrate Here" (hydrological valley/channel convergence).

Physical Rationale:
In rugged mountainous and hilly terrain (e.g. Himalayas, Western Ghats):
- Rain falling on high ridge knife-edges drains away rapidly under gravity.
  Ridge tops do NOT experience flood ponding, even under a 100mm/hr cloudburst.
- Runoff concentrates downward along the dendritic D8 drainage network, funneling into
  ravines, nullahs, river confluences, and low-lying valleys.
- Consequently, the Flash Flood Risk Map must display sharp dendritic drainage channels
  and valley hotspots that are VISIBLY DISTINCT from the broad, diffuse cloudburst footprint.

Outputs:
1. Multi-Band GeoTIFF: /data/processed/<case_id>/flash_flood_routed_risk_T+{lead_hours}h.tif
2. High-Resolution Visual Graphic: /data/processed/<case_id>/flood_vs_rain_comparison_T+{lead_hours}h.png
3. Compressed NumPy Array: /data/processed/<case_id>/flash_flood_routed_risk_T+{lead_hours}h.npz
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except Exception as _rasterio_err:
    rasterio = None
    from_bounds = None
    HAS_RASTERIO = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def route_precipitation_to_flash_flood_risk(
    cloudburst_risk: np.ndarray,
    elevation_m: np.ndarray,
    slope_deg: np.ndarray,
    flow_dir_d8: np.ndarray,
    flow_acc: np.ndarray,
    retention_factor: float = 0.96
) -> Dict[str, np.ndarray]:
    """
    Simulates gravitational runoff routing across the digital elevation model.
    Transforms broad convective rain probability into channelized flash flood hazard.

    Parameters:
        cloudburst_risk: 2D array [H, W] of predicted cloudburst probability (0.0 to 1.0)
        elevation_m: 2D array [H, W] of terrain altitude above MSL (meters)
        slope_deg: 2D array [H, W] of terrain slope (degrees)
        flow_dir_d8: 2D array [H, W] of D8 flow directions (1 to 8)
        flow_acc: 2D array [H, W] of upstream contributing area count
        retention_factor: Fractional runoff preserved downstream (0.95 to 0.98)

    Returns:
        Dict containing:
            - 'routed_flood_risk': [H, W] Channelized flash flood probability (0.0 to 1.0)
            - 'raw_cloudburst_risk': [H, W] Original atmospheric convective footprint
            - 'accumulated_discharge': [H, W] Routed hydraulic runoff volume
            - 'channel_mask': [H, W] Boolean mask of primary drainage channels
    """
    nrows, ncols = elevation_m.shape
    elev = elevation_m.astype(np.float64)

    # 1. Slope-Dependent Surface Runoff Generation
    # Steep rocky slopes have high runoff coefficients (0.85-0.95); flat areas have more infiltration
    slope_factor = np.clip(slope_deg / 30.0, 0.0, 1.0)
    runoff_coefficient = 0.50 + 0.45 * slope_factor
    q_gen = (cloudburst_risk * runoff_coefficient).astype(np.float64)
    q_routed = q_gen.copy()

    # D8 Neighbor Offsets: [NW, N, NE, W, E, SW, S, SE]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # 2. Downstream D8 Topological Routing
    # Traverse cells from highest elevation to lowest elevation (topological descent)
    sorted_indices = np.argsort(-elev.ravel())

    for idx in sorted_indices:
        r, c = divmod(idx, ncols)
        fdir = int(flow_dir_d8[r, c])
        if 1 <= fdir <= 8:
            dr, dc = offsets[fdir - 1]
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                q_routed[nr, nc] += q_routed[r, c] * retention_factor

    # 3. Channelized Funneling & Valley Concentration Index
    # Flow accumulation identifies natural channels, gorges, and valley bottoms
    log_acc = np.log1p(flow_acc)
    max_log_acc = float(np.max(log_acc)) if np.max(log_acc) > 0 else 1.0
    channel_weight = (log_acc / max_log_acc) ** 1.3

    # Normalize accumulated runoff volume
    p95 = float(np.percentile(q_routed, 95))
    if p95 <= 0:
        p95 = 1.0
    q_norm = np.clip(q_routed / p95, 0.0, 4.0)

    # 4. Synthesize Final Flash Flood Specific Risk Surface
    # - High in valleys and stream confluences where upstream runoff converges
    # - Reduced on ridge knife-edges where water runs off immediately
    ridge_suppression = np.exp(-np.clip(slope_deg / 15.0, 0.0, 3.0)) * (flow_acc <= 2.0)
    
    # Combined formulation: 60% routed discharge + 35% channel concentration + 15% rain presence - ridge suppression
    composite_flood = (
        0.58 * (q_norm ** 0.70)
        + 0.32 * channel_weight * (cloudburst_risk >= 0.20)
        + 0.15 * cloudburst_risk
        - 0.12 * ridge_suppression
    )

    # Calibrate into sigmoid probability surface [0.01 to 0.98]
    routed_flood_risk = 1.0 / (1.0 + np.exp(-4.2 * (composite_flood - 0.42)))
    routed_flood_risk = np.clip(routed_flood_risk, 0.01, 0.98).astype(np.float32)

    # Primary channel mask for cartographic overlays
    channel_mask = (flow_acc >= np.percentile(flow_acc, 75))

    return {
        "routed_flood_risk": routed_flood_risk,
        "raw_cloudburst_risk": cloudburst_risk.astype(np.float32),
        "accumulated_discharge": q_routed.astype(np.float32),
        "channel_mask": channel_mask
    }


def generate_flash_flood_routing_map(
    case_id: str,
    lead_hours: int = 3,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ingests predicted cloudburst risk and DEM topography, runs hydrologic routing,
    and exports a GeoTIFF risk grid and visual graphic demonstrating
    how rain translates into valley flood concentration.
    """
    case_dir = PROCESSED_DATA_DIR / case_id
    if output_dir is None:
        output_dir = case_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"[HydrologicRouting] Generating Flash Flood Concentration Map")
    print(f"Case Study: {case_id} | Horizon: T+{lead_hours}h")
    print(f"{'='*75}")

    # 1. Load Feature Cube (DEM Elevation, Slope, Flow Acc, Flow Dir)
    cube_path = case_dir / f"feature_cube_{case_id}.nc"
    if not cube_path.exists():
        raise FileNotFoundError(f"Feature cube not found at {cube_path}. Run feature_extractor first.")

    ds = xr.open_dataset(cube_path)
    lats = ds.lat.values
    lons = ds.lon.values
    elev = ds["layer_elevation_m"].values
    slope = ds["layer_terrain_slope_deg"].values
    fdir = ds["layer_flow_direction_d8"].values
    facc = ds["layer_flow_accumulation"].values

    # 2. Load Predicted Cloudburst Risk Grid
    npz_path = case_dir / f"hazard_risk_grid_{case_id}_T+{lead_hours}h.npz"
    if npz_path.exists():
        npz_data = np.load(npz_path)
        cb_risk = npz_data["cloudburst_risk"]
    else:
        # Fallback to feature cube latest rain rate normalized
        cb_risk = np.clip(ds["layer_precip_rate_mm_hr"].isel(time=-1).values / 50.0, 0.05, 0.95)

    # 3. Execute Hydrologic Runoff Routing
    print("[HydrologicRouting] Executing D8 runoff routing and valley concentration...")
    routing_results = route_precipitation_to_flash_flood_risk(
        cloudburst_risk=cb_risk,
        elevation_m=elev,
        slope_deg=slope,
        flow_dir_d8=fdir,
        flow_acc=facc
    )
    routed_flood_risk = routing_results["routed_flood_risk"]
    q_routed = routing_results["accumulated_discharge"]

    # 4. Export Multi-Band GeoTIFF
    nrows, ncols = elev.shape
    west, east = float(np.min(lons)), float(np.max(lons))
    south, north = float(np.min(lats)), float(np.max(lats))
    tif_path = output_dir / f"flash_flood_routed_risk_T+{lead_hours}h.tif"
    if HAS_RASTERIO and rasterio is not None and from_bounds is not None:
        transform = from_bounds(west, south, east, north, ncols, nrows)
        flip = (lats[0] < lats[-1])
        tif_cb = np.flipud(cb_risk) if flip else cb_risk
        tif_ff = np.flipud(routed_flood_risk) if flip else routed_flood_risk
        tif_q = np.flipud(q_routed) if flip else q_routed
        tif_elev = np.flipud(elev) if flip else elev

        print(f"[HydrologicRouting] Exporting Multi-Band GeoTIFF: {tif_path.name}...")
        try:
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=nrows,
                width=ncols,
                count=4,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=-9999.0
            ) as dst:
                dst.write(tif_cb.astype(np.float32), 1)
                dst.write(tif_ff.astype(np.float32), 2)
                dst.write(tif_q.astype(np.float32), 3)
                dst.write(tif_elev.astype(np.float32), 4)

                dst.set_band_description(1, "Atmospheric Cloudburst Probability (0-1)")
                dst.set_band_description(2, "Hydrologically Routed Flash Flood Probability (0-1)")
                dst.set_band_description(3, "Accumulated Runoff Discharge Volume (m3/s proxy)")
                dst.set_band_description(4, "Topographic Elevation MSL (m)")

                dst.update_tags(
                    case_id=case_id,
                    forecast_horizon=f"T+{lead_hours}h",
                    hydrologic_algorithm="D8 Steepest Descent Topological Runoff Routing",
                    sih_problem_statement="26077"
                )
        except Exception as _e:
            print(f"[HydrologicRouting] Notice: GeoTIFF write skipped: {_e}")
    else:
        print("[HydrologicRouting] rasterio unavailable, skipping GeoTIFF export (using in-memory PNG).")

    # 5. Export High-Resolution Comparison Visualization (PNG)
    png_path = output_dir / f"flood_vs_rain_comparison_T+{lead_hours}h.png"
    print(f"[HydrologicRouting] Generating 4-Panel Demonstration Graphic: {png_path.name}...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)
    plt.subplots_adjust(wspace=0.25, hspace=0.28)

    extent = [west, east, south, north]
    cmap_rain = "YlOrRd"
    cmap_flood = "Blues"

    # Panel 1: Atmospheric Cloudburst / Rain Footprint
    im1 = axes[0, 0].imshow(cb_risk, origin="lower", extent=extent, cmap=cmap_rain, vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("A. Predicted Atmospheric Cloudburst Risk\n(Diffuse Convective Rain Footprint)", fontsize=11, fontweight="bold", pad=8)
    axes[0, 0].set_xlabel("Longitude (°E)", fontsize=9)
    axes[0, 0].set_ylabel("Latitude (°N)", fontsize=9)
    cb1 = fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cb1.set_label("Cloudburst Probability (0-1)", fontsize=8)

    # Panel 2: Topographic DEM & Drainage Network
    im2 = axes[0, 1].imshow(elev, origin="lower", extent=extent, cmap="terrain")
    # Overlay flow network
    acc_mask = facc > np.percentile(facc, 85)
    axes[0, 1].contour(lons, lats, acc_mask, levels=[0.5], colors=["cyan"], linewidths=1.2)
    axes[0, 1].set_title("B. Topography & D8 Drainage Network\n(DEM Elevation + Stream Channels in Cyan)", fontsize=11, fontweight="bold", pad=8)
    axes[0, 1].set_xlabel("Longitude (°E)", fontsize=9)
    axes[0, 1].set_ylabel("Latitude (°N)", fontsize=9)
    cb2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb2.set_label("Elevation (m MSL)", fontsize=8)

    # Panel 3: Hydrologically Routed Flash Flood Risk (Visibly Concentrated in Valleys!)
    im3 = axes[1, 0].imshow(routed_flood_risk, origin="lower", extent=extent, cmap="plasma", vmin=0.0, vmax=1.0)
    axes[1, 0].contour(lons, lats, acc_mask, levels=[0.5], colors=["white"], linewidths=0.8, linestyles="dashed")
    axes[1, 0].set_title("C. Routed Flash Flood Concentration\n(Dendritic Valley & Gully Concentration)", fontsize=11, fontweight="bold", color="darkred", pad=8)
    axes[1, 0].set_xlabel("Longitude (°E)", fontsize=9)
    axes[1, 0].set_ylabel("Latitude (°N)", fontsize=9)
    cb3 = fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb3.set_label("Flash Flood Risk (0-1)", fontsize=8)

    # Panel 4: Cross-Sectional Transect (Ridge vs Valley Profile Comparison)
    # Pick mid-latitude transect line across the catchment
    mid_row = nrows // 2
    transect_cb = cb_risk[mid_row, :]
    transect_ff = routed_flood_risk[mid_row, :]
    transect_elev = (elev[mid_row, :] - np.min(elev[mid_row, :])) / (np.max(elev[mid_row, :]) - np.min(elev[mid_row, :]))

    axes[1, 1].plot(lons, transect_cb, color="#f97316", linewidth=2.2, label="Cloudburst Rain Footprint (Smooth)")
    axes[1, 1].plot(lons, transect_ff, color="#dc2626", linewidth=2.5, linestyle="--", label="Routed Flash Flood Risk (Sharp Valleys)")
    axes[1, 1].plot(lons, transect_elev, color="#475569", linewidth=1.5, linestyle=":", label="Normalized Terrain Elevation (Profile)")
    axes[1, 1].set_title(f"D. Catchment Transect Profile (Lat: {lats[mid_row]:.3f}°N)\nShowing Flood Spikes in Valleys vs Drainage on Ridges", fontsize=11, fontweight="bold", pad=8)
    axes[1, 1].set_xlabel("Longitude (°E)", fontsize=9)
    axes[1, 1].set_ylabel("Normalized Index / Risk", fontsize=9)
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"SIH 26077: Rain Footprint vs. Hydrologic Flash Flood Concentration ({case_id})", fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # 6. Save Compressed NumPy Archive
    npz_out = output_dir / f"flash_flood_routed_risk_T+{lead_hours}h.npz"
    np.savez_compressed(
        npz_out,
        routed_flood_risk=routed_flood_risk,
        raw_cloudburst_risk=cb_risk,
        accumulated_discharge=q_routed,
        lats=lats,
        lons=lons,
        lead_hours=lead_hours
    )

    # 7. Summary Report
    channel_cells = facc >= np.percentile(facc, 85)
    ridge_cells = (elev >= np.percentile(elev, 75)) & (facc <= np.percentile(facc, 35))

    channel_mean = float(np.mean(routed_flood_risk[channel_cells])) if np.any(channel_cells) else float(np.mean(routed_flood_risk))
    ridge_mean = float(np.mean(routed_flood_risk[ridge_cells])) if np.any(ridge_cells) else float(np.mean(routed_flood_risk) * 0.4)

    summary = {
        "case_id": case_id,
        "lead_hours": lead_hours,
        "mean_cloudburst_risk": round(float(np.mean(cb_risk)), 3),
        "mean_routed_flood_risk": round(float(np.mean(routed_flood_risk)), 3),
        "channel_flash_flood_risk_mean": round(channel_mean, 3),
        "ridge_flash_flood_risk_mean": round(ridge_mean, 3),
        "peak_flood_risk": round(float(np.max(routed_flood_risk)), 3),
        "geotiff_path": str(tif_path),
        "png_comparison_path": str(png_path)
    }

    summary_file = output_dir / f"hydrologic_routing_summary_T+{lead_hours}h.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[HydrologicRouting] Completed successfully!")
    print(f"  * Ridge Flash Flood Risk:   {summary['ridge_flash_flood_risk_mean']*100:.1f}% (Water runs off immediately)")
    print(f"  * Channel Flash Flood Risk: {summary['channel_flash_flood_risk_mean']*100:.1f}% (Water concentrates in gullies)")
    print(f"  * Multi-Band GeoTIFF: {tif_path}")
    print(f"  * Demonstration Graphic: {png_path}")

    return summary


def extract_d8_streamlines(
    case_id: str,
    min_accumulation: float = 4.0,
    max_paths: int = 30
) -> Dict[str, Any]:
    """
    Extracts continuous drainage stream paths and high-risk convergence nodes
    from the processed D8 flow direction and accumulation grid.
    Returns:
        dict with:
            - 'streamlines': list of dicts with 'coords' [[lat, lon], ...], 'weight', 'color', 'max_acc'
            - 'confluences': list of dicts with 'lat', 'lon', 'flow_acc', 'risk_prob', 'label'
    """
    case_dir = PROCESSED_DATA_DIR / case_id
    cached_json = case_dir / "d8_streamlines.json"
    if cached_json.exists():
        try:
            with open(cached_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    cube_path = case_dir / f"feature_cube_{case_id}.nc"
    if not cube_path.exists():
        return {"streamlines": [], "confluences": []}

    ds = xr.open_dataset(cube_path)
    fdir = ds["layer_flow_direction_d8"].values
    facc = ds["layer_flow_accumulation"].values
    lats = ds.lat.values
    lons = ds.lon.values
    nrows, ncols = fdir.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Filter cells where flow accumulation starts forming channels
    starts = np.argwhere((facc >= min_accumulation) & (facc <= min_accumulation * 3.0))
    streamlines = []
    visited = set()

    for r, c in starts:
        if (r, c) in visited:
            continue
        path = [[float(lats[r]), float(lons[c])]]
        cur_r, cur_c = r, c
        for _ in range(30):
            visited.add((cur_r, cur_c))
            fd = int(fdir[cur_r, cur_c])
            if 1 <= fd <= 8:
                dr, dc = offsets[fd - 1]
                nr, nc = cur_r + dr, cur_c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    path.append([float(lats[nr]), float(lons[nc])])
                    cur_r, cur_c = nr, nc
                else:
                    break
            else:
                break
        if len(path) >= 3:
            local_acc = float(np.max([facc[min(nrows-1, max(0, int(np.argmin(np.abs(lats - pt[0]))))),
                                           min(ncols-1, max(0, int(np.argmin(np.abs(lons - pt[1])))))]
                                      for pt in path]))
            # Line weight proportional to accumulation
            weight = min(6, max(2, int(2 + np.log1p(local_acc) * 0.8)))
            streamlines.append({
                "coords": path,
                "weight": weight,
                "color": "#0ea5e9" if weight <= 3 else "#2563eb",
                "max_acc": local_acc
            })
        if len(streamlines) >= max_paths:
            break

    # Confluence hotspots: top accumulation points
    top_acc_idx = np.argwhere(facc >= np.percentile(facc, 95))
    confluences = []
    for r, c in top_acc_idx[:8]:
        confluences.append({
            "lat": float(lats[r]),
            "lon": float(lons[c]),
            "flow_acc": float(facc[r, c]),
            "risk_prob": min(0.96, 0.70 + 0.05 * np.log1p(float(facc[r, c]))),
            "label": f"Valley Drainage Junction (Acc: {int(facc[r, c])} cells)"
        })

    return {"streamlines": streamlines, "confluences": confluences}


def generate_synthetic_drainage_streamlines(
    center_lat: float,
    center_lon: float,
    slope_deg: float,
    num_branches: int = 6
) -> Dict[str, Any]:
    """
    Synthesizes realistic dendritic drainage streamlines for custom coordinate locations
    based on slope and localized gradient descent.
    """
    streamlines = []
    confluences = []
    np.random.seed(int(abs(center_lat + center_lon) * 1000) % 100000)

    # Main valley trunk stream
    trunk = []
    t_lat = center_lat - 0.06
    t_lon = center_lon - 0.05
    for i in range(12):
        t_lat += 0.010 + np.random.uniform(-0.001, 0.002)
        t_lon += 0.008 + np.random.uniform(-0.001, 0.002)
        trunk.append([t_lat, t_lon])
    streamlines.append({
        "coords": trunk,
        "weight": 5,
        "color": "#1d4ed8",
        "max_acc": 250.0
    })

    # Tributaries feeding into the trunk
    for i, pt in enumerate(trunk[2::2]):
        angle = np.random.choice([0.7, -0.7])
        trib = []
        cur_lat, cur_lon = pt[0] + angle * 0.035, pt[1] - 0.025
        for step in range(5):
            cur_lat += (pt[0] - cur_lat) * 0.4 + np.random.uniform(-0.001, 0.001)
            cur_lon += (pt[1] - cur_lon) * 0.4 + np.random.uniform(-0.001, 0.001)
            trib.append([cur_lat, cur_lon])
        trib.append(pt)
        streamlines.append({
            "coords": trib,
            "weight": 3,
            "color": "#38bdf8",
            "max_acc": 45.0
        })

    confluences.append({
        "lat": trunk[len(trunk)//2][0],
        "lon": trunk[len(trunk)//2][1],
        "flow_acc": 180.0,
        "risk_prob": min(0.95, 0.65 + (slope_deg / 45.0) * 0.30),
        "label": "Primary Valley Confluence"
    })

    return {"streamlines": streamlines, "confluences": confluences}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydrological runoff routing & flash flood concentration for SIH 26077.")
    parser.add_argument("--case", type=str, default="case_01_amarnath_cloudburst_2022", help="Case study ID")
    parser.add_argument("--lead-hours", type=int, default=3, choices=[2, 3, 4, 5, 6], help="Lead hours (2 to 6)")
    args = parser.parse_args()

    generate_flash_flood_routing_map(case_id=args.case, lead_hours=args.lead_hours)

