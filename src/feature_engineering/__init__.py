"""
Feature engineering module for calculating thermodynamic indices,
spatial gradients, and convective severity indicators.
"""

from .atmospheric_indices import (
    classify_cape_instability,
    cloud_top_convective_score,
    calculate_cloudburst_risk,
    calculate_flash_flood_risk,
)
from .feature_extractor import (
    compute_integrated_water_vapor,
    compute_metpy_instability_cape_cin,
    compute_wind_convergence_and_shear,
    compute_cloud_top_temperature_and_drop_rate,
    compute_hydrological_dem_features,
    extract_case_study_features,
)
from .hydrologic_routing import (
    route_precipitation_to_flash_flood_risk,
    generate_flash_flood_routing_map,
    extract_d8_streamlines,
    generate_synthetic_drainage_streamlines,
)

__all__ = [
    "classify_cape_instability",
    "cloud_top_convective_score",
    "calculate_cloudburst_risk",
    "calculate_flash_flood_risk",
    "compute_integrated_water_vapor",
    "compute_metpy_instability_cape_cin",
    "compute_wind_convergence_and_shear",
    "compute_cloud_top_temperature_and_drop_rate",
    "compute_hydrological_dem_features",
    "extract_case_study_features",
    "route_precipitation_to_flash_flood_risk",
    "generate_flash_flood_routing_map",
    "extract_d8_streamlines",
    "generate_synthetic_drainage_streamlines",
]
