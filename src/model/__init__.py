"""
Spatio-Temporal Nowcasting Models for 2-6 hour severe convective weather forecasting.
Lazy module loading (PEP 562) to enable near-instantaneous startup without eagerly
loading PyTorch, Rasterio, or heavy training modules on package initialization.
"""

from typing import Any

# Scripted scenarios are ultra-lightweight pure python dictionaries, safe to import eagerly
from .scripted_scenarios import (
    SCRIPTED_SCENARIOS,
    TRANSPARENCY_NOTE,
    list_scripted_scenarios,
    get_scenario_by_id,
    get_scenario_stage,
)

__all__ = [
    "SpatioTemporalNowcastNet",
    "run_nowcast_inference",
    "MultiTaskSevereWeatherModel",
    "FEATURE_COLUMNS",
    "train_and_evaluate_model",
    "run_hazard_grid_inference",
    "load_model",
    "SCRIPTED_SCENARIOS",
    "TRANSPARENCY_NOTE",
    "list_scripted_scenarios",
    "get_scenario_by_id",
    "get_scenario_stage",
]


def __getattr__(name: str) -> Any:
    """Lazy imports for heavy modules only when explicitly accessed."""
    if name == "SpatioTemporalNowcastNet":
        from .nowcast_net import SpatioTemporalNowcastNet
        return SpatioTemporalNowcastNet
    if name == "run_nowcast_inference":
        from .inference import run_nowcast_inference
        return run_nowcast_inference
    if name == "MultiTaskSevereWeatherModel":
        from .multitask_model import MultiTaskSevereWeatherModel
        return MultiTaskSevereWeatherModel
    if name == "FEATURE_COLUMNS":
        from .multitask_model import FEATURE_COLUMNS
        return FEATURE_COLUMNS
    if name == "train_and_evaluate_model":
        from .train_multitask_model import train_and_evaluate_model
        return train_and_evaluate_model
    if name == "run_hazard_grid_inference":
        from .predict_hazard_grid import run_hazard_grid_inference
        return run_hazard_grid_inference
    if name == "load_model":
        from .predict_hazard_grid import load_model
        return load_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
