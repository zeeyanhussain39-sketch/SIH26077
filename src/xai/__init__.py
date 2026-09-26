"""
Explainable AI (XAI) Module for Severe Weather Nowcasting.
Uses SHAP / Gradient / Attribution explanations to break down hazard predictions.
"""

from .explainability import (
    SevereWeatherShapExplainer,
    get_default_shap_explainer,
    generate_shap_bar_chart,
    explain_case_study_cell,
    compute_hazard_attributions,
    explain_prediction_summary,
)

__all__ = [
    "SevereWeatherShapExplainer",
    "get_default_shap_explainer",
    "generate_shap_bar_chart",
    "explain_case_study_cell",
    "compute_hazard_attributions",
    "explain_prediction_summary",
]

