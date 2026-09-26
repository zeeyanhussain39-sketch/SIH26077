"""
Explainable AI (XAI) Module for Severe Weather Nowcasting.
Uses SHAP / Gradient / Attribution explanations to break down hazard predictions.
"""

from .explainability import compute_hazard_attributions, explain_prediction_summary

__all__ = [
    "compute_hazard_attributions",
    "explain_prediction_summary",
]
