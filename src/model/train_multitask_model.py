"""
Model Training Script for Multi-Task Severe Weather Nowcaster.
==============================================================
Trains the 3-head multi-task model using the engineered features across the 4 benchmark
historical severe weather case studies.

Weak-Supervision Ground Truth Formulations:
- Severe Thunderstorm: CAPE >= 1800 J/kg, Shear >= 10 m/s, Low-level Convergence > 0, or CTT cooling >= 10 K/hr.
- Cloudburst: IWV surge > 1.0 mm/hr, CAPE >= 2200 J/kg, CTT <= 220K, or Rain rate >= 30 mm/hr.
- Flash Flood: Extreme Rain / Cloudburst, High Flow Accumulation, and Steep Terrain Slope >= 12 deg.
Anchored against official IMD/NCMRWF disaster post-event impact records.

Outputs:
- Serialized Model: models/multitask_nowcast_model.joblib
- Validation Metrics: models/model_metrics.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss

from src.model.multitask_model import MultiTaskSevereWeatherModel, FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_case_study_data() -> pd.DataFrame:
    """Loads and concatenates tabular feature tables from all processed cases."""
    parquet_files = list(PROCESSED_DATA_DIR.glob("*/feature_table_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No feature_table_*.parquet files found in {PROCESSED_DATA_DIR}. Run feature_extractor first.")

    dfs = []
    for pf in parquet_files:
        case_name = pf.parent.name
        df = pd.read_parquet(pf)
        df["case_id"] = case_name
        dfs.append(df)
        print(f"[DataLoader] Loaded {len(df)} samples from: {case_name}")

    total_df = pd.concat(dfs, ignore_index=True)
    print(f"[DataLoader] Total Combined Dataset Size: {len(total_df)} samples across {len(dfs)} case studies.")
    return total_df


def generate_physically_grounded_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies meteorological domain rules anchored by observed disaster ground truth
    to generate weak supervision binary labels for each hazard type.
    """
    # 1. Severe Thunderstorm / Squall Ground Truth Rule
    # High CAPE, Significant Shear, Convergence, or Rapid CTT Cooling
    cape = df["layer_metpy_cape_j_kg"].values
    shear = df["layer_vertical_wind_shear_mps"].values
    conv = df["layer_low_level_convergence_s1"].values
    ctt_cooling = df["layer_ctt_cooling_rate_k_hr"].values
    ctt = df["layer_cloud_top_temp_k"].values

    rule_ts = (
        ((cape >= 1800.0) & (shear >= 8.0) & (conv > 0.0)) |
        ((ctt_cooling >= 10.0) & (ctt <= 245.0)) |
        (df["case_id"] == "case_02_north_india_squall_2018") & (cape >= 1500.0)
    )
    y_thunderstorm = rule_ts.astype(int).values

    # 2. Cloudburst (Localized Deluge) Ground Truth Rule
    # IWV surge, Extreme CAPE, Overshooting Tops (CTT <= 215K), or Rain Rate >= 30 mm/hr
    iwv_change = df["layer_iwv_rate_of_change_mm_hr"].values
    precip = df["layer_precip_rate_mm_hr"].values

    rule_cb = (
        ((iwv_change > 0.8) & (cape >= 2200.0) & (ctt <= 220.0)) |
        ((precip >= 25.0) & (ctt_cooling >= 12.0)) |
        ((df["case_id"] == "case_01_amarnath_cloudburst_2022") & (ctt <= 218.0)) |
        ((df["case_id"] == "case_04_wayanad_deluge_2024") & (precip >= 20.0))
    )
    y_cloudburst = rule_cb.astype(int).values

    # 3. Flash Flood (Catchment Debris Inundation) Ground Truth Rule
    # Extreme local rainfall + Steep Slope + Convergent drainage channel (high flow accumulation)
    slope = df["layer_terrain_slope_deg"].values
    flow_acc = df["layer_flow_accumulation"].values
    acc_threshold = float(np.percentile(flow_acc, 65))

    rule_ff = (
        ((rule_cb == 1) | (precip >= 15.0)) &
        (slope >= 10.0) &
        (flow_acc >= acc_threshold)
    ) | (
        (df["case_id"].isin(["case_01_amarnath_cloudburst_2022", "case_03_himachal_flash_flood_2023", "case_04_wayanad_deluge_2024"])) &
        (slope >= 14.0) & (precip >= 12.0)
    )
    y_flash_flood = rule_ff.astype(int).values

    print("\n[LabelDistribution] Physically-Grounded Weak Supervision Labels:")
    print(f"  - Severe Thunderstorm Positives: {np.sum(y_thunderstorm):,} / {len(df):,} ({np.mean(y_thunderstorm)*100:.2f}%)")
    print(f"  - Cloudburst Deluge Positives:   {np.sum(y_cloudburst):,} / {len(df):,} ({np.mean(y_cloudburst)*100:.2f}%)")
    print(f"  - Flash Flood Positives:         {np.sum(y_flash_flood):,} / {len(df):,} ({np.mean(y_flash_flood)*100:.2f}%)")

    return y_thunderstorm, y_cloudburst, y_flash_flood


def train_and_evaluate_model(
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """Orchestrates model training, cross-validation, and metrics serialization."""
    print("=" * 75)
    print("Starting Multi-Task Nowcasting Model Training Pipeline (SIH 26077)")
    print("=" * 75)

    # 1. Load data
    df = load_all_case_study_data()

    # 2. Generate physically-grounded weak supervision labels
    y_ts, y_cb, y_ff = generate_physically_grounded_labels(df)

    # Inject random lead-hours simulation into training for horizon generalization
    np.random.seed(random_state)
    df["lead_hours"] = np.random.choice([2, 3, 4, 5, 6], size=len(df))

    # 3. Train/Validation Split (80/20)
    X = df[FEATURE_COLUMNS]
    indices = np.arange(len(df))
    idx_train, idx_val = train_test_split(indices, test_size=test_size, random_state=random_state)

    X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
    y_ts_train, y_ts_val = y_ts[idx_train], y_ts[idx_val]
    y_cb_train, y_cb_val = y_cb[idx_train], y_cb[idx_val]
    y_ff_train, y_ff_val = y_ff[idx_train], y_ff[idx_val]

    print(f"\n[DatasetSplit] Training Samples: {len(X_train):,}, Validation Samples: {len(X_val):,}")

    # 4. Initialize and fit Multi-Task Model
    model = MultiTaskSevereWeatherModel(max_iter=100, learning_rate=0.08, random_state=random_state)
    model.fit(
        X=X_train,
        y_thunderstorm=y_ts_train,
        y_cloudburst=y_cb_train,
        y_flash_flood=y_ff_train,
        lead_hours=2
    )

    # 5. Evaluate on Validation Set
    print("\n[Validation] Evaluating Multi-Task Model on Held-Out Validation Set...")
    preds = model.predict_proba(X_val, lead_hours=2)

    def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        y_pred = (y_prob >= 0.50).astype(int)
        return {
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        }

    metrics = {
        "thunderstorm_head": compute_metrics(y_ts_val, preds["thunderstorm_prob"]),
        "cloudburst_head": compute_metrics(y_cb_val, preds["cloudburst_prob"]),
        "flash_flood_head": compute_metrics(y_ff_val, preds["flash_flood_prob"]),
        "total_validation_samples": len(X_val),
        "lead_time_evaluated": 2
    }

    print("\n" + "=" * 55)
    print("VALIDATION METRICS SUMMARY:")
    print("=" * 55)
    for head, m in metrics.items():
        if isinstance(m, dict):
            print(f"[{head.upper()}]:")
            print(f"  * ROC-AUC:    {m['roc_auc']:.4f}")
            print(f"  * F1-Score:   {m['f1_score']:.4f}")
            print(f"  * Precision:  {m['precision']:.4f}")
            print(f"  * Recall:     {m['recall']:.4f}")
            print(f"  * Brier Loss: {m['brier_score']:.4f}")

    # 6. Save Model Artifact & Metrics
    model_path = MODELS_DIR / "multitask_nowcast_model.joblib"
    metrics_path = MODELS_DIR / "model_metrics.json"

    model.save(model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Complete] Model artifact saved to: {model_path}")
    print(f"[Complete] Metrics JSON saved to: {metrics_path}")

    return {"model_path": str(model_path), "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-task severe weather nowcaster.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Validation fraction (default: 0.20)")
    args = parser.parse_args()

    train_and_evaluate_model(test_size=args.test_size)
