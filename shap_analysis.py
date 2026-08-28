# ============================================================
#  shap_analysis.py  —  SHAP feature importance explanations
# ============================================================

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import FEATURES_CSV, FEATURE_COLS, TARGET_COL, MODELS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Load data & model
# ------------------------------------------------------------------

def load_data():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def compute_shap(model_name: str = "best_model"):
    df    = load_data()
    X     = df[FEATURE_COLS]

    # Load the inner model from pipeline (SHAP needs raw model not pipeline)
    pipeline = joblib.load(f"{MODELS_DIR}/{model_name}.pkl")
    scaler   = pipeline.named_steps["scaler"]
    model    = pipeline.named_steps["model"]
    X_scaled = scaler.transform(X)

    print(f"Computing SHAP values for: {model_name} ...")

    # Use a sample of 200 rows for speed
    sample = shap.sample(X_scaled, 200, random_state=42)

    # Use KernelExplainer — works for all model types
    explainer   = shap.KernelExplainer(model.predict, sample)
    shap_values = explainer.shap_values(sample, nsamples=100)

    return shap_values, sample, FEATURE_COLS


def save_shap_plots(model_name: str = "best_model"):
    shap_values, X_sample, feature_names = compute_shap(model_name)

    # 1. Bar plot — mean absolute SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name.replace('_',' ').title()}")
    plt.tight_layout()
    path1 = f"{MODELS_DIR}/shap_importance.png"
    plt.savefig(path1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path1}")

    # 2. Dot plot — shows direction of impact
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names,
                      show=False)
    plt.title(f"SHAP Summary — {model_name.replace('_',' ').title()}")
    plt.tight_layout()
    path2 = f"{MODELS_DIR}/shap_summary.png"
    plt.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path2}")

    return path1, path2


def get_shap_values_df(model_name: str = "best_model") -> pd.DataFrame:
    """Return mean absolute SHAP values as a DataFrame — used by dashboard."""
    shap_values, X_sample, feature_names = compute_shap(model_name)
    mean_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "Feature":    feature_names,
        "SHAP Value": mean_shap,
    }).sort_values("SHAP Value", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Running SHAP analysis on best model...")
    p1, p2 = save_shap_plots("best_model")
    print("\nTop feature importances:")
    print(get_shap_values_df("best_model").to_string(index=False))
    print(f"\nPlots saved to models/ folder")
