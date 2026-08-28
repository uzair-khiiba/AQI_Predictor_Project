# ============================================================
#  run_pipeline.py  —  Run the entire AQI pipeline end-to-end
#  Usage: python run_pipeline.py [--skip-fetch] [--skip-train]
# ============================================================

import argparse, logging, sys, os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log"),
    ]
)
log = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)


def step_banner(title: str):
    log.info("")
    log.info("=" * 55)
    log.info(f"  STEP: {title}")
    log.info("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="AQI Prediction Pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching (use existing CSV)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing models)")
    args = parser.parse_args()

    # ── Step 1: Fetch historical data ────────────────────────────
    if not args.skip_fetch:
        step_banner("Fetching Historical Data")
        from fetch_data import fetch_historical
        df = fetch_historical()
        log.info(f"  Fetched {len(df)} records  [{df['date'].min()} → {df['date'].max()}]")
    else:
        log.info("SKIPPED: data fetch")

    # ── Step 2: Feature engineering ──────────────────────────────
    step_banner("Building Features")
    import pandas as pd
    from config           import HISTORICAL_CSV, FEATURES_CSV
    from feature_pipeline import build_features

    raw_df  = pd.read_csv(HISTORICAL_CSV)
    feat_df = build_features(raw_df)
    feat_df.to_csv(FEATURES_CSV, index=False)
    log.info(f"  Feature matrix: {feat_df.shape}  → {FEATURES_CSV}")

    # ── Step 3: Train models ──────────────────────────────────────
    if not args.skip_train:
        step_banner("Training Models")
        from train import train_all, print_summary
        meta = train_all()
        print_summary(meta)
    else:
        log.info("SKIPPED: model training")
        from train    import load_metadata
        from config   import MODELS_DIR
        import json
        with open(f"{MODELS_DIR}/metadata.json") as f:
            meta = json.load(f)

    # ── Step 4: Predict next 3 days ───────────────────────────────
    step_banner("Generating 3-Day Forecast")
    from predict import load_best_model, forecast_next_days, forecast_all_models, print_forecast
    from config  import HISTORICAL_CSV

    history      = pd.read_csv(HISTORICAL_CSV, parse_dates=["date"])
    best_model   = load_best_model()
    forecasts    = forecast_next_days(best_model, history)
    print_forecast(forecasts, meta["best_model"])

    log.info("")
    log.info("All-model comparison:")
    all_f = forecast_all_models(history)
    for mname, fc in all_f.items():
        preds = "  |  ".join([f"{d['date']} PM2.5={d['pm25']}" for d in fc])
        log.info(f"  [{mname}]  {preds}")

    # ── Step 5: Launch dashboard ──────────────────────────────────
    log.info("")
    log.info("=" * 55)
    log.info("  Pipeline complete! Launch the dashboard with:")
    log.info("    streamlit run dashboard.py")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
