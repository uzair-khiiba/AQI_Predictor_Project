import os
import json
import joblib
import glob
import traceback
import numpy as np
import pandas as pd
from datetime import timedelta

from config           import MODELS_DIR, FEATURE_COLS, FORECAST_DAYS
from feature_pipeline import aqi_category, load_features


def load_best_model():
    path = f"{MODELS_DIR}/best_model.pkl"
    try:
        return joblib.load(path)
    except Exception as e:
        print("\n!!! CRITICAL PICKLE ERROR LOADING BEST MODEL !!!")
        print(f"Error Message: {str(e)}")
        print("--- FULL ERROR TRACEBACK TO UNCOVER HIDDEN MODULE ---")
        traceback.print_exc()
        print("----------------------------------------------------\n")
        raise e


def load_all_models():
    models = {}
    for path in glob.glob(f"{MODELS_DIR}/*.pkl"):
        name = os.path.basename(path).replace(".pkl", "").replace("_", " ").title()
        if name == "Best Model":
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            print(f"\n!!! CRITICAL PICKLE ERROR LOADING MODEL '{name}' !!!")
            print(f"Error Message: {str(e)}")
            print("--- FULL ERROR TRACEBACK TO UNCOVER HIDDEN MODULE ---")
            traceback.print_exc()
            print("----------------------------------------------------\n")
            raise e
    return models


def load_metadata():
    with open(f"{MODELS_DIR}/metadata.json") as f:
        return json.load(f)


def _build_feature_row(df, target_date):
    """
    Dynamically slices context histories and generates ALL necessary ML features,
    including weather metrics, lags, and seasonal curves to match training features.
    """
    day_of_year = target_date.timetuple().tm_yday
    
    return {
        "aqi_lag1":        df["aqi"].iloc[-1],
        "aqi_lag2":        df["aqi"].iloc[-2] if len(df) > 1 else df["aqi"].iloc[-1],
        "aqi_lag3":        df["aqi"].iloc[-3] if len(df) > 2 else df["aqi"].iloc[-1],
        "aqi_lag7":        df["aqi"].iloc[-7] if len(df) > 6 else df["aqi"].iloc[-1],
        "aqi_lag14":       df["aqi"].iloc[-14] if len(df) > 13 else df["aqi"].iloc[-1],
        
        "aqi_roll3":       df["aqi"].iloc[-3:].mean(),
        "aqi_roll7":       df["aqi"].iloc[-7:].mean(),
        "aqi_change_rate": df["aqi"].iloc[-1] - df["aqi"].iloc[-2] if len(df) > 1 else 0,
        
        "day_of_week":     target_date.dayofweek,
        "month":           target_date.month,
        "is_weekend":      int(target_date.dayofweek >= 5),
        
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.0),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.0),
        
        "temperature_2m":       df["temperature_2m"].iloc[-1] if "temperature_2m" in df.columns else 0.0,
        "relative_humidity_2m": df["relative_humidity_2m"].iloc[-1] if "relative_humidity_2m" in df.columns else 0.0,
        "wind_speed_10m":       df["wind_speed_10m"].iloc[-1] if "wind_speed_10m" in df.columns else 0.0,
        "precipitation":        df["precipitation"].iloc[-1] if "precipitation" in df.columns else 0.0,
        "surface_pressure":     df["surface_pressure"].iloc[-1] if "surface_pressure" in df.columns else 0.0
    }


def forecast_next_days(model, features_df=None, days=FORECAST_DAYS):
    """Generates recursive forward predictions, passing features_df directly to prevent database stalls."""
    if features_df is None:
        features_df = load_features()
        
    df = features_df.sort_values("datetime").reset_index(drop=True)
    last_date = pd.Timestamp(df["datetime"].iloc[-1])
    forecasts = []

    for step in range(1, days + 1):
        forecast_date = last_date + timedelta(days=step)
        fv            = _build_feature_row(df, forecast_date)
        
        X             = np.array([[fv[c] for c in FEATURE_COLS]])
        aqi_pred      = max(0, float(model.predict(X)[0]))
        label, color  = aqi_category(aqi_pred)

        new_row = fv.copy()
        new_row["datetime"] = forecast_date
        new_row["aqi"] = aqi_pred
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        forecasts.append({
            "date":     forecast_date.strftime("%Y-%m-%d"),
            "day":      forecast_date.strftime("%A"),
            "aqi":      round(aqi_pred, 1),
            "category": label,
            "color":    color,
        })

    return forecasts


def forecast_all_models(features_df=None, days=FORECAST_DAYS):
    """Reuses the single feature frame context across every model instance."""
    if features_df is None:
        features_df = load_features()
        
    models = load_all_models()
    return {name: forecast_next_days(model, features_df, days) for name, model in models.items()}


if __name__ == "__main__":
    shared_features = load_features()
    best_model = load_best_model()
    meta       = load_metadata()
    forecasts  = forecast_next_days(best_model, shared_features)
    
    print(f"\n{'='*55}")
    print(f"  3-DAY AQI FORECAST - Karachi  [{meta['best_model']}]")
    print(f"{'='*55}")
    for f in forecasts:
        print(f"  {f['date']}  {f['day']:<10}  AQI={f['aqi']:>6.1f}  {f['category']}")
    print(f"{'='*55}\n")

    print("All-model comparison:")
    for mname, fc in forecast_all_models(shared_features).items():
        print(f"\n  [{mname}]")
        for day in fc:
            print(f"    {day['date']}  AQI={day['aqi']}  {day['category']}")