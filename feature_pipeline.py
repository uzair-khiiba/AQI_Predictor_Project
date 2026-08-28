import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
import certifi
from config import HISTORICAL_CSV, MONGO_URI, MONGO_DB, FEATURES_COLLECTION, FEATURE_COLS, TARGET_COL, WEATHER_COLS


def aqi_category(value):
    if value <= 50:
        return "Good", "#00e400"
    elif value <= 100:
        return "Moderate", "#ffff00"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif value <= 200:
        return "Unhealthy", "#ff0000"
    elif value <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


def get_collection():
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=30000
    )
    client.admin.command("ping")
    return client[MONGO_DB][FEATURES_COLLECTION]


def build_features(df):
    from fetch_data import fetch_historical_weather

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date").agg(aqi=("aqi", "mean")).reset_index()
    daily["datetime"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("datetime").reset_index(drop=True)

    daily["aqi"] = daily["aqi"].ffill(limit=3)

    # --- merge historical weather ---
    try:
        weather = fetch_historical_weather()
        weather["date"] = pd.to_datetime(weather["date"]).dt.date
        daily = daily.merge(weather, on="date", how="left")
        for col in WEATHER_COLS:
            daily[col] = daily[col].ffill(limit=3)
        print("Historical weather merged successfully")
    except Exception as e:
        print(f"Weather fetch failed: {e}. Filling weather cols with 0.")
        for col in WEATHER_COLS:
            daily[col] = 0.0

    daily["day_of_week"] = daily["datetime"].dt.dayofweek
    daily["month"]       = daily["datetime"].dt.month
    daily["is_weekend"]  = (daily["day_of_week"] >= 5).astype(int)

    # seasonal cyclical features
    doy = daily["datetime"].dt.dayofyear
    daily["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
    daily["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)

    daily["aqi_lag1"]  = daily["aqi"].shift(1)
    daily["aqi_lag2"]  = daily["aqi"].shift(2)
    daily["aqi_lag3"]  = daily["aqi"].shift(3)
    daily["aqi_lag7"]  = daily["aqi"].shift(7)
    daily["aqi_lag14"] = daily["aqi"].shift(14)

    daily["aqi_roll3"] = daily["aqi"].shift(1).rolling(3).mean()
    daily["aqi_roll7"] = daily["aqi"].shift(1).rolling(7).mean()

    daily["aqi_change_rate"] = daily["aqi"].diff()

    daily = daily.dropna(subset=FEATURE_COLS + [TARGET_COL])
    daily = daily.reset_index(drop=True)

    print(f"Feature engineering done: {len(daily)} daily rows")
    return daily


def save_features(df):
    try:
        collection = get_collection()
        collection.drop()
        records = df.to_dict("records")
        for r in records:
            r["datetime"] = str(r["datetime"])
            r.pop("date", None)
        collection.insert_many(records)
        print(f"Saved {len(records)} feature rows to MongoDB")
    except Exception as e:
        print(f"MongoDB save failed: {e}. Saving to CSV only.")
        df.to_csv("data/karachi_features.csv", index=False)


def load_features():
    try:
        collection = get_collection()
        records    = list(collection.find({}, {"_id": 0}))
        df         = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"Loaded {len(df)} rows from MongoDB")
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        print(f"MongoDB load failed: {e}. Using CSV fallback.")
        df = pd.read_csv("data/karachi_features.csv", parse_dates=["datetime"])
        print(f"Loaded {len(df)} rows from CSV")
        return df.sort_values("datetime").reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(HISTORICAL_CSV)
    print(f"Loaded {len(raw)} raw records")
    features_df = build_features(raw)
    save_features(features_df)
    print(f"\nShape: {features_df.shape}")
    print(f"\nCorrelation with AQI:")
    corr = features_df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    print(corr.abs().sort_values(ascending=False).round(3))