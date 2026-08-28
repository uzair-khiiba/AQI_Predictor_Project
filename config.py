import os

CITY_NAME = "Karachi"
LAT       = 24.8607
LON       = 67.0011

HISTORICAL_START    = "2025-01-01"
DATA_DIR            = "data"
HISTORICAL_CSV      = f"{DATA_DIR}/karachi_historical.csv"
FEATURES_CSV = f"{DATA_DIR}/karachi_features.csv"
MONGO_URI           = os.environ.get("MONGO_URI", "")
MONGO_DB            = "aqi_predictor"
RAW_COLLECTION      = "raw_data"
FEATURES_COLLECTION = "features"

FEATURE_COLS = [
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "aqi_lag7", "aqi_lag14",
    "aqi_roll3", "aqi_roll7",
    "aqi_change_rate",
    "day_of_week", "month", "is_weekend",
    "day_of_year_sin", "day_of_year_cos",
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "precipitation",
    "surface_pressure",
]

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "precipitation",
    "surface_pressure",
]

TARGET_COL    = "aqi"
MODELS_DIR    = "models"
FORECAST_DAYS = 3
TEST_SIZE     = 0.2
RANDOM_STATE  = 42