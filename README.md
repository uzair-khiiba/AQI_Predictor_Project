# 🌫️ AQI Predictor — Karachi

**Live Dashboard:** https://aqipredictor-jwzqttnrgueww7a6a5adm3.streamlit.app/

**GitHub:** https://github.com/MuzamilMM/aqi_predictor

An end-to-end machine learning pipeline that predicts Karachi's Air Quality Index (AQI) for the next 3 days. The system fetches real data every hour, retrains models every day, and serves predictions through a publicly deployed Streamlit dashboard — fully automated via GitHub Actions with zero manual work after initial setup.

---

## 📊 Live Results (June 6, 2026)

| Date | Day | Predicted AQI | Category |
|------|-----|--------------|----------|
| 2026-06-06 | Saturday | 92.9 | Moderate |
| 2026-06-07 | Sunday | 93.8 | Moderate |
| 2026-06-08 | Monday | 94.1 | Moderate |

**Current AQI:** 89 (Moderate) | **Best Model:** Ridge Regression | **RMSE:** 0.921 | **R²:** 0.9972

---

## 🎯 Project Choice & Goal

This repository contains the project I developed during my data science internship in the Shine Program at 10Pearls. 
The goal was to build a complete, production-ready machine learning system from scratch that targets a real-world problem. I chose to focus on Karachi's air quality because we were asked to predict the AQI of the city we live in.

---

## 🏗️ System Architecture

```
OpenMeteo Air Quality API  +  OpenMeteo Weather Archive API
                ↓
         fetch_data.py  ←── GitHub Actions (every hour)
                ↓
    MongoDB Atlas — raw_data collection (29,976 records)
                ↓
      feature_pipeline.py  ←── GitHub Actions (every hour)
                ↓
    MongoDB Atlas — features collection (1,235 daily rows)
                ↓
         train.py  ←── GitHub Actions (every day at midnight)
                ↓
          models/best_model.pkl
                ↓
    predict.py → dashboard.py → Streamlit Cloud (public URL)
```
## 📊 Exploratory Data Analysis & Feature Insights

Because this project was developed as a modular, script-based production pipeline rather than an isolated Jupyter notebook, the Exploratory Data Analysis (EDA) phase was integrated directly into the system architecture and live dashboard:

* **Trend Identification & Seasonality:** During initial data exploration, strong annual pollution cycles and daily variations were identified. To capture these trends mathematically without overfitting, cyclical sine/cosine encodings (`day_of_year_sin`, `day_of_year_cos`) and specific monthly/weekend calendar features were engineered into the pipeline.
* **Feature Correlations:** Deep analysis of historical data revealed that yesterday's air quality has the highest correlation with today's conditions. This insight drove the creation of the autoregressive lag features (`aqi_lag1` through `aqi_lag14`) and rolling averages to capture short-term atmospheric momentum.
* **Post-Model Explainability (SHAP):** Instead of static offline EDA plots, **SHAP (SHapley Additive exPlanations)** is utilized to dynamically interpret how features impact predictions. This allows us to see exactly how weather variations and rolling trends affect Karachi's AQI values across different seasons.
* **Live Visualizations:** Actual-vs-predicted validation curves (for the last 60 days) and long-term historical AQI trends from January 2023 to the present day are served interactively on the live Streamlit dashboard for continuous analytical tracking.

---

## 🗂️ Project Structure

```
aqi_predictor/
├── config.py                        # Central settings — coordinates, MongoDB, 18 feature columns
├── fetch_data.py                    # Fetches AQI + weather from OpenMeteo, stores in MongoDB
├── feature_pipeline.py              # Engineers 18 features, saves to MongoDB feature store
├── train.py                         # Trains 4 ML models, saves best model
├── predict.py                       # Recursive 3-day forecast with weather features
├── dashboard.py                     # Streamlit dashboard (deployed on Streamlit Cloud)
├── alerts.py                        # AQI health alert system
├── shap_analysis.py                 # SHAP feature importance analysis
├── run_pipeline.py                  # Master script to run full pipeline locally
├── requirements.txt
├── .github/workflows/
│   ├── feature_pipeline.yml         # Runs every hour
│   └── training_pipeline.yml        # Runs every day at midnight
└── models/
    ├── best_model.pkl
    ├── metadata.json
    ├── model_comparison.png
    ├── shap_importance.png          # SHAP feature importance bar chart
    └── shap_summary.png             # SHAP summary dot plot
```

---

## 📦 Tech Stack

| Category | Technology |
|----------|-----------|
| AQI Data | OpenMeteo Air Quality API (free, no API key) |
| Weather Data | OpenMeteo Archive + Forecast API |
| Feature Store | MongoDB Atlas — GCP us-east-1 |
| ML Framework | Scikit-learn |
| CI/CD | GitHub Actions (hourly + daily) |
| Dashboard | Streamlit Cloud (public deployment) |
| Explainability | SHAP KernelExplainer |
| Language | Python 3.11 |

---

## 🔧 Features Used (18 Total)

| Feature | Type | Description |
|---------|------|-------------|
| aqi_lag1, lag2, lag3, lag7, lag14 | Lag | Past AQI values (1, 2, 3, 7, 14 days ago) |
| aqi_roll3, aqi_roll7 | Rolling | 3-day and 7-day rolling averages |
| aqi_change_rate | Derived | Day-over-day AQI trend (rising or falling) |
| day_of_week, month, is_weekend | Time | Calendar-based features |
| day_of_year_sin, day_of_year_cos | Seasonal | Cyclical encoding — captures annual pollution cycles |
| temperature_2m | Weather | Daily average temperature |
| relative_humidity_2m | Weather | Daily average humidity |
| wind_speed_10m | Weather | Daily average wind speed |
| precipitation | Weather | Daily total rainfall |
| surface_pressure | Weather | Daily average atmospheric pressure |

---

### 📈 Model Evaluation & Performance

Model selection and final evaluation were strictly driven by a chronological holdout test set split to simulate real-world production forecasting performance on unseen future data. 

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Ridge Regression** | **0.921** | **0.684** | **0.9972** |
| Gradient Boosting | 3.506 | 2.114 | 0.9610 |
| Random Forest | 4.208 | 2.894 | 0.9421 |
| Lasso Regression | 5.112 | 3.941 | 0.9103 |

- **Training data:** 1,235 daily rows (January 2023 — June 2026)
- **Train/Test split:** 988 training / 247 test rows (chronological, no shuffling)
- **Cross-validation:** TimeSeriesSplit with 5 folds
> 💡 **Validation Strategy Note:** Because this forecasting system relies heavily on autoregressive lag features (`aqi_lag1` to `aqi_lag14`), standard cross-validation folds are bypassed to prevent temporal distortion. Evaluation on the sequential chronological test split ensures that metrics genuinely reflect production-grade stability and reliability.

---

## 🔮 Forecasting Approach

Recursive multi-step prediction:
1. Day+1 predicted using real historical lags + tomorrow's weather forecast from OpenMeteo
2. Day+1 prediction becomes lag1 for Day+2
3. Day+2 prediction becomes lag1 for Day+3

Each day uses its own weather forecast — producing different predictions per day.

**Before weather features:** 93.9, 93.9, 93.9 (identical — not useful)
**After weather features:** 92.9, 93.8, 94.1 (different — Useful)

---

## 🔍 SHAP Feature Importance

SHAP (SHapley Additive exPlanations) was used to explain what drives model predictions.

| Feature | SHAP Value | Interpretation |
|---------|-----------|----------------|
| aqi_lag1 | 25.34 | Yesterday's AQI is the strongest single predictor |
| aqi_change_rate | 13.84 | Rising vs falling trend matters more than the absolute value |
| aqi_roll3 | 8.37 | Recent 3-day average adds useful context |
| Weather features | varies | Temperature and pressure affect AQI differently by season |

**Key finding:** Day of week and is_weekend scored near zero — Karachi's AQI has no weekly pattern unlike traffic-driven cities. SHAP plots saved in `models/shap_importance.png` and `models/shap_summary.png`.

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| Feature Pipeline | Every hour | Fetches new AQI from OpenMeteo, fetches weather, rebuilds 18 features, saves to MongoDB |
| Training Pipeline | Daily at midnight | Retrains all 4 models on latest data, saves best model as GitHub artifact |

**33+ successful workflow runs** visible in the Actions tab.

MongoDB URI stored as GitHub Actions Secret and Streamlit Cloud Secret — never written in source code.

---

## 🚧 Challenges Faced and How They Were Solved

### 1. Data Leakage (R² = 0.9997 — Too Perfect)
* **The Problem:** Initially, I set up the pipeline to train on hourly data, which made the model's `lag1` feature look at "1 hour ago." Because air quality rarely changes drastically from one hour to the next, the model learned a shortcut: it just copied the previous hour's value instead of actually learning the underlying patterns. This resulted in a suspiciously perfect R² score of 0.9997.
* **The Solution:** I fixed this by switching the feature pipeline to daily aggregation, averaging the 24 hourly readings into a single daily average. This changed `lag1` to mean "yesterday's average," turning it into a genuine 24-hour prediction challenge. After this change, the R² reduced slightly to a much more reliable 0.9972, which makes sense given that our large dataset size helps the model capture strong, genuine historical patterns.

### 2. Identical 3-Day Predictions
* **The Problem:** In my first forecasting attempts, the predictions for all three forward days came out almost identical (e.g., 93.9, 93.9, 93.9). I discovered the root cause was that only the day-of-week feature was changing between day 1, day 2, and day 3. The model had rightly learned that the day of the week has very little correlation with Karachi's overall pollution, so it outputted a flat baseline.
* **The Solution:** To make the forecasts dynamic, I integrated OpenMeteo’s free 3-day weather forecast directly into the prediction loop as input features. Now, because each day passes different forecasted values for temperature, humidity, wind speed, and pressure, the model successfully reflects changing atmospheric conditions for each specific day.

### 3. MongoDB SSL Error on GitHub Actions
* **The Problem:** While testing the automation, my GitHub Actions runner consistently failed to connect to MongoDB Atlas, throwing a silent TLS/SSL handshake error. After a lot of troubleshooting, I realized the issue was geographical: my initial MongoDB cluster was hosted in the Mumbai region (ap-south-1), while GitHub’s automated runners operate out of AWS us-east-1, causing a network mismatch. 
* **The Solution:** I resolved this by spinning up a brand new MongoDB cluster on Google Cloud in the us-east-1 region to physically align it with GitHub’s servers. I also added `0.0.0.0/0` to the MongoDB Network Access list since GitHub Actions uses dynamic IP addresses that change with every run.

### 4. Credentials Accidentally Committed to Code
* **The Problem:** Early in development, I accidentally left my live MongoDB connection string—including the plaintext database password—hardcoded inside `config.py`. As soon as I pushed the code, GitHub's automated security scanner flags caught the exposure and blocked the push.
* **The Solution:** I immediately rotated and changed the database password on MongoDB Atlas. Then, I cleaned out the hardcoded string from my local scripts and git history, moving all sensitive credentials over to GitHub Actions Secrets and Streamlit Cloud Secrets. The `config.py` file was updated to pull securely using `os.environ.get("MONGO_URI", "")`, ensuring safe, credential-free code.

### 5. Choosing the Right Data Source
* **The Problem:** Finding a reliable data source was tricky. I initially spent time registering for platforms like AQICN and OpenWeather, only to discover deep into testing that their historical data access required expensive paid subscriptions. 
* **The Solution:** While looking for options, I saw a discussion on Discord where another student mentioned using the OpenMeteo API. I decided to try it out and found it to be a perfect fit. It is completely free, does not require an API key, and lets me pull historical data all the way back to 2020. Because it provides both the Air Quality and Weather Forecast data through simple endpoints, it served as the perfect single-source solution for the entire pipeline.

## 📈 Dashboard Features

**Live at:** https://aqipredictor-jwzqttnrgueww7a6a5adm3.streamlit.app/

- Live AQI gauge with current reading
- Health alert banners — color-coded by AQI category with action advice
- 3-day forecast bar chart — different color per AQI category
- Forecast cards — day name, date, predicted AQI, category
- Model validation chart — actual vs predicted for last 60 days (MAE, RMSE, R²)
- Historical AQI chart — Karachi from January 2023 to today
- All-models comparison — side-by-side forecast from all 4 models
- Model performance table — RMSE, MAE, R², CV RMSE for all 4 models
- SHAP feature importance plots

---

## 🚨 AQI Scale

| AQI | Category | Health Action |
|-----|----------|--------------|
| 0–50 | Good | No action needed |
| 51–100 | Moderate | Sensitive people reduce outdoor activity |
| 101–150 | Unhealthy for Sensitive Groups | Sensitive groups stay indoors |
| 151–200 | Unhealthy | Everyone reduce outdoor activity |
| 201–300 | Very Unhealthy | Everyone stay indoors |
| 301–500 | Hazardous | Health emergency |

---

## 🛠️ Local Setup

```bash
pip install -r requirements.txt

$env:MONGO_URI = "your_mongodb_connection_string"

python fetch_data.py
python feature_pipeline.py
python train.py
python predict.py
streamlit run dashboard.py
```

---

## 🔭 Future Improvements

- Add XGBoost/LightGBM for potentially better performance
- Show prediction confidence intervals instead of single-point estimates

---

*Data: OpenMeteo · City: Karachi, Pakistan (24.8607°N, 67.0011°E) · Updated: June 2026*