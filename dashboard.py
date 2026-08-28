import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from config           import MODELS_DIR, HISTORICAL_CSV, CITY_NAME, FEATURE_COLS, TARGET_COL
from fetch_data       import fetch_live
from feature_pipeline import aqi_category
from predict          import load_best_model, load_all_models, load_metadata, forecast_next_days, forecast_all_models
from alerts           import get_alert, check_forecast_alerts

st.set_page_config(page_title=f"AQI Forecaster — {CITY_NAME}", page_icon="🌫️", layout="wide")

st.markdown("""
<style>
.forecast-card { background: #2a2a3e; border-radius: 10px; padding: 20px; text-align: center; }
.aqi-value { font-size: 3rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_history():
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_resource
def load_model_meta():
    meta   = load_metadata()
    model  = load_best_model()
    models = load_all_models()
    return model, models, meta


@st.cache_data(ttl=600)
def get_live():
    try:
        return fetch_live()
    except:
        return None


def sidebar():
    st.sidebar.header("⚙️ Settings")
    model_choice    = st.sidebar.selectbox("Forecast model",
        ["Best (auto)", "Ridge Regression", "Lasso Regression", "Random Forest", "Gradient Boosting"])
    show_all        = st.sidebar.checkbox("Compare all models", value=False)
    show_validation = st.sidebar.checkbox("Show model validation", value=True)
    show_shap       = st.sidebar.checkbox("Show SHAP analysis", value=False)
    refresh         = st.sidebar.button("🔄 Refresh")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**City:** {CITY_NAME}")
    st.sidebar.markdown(f"**Source:** OpenMeteo API")
    st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return model_choice, show_all, show_validation, show_shap, refresh


def gauge(value):
    label, color = aqi_category(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"Current AQI<br><span style='font-size:0.85em;color:{color}'>{label}</span>", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 500]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,   50],  "color": "#00e400"},
                {"range": [50,  100], "color": "#ffff00"},
                {"range": [100, 150], "color": "#ff7e00"},
                {"range": [150, 200], "color": "#ff0000"},
                {"range": [200, 300], "color": "#8f3f97"},
                {"range": [300, 500], "color": "#7e0023"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def forecast_chart(forecasts, title):
    fig = go.Figure(go.Bar(
        x=[f["date"] for f in forecasts],
        y=[f["aqi"]  for f in forecasts],
        marker_color=[f["color"] for f in forecasts],
        text=[f"{f['aqi']}<br>{f['category']}" for f in forecasts],
        textposition="outside",
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="AQI",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", yaxis=dict(gridcolor="rgba(255,255,255,0.1)"))
    return fig


def historical_chart(df):
    daily = df.copy()
    daily["date"] = daily["datetime"].dt.date
    daily = daily.groupby("date")["aqi"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["aqi"], mode="lines",
        name="AQI", line=dict(color="#4fc3f7", width=1.5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.1)"))

    for val, color, label in [(50,"#00e400","Good"),(100,"#ffff00","Moderate"),
                               (150,"#ff7e00","USG"),(200,"#ff0000","Unhealthy")]:
        fig.add_hline(y=val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right",
                      line_width=0.8, opacity=0.6)
    fig.update_layout(title="Historical AQI — Karachi (Daily Average)",
        xaxis_title="Date", yaxis_title="AQI", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"))
    return fig


def validation_chart(model):
    from feature_pipeline import load_features
    df = load_features().dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    df = df.sort_values("datetime").tail(60)
    df["predicted"] = model.predict(df[FEATURE_COLS]).clip(min=0)
    mae  = round((df[TARGET_COL] - df["predicted"]).abs().mean(), 2)
    rmse = round(((df[TARGET_COL] - df["predicted"]) ** 2).mean() ** 0.5, 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df[TARGET_COL],
        mode="lines", name="Actual AQI", line=dict(color="#4fc3f7", width=2)))
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["predicted"],
        mode="lines", name="Predicted AQI", line=dict(color="#FF5722", width=2, dash="dash")))
    fig.update_layout(
        title=f"Model Validation — Last 60 Days (MAE={mae}  |  RMSE={rmse})",
        xaxis_title="Date", yaxis_title="AQI", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.1)"))
    return fig, mae, rmse


def comparison_chart(all_forecasts):
    colors = {"Lasso Regression":"#4fc3f7","Ridge Regression":"#81c784",
              "Random Forest":"#ffb74d","Gradient Boosting":"#f06292"}
    fig = go.Figure()
    for name, fc in all_forecasts.items():
        fig.add_trace(go.Scatter(x=[f["date"] for f in fc], y=[f["aqi"] for f in fc],
            mode="lines+markers", name=name,
            line=dict(color=colors.get(name,"#fff"), width=2), marker=dict(size=8)))
    fig.update_layout(title="All Models — 3-Day AQI Forecast",
        xaxis_title="Date", yaxis_title="AQI", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"))
    return fig


def main():
    model_choice, show_all, show_validation, show_shap, refresh = sidebar()

    st.title(f"🌫️ AQI Forecaster — {CITY_NAME}")
    st.markdown("**3-day Air Quality Index prediction for Karachi**")
    st.markdown("---")

    if show_shap:
        st.subheader("🔍 SHAP Feature Importance")
        c_shap1, c_shap2 = st.columns(2)
        with c_shap1:
            st.image("models/shap_importance.png", caption="Feature Importance")
        with c_shap2:
            st.image("models/shap_summary.png", caption="SHAP Summary")
        st.markdown("---")

    history = load_history()
    best_model, all_models, meta = load_model_meta()

    if refresh:
        st.cache_data.clear()

    live = get_live()

    if model_choice == "Best (auto)":
        selected_model = best_model
        selected_name  = meta["best_model"]
    else:
        selected_model = all_models.get(model_choice, best_model)
        selected_name  = model_choice

    forecasts     = forecast_next_days(selected_model)
    all_forecasts = forecast_all_models() if show_all else {}

    latest      = history.sort_values("datetime").iloc[-1]
    current_aqi = live["aqi"] if live else latest["aqi"]
    label, _    = aqi_category(current_aqi)

    alert = get_alert(current_aqi)
    if alert["is_hazard"]:
        st.error(f"{alert['emoji']} **HAZARDOUS — AQI {round(current_aqi)}** | {alert['action']}")
    elif alert["is_alert"]:
        st.warning(f"{alert['emoji']} **{label} — AQI {round(current_aqi)}** | {alert['action']}")
    else:
        st.success(f"{alert['emoji']} **{label} — AQI {round(current_aqi)}** | {alert['action']}")

    for a in check_forecast_alerts(forecasts):
        st.warning(f"{a['emoji']} **{a['day']} ({a['date']})** — Forecast AQI {a['aqi']} ({a['label']})")

    st.markdown("---")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(gauge(current_aqi), use_container_width=True)
    with c2:
        st.plotly_chart(forecast_chart(forecasts, f"3-Day Forecast [{selected_name}]"), use_container_width=True)

    st.subheader("📅 3-Day Forecast")
    cols = st.columns(3)
    for i, fc in enumerate(forecasts):
        _, color = aqi_category(fc["aqi"])
        with cols[i]:
            st.markdown(f"""
<div class="forecast-card" style="border-top:4px solid {color}">
  <div style="font-size:1.1rem;font-weight:600">{fc['day']}</div>
  <div style="color:#aaa;font-size:0.85rem">{fc['date']}</div>
  <div class="aqi-value" style="color:{color}">{fc['aqi']}</div>
  <div style="font-size:0.8rem;color:#ccc">AQI</div>
  <div style="margin-top:8px;font-size:0.85rem">{fc['category']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    if show_validation:
        st.subheader("✅ Model Validation — Actual vs Predicted (Last 60 Days)")
        fig_val, mae, rmse = validation_chart(selected_model)
        st.plotly_chart(fig_val, use_container_width=True)
        v1, v2, v3 = st.columns(3)
        v1.metric("MAE",  f"{mae} AQI units")
        v2.metric("RMSE", f"{rmse} AQI units")
        v3.metric("R²",   meta["summary"][selected_name]["R2"])
        st.markdown("---")

    st.plotly_chart(historical_chart(history), use_container_width=True)

    if show_all and all_forecasts:
        st.plotly_chart(comparison_chart(all_forecasts), use_container_width=True)

    st.subheader("📊 Model Performance")
    rows = [{"Model": n, "RMSE": m["RMSE"], "MAE": m["MAE"],
             "R²": m["R2"], "CV RMSE": m["CV_RMSE"],
             "Best": "✅" if n == meta["best_model"] else ""}
            for n, m in meta["summary"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.markdown(f"<div style='text-align:center;color:#666;font-size:0.8rem'>Data: OpenMeteo · Karachi · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()