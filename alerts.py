# ============================================================
#  alerts.py  —  AQI hazard alert system
# ============================================================

from feature_pipeline import aqi_category


# AQI thresholds for alerts
ALERT_LEVELS = {
    "Good":                           {"min": 0,   "max": 50,  "color": "#00e400", "emoji": "✅", "action": "Air quality is satisfactory. No precautions needed."},
    "Moderate":                       {"min": 51,  "max": 100, "color": "#ffff00", "emoji": "⚠️", "action": "Unusually sensitive people should consider reducing prolonged outdoor exertion."},
    "Unhealthy for Sensitive Groups": {"min": 101, "max": 150, "color": "#ff7e00", "emoji": "🟠", "action": "People with heart/lung disease, elderly and children should reduce prolonged outdoor exertion."},
    "Unhealthy":                      {"min": 151, "max": 200, "color": "#ff0000", "emoji": "🔴", "action": "Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor activity."},
    "Very Unhealthy":                 {"min": 201, "max": 300, "color": "#8f3f97", "emoji": "🟣", "action": "Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors."},
    "Hazardous":                      {"min": 301, "max": 500, "color": "#7e0023", "emoji": "☠️", "action": "HEALTH EMERGENCY. Everyone should avoid all outdoor activity."},
}


def get_alert(aqi_value: float) -> dict:
    """Return alert info for a given AQI value."""
    label, color = aqi_category(aqi_value)
    level = ALERT_LEVELS.get(label, ALERT_LEVELS["Hazardous"])
    return {
        "aqi":      aqi_value,
        "label":    label,
        "color":    color,
        "emoji":    level["emoji"],
        "action":   level["action"],
        "is_alert": aqi_value > 100,       # trigger alert above 100
        "is_hazard": aqi_value > 200,      # trigger hazard above 200
    }


def check_forecast_alerts(forecasts: list) -> list:
    """Check all forecast days and return alerts for dangerous days."""
    alerts = []
    for fc in forecasts:
        info = get_alert(fc["aqi"])
        if info["is_alert"]:
            alerts.append({
                "date":    fc["date"],
                "day":     fc["day"],
                "aqi":     fc["aqi"],
                "label":   info["label"],
                "color":   info["color"],
                "emoji":   info["emoji"],
                "action":  info["action"],
                "is_hazard": info["is_hazard"],
            })
    return alerts


if __name__ == "__main__":
    # Test with sample values
    test_values = [45, 85, 130, 175, 250, 350]
    print("AQI Alert Test:")
    print("=" * 50)
    for val in test_values:
        alert = get_alert(val)
        print(f"AQI {val:>3}  {alert['emoji']}  {alert['label']}")
        print(f"       → {alert['action']}")
        print()
