import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection  import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import Ridge, Lasso
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics          import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline         import Pipeline

from config           import MODELS_DIR, FEATURE_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE
from feature_pipeline import load_features

os.makedirs(MODELS_DIR, exist_ok=True)


def get_models():
    return {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=1.0, max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(n_estimators=200, max_depth=12,
                        random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                        max_depth=5, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(y_true, y_pred):
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 3),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


def train_all():
    df = load_features()
    df = df.sort_values("datetime").dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"Loaded {len(df)} rows from MongoDB")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    tscv    = TimeSeriesSplit(n_splits=5)
    models  = get_models()
    results = {}
    best_name, best_rmse = None, float("inf")

    for name, pipeline in models.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=tscv, scoring="neg_root_mean_squared_error")
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics["CV_RMSE"] = round(-cv_scores.mean(), 3)

        results[name] = {"metrics": metrics, "y_test": y_test.tolist(),
                         "y_pred": y_pred.tolist(), "pipeline": pipeline}

        print(f"{name}: RMSE={metrics['RMSE']} MAE={metrics['MAE']} R2={metrics['R2']}")

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_name = name

    print(f"\nBest model: {best_name} (RMSE={best_rmse})")

    summary = {}
    for name, result in results.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(result["pipeline"], f"{MODELS_DIR}/{safe}.pkl")
        summary[name] = result["metrics"]

    joblib.dump(results[best_name]["pipeline"], f"{MODELS_DIR}/best_model.pkl")

    meta = {
        "best_model":   best_name,
        "best_rmse":    best_rmse,
        "feature_cols": FEATURE_COLS,
        "target":       TARGET_COL,
        "train_rows":   len(X_train),
        "test_rows":    len(X_test),
        "summary":      summary,
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    _plot_results(results, df["datetime"].iloc[split_idx:].values)
    return meta


def _plot_results(results, dates):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("AQI Prediction - Karachi (Actual vs Predicted)", fontsize=14)
    axes = axes.flatten()

    for ax, (name, result) in zip(axes, results.items()):
        m = result["metrics"]
        ax.plot(dates, result["y_test"], label="Actual",    color="#2196F3", linewidth=1.5)
        ax.plot(dates, result["y_pred"], label="Predicted", color="#FF5722", linewidth=1.5, linestyle="--")
        ax.set_title(f"{name}\nRMSE={m['RMSE']} MAE={m['MAE']} R2={m['R2']}", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/model_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {MODELS_DIR}/model_comparison.png")


def print_summary(meta):
    print(f"\n{'='*55}")
    print(f"  TRAINING SUMMARY - {meta['train_rows']} train / {meta['test_rows']} test rows")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'RMSE':>7} {'MAE':>7} {'R2':>7}")
    print(f"  {'-'*50}")
    for name, m in meta["summary"].items():
        star = " *" if name == meta["best_model"] else ""
        print(f"  {name:<25} {m['RMSE']:>7} {m['MAE']:>7} {m['R2']:>7}{star}")
    print(f"\n  Best: {meta['best_model']} (RMSE={meta['best_rmse']})")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    meta = train_all()
    print_summary(meta)