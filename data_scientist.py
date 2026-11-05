"""Agent 1: Data Scientist. Deterministic Python, so metrics can't be hallucinated."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import OUT, TARGET


def summarize(df, target):
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "target": target,
        "target_stats": {k: round(float(v), 2) for k, v in df[target].describe().items()},
        "categorical_columns": df.select_dtypes("object").columns.tolist(),
        "numeric_columns": df.select_dtypes("number").columns.tolist(),
        "pct_zero_final_grade": round(float((df[target] == 0).mean() * 100), 1),
    }


def build_model(df, target, drop_cols, name):
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]
    cat = X.select_dtypes("object").columns.tolist()
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough"
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "ridge": Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [("pre", pre), ("m", RandomForestRegressor(n_estimators=300, random_state=42))]
        ),
    }
    results = {}
    for key, pipe in candidates.items():
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="r2")
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[key] = {
            "cv_r2_mean": round(float(cv.mean()), 3),
            "test_r2": round(float(r2_score(yte, pred)), 3),
            "test_mae": round(float(mean_absolute_error(yte, pred)), 3),
            "test_rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 3),
        }
    best = max(results, key=lambda k: results[k]["cv_r2_mean"])
    model = candidates[best]

    # Permutation importance on the ORIGINAL columns
    pi = permutation_importance(model, Xte, yte, n_repeats=10, random_state=42)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7, 5))
    imp[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top features ({name})")
    ax.set_xlabel("Permutation importance (drop in R²)")
    fig.tight_layout()
    fig.savefig(OUT / f"importance_{name}.png", dpi=150)
    plt.close(fig)

    pred = model.predict(Xte)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yte, pred, alpha=0.5)
    ax.plot([0, 20], [0, 20], "r--")
    ax.set_xlabel("Actual final grade")
    ax.set_ylabel("Predicted final grade")
    ax.set_title(f"Actual vs predicted ({name})")
    fig.tight_layout()
    fig.savefig(OUT / f"pred_vs_actual_{name}.png", dpi=150)
    plt.close(fig)

    return {
        "features_dropped": drop_cols,
        "best_model": best,
        "all_models": results,
        "top_features": {k: round(float(v), 4) for k, v in imp.items()},
        "charts": [f"importance_{name}.png", f"pred_vs_actual_{name}.png"],
    }


def run_data_scientist(df, target=TARGET):
    OUT.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    df[target].hist(bins=20, ax=ax)
    ax.set_title("Final grade distribution")
    ax.set_xlabel("G3 (0-20)")
    fig.savefig(OUT / "target_dist.png", dpi=150)
    plt.close(fig)

    results = {
        "dataset_summary": summarize(df, target),
        "distribution_chart": "target_dist.png",
        # Uses earlier-period grades: very accurate, but less actionable
        "model_with_prior_grades": build_model(df, target, [], "with_G1_G2"),
        # Excludes G1/G2: reveals behavioural and demographic drivers
        "model_without_prior_grades": build_model(df, target, ["G1", "G2"], "no_G1_G2"),
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
