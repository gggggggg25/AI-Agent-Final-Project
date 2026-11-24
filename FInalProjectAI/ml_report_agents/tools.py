"""Tools used by the Data Scientist agent and the Report agent.

All heavy lifting (profiling, modelling, charts) is done here in plain Python so
results are reproducible and numbers in the report come from code, not from the LLM.

Run state is persisted on disk under ``outputs/`` so the two agents can hand off
work to each other even though they run in separate ADK sessions (AgentTool).
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # noqa: E402

try:  # ToolContext is only needed inside ADK; keep the module importable without it
    from google.adk.tools import ToolContext
    from google.genai import types
except Exception:  # pragma: no cover
    ToolContext = object  # type: ignore
    types = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
def _writable_dir(env_var: str, preferred: Path, name: str) -> Path:
    """First writable of: $env_var, preferred, system temp dir. Never fails on read-only app folders."""
    for cand in (Path(os.getenv(env_var)) if os.getenv(env_var) else None, preferred,
                 Path(tempfile.gettempdir()) / "ml_report_agents" / name):
        if cand is None:
            continue
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return cand
        except OSError:
            continue
    raise RuntimeError(f"No writable directory available for {name}")


BUNDLED_DATA_DIRS = [Path(__file__).resolve().parent / "data", ROOT / "data"]  # files shipped with the app (read-only is fine)
DATA_DIR = _writable_dir("ML_AGENT_DATA_DIR", ROOT / "data", "data")
OUT_DIR = _writable_dir("ML_AGENT_OUT_DIR", ROOT / "outputs", "outputs")
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEFAULT_FILES = {"default": "student-mat.csv", "default_math": "student-mat.csv", "default_por": "student-por.csv"}
RUN_POINTER = OUT_DIR / "current_run.json"


# --------------------------------------------------------------------------- helpers
def _json_safe(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_safe))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _extract_zip(buf: io.BytesIO) -> None:
    with zipfile.ZipFile(buf) as z:
        for name in z.namelist():
            data = z.read(name)
            low = name.lower()
            if low.endswith(".zip"):
                _extract_zip(io.BytesIO(data))
            elif low.endswith((".csv", ".txt")):
                (DATA_DIR / Path(name).name).write_bytes(data)


DATA_BUCKET = os.getenv("DATA_BUCKET", "").strip()   # optional: bucket NAME holding user datasets
DATA_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


def _fetch_from_gcs(bucket: str, blob_name: str) -> Path:
    """Download gs://bucket/blob_name into the local cache (always fresh) and return the path."""
    from google.cloud import storage  # lazy import: only needed when a bucket is used

    blob = storage.Client().bucket(bucket).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{blob_name} does not exist.")
    dest = DATA_DIR / "gcs_cache" / Path(blob_name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))
    return dest


def _resolve_source(source: str) -> Path:
    """'default' -> UCI Student Performance (math). 'gs://bucket/file.csv' -> Cloud Storage.
    Otherwise a file name/path: local folders first, then the DATA_BUCKET bucket."""
    key = (source or "default").strip().lower()
    if key in DEFAULT_FILES:
        fname = DEFAULT_FILES[key]
        path = DATA_DIR / fname
        for d in BUNDLED_DATA_DIRS:
            if not path.exists() and (d / fname).exists():
                return d / fname
        if not path.exists():
            raw = urllib.request.urlopen(UCI_URL, timeout=60).read()
            _extract_zip(io.BytesIO(raw))
        if not path.exists():
            raise FileNotFoundError(
                f"Could not obtain {path.name}. Download the dataset from UCI and place it in {DATA_DIR}."
            )
        return path
    src = source.strip()
    if src.startswith("gs://"):
        bucket, _, blob = src[5:].partition("/")
        return _fetch_from_gcs(bucket, blob)
    p = Path(src).expanduser()
    for cand in (p, Path.cwd() / p, DATA_DIR / p, DATA_DIR / "uploads" / p, *[d / p for d in BUNDLED_DATA_DIRS], ROOT / p):
        if cand.exists():
            return cand
    if DATA_BUCKET:
        try:
            return _fetch_from_gcs(DATA_BUCKET, src)
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001  (auth/permission/network problems)
            raise RuntimeError(f"Could not read gs://{DATA_BUCKET}/{src}: {e}") from e
    where = f"gs://{DATA_BUCKET}/" if DATA_BUCKET else "(no DATA_BUCKET configured)"
    raise FileNotFoundError(
        f"File '{source}' not found in the app's data folders or in {where}. "
        "Use list_available_datasets to see what exists, or upload the file to the bucket."
    )


def list_available_datasets() -> dict:
    """Lists the datasets the agent can analyse: the built-in UCI Student Performance files,
    files shipped inside the app, and files uploaded to the configured Cloud Storage bucket
    (uploading there needs no redeploy). Use the names exactly as shown as the `source`."""
    out = {"app_build": os.getenv("BUILD_STAMP", "unknown (old deploy script)"), "built_in": ["default (student-mat.csv, math)", "default_por (student-por.csv, Portuguese)"],
           "bundled_files": sorted({f.name for d in BUNDLED_DATA_DIRS if d.exists()
                                    for f in d.iterdir() if f.suffix.lower() in DATA_SUFFIXES}),
           "bucket": f"gs://{DATA_BUCKET}/" if DATA_BUCKET else "not configured", "bucket_files": []}
    if DATA_BUCKET:
        try:
            from google.cloud import storage

            out["bucket_files"] = sorted(b.name for b in storage.Client().list_blobs(DATA_BUCKET)
                                         if b.name.lower().endswith(DATA_SUFFIXES))[:100]
        except Exception as e:  # noqa: BLE001
            out["bucket_error"] = str(e)
    return out


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")  # auto-detects ; , \t
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _match_col(name: str, cols) -> str | None:
    name = name.strip()
    if name in cols:
        return name
    low = {c.lower(): c for c in cols}
    return low.get(name.lower())


def _split_list(s: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,;\n]", s or "") if x.strip()]


def _detect_task(y: pd.Series, task: str, threshold: float) -> str:
    task = (task or "auto").lower()
    if threshold and threshold > 0:
        return "classification"
    if task in ("regression", "classification"):
        return task
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if y.nunique() <= 10 and np.allclose(y.dropna() % 1, 0):
        return "classification"
    return "regression"


def _load_run() -> dict:
    if not RUN_POINTER.exists():
        raise RuntimeError("No dataset loaded yet. Call load_and_profile_dataset first.")
    return _read_json(RUN_POINTER)


def _prepare(run: dict, exclude: list[str] | None = None):
    df = _read_table(Path(run["source_path"]))
    target = run["target"]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    feats = [f for f in run["features"] if f not in (exclude or [])]
    X = df[feats].copy()
    y_raw = df[target]
    class_names = None
    if run["task"] == "classification":
        thr = run.get("binarize_threshold") or 0
        if thr > 0:
            y = (y_raw.astype(float) >= thr).astype(int)
            class_names = [f"{target} < {thr:g}", f"{target} >= {thr:g}"]
        else:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target)
            class_names = [str(c) for c in le.classes_]
    else:
        y = y_raw.astype(float)
    return X, y, class_names


async def _publish(tool_context, path: Path) -> None:
    """Expose a file as an ADK artifact so the web UI can display it (best effort)."""
    if tool_context is None or types is None:
        return
    mime = "image/png" if path.suffix == ".png" else "text/markdown" if path.suffix == ".md" else "text/html"
    try:
        await tool_context.save_artifact(path.name, types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
    except Exception:
        pass  # no artifact service configured - files are still on disk


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- Data Scientist tool 1
async def load_and_profile_dataset(
    source: str = "default",
    target: str = "G3",
    features: str = "",
    task: str = "auto",
    binarize_threshold: float = 0.0,
    tool_context: ToolContext = None,
) -> dict:
    """Loads a dataset, validates the target/features, profiles it and draws EDA charts.

    Args:
        source: "default" (UCI Student Performance, Math: student-mat.csv), "default_por"
            (Portuguese course) or a path to a user CSV/TSV/XLSX file (absolute, or relative to
            the project's data/ folder).
        target: Name of the column to predict. For the default dataset use "G3" (final grade).
        features: Comma-separated feature column names. Empty string = use every column except the target.
        task: "auto", "regression" or "classification".
        binarize_threshold: If > 0, a numeric target is converted to a binary classification
            target (value >= threshold -> 1). Example: target G3 with threshold 10 = pass/fail.
            Use 0 to disable.

    Returns:
        A dict with the dataset summary, or {"status": "error", ...} explaining what to fix.
    """
    try:
        path = _resolve_source(source)
        df = _read_table(path)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}

    tgt = _match_col(target, df.columns)
    if tgt is None:
        return {"status": "error", "message": f"Target '{target}' not found.", "available_columns": list(df.columns)}

    if features.strip():
        feats, bad = [], []
        for f in _split_list(features):
            m = _match_col(f, df.columns)
            (feats if m else bad).append(m or f)
        if bad:
            return {"status": "error", "message": f"Unknown feature columns: {bad}", "available_columns": list(df.columns)}
        feats = [f for f in dict.fromkeys(feats) if f != tgt]
    else:
        feats = [c for c in df.columns if c != tgt]

    # drop identifier-like columns (all-unique strings) and constant columns
    dropped = {}
    for c in list(feats):
        if df[c].nunique(dropna=True) <= 1:
            dropped[c] = "constant"
        elif not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == len(df):
            dropped[c] = "looks like an identifier"
    feats = [c for c in feats if c not in dropped]

    n_before = len(df)
    df = df.dropna(subset=[tgt]).reset_index(drop=True)
    ytype = _detect_task(df[tgt], task, binarize_threshold)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / run_id
    run = {
        "run_id": run_id, "run_dir": str(run_dir), "source_path": str(path), "target": tgt,
        "features": feats, "task": ytype, "binarize_threshold": float(binarize_threshold or 0),
    }
    _write_json(RUN_POINTER, run)

    X = df[feats]
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in feats if c not in num]

    if ytype == "regression":
        t = df[tgt].astype(float)
        target_summary = {k: round(float(v), 3) for k, v in
                          t.describe()[["mean", "std", "min", "50%", "max"]].rename({"50%": "median"}).items()}
        y_num = t
    else:
        counts = df[tgt].value_counts()
        if binarize_threshold and binarize_threshold > 0:
            b = (df[tgt].astype(float) >= binarize_threshold).astype(int)
            counts = b.map({0: f"< {binarize_threshold:g}", 1: f">= {binarize_threshold:g}"}).value_counts()
            y_num = b.astype(float)
        else:
            y_num = pd.Series(pd.factorize(df[tgt])[0], dtype=float)
        target_summary = {"class_counts": {str(k): int(v) for k, v in counts.items()},
                          "majority_class_share": round(float(counts.max() / counts.sum()), 3)}

    corr = {}
    if num:
        c = X[num].apply(pd.to_numeric, errors="coerce").corrwith(y_num).dropna()
        corr = {k: round(float(v), 3) for k, v in c.reindex(c.abs().sort_values(ascending=False).index).head(10).items()}

    missing = {k: int(v) for k, v in df[feats + [tgt]].isna().sum().items() if v > 0}
    summary = {
        "status": "ok", "source_file": path.name, "run_id": run_id,
        "n_rows": int(len(df)), "rows_dropped_for_missing_target": int(n_before - len(df)),
        "n_features": len(feats), "target": tgt, "task": ytype,
        "binarize_threshold": run["binarize_threshold"] or None,
        "numeric_features": num, "categorical_features": cat,
        "categorical_cardinality": {c: int(df[c].nunique()) for c in cat},
        "missing_values": missing or "none", "duplicate_rows": int(df.duplicated().sum()),
        "dropped_columns": dropped or "none", "target_summary": target_summary,
        "top_numeric_correlations_with_target": corr,
        "numeric_describe": X[num].describe().round(2).T[["mean", "std", "min", "max"]].head(15).to_dict("index") if num else {},
    }

    # ---- EDA charts
    chart_dir = run_dir / "charts"
    charts = []
    fig, ax = plt.subplots(figsize=(6, 4))
    if ytype == "regression":
        ax.hist(df[tgt].astype(float), bins=20, color="#4C78A8", edgecolor="white")
        ax.set_xlabel(tgt)
    else:
        vc = pd.Series(y_num).map(lambda v: int(v)).value_counts().sort_index() if binarize_threshold else df[tgt].value_counts()
        ax.bar([str(i) for i in vc.index], vc.values, color="#4C78A8")
        ax.set_xlabel(tgt)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of target ({tgt})")
    p = chart_dir / "target_distribution.png"
    _savefig(fig, p)
    charts.append(p)

    if num:
        top = list(corr.keys())[:11]
        cm_df = X[top].apply(pd.to_numeric, errors="coerce").assign(**{tgt: y_num}).corr() if top else None
        if cm_df is not None and len(cm_df) > 1:
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cm_df.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cm_df)), cm_df.columns, rotation=60, ha="right")
            ax.set_yticks(range(len(cm_df)), cm_df.columns)
            for i in range(len(cm_df)):
                for j in range(len(cm_df)):
                    ax.text(j, i, f"{cm_df.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im)
            ax.set_title("Correlation: target and most related numeric features")
            p = chart_dir / "correlation_heatmap.png"
            _savefig(fig, p)
            charts.append(p)

    for p in charts:
        await _publish(tool_context, p)
    summary["charts"] = [str(p) for p in charts]
    _write_json(run_dir / "results.json", {"dataset_summary": summary, "scenarios": {}})
    return summary


# --------------------------------------------------------------------------- Data Scientist tool 2
def _models(task: str, seed: int) -> dict:
    if task == "regression":
        return {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }


async def train_and_evaluate_model(
    scenario_name: str = "full_model",
    exclude_features: str = "",
    test_size: float = 0.2,
    random_state: int = 42,
    tool_context: ToolContext = None,
) -> dict:
    """Trains and compares several models on the dataset loaded by load_and_profile_dataset.

    Models are compared with 5-fold cross-validation on the training split; the best one is
    evaluated once on a held-out test split. Also computes permutation feature importance and
    saves charts. Call again with a different scenario_name/exclude_features to compare variants
    (e.g. excluding G1 and G2 to predict grades *early*).

    Args:
        scenario_name: Short label for this run, e.g. "full_model" or "early_warning_no_G1_G2".
        exclude_features: Comma-separated features to leave out in this scenario ("" = none).
        test_size: Fraction held out for testing (0.1-0.4).
        random_state: Seed for reproducibility.

    Returns:
        Dict with model comparison, test metrics, baseline, feature importance and chart paths.
    """
    try:
        run = _load_run()
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}
    task = run["task"]
    excl = [m for m in (_match_col(f, run["features"]) for f in _split_list(exclude_features)) if m]
    X, y, class_names = _prepare(run, excl)
    if X.shape[1] == 0:
        return {"status": "error", "message": "No features left after exclusions."}

    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    steps = []
    if num:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    pre = ColumnTransformer(steps)

    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
    scoring = "r2" if task == "regression" else "f1_weighted"
    cv = (KFold if task == "regression" or strat is None else StratifiedKFold)(5, shuffle=True, random_state=random_state)

    comparison, fitted = {}, {}
    for name, est in _models(task, random_state).items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        sc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=scoring)
        comparison[name] = {"cv_mean": round(float(sc.mean()), 4), "cv_std": round(float(sc.std()), 4)}
        fitted[name] = pipe
    best_name = max(comparison, key=lambda k: comparison[k]["cv_mean"])
    best = fitted[best_name].fit(Xtr, ytr)
    pred = best.predict(Xte)

    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        base_rmse = float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))))
        metrics = {"R2": round(float(r2_score(yte, pred)), 4), "RMSE": round(rmse, 4),
                   "MAE": round(float(mean_absolute_error(yte, pred)), 4)}
        baseline = {"strategy": "always predict training mean", "RMSE": round(base_rmse, 4), "R2": 0.0}
        metrics["RMSE_improvement_over_baseline_pct"] = round(100 * (1 - rmse / base_rmse), 1)
    else:
        metrics = {"accuracy": round(float(accuracy_score(yte, pred)), 4),
                   "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
                   "f1_weighted": round(float(f1_score(yte, pred, average="weighted")), 4)}
        if yte.nunique() == 2:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(yte, best.predict_proba(Xte)[:, 1])), 4)
            except Exception:  # noqa: BLE001
                pass
        maj = ytr.value_counts().idxmax()
        baseline = {"strategy": "always predict majority class",
                    "accuracy": round(float(accuracy_score(yte, np.full(len(yte), maj))), 4)}

    pi = permutation_importance(best, Xte, yte, n_repeats=15, random_state=random_state, scoring=scoring)
    imp = (pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "std": pi.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    top_imp = [{"feature": r.feature, "importance": round(float(r.importance), 4), "std": round(float(r.std), 4)}
               for r in imp.head(15).itertuples()]

    # ---- charts
    tag = re.sub(r"\W+", "_", scenario_name).strip("_") or "scenario"
    chart_dir = Path(run["run_dir"]) / "charts"
    charts = []

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(comparison)
    ax.barh(names, [comparison[n]["cv_mean"] for n in names], xerr=[comparison[n]["cv_std"] for n in names], color="#72B7B2")
    ax.set_xlabel(f"5-fold CV {scoring}")
    ax.set_title(f"Model comparison - {scenario_name}")
    p = chart_dir / f"{tag}_model_comparison.png"
    _savefig(fig, p)
    charts.append(p)

    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#F58518")
    ax.set_xlabel(f"Permutation importance (drop in {scoring} when shuffled)")
    ax.set_title(f"Top features - {best_name} ({scenario_name})")
    p = chart_dir / f"{tag}_feature_importance.png"
    _savefig(fig, p)
    charts.append(p)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if task == "regression":
        ax.scatter(yte, pred, alpha=0.7, color="#4C78A8")
        lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        ax.set_title(f"Actual vs predicted (test set) - {scenario_name}")
        p = chart_dir / f"{tag}_actual_vs_predicted.png"
    else:
        cmx = confusion_matrix(yte, pred)
        ax.imshow(cmx, cmap="Blues")
        labels = class_names if class_names and len(class_names) == cmx.shape[0] else range(cmx.shape[0])
        ax.set_xticks(range(len(cmx)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(cmx)), labels)
        for i in range(cmx.shape[0]):
            for j in range(cmx.shape[1]):
                ax.text(j, i, cmx[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix (test set) - {scenario_name}")
        p = chart_dir / f"{tag}_confusion_matrix.png"
    _savefig(fig, p)
    charts.append(p)
    for c in charts:
        await _publish(tool_context, c)

    result = {
        "status": "ok", "scenario": scenario_name, "task": task,
        "features_used": list(X.columns), "features_excluded": excl,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "cv_scoring": scoring, "model_comparison": comparison, "best_model": best_name,
        "test_metrics": metrics, "baseline": baseline, "top_feature_importance": top_imp,
        "charts": [str(c) for c in charts],
    }
    res_path = Path(run["run_dir"]) / "results.json"
    allres = _read_json(res_path)
    allres["scenarios"][scenario_name] = result
    _write_json(res_path, allres)
    return result


# --------------------------------------------------------------------------- Report agent tools
def get_analysis_results() -> dict:
    """Returns everything the Data Scientist produced for the current run: the dataset summary and
    every trained scenario (model comparison, test metrics, baseline, feature importance, chart paths).
    Chart paths are given relative to the report folder (e.g. charts/full_model_feature_importance.png)."""
    try:
        run = _load_run()
        res = _read_json(Path(run["run_dir"]) / "results.json")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}
    rel = lambda p: "charts/" + Path(p).name  # noqa: E731
    res["dataset_summary"]["charts"] = [rel(p) for p in res["dataset_summary"].get("charts", [])]
    for s in res["scenarios"].values():
        s["charts"] = [rel(p) for p in s["charts"]]
    res["status"] = "ok" if res["scenarios"] else "no_models_trained_yet"
    return res


async def save_report(markdown_report: str, tool_context: ToolContext = None) -> dict:
    """Saves the final report (Markdown) as report.md and report.html in the current run folder.

    Args:
        markdown_report: The complete report in Markdown. Reference charts with relative paths such as
            ![Feature importance](charts/full_model_feature_importance.png).
    """
    try:
        run = _load_run()
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": str(e)}
    run_dir = Path(run["run_dir"])
    md_path = run_dir / "report.md"
    md_path.write_text(markdown_report, encoding="utf-8")
    try:
        import markdown as md

        body = md.markdown(markdown_report, extensions=["tables", "fenced_code"])
    except Exception:  # noqa: BLE001
        body = f"<pre>{markdown_report}</pre>"
    html = ("<!doctype html><meta charset='utf-8'><title>ML Report</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.55}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 10px}img{max-width:100%}"
            f"</style>{body}")
    html_path = run_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    await _publish(tool_context, md_path)
    await _publish(tool_context, html_path)
    return {"status": "ok", "report_markdown": str(md_path), "report_html": str(html_path)}
