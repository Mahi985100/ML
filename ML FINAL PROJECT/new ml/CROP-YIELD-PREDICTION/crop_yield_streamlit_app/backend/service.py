from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"

def _resolve_data_path() -> Path:
    candidates = [
        ROOT_DIR / "Crop_Yield_final.csv",
        ROOT_DIR / "CROP_YEILD.csv",
        ROOT_DIR / "Crop_Yield_Added_Columns.csv",
        ROOT_DIR / "Crop_Yield_Added_Columns(1).csv",
        ROOT_DIR / "Crop_Yield_with_season.csv",
        ROOT_DIR / "Crop_Yield_Enhanced.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    csv_files = sorted(ROOT_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return ROOT_DIR / "Crop_Yield_final.csv"

DATA_PATH = _resolve_data_path()
MODEL_PATH = ARTIFACT_DIR / "crop_yield_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
NOTEBOOK_PATH = ROOT_DIR / "FINAL_CODE.ipynb"

DISPLAY_COLUMNS = ["fertilizer", "temp", "n", "p", "k", "rainfall", "humidity", "yield"]
FEATURE_COLUMNS = ["fertilizer", "temp", "n", "p", "k", "crop_season", "rainfall", "humidity", "crop_name"]
ALL_REQUIRED_COLUMNS = ["fertilizer", "temp", "n", "p", "k", "yield", "crop_season", "rainfall", "humidity", "crop_name"]
SEASON_MAP = {"Kharif": 0, "Rabi": 1}
CROP_MAP = {"Cotton": 0, "Maize": 1, "Rice": 2, "Wheat": 3}


@dataclass
class RuntimeBundle:
    raw_df: pd.DataFrame
    processed_df: pd.DataFrame
    pipeline: Pipeline
    metrics: Dict[str, float]
    model_comparison: Dict[str, Dict[str, float]]
    feature_ranges: Dict[str, Dict[str, float]]
    sample_predictions: pd.DataFrame
    freshness: Dict[str, float]


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.feature_columns_: list[str] | None = None
        self.medians_: pd.Series | None = None

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = out.columns.str.strip().str.lower()
        if "yeild" in out.columns:
            out = out.rename(columns={"yeild": "yield"})
        if "fertilizer" not in out.columns and "fertilizer" in [c.lower() for c in df.columns]:
            pass
        if "crop_season" in out.columns:
            out["crop_season"] = out["crop_season"].replace(SEASON_MAP)
        if "crop_name" in out.columns:
            out["crop_name"] = out["crop_name"].replace(CROP_MAP)
        numeric_like = [c for c in out.columns if c not in {"crop_season", "crop_name"}]
        for col in numeric_like:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].abs()
        for col in ["crop_season", "crop_name"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    @staticmethod
    def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["npk_sum"] = out["n"] + out["p"] + out["k"]
        out["npk_mean"] = (out["n"] + out["p"] + out["k"]) / 3
        out["fert_temp"] = out["fertilizer"] * out["temp"]
        out["n_p"] = out["n"] * out["p"]
        out["p_k"] = out["p"] * out["k"]
        out["n_k"] = out["n"] * out["k"]
        if "rainfall" in out.columns and "humidity" in out.columns:
            out["rain_humidity"] = out["rainfall"] * out["humidity"]
        if "crop_season" in out.columns and "temp" in out.columns:
            out["season_temp"] = out["crop_season"] * out["temp"]
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataCleaner":
        df = self._normalize(X)
        if y is not None:
            df = df.copy()
            df["yield"] = pd.to_numeric(pd.Series(y), errors="coerce").abs()
        df = df.fillna(df.median(numeric_only=True))
        base = df.drop(columns=["yield"], errors="ignore")
        engineered = self._add_engineered_features(base)
        self.feature_columns_ = list(engineered.columns)
        self.medians_ = df.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns_ is None or self.medians_ is None:
            raise ValueError("DataCleaner must be fit before transform.")
        df = self._normalize(X)
        df = df.fillna(self.medians_.reindex(df.columns).fillna(0))
        engineered = self._add_engineered_features(df)
        for col in self.feature_columns_:
            if col not in engineered.columns:
                engineered[col] = 0
        return engineered[self.feature_columns_]


def get_project_paths() -> Dict[str, str]:
    return {
        "root": str(ROOT_DIR),
        "data": str(DATA_PATH),
        "model": str(MODEL_PATH),
        "metadata": str(METADATA_PATH),
        "notebook": str(NOTEBOOK_PATH),
    }


def _freshness_signature() -> Dict[str, float]:
    return {
        "data_mtime": DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0,
        "service_mtime": Path(__file__).stat().st_mtime,
    }


def _load_raw_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, low_memory=False)


def _prepare_training_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    cleaner = DataCleaner()
    normalized = cleaner._normalize(raw_df)

    for col in ALL_REQUIRED_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = np.nan

    numeric_cols = ["fertilizer", "temp", "n", "p", "k", "yield", "rainfall", "humidity"]
    normalized[numeric_cols] = normalized[numeric_cols].apply(pd.to_numeric, errors="coerce")
    normalized[numeric_cols] = normalized[numeric_cols].abs()
    normalized[numeric_cols] = normalized[numeric_cols].fillna(normalized[numeric_cols].median())

    normalized["crop_season"] = pd.to_numeric(normalized["crop_season"].replace(SEASON_MAP), errors="coerce").fillna(0)
    normalized["crop_name"] = pd.to_numeric(normalized["crop_name"].replace(CROP_MAP), errors="coerce").fillna(0)

    normalized = normalized.dropna(subset=["yield"])
    cols_for_iqr = [c for c in DISPLAY_COLUMNS if c in normalized.columns]
    q1 = normalized[cols_for_iqr].quantile(0.25)
    q3 = normalized[cols_for_iqr].quantile(0.75)
    iqr = q3 - q1
    mask = ~((normalized[cols_for_iqr] < (q1 - 1.5 * iqr)) | (normalized[cols_for_iqr] > (q3 + 1.5 * iqr))).any(axis=1)
    filtered = normalized.loc[mask, ALL_REQUIRED_COLUMNS].copy()
    return cleaner._add_engineered_features(filtered)


def _build_pipeline() -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=40,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("cleaner", DataCleaner()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _build_artifacts() -> RuntimeBundle:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = _load_raw_dataset()
    processed_df = _prepare_training_dataframe(raw_df)

    X = processed_df.drop(columns=["yield"])
    y = processed_df["yield"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    train_pred = pipeline.predict(X_train)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "overfit_gap": float(r2_score(y_train, train_pred) - r2_score(y_test, y_pred)),
        "rows_raw": int(len(raw_df)),
        "rows_processed": int(len(processed_df)),
    }

    baseline_model = RandomForestRegressor(n_estimators=15, random_state=42, n_jobs=-1)
    baseline_pipeline = Pipeline(
        steps=[
            ("cleaner", DataCleaner()),
            ("poly", PolynomialFeatures(degree=1, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", baseline_model),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)

    model_comparison = {
        "Baseline": {
            "MAE": float(mean_absolute_error(y_test, baseline_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
            "R2": float(r2_score(y_test, baseline_pred)),
        },
        "Final Model": {
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"],
        },
    }

    sample_predictions = pd.DataFrame(
        {
            "Actual Yield": y_test.reset_index(drop=True),
            "Predicted Yield": pd.Series(y_pred).reset_index(drop=True),
        }
    ).sample(n=min(160, len(y_test)), random_state=42)

    feature_ranges = {}
    for col in FEATURE_COLUMNS:
        if col not in processed_df.columns:
            continue
        s = processed_df[col]
        feature_ranges[col] = {
            "min": float(np.floor(s.min())),
            "max": float(np.ceil(s.max())),
            "default": float(round(s.median(), 2)),
            "step": 1.0,
        }

    freshness = _freshness_signature()
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "freshness": freshness,
                "metrics": metrics,
                "model_comparison": model_comparison,
                "feature_ranges": feature_ranges,
                "sample_predictions": sample_predictions.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return RuntimeBundle(
        raw_df=raw_df,
        processed_df=processed_df,
        pipeline=pipeline,
        metrics=metrics,
        model_comparison=model_comparison,
        feature_ranges=feature_ranges,
        sample_predictions=sample_predictions,
        freshness=freshness,
    )


def _artifacts_are_fresh() -> bool:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return False
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return meta.get("freshness") == _freshness_signature()


@lru_cache(maxsize=1)
def get_runtime() -> RuntimeBundle:
    if not _artifacts_are_fresh():
        return _build_artifacts()

    raw_df = _load_raw_dataset()
    processed_df = _prepare_training_dataframe(raw_df)
    pipeline = joblib.load(MODEL_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return RuntimeBundle(
        raw_df=raw_df,
        processed_df=processed_df,
        pipeline=pipeline,
        metrics=meta["metrics"],
        model_comparison=meta["model_comparison"],
        feature_ranges=meta["feature_ranges"],
        sample_predictions=pd.DataFrame(meta.get("sample_predictions", [])),
        freshness=meta.get("freshness", _freshness_signature()),
    )


def refresh_runtime() -> RuntimeBundle:
    get_runtime.cache_clear()
    return get_runtime()


def get_feature_ranges() -> Dict[str, Dict[str, float]]:
    return get_runtime().feature_ranges


def get_dataset_preview(rows: int = 10) -> pd.DataFrame:
    return get_runtime().raw_df.head(rows)


def predict_yield(
    fertilizer: float,
    temp: float,
    n: float,
    p: float,
    k: float,
    crop_season: float = 0,
    rainfall: float = 0,
    humidity: float = 0,
    crop_name: float = 0,
) -> float:
    runtime = get_runtime()
    input_df = pd.DataFrame([
        {
            "fertilizer": fertilizer,
            "temp": temp,
            "n": n,
            "p": p,
            "k": k,
            "crop_season": crop_season,
            "rainfall": rainfall,
            "humidity": humidity,
            "crop_name": crop_name,
        }
    ])
    return float(runtime.pipeline.predict(input_df)[0])


def _yield_distribution_chart(processed_df: pd.DataFrame):
    fig = px.histogram(
        processed_df,
        x="yield",
        nbins=24,
        marginal="box",
        title="Yield distribution",
        labels={"yield": "Yield"},
    )
    fig.update_layout(bargap=0.08)
    return fig


def _yield_by_temp_chart(processed_df: pd.DataFrame):
    grouped = (
        processed_df.groupby(processed_df["temp"].round())["yield"]
        .mean()
        .reset_index()
        .rename(columns={"temp": "Temperature", "yield": "Average Yield"})
        .sort_values("Temperature")
    )
    return px.line(
        grouped,
        x="Temperature",
        y="Average Yield",
        markers=True,
        title="Average yield by temperature",
    )


def _fertilizer_vs_yield_chart(processed_df: pd.DataFrame):
    sample = processed_df[["fertilizer", "yield"]].sample(n=min(300, len(processed_df)), random_state=42)
    return px.scatter(
        sample,
        x="fertilizer",
        y="yield",
        trendline="ols",
        title="Fertilizer vs yield",
        labels={"fertilizer": "Fertilizer", "yield": "Yield"},
    )


def _nutrient_balance_chart(processed_df: pd.DataFrame):
    avg_values = processed_df[["n", "p", "k"]].mean().reset_index()
    avg_values.columns = ["Nutrient", "Average Value"]
    return px.bar(
        avg_values,
        x="Nutrient",
        y="Average Value",
        text_auto=".1f",
        title="Average nutrient balance",
    )


def _comparison_chart(model_comparison: Dict[str, Dict[str, float]]):
    rows = [
        {"Model": model_name, "Metric": metric_name, "Value": value}
        for model_name, values in model_comparison.items()
        for metric_name, value in values.items()
    ]
    comparison_df = pd.DataFrame(rows)
    return px.bar(
        comparison_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Model performance comparison",
        text_auto=".3f",
    )


def _actual_vs_predicted_chart(sample_predictions: pd.DataFrame):
    return px.scatter(
        sample_predictions,
        x="Actual Yield",
        y="Predicted Yield",
        trendline="ols",
        title="Actual vs predicted yield",
    )


def _feature_relationship_chart(processed_df: pd.DataFrame):
    cols = [c for c in ["fertilizer", "temp", "n", "p", "k", "crop_season", "rainfall", "humidity", "crop_name", "yield"] if c in processed_df.columns]
    corr = (
        processed_df[cols]
        .corr(numeric_only=True)["yield"]
        .drop("yield")
        .reset_index()
    )
    corr.columns = ["Feature", "Relationship with Yield"]
    corr["Relationship with Yield"] = corr["Relationship with Yield"].round(3)
    return px.bar(
        corr,
        x="Feature",
        y="Relationship with Yield",
        text_auto=True,
        title="How each input relates to yield",
    )


def get_home_charts() -> Dict[str, Any]:
    runtime = get_runtime()
    return {
        "yield_distribution": _yield_distribution_chart(runtime.processed_df),
        "yield_by_temp": _yield_by_temp_chart(runtime.processed_df),
        "fertilizer_vs_yield": _fertilizer_vs_yield_chart(runtime.processed_df),
        "nutrient_balance": _nutrient_balance_chart(runtime.processed_df),
    }


def get_about_charts() -> Dict[str, Any]:
    runtime = get_runtime()
    return {
        "comparison": _comparison_chart(runtime.model_comparison),
        "actual_vs_predicted": _actual_vs_predicted_chart(runtime.sample_predictions),
        "feature_relationship": _feature_relationship_chart(runtime.processed_df),
    }
