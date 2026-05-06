from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel, Field

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "AdidasDataset2023to2025.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "lstm_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
META_PATH = ARTIFACTS_DIR / "meta.joblib"


app = FastAPI(
    title="Adidas Analytics Demo API",
    description="Demo-ready endpoints for forecast, inventory health, and segment analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastPoint(BaseModel):
    month: str = Field(..., examples=["Jan"])
    forecast: float = Field(..., examples=[12000.5])
    actual: float | None = Field(default=None, examples=[11890.2])


class ForecastResponse(BaseModel):
    year: int
    model: str = "LSTM"
    points: list[ForecastPoint]


class InventoryHealthItem(BaseModel):
    product_id: str
    product_name: str
    category: str
    inventory_turnover: float
    sales_velocity: float
    tier: Literal["High Performer", "Steady", "At Risk"]
    action: Literal["Hold", "Monitor", "Discount"]


class ClusterPoint(BaseModel):
    product_id: str
    product_name: str
    cluster: str
    x: float
    y: float
    category: str


def aggregate_daily(df: pd.DataFrame, target: str = "Profit") -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not in DataFrame")
    if "Order_Date" not in df.columns:
        raise ValueError("'Order_Date' column required for daily aggregation")

    series_df = df.copy()
    series_df["Order_Date"] = pd.to_datetime(series_df["Order_Date"])
    daily = series_df.groupby("Order_Date")[target].sum().to_frame(target)

    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0)
    daily.index.name = "Order_Date"
    return daily


def _demo_forecast(year: int) -> ForecastResponse:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    base = np.array([11200, 11950, 12600, 12100, 13250, 13620, 14110, 13840, 14520, 14980, 15650, 16300], dtype=float)
    points = [
        ForecastPoint(month=month, forecast=float(value), actual=float(value * 0.97))
        for month, value in zip(months, base)
    ]
    return ForecastResponse(year=year, model="LSTM", points=points)


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[object, object, dict]:
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()):
        raise FileNotFoundError("Exported LSTM artifacts are missing from backend/artifacts.")

    model = load_model(MODEL_PATH, compile=False)
    scaler = load(SCALER_PATH)
    meta = load(META_PATH)
    return model, scaler, meta


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df = df.dropna(subset=["Order_Date"]).copy()
    df["Year"] = df["Order_Date"].dt.year
    df["Month"] = df["Order_Date"].dt.month
    return df


def build_forecast(year: int) -> ForecastResponse:
    try:
        df = load_dataset()
        model, scaler, meta = load_artifacts()
    except FileNotFoundError:
        return _demo_forecast(year)

    target = meta.get("target", "Profit")
    window = int(meta.get("window", 10))

    daily = aggregate_daily(df, target=target)
    if daily.empty:
        return _demo_forecast(year)

    min_year = int(daily.index.min().year)
    max_actual_date = pd.Timestamp(daily.index.max())
    max_supported_year = max_actual_date.year + 1
    if year < min_year or year > max_supported_year:
        raise HTTPException(
            status_code=404,
            detail=f"Year {year} is outside the supported forecast range: {min_year}-{max_supported_year}.",
        )

    actual_series = daily[target].astype(float)
    scaled_actual = scaler.transform(actual_series.to_frame().values).reshape(-1).tolist()
    target_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    forecast_by_date: dict[pd.Timestamp, float] = {}
    history_scaled = list(scaled_actual)
    actual_index = set(actual_series.index)

    # Roll through the requested year one day at a time. For dates with observed
    # data, the next prediction uses only prior actuals. For future-only dates,
    # predictions feed back recursively.
    for target_date in target_dates:
        if target_date <= max_actual_date:
            history_end = target_date - pd.Timedelta(days=1)
            history_values = actual_series.loc[:history_end]
            if len(history_values) < window:
                continue
            scaled_history = scaler.transform(history_values.to_frame().values).reshape(-1).tolist()
            input_window = np.array(scaled_history[-window:], dtype=np.float32).reshape(1, window, 1)
            pred_scaled = float(model.predict(input_window, verbose=0).reshape(-1)[0])
            pred_value = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            forecast_by_date[target_date] = pred_value
        else:
            if len(history_scaled) < window:
                break
            input_window = np.array(history_scaled[-window:], dtype=np.float32).reshape(1, window, 1)
            pred_scaled = float(model.predict(input_window, verbose=0).reshape(-1)[0])
            pred_value = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            forecast_by_date[target_date] = pred_value
            history_scaled.append(pred_scaled)

    forecast_series = pd.Series(forecast_by_date).sort_index()
    if forecast_series.empty:
        return _demo_forecast(year)

    forecast_monthly = forecast_series.groupby(forecast_series.index.month).sum()
    actual_year = actual_series[actual_series.index.year == year]
    actual_monthly = actual_year.groupby(actual_year.index.month).sum()

    points = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for month_number in range(1, 13):
        forecast_value = float(forecast_monthly.get(month_number, 0.0))
        actual_value = actual_monthly.get(month_number)
        points.append(
            ForecastPoint(
                month=month_names[month_number - 1],
                forecast=round(forecast_value, 2),
                actual=None if actual_value is None or pd.isna(actual_value) else round(float(actual_value), 2),
            )
        )

    return ForecastResponse(year=year, model="LSTM", points=points)


def build_inventory_health() -> list[InventoryHealthItem]:
    df = load_dataset()
    sku_perf = (
        df.groupby(["SKU", "Product_Name", "Category"], as_index=False)
        .agg(
            total_units=("Units_Sold", "sum"),
            total_revenue=("Revenue", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order_ID", "count"),
            avg_discount=("Discount", "mean"),
        )
    )

    sku_perf["sales_velocity"] = sku_perf["total_units"] / sku_perf["order_count"].clip(lower=1)
    sku_perf["inventory_turnover"] = sku_perf["total_revenue"] / sku_perf["total_units"].clip(lower=1)

    velocity_top = sku_perf["sales_velocity"].quantile(0.66)
    velocity_low = sku_perf["sales_velocity"].quantile(0.33)
    profit_top = sku_perf["total_profit"].quantile(0.66)
    profit_low = sku_perf["total_profit"].quantile(0.33)

    def classify(row: pd.Series) -> tuple[str, str]:
        if row["sales_velocity"] >= velocity_top and row["total_profit"] >= profit_top:
            return "High Performer", "Hold"
        if row["sales_velocity"] <= velocity_low or row["total_profit"] <= profit_low:
            return "At Risk", "Discount"
        return "Steady", "Monitor"

    classifications = sku_perf.apply(classify, axis=1, result_type="expand")
    sku_perf["tier"] = classifications[0]
    sku_perf["action"] = classifications[1]

    # Return a balanced demo sample so the dashboard shows all action states.
    per_tier = {
        "High Performer": 6,
        "Steady": 7,
        "At Risk": 7,
    }
    tier_frames = []
    for tier_name, sample_size in per_tier.items():
        tier_frame = sku_perf[sku_perf["tier"] == tier_name].sort_values(
            ["total_profit", "sales_velocity"], ascending=[False, False]
        )
        tier_frames.append(tier_frame.head(sample_size))

    sku_perf = (
        pd.concat(tier_frames, ignore_index=True)
        .sort_values(["tier", "total_profit"], ascending=[True, False])
    )

    return [
        InventoryHealthItem(
            product_id=row.SKU,
            product_name=row.Product_Name,
            category=row.Category,
            inventory_turnover=round(float(row.inventory_turnover), 2),
            sales_velocity=round(float(row.sales_velocity), 2),
            tier=row.tier,
            action=row.action,
        )
        for row in sku_perf.itertuples(index=False)
    ]


def build_cluster_analysis() -> list[ClusterPoint]:
    df = load_dataset()
    sku_perf = (
        df.groupby(["SKU", "Product_Name", "Category"], as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            profit=("Profit", "sum"),
            units=("Units_Sold", "sum"),
            avg_discount=("Discount", "mean"),
        )
    )

    sku_perf["x"] = sku_perf["revenue"] / sku_perf["units"].clip(lower=1)
    sku_perf["y"] = sku_perf["profit"] / sku_perf["units"].clip(lower=1)

    revenue_hi = sku_perf["revenue"].quantile(0.66)
    profit_hi = sku_perf["profit"].quantile(0.66)
    discount_hi = sku_perf["avg_discount"].quantile(0.66)

    def assign_cluster(row: pd.Series) -> str:
        if row["revenue"] >= revenue_hi and row["profit"] >= profit_hi:
            return "Premium Loyalists"
        if row["avg_discount"] >= discount_hi:
            return "Promotion Driven"
        return "Core Value"

    sku_perf["cluster"] = sku_perf.apply(assign_cluster, axis=1)

    # Return a balanced demo sample so the scatterplot displays all cluster types.
    per_cluster = {
        "Premium Loyalists": 14,
        "Promotion Driven": 13,
        "Core Value": 13,
    }
    cluster_frames = []
    for cluster_name, sample_size in per_cluster.items():
        cluster_frame = sku_perf[sku_perf["cluster"] == cluster_name].sort_values(
            ["revenue", "profit"], ascending=[False, False]
        )
        cluster_frames.append(cluster_frame.head(sample_size))

    sku_perf = (
        pd.concat(cluster_frames, ignore_index=True)
        .sort_values(["cluster", "revenue"], ascending=[True, False])
    )

    return [
        ClusterPoint(
            product_id=row.SKU,
            product_name=row.Product_Name,
            cluster=row.cluster,
            x=round(float(row.x), 2),
            y=round(float(row.y), 2),
            category=row.Category,
        )
        for row in sku_perf.itertuples(index=False)
    ]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Adidas analytics demo API is running."}


@app.get("/forecast/{year}", response_model=ForecastResponse)
def get_forecast(year: int) -> ForecastResponse:
    return build_forecast(year)


@app.get("/inventory-health", response_model=list[InventoryHealthItem])
def get_inventory_health() -> list[InventoryHealthItem]:
    return build_inventory_health()


@app.get("/cluster-analysis", response_model=list[ClusterPoint])
def get_cluster_analysis() -> list[ClusterPoint]:
    return build_cluster_analysis()
