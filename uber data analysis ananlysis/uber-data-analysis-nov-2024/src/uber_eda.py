from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class DatasetMeta:
    datetime_col: str
    lat_col: Optional[str]
    lon_col: Optional[str]
    surge_col: Optional[str]


def _normalize_colname(name: str) -> str:
    # Helps match variants like "Date/Time", "pickup_datetime", etc.
    return (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace("/", "_")
    )


def _find_first_column(df: pd.DataFrame, candidates: list[str], *, contains_tokens: bool = False) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in cols}
    for cand in candidates:
        cand_norm = _normalize_colname(cand)
        if cand_norm in norm_map:
            return norm_map[cand_norm]

    if contains_tokens:
        for c in cols:
            cn = _normalize_colname(c)
            for cand in candidates:
                token = _normalize_colname(cand)
                if token and token in cn:
                    return c
    return None


def detect_columns(df: pd.DataFrame) -> DatasetMeta:
    datetime_col = _find_first_column(
        df,
        candidates=["Date/Time", "datetime", "pickup_datetime", "timestamp"],
        contains_tokens=True,
    )
    if not datetime_col:
        raise ValueError(
            "Could not auto-detect the datetime column. Expected something like 'Date/Time', "
            "'datetime', 'pickup_datetime', or 'timestamp'."
        )

    lat_col = _find_first_column(df, candidates=["Lat", "Latitude", "lat"], contains_tokens=True)
    lon_col = _find_first_column(df, candidates=["Lon", "Longitude", "lon"], contains_tokens=True)

    # Surge/pricing-like columns vary a lot across datasets; detect a "signal" column if present.
    surge_col = _find_first_column(
        df,
        candidates=["Surge", "surge_multiplier", "surge", "fare_amount", "price", "total_amount", "fare"],
        contains_tokens=True,
    )

    return DatasetMeta(
        datetime_col=datetime_col,
        lat_col=lat_col,
        lon_col=lon_col,
        surge_col=surge_col,
    )


def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}. Use CSV or Parquet.")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetMeta]:
    df = df.copy()
    meta = detect_columns(df)

    df[meta.datetime_col] = pd.to_datetime(df[meta.datetime_col], errors="coerce")
    df = df.dropna(subset=[meta.datetime_col])

    dt = df[meta.datetime_col]
    df["date"] = dt.dt.date.astype("object")
    df["hour"] = dt.dt.hour.astype(int)
    df["day_of_week"] = dt.dt.day_name()
    df["weekday_index"] = dt.dt.dayofweek.astype(int)
    df["month"] = dt.dt.month.astype(int)

    # Ensure lat/lon are numeric if present.
    if meta.lat_col:
        df[meta.lat_col] = pd.to_numeric(df[meta.lat_col], errors="coerce")
    if meta.lon_col:
        df[meta.lon_col] = pd.to_numeric(df[meta.lon_col], errors="coerce")

    if meta.lat_col and meta.lon_col:
        df = df.dropna(subset=[meta.lat_col, meta.lon_col])

    # Clean surge/pricing signal if detected.
    if meta.surge_col:
        df[meta.surge_col] = pd.to_numeric(df[meta.surge_col], errors="coerce")
        df = df.dropna(subset=[meta.surge_col])

    # Lightweight deduplication to avoid repeated rows ruining plots.
    dedup_cols = [meta.datetime_col]
    if meta.lat_col and meta.lon_col:
        dedup_cols += [meta.lat_col, meta.lon_col]
    df = df.drop_duplicates(subset=dedup_cols)

    return df, meta


def add_geo_bins(df: pd.DataFrame, lat_col: str, lon_col: str, precision: int = 2) -> pd.DataFrame:
    # Bins rides into a grid so we can find "high traffic geo-locations".
    df = df.copy()
    df["lat_bin"] = df[lat_col].round(precision)
    df["lon_bin"] = df[lon_col].round(precision)
    df["geo_bin"] = df["lat_bin"].astype(str) + "," + df["lon_bin"].astype(str)
    return df


def _save_fig(fig: plt.Figure, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_peak_hours(df: pd.DataFrame, outdir: str) -> dict[str, Any]:
    sns.set_theme(style="whitegrid")

    counts = df.groupby("hour").size().sort_index()
    top_hour = int(counts.idxmax())

    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Demand by Hour (Peak Hours)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Rides")

    _save_fig(fig, os.path.join(outdir, "01_peak_hours.png"))

    return {
        "peak_hour": top_hour,
        "peak_hour_rides": int(counts.loc[top_hour]),
        "demand_by_hour": counts.to_dict(),
    }


def plot_demand_time_series(df: pd.DataFrame, outdir: str) -> None:
    sns.set_theme(style="whitegrid")

    daily = df.groupby("date").size().rename("rides").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["rolling_7d"] = daily["rides"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["rides"], label="Daily demand", linewidth=1)
    ax.plot(daily["date"], daily["rolling_7d"], label="7-day rolling avg", linewidth=2)
    ax.set_title("Time-Series Demand (Daily Rides)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Rides")
    ax.legend()

    _save_fig(fig, os.path.join(outdir, "02_time_series_daily_demand.png"))


def plot_geo_hotspots(
    df: pd.DataFrame,
    outdir: str,
    *,
    top_n: int = 10,
    precision: int = 2,
) -> list[dict[str, Any]]:
    if not (("lat_col" in df.columns) and ("lon_col" in df.columns)):
        # Expect bins to already be present.
        pass

    if "lat_bin" not in df.columns or "lon_bin" not in df.columns:
        # If bins are missing, we can't plot geo hotspots.
        raise ValueError("Geo bins not found. Call add_geo_bins() first.")

    sns.set_theme(style="whitegrid")

    counts = df.groupby(["lat_bin", "lon_bin"]).size().rename("rides").reset_index()
    counts = counts.sort_values("rides", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(counts["lon_bin"], counts["lat_bin"], s=counts["rides"] * 6, alpha=0.75)
    ax.set_title(f"High-Traffic Geo Hotspots (Top {top_n} Bins)")
    ax.set_xlabel(f"Longitude (rounded to {precision} decimals)")
    ax.set_ylabel(f"Latitude (rounded to {precision} decimals)")

    _save_fig(fig, os.path.join(outdir, "03_geo_hotspots.png"))

    return [
        {"lat_bin": float(r.lat_bin), "lon_bin": float(r.lon_bin), "rides": int(r.rides)}
        for r in counts.itertuples(index=False)
    ]


def plot_correlation_heatmap(df: pd.DataFrame, outdir: str) -> None:
    sns.set_theme(style="whitegrid")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")

    _save_fig(fig, os.path.join(outdir, "04_correlation_heatmap.png"))


def _pick_surge_signal_column(df: pd.DataFrame) -> Optional[str]:
    # Keep this simple: if a column name suggests surge/pricing and it's numeric, use it.
    surge_tokens = ["surge", "fare", "price"]
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    for c in df.columns:
        if c in numeric_cols:
            lc = _normalize_colname(c)
            if any(tok in lc for tok in surge_tokens):
                return c
    return None


def plot_surge_or_pricing_patterns(df: pd.DataFrame, outdir: str, *, surge_col: Optional[str]) -> Optional[dict[str, Any]]:
    sns.set_theme(style="whitegrid")

    if not surge_col or surge_col not in df.columns:
        # No surge/pricing signal, so we instead show a demand heatmap by day/hour.
        pivot = df.pivot_table(index="day_of_week", columns="hour", values="hour", aggfunc="size", fill_value=0)
        # Order days properly (Mon..Sun)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex(day_order)

        fig, ax = plt.subplots(figsize=(12, 4.5))
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
        ax.set_title("Demand Heatmap (Day vs Hour)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Day of Week")
        _save_fig(fig, os.path.join(outdir, "05_surging_or_pricing_patterns.png"))
        return None

    signal = df[[surge_col, "day_of_week", "hour"]].dropna()
    pivot = signal.pivot_table(index="day_of_week", columns="hour", values=surge_col, aggfunc="mean")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
    ax.set_title(f"Surge/Pricing Pattern (Average {surge_col}) by Day & Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day of Week")
    _save_fig(fig, os.path.join(outdir, "05_surging_or_pricing_patterns.png"))

    # Find the most "expensive" (highest mean) hour/day combination.
    max_loc = pivot.stack().idxmax() if not pivot.empty else None
    return {
        "surge_col": surge_col,
        "highest_avg_day": str(max_loc[0]) if max_loc else None,
        "highest_avg_hour": int(max_loc[1]) if max_loc else None,
        "highest_avg_value": float(pivot.stack().max()) if not pivot.empty else None,
    }


def analyze(df: pd.DataFrame, *, output_dir: str, geo_precision: int = 2, top_n_geo: int = 10) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    df, meta = clean_data(df)

    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "datetime_col": meta.datetime_col,
        "lat_col": meta.lat_col,
        "lon_col": meta.lon_col,
        "surge_col": meta.surge_col,
        "date_min": str(pd.to_datetime(df[meta.datetime_col]).min().date()),
        "date_max": str(pd.to_datetime(df[meta.datetime_col]).max().date()),
    }

    peak_info = plot_peak_hours(df, output_dir)
    summary.update(peak_info)

    plot_demand_time_series(df, output_dir)

    if meta.lat_col and meta.lon_col:
        binned = add_geo_bins(df, meta.lat_col, meta.lon_col, precision=geo_precision)
        top_geo = plot_geo_hotspots(
            binned,
            output_dir,
            top_n=top_n_geo,
            precision=geo_precision,
        )
        summary["top_geo_hotspots"] = top_geo
    else:
        summary["top_geo_hotspots"] = []

    plot_correlation_heatmap(df, output_dir)

    # Surge/pricing heatmap (or fallback demand heatmap).
    surge_info = plot_surge_or_pricing_patterns(df, output_dir, surge_col=meta.surge_col)
    summary["surge_pattern_summary"] = surge_info

    return summary

