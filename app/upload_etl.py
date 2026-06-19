import json
import os
import numpy as np
import pandas as pd

from app.dataset_state import set_active_source
from app.data_loader import get_processed_dir
from app.analytics import build_climate_analytics, build_yearly_climate_data
from app.data_loader import apply_region_corrections

# Expected climate columns — same as existing climate_data.csv
REQUIRED_COLUMNS = {
    "country": ["country", "Country"],
    "region":  ["region",  "Region"],
    "year":    ["year",    "Year"],
    "avgtemp": ["avgtemp", "AvgTemp", "avgtemperature", "AvgTemperature", "avg_temp", "temperature"],
}


def _detect_column(df: pd.DataFrame, candidates: list) -> str | None:
    """Find first matching column name (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def run_upload_etl(input_path: str) -> dict:
    output_dir = get_processed_dir().parent / "upload"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # ── Detect required columns ──────────────────────────────
    col_map = {}
    missing = []
    for standard, candidates in REQUIRED_COLUMNS.items():
        found = _detect_column(df, candidates)
        if found:
            col_map[standard] = found
        else:
            missing.append(standard)

    if missing:
        raise ValueError(
            f"Uploaded CSV is missing required climate columns: {missing}. "
            f"Expected columns: country, region, year, avgtemp (or similar names). "
            f"Found columns: {list(df.columns)}"
        )

    # ── Standardise column names to match existing pipeline ──
    df = df.rename(columns={
        col_map["country"]: "Country",
        col_map["region"]:  "Region",
        col_map["year"]:    "Year",
        col_map["avgtemp"]: "AvgTemperature",
    })

    # ── Clean data — same as existing ETL ────────────────────
    df = df.dropna(how="all")

    for col in ["Country", "Region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    df["AvgTemperature"] = (
        df["AvgTemperature"]
        .astype(str)
        .str.replace(r"°C|deg|degree", "", regex=True, flags=0)
        .str.replace(",", "")
        .str.strip()
        .replace("", np.nan)
    )
    df["AvgTemperature"] = pd.to_numeric(df["AvgTemperature"], errors="coerce")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year", "Country", "AvgTemperature"])
    df["Year"] = df["Year"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["AvgTemperature"])

    # ── Apply region corrections ──────────────────────────────
    df = apply_region_corrections(df)

    # ── Run existing analytics pipeline ──────────────────────
    yearly = build_yearly_climate_data(df)
    yearly = yearly.sort_values(["Country", "Year"]).reset_index(drop=True)
    analytics = build_climate_analytics(yearly)

    # ── Save outputs to upload/ folder ───────────────────────
    yearly.to_csv(output_dir / "processed_climate_data.csv", index=False)
    analytics["country_trends"].to_csv(output_dir / "country_warming_trends.csv", index=False)
    analytics["regional_trends"].to_csv(output_dir / "regional_warming_trends.csv", index=False)
    analytics["fastest_warming_regions"].to_csv(output_dir / "fastest_warming_regions.csv", index=False)
    analytics["hottest_countries"].to_csv(output_dir / "hottest_countries.csv", index=False)

    # ── Save schema for AI context ───────────────────────────
    countries = sorted(df["Country"].dropna().unique().tolist())
    schema = {
        "columns": list(df.columns),
        "row_count": len(df),
        "countries": countries,
        "year_range": [int(df["Year"].min()), int(df["Year"].max())],
        "filename": os.path.basename(input_path),
        "categorical_columns": ["Country", "Region"],
        "numeric_columns": ["AvgTemperature"],
    }
    with open(output_dir / "schema.json", "w") as f:
        json.dump(schema, f)

    # ── Set active source ─────────────────────────────────────
    set_active_source("upload")

    return {
        "rows_processed": len(df),
        "countries": len(countries),
        "year_range": schema["year_range"],
        "columns_detected": col_map,
    }