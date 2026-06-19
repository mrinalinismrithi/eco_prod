import numpy as np
import pandas as pd
import json

from app.analytics import build_climate_analytics, build_yearly_climate_data
from app.data_loader import (
    get_country_trends_path,
    get_fastest_regions_path,
    get_hottest_countries_path,
    get_regional_trends_path,
    PROCESSED_DATA_PATH,
    load_raw_data,
)


def run_etl():
    """Run the full ETL pipeline: clean raw data, process it, and save analytics."""
    print("Starting ETL Pipeline...")

    df = load_raw_data().copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")

    for col in ["Country", "Region"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    if "AvgTemperature" in df.columns:
        df["AvgTemperature"] = (
            df["AvgTemperature"]
            .astype(str)
            .str.replace(r"Â°C|Ã‚Â°C|Â°", "", regex=True)
            .str.replace(",", "")
            .str.strip()
            .replace("", np.nan)
        )
        df["AvgTemperature"] = pd.to_numeric(df["AvgTemperature"], errors="coerce")

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

    required_cols = ["Country", "Region", "Year", "AvgTemperature"]
    available_required = [col for col in required_cols if col in df.columns]

    df = df.dropna(subset=available_required)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["AvgTemperature"])

    print(f"Processed raw data | Shape: {df.shape}")

    yearly = build_yearly_climate_data(df)
    yearly = yearly.sort_values(["Country", "Year"]).reset_index(drop=True)
    analytics = build_climate_analytics(yearly)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data -> {PROCESSED_DATA_PATH.name}")

    # use dynamic path functions so output goes to active folder
    analytics["country_trends"].to_csv(get_country_trends_path(), index=False)
    analytics["regional_trends"].to_csv(get_regional_trends_path(), index=False)
    analytics["fastest_warming_regions"].to_csv(get_fastest_regions_path(), index=False)
    analytics["hottest_countries"].to_csv(get_hottest_countries_path(), index=False)

    # Save schema for default dataset
    schema = {
        "active_source": "default",
        "filename": "climate_data.csv",
        "row_count": len(yearly),
        "columns": list(yearly.columns),
        "numeric_columns": [c for c in yearly.columns if pd.api.types.is_numeric_dtype(yearly[c])],
        "categorical_columns": [c for c in yearly.columns if not pd.api.types.is_numeric_dtype(yearly[c])],
        "date_columns": []
    }
    schema_path = PROCESSED_DATA_PATH.parent / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print("ETL Pipeline completed successfully!")
    return analytics


if __name__ == "__main__":
    run_etl() 