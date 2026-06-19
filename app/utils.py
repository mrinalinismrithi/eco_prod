import pandas as pd
from typing import Dict, Optional


def calculate_region_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate average temperature for each region.
    Returns a dictionary sorted by temperature (highest first).
    """
    if df.empty or "Region" not in df.columns or "AvgTemperature" not in df.columns:
        return {}

    region_stats = (
        df.groupby("Region")["AvgTemperature"]
        .mean()
        .round(2)                    # Round for cleaner output
        .sort_values(ascending=False)
    )

    return region_stats.to_dict()


def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Filter dataframe by region (case-insensitive).
    """
    if df.empty or "Region" not in df.columns:
        return pd.DataFrame()

    # Case-insensitive matching
    mask = df["Region"].str.lower() == region.lower().strip()
    return df[mask].copy()


# Optional: Add more useful utility functions
def get_top_n_hottest_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top N hottest countries based on average temperature."""
    if "AvgTemperature" not in df.columns or "Country" not in df.columns:
        return pd.DataFrame()
    
    return (df.groupby("Country")["AvgTemperature"]
            .mean()
            .round(2)
            .sort_values(ascending=False)
            .head(n)
            .reset_index())  
def get_top_countries(df, n=10, sort_by="TemperatureChange"):
    df_clean = df.dropna(subset=[sort_by])

    result = (
        df_clean
        .groupby("Country", as_index=False)
        .agg({
            "TemperatureChange": "mean",
            "LatestTemperature": "max",
            "WarmingTrendPerYear": "mean"
        })
        .sort_values(sort_by, ascending=False)
        .head(n)
    )

    return result 