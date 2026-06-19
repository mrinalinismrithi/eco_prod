import pandas as pd
from typing import Dict


def _trend_slope(group: pd.DataFrame) -> float:
    """Calculate linear warming trend (slope) per year."""
    if len(group) < 2:
        return 0.0

    years = group["Year"].astype(float)
    temps = group["AvgTemperature"].astype(float)

    # Linear regression slope
    numerator = ((years - years.mean()) * (temps - temps.mean())).sum()
    denominator = ((years - years.mean()) ** 2).sum()
    
    return float(numerator / denominator) if denominator != 0 else 0.0


def build_yearly_climate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build country/region/year average-temperature data with derived changes."""
    yearly = (
        df.groupby(["Country", "Region", "Year"], as_index=False)["AvgTemperature"]
        .mean()
        .sort_values(by=["Country", "Year"])
        .reset_index(drop=True)
    )
    yearly["AvgTemperature"] = yearly["AvgTemperature"].round(3)

    # Country-level changes
    yearly["TempChange"] = yearly.groupby("Country")["AvgTemperature"].diff().fillna(0)
    yearly["BaselineTemperature"] = yearly.groupby("Country")["AvgTemperature"].transform("first")
    yearly["TotalTempChange"] = yearly["AvgTemperature"] - yearly["BaselineTemperature"]

    # Regional averages
    regional_yearly = (
        yearly.groupby(["Region", "Year"], as_index=False)["AvgTemperature"]
        .mean()
        .rename(columns={"AvgTemperature": "RegionalAverageTemperature"})
    )
    regional_yearly["RegionalTempChange"] = (
        regional_yearly.sort_values(["Region", "Year"])
        .groupby("Region")["RegionalAverageTemperature"]
        .diff()
        .fillna(0)
    )

    yearly = yearly.merge(regional_yearly, on=["Region", "Year"], how="left")
    return yearly


def calculate_country_warming_trends(yearly: pd.DataFrame) -> pd.DataFrame:
    """Calculate detailed warming statistics per country."""
    country_stats = []

    for (region, country), group in yearly.groupby(["Region", "Country"]):
        group = group.sort_values("Year")
        first = group.iloc[0]
        latest = group.iloc[-1]
        yearly_changes = group["AvgTemperature"].diff().dropna()

        country_stats.append({
            "Region": region,
            "Country": country,
            "FirstYear": int(first["Year"]),
            "LatestYear": int(latest["Year"]),
            "FirstTemperature": round(float(first["AvgTemperature"]), 3),
            "LatestTemperature": round(float(latest["AvgTemperature"]), 3),
            "AverageTemperature": round(float(group["AvgTemperature"].mean()), 3),
            "TemperatureChange": round(float(latest["AvgTemperature"] - first["AvgTemperature"]), 3),
            "WarmingTrendPerYear": round(_trend_slope(group), 4),
            "VolatilityScore": round(float(yearly_changes.std()) if len(yearly_changes) else 0.0, 4),
            "HottestYear": int(group.loc[group["AvgTemperature"].idxmax(), "Year"]),
            "HottestTemperature": round(float(group["AvgTemperature"].max()), 3),
        })

    return pd.DataFrame(country_stats).sort_values(
        by=["TemperatureChange", "WarmingTrendPerYear"], ascending=False
    ).reset_index(drop=True)


def calculate_regional_warming_trends(yearly: pd.DataFrame) -> pd.DataFrame:
    """Calculate warming statistics per region."""
    regional_yearly = (
        yearly.groupby(["Region", "Year"], as_index=False)["AvgTemperature"]
        .mean()
        .rename(columns={"AvgTemperature": "RegionalAverageTemperature"})
    )

    regional_stats = []
    for region, group in regional_yearly.groupby("Region"):
        group = group.sort_values("Year")
        first = group.iloc[0]
        latest = group.iloc[-1]
        yearly_changes = group["RegionalAverageTemperature"].diff().dropna()

        regional_stats.append({
            "Region": region,
            "FirstYear": int(first["Year"]),
            "LatestYear": int(latest["Year"]),
            "FirstRegionalTemperature": round(float(first["RegionalAverageTemperature"]), 3),
            "LatestRegionalTemperature": round(float(latest["RegionalAverageTemperature"]), 3),
            "AverageRegionalTemperature": round(float(group["RegionalAverageTemperature"].mean()), 3),
            "RegionalTemperatureChange": round(float(latest["RegionalAverageTemperature"] - first["RegionalAverageTemperature"]), 3),
            "RegionalWarmingTrendPerYear": round(_trend_slope(group.rename(columns={"RegionalAverageTemperature": "AvgTemperature"})), 4),
            "RegionalVolatilityScore": round(float(yearly_changes.std()) if len(yearly_changes) else 0.0, 4),
        })

    return pd.DataFrame(regional_stats).sort_values(
        by=["RegionalTemperatureChange", "RegionalWarmingTrendPerYear"], ascending=False
    ).reset_index(drop=True)


def get_fastest_warming_regions(regional_trends: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    return regional_trends.head(limit).reset_index(drop=True)


def get_hottest_countries(country_trends: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    return country_trends.sort_values(
        by=["LatestTemperature", "AverageTemperature"], ascending=False
    ).head(limit).reset_index(drop=True)


def build_climate_analytics(yearly: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Main analytics builder."""
    country_trends = calculate_country_warming_trends(yearly)
    regional_trends = calculate_regional_warming_trends(yearly)

    return {
        "yearly": yearly,
        "country_trends": country_trends,
        "regional_trends": regional_trends,
        "fastest_warming_regions": get_fastest_warming_regions(regional_trends),
        "hottest_countries": get_hottest_countries(country_trends),
    } 
