from app.intent import Intent
from app.weather import extract_location, format_weather_context, get_current_weather
from difflib import SequenceMatcher
import re


# ----------------------------
# SAFE HELPERS
# ----------------------------

def _match_column(df, column, question):
    """Safe word-boundary match (avoids partial matching bugs)."""
    if df is None or column not in df.columns:
        return df.iloc[0:0] if df is not None else None

    q = question.lower().strip()

    def normalize(value):
        return " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", str(value).lower()).split())

    def fuzzy_phrase_match(value):
        candidate = normalize(value)
        query = normalize(q)
        if not candidate or not query:
            return False
        if candidate in query:
            return True

        words = query.split()
        candidate_words = candidate.split()
        width = max(1, len(candidate_words))
        for i in range(0, len(words) - width + 1):
            phrase = " ".join(words[i:i + width])
            if SequenceMatcher(None, candidate, phrase).ratio() >= 0.82:
                return True

        if width == 1:
            for word in words:
                if len(word) >= 4 and SequenceMatcher(None, candidate, word).ratio() >= 0.82:
                    return True
        return False

    return df[
        df[column].apply(
            lambda x: isinstance(x, str)
            and (
                bool(re.search(rf"\b{re.escape(x.lower())}\b", q))
                or fuzzy_phrase_match(x)
            )
        )
    ]


def _safe_df(df, n=10):
    if df is None or df.empty:
        return None
    return df.head(n)


def _latest_yearly_rows(yearly_df, countries, rows_per_country=5):
    if yearly_df is None or not countries:
        return None

    country_set = {c.lower() for c in countries if isinstance(c, str)}

    if "Country" not in yearly_df.columns:
        return None

    filtered = yearly_df[
        yearly_df["Country"].apply(
            lambda c: isinstance(c, str) and c.lower() in country_set
        )
    ]

    if filtered.empty:
        return None

    return (
        filtered.sort_values(["Country", "Year"])
        .groupby("Country", group_keys=False)
        .tail(rows_per_country)
    )


def _df_to_text(df):
    if df is None or df.empty:
        return "No match found"
    return df.to_string(index=False)


# ----------------------------
# WEATHER CHAIN
# ----------------------------

def build_weather_chain(question, location, country_trends):

    loc = extract_location(question) or location
    weather = get_current_weather(loc)

    if not weather:
        return {
            "chain_name": "weather",
            "evidence": "Weather API failed or location not found.",
            "structured_facts": {}
        }

    country_avg = None
    country = weather.get("location", {}).get("country")

    if country and country_trends is not None:
        match = country_trends[
            country_trends["Country"].apply(
                lambda c: isinstance(c, str) and c.lower() == country.lower()
            )
        ]

        if not match.empty:
            country_avg = float(match.iloc[0]["AverageTemperature"])

    return {
        "chain_name": "weather",
        "evidence": format_weather_context(
            weather,
            country_average_temperature=country_avg
        ),
        "structured_facts": {
            "weather": weather,
            "country_average": country_avg
        }
    }


# ----------------------------
# HYBRID CHAIN (WEATHER + CLIMATE)
# ----------------------------

def build_weather_and_climate_chain(
    question,
    location,
    yearly_df,
    country_trends,
    regional_trends
):

    loc = extract_location(question) or location
    weather = get_current_weather(loc)

    if not weather:
        return {
            "chain_name": "hybrid",
            "evidence": "Weather API failed.",
            "structured_facts": {}
        }

    country_rows = _match_column(country_trends, "Country", question)
    region_rows = _match_column(regional_trends, "Region", question)

    weather_country = weather.get("location", {}).get("country")

    # fallback country match from weather API
    if (country_rows is None or country_rows.empty) and weather_country:
        country_rows = country_trends[
            country_trends["Country"].apply(
                lambda c: isinstance(c, str)
                and c.lower() == weather_country.lower()
            )
        ] if country_trends is not None else None

    # fallback region mapping
    if (region_rows is None or region_rows.empty) and country_rows is not None and not country_rows.empty:
        regions = country_rows["Region"].dropna().unique().tolist()
        region_rows = regional_trends[
            regional_trends["Region"].isin(regions)
        ] if regional_trends is not None else None

    countries = (
        country_rows["Country"].dropna().unique().tolist()
        if country_rows is not None and not country_rows.empty
        else []
    )

    yearly = _latest_yearly_rows(yearly_df, countries)

    return {
        "chain_name": "hybrid",
        "evidence": "\n\n".join([
            "LIVE WEATHER:",
            format_weather_context(weather),

            "\nCSV COUNTRY DATA:",
            _df_to_text(country_rows),

            "\nCSV REGION DATA:",
            _df_to_text(region_rows),

            "\nYEARLY DATA:",
            _df_to_text(yearly)
        ]),
        "structured_facts": {
            "weather": weather,
            "country_rows": country_rows.to_dict(orient="records")
            if country_rows is not None and not country_rows.empty else [],

            "region_rows": region_rows.to_dict(orient="records")
            if region_rows is not None and not region_rows.empty else [],

            "yearly_rows": yearly.to_dict(orient="records")
            if yearly is not None and not yearly.empty else []
        }
    }


# ----------------------------
# CLIMATE CHAIN
# ----------------------------

def build_historical_climate_chain(
    question,
    yearly_df,
    country_trends,
    regional_trends
):

    country_rows = _match_column(country_trends, "Country", question)
    region_rows = _match_column(regional_trends, "Region", question)

    if country_rows is None or country_rows.empty:
        country_rows = _safe_df(country_trends, 10)

    if region_rows is None or region_rows.empty:
        region_rows = _safe_df(regional_trends, 6)

    return {
        "chain_name": "climate",
        "evidence": "\n\n".join([
            "COUNTRY DATA:",
            _df_to_text(country_rows),

            "\nREGION DATA:",
            _df_to_text(region_rows),

            "\nYEARLY SAMPLE:",
            _df_to_text(_safe_df(yearly_df, 20))
        ]),
        "structured_facts": {
            "country_rows": country_rows.to_dict(orient="records")
            if country_rows is not None and not country_rows.empty else [],

            "region_rows": region_rows.to_dict(orient="records")
            if region_rows is not None and not region_rows.empty else []
        }
    }


# ----------------------------
# VOLATILITY CHAIN
# ----------------------------

def build_volatility_chain(country_trends, regional_trends):

    top_countries = None
    top_regions = None

    if country_trends is not None and "VolatilityScore" in country_trends.columns:
        top_countries = country_trends.sort_values(
            "VolatilityScore",
            ascending=False
        ).head(10)

    if regional_trends is not None and "RegionalVolatilityScore" in regional_trends.columns:
        top_regions = regional_trends.sort_values(
            "RegionalVolatilityScore",
            ascending=False
        ).head(10)

    return {
        "chain_name": "volatility",
        "evidence": "\n\n".join([
            "TOP VOLATILE COUNTRIES:",
            _df_to_text(top_countries),

            "\nTOP VOLATILE REGIONS:",
            _df_to_text(top_regions)
        ]),
        "structured_facts": {}
    }


# ----------------------------
# COMPARISON CHAIN
# ----------------------------

def build_comparison_chain(question, country_trends, regional_trends):

    country_rows = _match_column(country_trends, "Country", question)
    region_rows = _match_column(regional_trends, "Region", question)

    if country_rows is None or country_rows.empty:
        country_rows = _safe_df(country_trends, 10)

    if region_rows is None or region_rows.empty:
        region_rows = _safe_df(regional_trends, 6)

    return {
        "chain_name": "comparison",
        "evidence": "\n\n".join([
            "COUNTRY COMPARISON:",
            _df_to_text(country_rows),

            "\nREGION COMPARISON:",
            _df_to_text(region_rows)
        ]),
        "structured_facts": {}
    }
# These two functions go into chains.py
# Paste them BEFORE the # ROUTER section

def build_prediction_chain(question: str, country_trends, regional_trends):
    """
    Build context for long-term climate temperature prediction.
    Uses WarmingTrendPerYear (linear regression slope) already in
    country_warming_trends.csv — no new model needed.
    Formula: predicted_temp = LatestTemperature + (WarmingTrendPerYear x years_ahead)
    """
    import re
    import pandas as pd
    from datetime import date
    from difflib import SequenceMatcher

    q = question.lower()
    current_year = date.today().year

    # Extract target year from question
    year_match = re.search(r"\b(202[5-9]|20[3-9]\d|2100)\b", q)
    target_year = int(year_match.group()) if year_match else None

    # Extract "next N years"
    years_ahead_match = re.search(r"next\s+(\d+)\s+years?", q)
    years_ahead = int(years_ahead_match.group(1)) if years_ahead_match else None

    if country_trends is None or getattr(country_trends, "empty", True):
        return {
            "chain_name": "prediction",
            "evidence": "Climate dataset unavailable for prediction.",
            "structured_facts": {}
        }

    # Try to match a specific country from the question
    matched_rows = None
    if "Country" in country_trends.columns:
        for country in country_trends["Country"].dropna().astype(str).unique():
            c_lower = country.lower()
            if c_lower in q or SequenceMatcher(None, c_lower, q).ratio() > 0.6:
                matched_rows = country_trends[
                    country_trends["Country"].astype(str).str.lower() == c_lower
                ]
                break

    # No specific country → use all for global/ranking prediction
    if matched_rows is None or getattr(matched_rows, "empty", True):
        matched_rows = country_trends

    lines = [
        "=== CLIMATE PREDICTION DATA ===",
        "Method: Linear Regression (WarmingTrendPerYear already computed from historical data)",
        "Formula: PredictedTemp = LatestTemperature + (WarmingTrendPerYear x YearsAhead)",
        "",
    ]

    predictions = []
    for _, row in matched_rows.iterrows():
        country   = row.get("Country", "Unknown")
        latest_t  = pd.to_numeric(row.get("LatestTemperature"), errors="coerce")
        latest_yr = pd.to_numeric(row.get("LatestYear"), errors="coerce")
        trend     = pd.to_numeric(row.get("WarmingTrendPerYear"), errors="coerce")

        if pd.isna(latest_t) or pd.isna(trend) or pd.isna(latest_yr):
            continue

        base_year = int(latest_yr)

        if target_year:
            yrs = max(target_year - base_year, 0)
            predicted = round(float(latest_t) + float(trend) * yrs, 3)
            predictions.append({
                "Country":                  country,
                "LatestTemp_C":             round(float(latest_t), 3),
                "LatestYear":               base_year,
                "WarmingTrend_C_per_year":  round(float(trend), 4),
                "TargetYear":               target_year,
                "YearsAhead":               yrs,
                "PredictedTemp_C":          predicted,
            })

        elif years_ahead:
            for yr in range(current_year + 1, current_year + years_ahead + 1):
                yrs = max(yr - base_year, 0)
                predictions.append({
                    "Country":                  country,
                    "LatestTemp_C":             round(float(latest_t), 3),
                    "WarmingTrend_C_per_year":  round(float(trend), 4),
                    "TargetYear":               yr,
                    "YearsAhead":               yrs,
                    "PredictedTemp_C":          round(float(latest_t) + float(trend) * yrs, 3),
                })

        else:
            # Default: next 5 years
            for yr in range(current_year + 1, current_year + 6):
                yrs = max(yr - base_year, 0)
                predictions.append({
                    "Country":                  country,
                    "LatestTemp_C":             round(float(latest_t), 3),
                    "WarmingTrend_C_per_year":  round(float(trend), 4),
                    "TargetYear":               yr,
                    "YearsAhead":               yrs,
                    "PredictedTemp_C":          round(float(latest_t) + float(trend) * yrs, 3),
                })

    if predictions:
        pred_df = pd.DataFrame(predictions)
        lines.append(pred_df.to_string(index=False))
        lines.append("")
        lines.append("NOTES:")
        lines.append("- WarmingTrend is degrees Celsius per year (positive = warming).")
        lines.append("- Predictions are projections — actual future temps may vary.")
        lines.append("- Label clearly as projections, not recorded data.")
    else:
        lines.append("No matching country data found. Ensure country name is in the dataset.")

    return {
        "chain_name": "prediction",
        "evidence": "\n".join(lines),
        "structured_facts": {"predictions": predictions}
    }


def build_weather_forecast_chain(question: str, location: str):
    """
    Build context for short-term weather forecast.
    Example: "What will be tomorrow's weather in Chennai?"
    Uses get_weather_forecast() from weather.py — same Open-Meteo API,
    no new API key needed.
    """
    from app.weather import extract_location, get_weather_forecast

    loc = extract_location(question) or location

    if not loc:
        return {
            "chain_name": "weather_forecast",
            "evidence": "No location found. Please include a city name.",
            "structured_facts": {}
        }

    q = question.lower()
    if "7 day" in q or "next week" in q or "this week" in q:
        days = 7
    elif "5 day" in q:
        days = 5
    elif "3 day" in q or "few days" in q or "weekend" in q:
        days = 3
    else:
        days = 1  # default: tomorrow only

    try:
        forecast = get_weather_forecast(loc, days=days)
    except Exception as exc:
        return {
            "chain_name": "weather_forecast",
            "evidence": f"Weather forecast unavailable: {exc}",
            "structured_facts": {}
        }

    place = forecast.get("location", {})
    location_label = ", ".join(
        p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p
    )

    lines = [f"=== WEATHER FORECAST FOR {location_label.upper()} ==="]
    for day in forecast.get("forecast_days", []):
        lines.append(
            f"Date: {day['date']} | "
            f"Max: {day['max_temperature_c']} C | "
            f"Min: {day['min_temperature_c']} C | "
            f"Condition: {day['condition']} | "
            f"Rain: {day['rain_sum_mm']} mm | "
            f"Rain probability: {day['precipitation_probability_percent']}% | "
            f"Wind: {day['wind_speed_kmh']} km/h | "
            f"Will rain: {day['will_rain']}"
        )

    return {
        "chain_name": "weather_forecast",
        "evidence": "\n".join(lines),
        "structured_facts": {
            "location": location_label,
            "forecast_days": forecast.get("forecast_days", [])
        }
    }


# ----------------------------
# ROUTER
# ----------------------------

def build_reasoning_chain(
    intent: Intent,
    question,
    location,
    yearly_df,
    country_trends,
    regional_trends
):

    if intent == Intent.WEATHER:
        return build_weather_chain(question, location, country_trends)

    if intent == Intent.WEATHER_AND_CLIMATE:
        return build_weather_and_climate_chain(
            question,
            location,
            yearly_df,
            country_trends,
            regional_trends
        )

    if intent == Intent.VOLATILITY:
        return build_volatility_chain(country_trends, regional_trends)

    if intent == Intent.COMPARISON:
        return build_comparison_chain(question, country_trends, regional_trends)

    if intent == Intent.CLIMATE or intent == Intent.CLIMATE_TREND:
        return build_historical_climate_chain(
            question,
            yearly_df,
            country_trends,
            regional_trends
        )

    return build_historical_climate_chain(
        question,
        yearly_df,
        country_trends,
        regional_trends
    ) 
