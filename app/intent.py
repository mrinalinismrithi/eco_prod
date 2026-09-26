import re
from enum import Enum


class Intent(str, Enum):
    WEATHER = "weather"
    CLIMATE = "climate"
    CLIMATE_TREND = "climate_trend"
    COMPARISON = "comparison"
    WEATHER_AND_CLIMATE = "weather_and_climate"
    VOLATILITY = "volatility"
    UNSUPPORTED = "unsupported"
    PREDICTION = "prediction"
    WEATHER_FORECAST = "weather_forecast"


def detect_intent(question: str) -> Intent:

    q = question.lower().strip()

    if not q:
        return Intent.UNSUPPORTED

    # ── Climate prediction ────────────────────────────────────────────────
    prediction_keywords = [
        "predict", "prediction", "future temperature",
        "will be in", "will it be", "expected temperature", "projected",
        "by 2025", "by 2026", "by 2027", "by 2028", "by 2029",
        "by 2030", "by 2035", "by 2040", "by 2050",
        "next 5 years", "next 10 years", "next year climate",
        "temperature in 2025", "temperature in 2026", "temperature in 2027",
        "temperature in 2028", "temperature in 2029", "temperature in 2030",
        "temperature in 2035", "temperature in 2040", "temperature in 2050",
        "what will", "how hot will", "how warm will",
    ]
    weather_forecast_kw = [
        "tomorrow", "next few days", "this week", "weekend",
        "next 3 days", "next 7 days", "next week",
    ]
    if any(k in q for k in prediction_keywords):
        if not any(k in q for k in weather_forecast_kw):
            return Intent.PREDICTION

    # ── Short-term weather forecast ───────────────────────────────────────
    weather_forecast_keywords = [
        "tomorrow", "tomorrow's weather", "tomorrow's temperature",
        "next few days", "weather this week", "weekend weather",
        "next 3 days", "next 7 days", "next week weather",
        "forecast for tomorrow", "rain tomorrow",
        "will it rain tomorrow", "weather forecast",
        "3 day forecast", "7 day forecast",
        "will it be hot tomorrow", "will it be cold tomorrow",
    ]
    if any(k in q for k in weather_forecast_keywords):
        return Intent.WEATHER_FORECAST

    # Historical year → always climate
    has_year = bool(
        re.search(r"\b(19\d{2}|20\d{2})\b", q)
    )
    if has_year:
        return Intent.CLIMATE

    weather_keywords = [
        "weather", "today", "now", "live", "forecast",
        "rain", "humidity", "wind", "temperature", "temp",
    ]

    climate_keywords = [
        "climate", "trend", "warming", "historical",
        "history", "average", "change", "dataset",
        "csv", "warmest", "volatility",
    ]

    is_weather = any(k in q for k in weather_keywords)
    is_climate = any(k in q for k in climate_keywords)

    if (
        is_weather
        and is_climate
        and "compare" in q
    ):
        return Intent.COMPARISON

    if is_climate:
        return Intent.CLIMATE

    if is_weather:
        return Intent.WEATHER

    return Intent.UNSUPPORTED