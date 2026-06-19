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


def detect_intent(question: str) -> Intent:

    q = question.lower().strip()

    if not q:
        return Intent.UNSUPPORTED


    # Historical year → always climate
    has_year = bool(
        re.search(r"\b(19\d{2}|20\d{2})\b", q)
    )

    if has_year:
        return Intent.CLIMATE


    weather_keywords = [
        "weather",
        "today",
        "now",
        "live",
        "forecast",
        "rain",
        "humidity",
        "wind",
        "temperature",
        "temp",
    ]

    climate_keywords = [
        "climate",
        "trend",
        "warming",
        "historical",
        "history",
        "average",
        "change",
        "dataset",
        "csv",
        "warmest",
        "volatility",
    ]


    is_weather = any(k in q for k in weather_keywords)

    is_climate = any(k in q for k in climate_keywords)


    # Only explicit comparison
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