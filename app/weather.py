from datetime import date
import re
import time
from urllib.parse import quote
import requests
from difflib import get_close_matches

from app.logging_config import logger


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WTTR_URL = "https://wttr.in"
REQUEST_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1


WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _get_json(url: str, params: dict, timeout, label: str, attempts: int = REQUEST_ATTEMPTS) -> dict:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            last_error = Exception(f"{label} failed: {response.text}")
            logger.warning("%s failed on attempt %s/%s: %s", label, attempt, attempts, response.text)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("%s error on attempt %s/%s: %s", label, attempt, attempts, exc)

        if attempt < attempts:
            time.sleep(RETRY_DELAY_SECONDS)

    raise last_error or Exception(f"{label} failed")


def _as_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# =========================
# WEATHER QUESTION DETECTION
# =========================

def looks_like_weather_question(question: str) -> bool:
    q = question.lower()

    weather_keywords = [
        "weather", "rain", "rainfall", "storm",
        "humidity", "forecast", "temperature", "temp",
        "hot", "cold", "wind", "climate", "sunny", "heat"
    ]

    if any(word in q for word in weather_keywords):
        return True

    location_patterns = [" in ", " at ", " of "]

    if any(p in q for p in location_patterns):
        return True

    return False


# =========================
# LOCATION EXTRACTION
# =========================

def extract_location(question: str) -> str | None:
    patterns = [
        r"\bin\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from)|\?|$)",
        r"\bfor\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from)|\?|$)",
        r"\bat\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from)|\?|$)",
        r"\bof\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from)|\?|$)",
        r"\b(?:weather|temperature|temp|forecast|condition|conditions)\s+(?:in\s+|for\s+|at\s+|of\s+)?([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from)|\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            if location and location.lower() not in {"today", "now", "current", "live"}:
                return location

    return None


# =========================
# NORMALIZATION
# =========================

def normalize_weather(weather):

    if not weather:
        return None

    current = weather.get("current", {})
    today = weather.get("today", {})
    location = weather.get("location", {})

    def first_present(*values):
        for value in values:
            if value is not None:
                return value
        return None

    condition = current.get("condition")

    if isinstance(condition, dict):
        condition = condition.get("text")

    return {

        "location": {
            "name": location.get("name"),
            "admin1": location.get("admin1"),
            "country": location.get("country"),
        },

        "current": {

            "temperature_c":
                first_present(
                    current.get("temperature_c"),
                    current.get("temp_c"),
                    current.get("temperature"),
                ),

            "feels_like_c":
                first_present(
                    current.get("feels_like_c"),
                    current.get("feelslike_c"),
                ),

            "humidity_percent":
                first_present(
                    current.get("humidity_percent"),
                    current.get("humidity"),
                ),

            "wind_speed_kmh":
                first_present(
                    current.get("wind_speed_kmh"),
                    current.get("wind_kph"),
                ),

            "condition": condition,
        },

        "today": {

            "min_temperature_c":
                first_present(
                    today.get("min_temperature_c"),
                    today.get("mintemp_c"),
                ),

            "max_temperature_c":
                first_present(
                    today.get("max_temperature_c"),
                    today.get("maxtemp_c"),
                ),

            "rain_sum_mm":
                first_present(
                    today.get("rain_sum_mm"),
                    today.get("totalprecip_mm"),
                ),
        }
    } 

def normalize_location(location: str) -> str:

    if not location:
        return ""

    location = location.strip()

    replacements = {
        "nyc": "New York",
        "usa": "United States",
        "uk": "United Kingdom",
    }

    lower_location = location.lower()

    return replacements.get(lower_location, location.title()) 
 


# =========================
# GEOCODING
# =========================

def geocode_location(location: str) -> dict:
    original_location = location.strip()
    location = normalize_location(original_location)

    search_terms = [location]

    for search in search_terms:
        try:
            data = _get_json(
                GEOCODING_URL,
                params={
                    "name": search,
                    "count": 10,
                    "language": "en",
                },
                timeout=(5, 10),
                label="Geocoding",
            )

            results = data.get("results", [])

            if results:
                place = results[0]

                # preserve user input
                place["name"] = original_location.title()

                return place

        except Exception as exc:
            logger.warning("Geocoding error for %s: %s", search, exc)

    raise ValueError(f"No weather location found for '{original_location}'.")


# =========================
# WEATHER FETCH
# =========================

def get_current_weather(location: str) -> dict:
    if not location:
        raise ValueError("Location is required for weather API")

    try:
        place = geocode_location(location)
    except Exception as exc:
        logger.warning("Open-Meteo geocoding failed; using wttr.in fallback: %s", exc)
        return get_wttr_weather(location)

    logger.info("Fetching weather forecast for %s", location)

    try:
        data = _get_json(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "rain",
                    "weather_code",
                    "wind_speed_10m",
                ]),
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "rain_sum",
                    "precipitation_probability_max",
                    "weather_code",
                ]),
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=(4, 8),
            label="Weather forecast",
            attempts=1,
        )
    except Exception as exc:
        logger.warning("Open-Meteo forecast failed; using wttr.in fallback: %s", exc)
        return get_wttr_weather(location, place)

    current = data.get("current")
    daily = data.get("daily")

    if not current or not daily:
        raise ValueError("Weather API returned incomplete response")

    weather_code = int(current.get("weather_code", -1))
    daily_weather_code = int(daily["weather_code"][0])

    # FIX: safe handling
    rain_sum = float(daily.get("rain_sum", [0])[0] or 0)
    precipitation_probability = daily.get("precipitation_probability_max", [None])[0]

    return {
        "location": {
            "name": place["name"],
            "country": place.get("country"),
            "admin1": place.get("admin1"),
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "timezone": data.get("timezone"),
        },
        "date": daily["time"][0] if daily.get("time") else date.today().isoformat(),
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": weather_code,
            "condition": WEATHER_CODES.get(weather_code, "unknown"),
        },
        "today": {
            "max_temperature_c": daily["temperature_2m_max"][0],
            "min_temperature_c": daily["temperature_2m_min"][0],
            "precipitation_sum_mm": daily["precipitation_sum"][0],
            "rain_sum_mm": rain_sum,
            "precipitation_probability_percent": precipitation_probability,
            "weather_code": daily_weather_code,
            "condition": WEATHER_CODES.get(daily_weather_code, "unknown"),
            "will_rain": rain_sum > 0 or (
                precipitation_probability is not None and precipitation_probability >= 50
            ),
        },
    }


def get_wttr_weather(location: str, place: dict | None = None) -> dict:
    data = _get_json(
        f"{WTTR_URL}/{quote(location)}",
        params={"format": "j1"},
        timeout=(5, 15),
        label="wttr.in weather",
        attempts=2,
    )
    current = (data.get("current_condition") or [{}])[0]
    today = (data.get("weather") or [{}])[0]
    description = ((current.get("weatherDesc") or [{}])[0]).get("value") or "unknown"
    temp = _as_float(current.get("temp_C"))
    feels_like = _as_float(current.get("FeelsLikeC"))
    humidity = _as_float(current.get("humidity"))
    wind_speed = _as_float(current.get("windspeedKmph"))
    rain = _as_float(current.get("precipMM"))
    min_temp = _as_float(today.get("mintempC"))
    max_temp = _as_float(today.get("maxtempC"))

    return {
        "location": {
            "name": (place or {}).get("name") or location.title(),
            "country": (place or {}).get("country"),
            "admin1": (place or {}).get("admin1"),
            "latitude": (place or {}).get("latitude"),
            "longitude": (place or {}).get("longitude"),
            "timezone": None,
        },
        "date": today.get("date") or date.today().isoformat(),
        "current": {
            "temperature_c": temp,
            "feels_like_c": feels_like,
            "humidity_percent": humidity,
            "precipitation_mm": rain,
            "rain_mm": rain,
            "wind_speed_kmh": wind_speed,
            "weather_code": _as_float(current.get("weatherCode")),
            "condition": description,
        },
        "today": {
            "max_temperature_c": max_temp if max_temp is not None else temp,
            "min_temperature_c": min_temp if min_temp is not None else temp,
            "precipitation_sum_mm": rain,
            "rain_sum_mm": rain,
            "precipitation_probability_percent": None,
            "weather_code": _as_float(current.get("weatherCode")),
            "condition": description,
            "will_rain": bool(rain and rain > 0),
        },
    }


# =========================
# WEATHER CONTEXT FORMATTER
# =========================

def format_weather_context(weather: dict) -> str:

    if not weather:
        return "Weather data unavailable."

    current = weather.get("current", {})
    location = weather.get("location", {})
    today = weather.get("today", {})

    return f"""
Location: {location.get('name')}, {location.get('country')}

Current Temperature: {current.get('temperature_c')}°C
Feels Like: {current.get('feels_like_c')}°C
Condition: {current.get('condition')}
Humidity: {current.get('humidity_percent')}%
Wind Speed: {current.get('wind_speed_kmh')} km/h

Today's Forecast:
Min Temp: {today.get('min_temperature_c')}°C
Max Temp: {today.get('max_temperature_c')}°C
Rain: {today.get('rain_sum_mm')} mm
""" 
