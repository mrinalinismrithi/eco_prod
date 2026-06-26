from datetime import date, datetime
import re
import time
from urllib.parse import quote
import requests
from difflib import get_close_matches

from app.logging_config import logger


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
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
# HISTORICAL DATE DETECTION
# =========================

def extract_historical_date(question: str) -> str | None:
    """
    Extract a historical date/period from a question.
    Returns a string in one of these forms:
        "YYYY-MM-DD"           → specific day
        "YYYY-MM"              → month query  (caller fetches whole month)
        "YYYY"                 → year query   (caller fetches whole year)
    Returns None if no historical date found.
    """
    q = question.lower()

    month_names = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }

    MONTH_PAT = (
        r"january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    )

    # ── 1. Full date: "december 12 2003" / "12 december 2003" ───────────
    m = re.search(
        rf"({MONTH_PAT})\s+(\d{{1,2}})[,\s]+(\d{{4}})", q
    )
    if m:
        month = month_names[m.group(1)]
        day   = m.group(2).zfill(2)
        year  = m.group(3)
        date_str = f"{year}-{month}-{day}"
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
            if parsed < date.today():
                return date_str
        except ValueError:
            pass

    m = re.search(
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PAT})\s+(\d{{4}})", q
    )
    if m:
        day   = m.group(1).zfill(2)
        month = month_names[m.group(2)]
        year  = m.group(3)
        date_str = f"{year}-{month}-{day}"
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
            if parsed < date.today():
                return date_str
        except ValueError:
            pass

    # ── 2. ISO / numeric full date: "2003-12-12" / "12/12/2003" ─────────
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", q)
    if m:
        date_str = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
            if parsed < date.today():
                return date_str
        except ValueError:
            pass

    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", q)
    if m:
        date_str = f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
            if parsed < date.today():
                return date_str
        except ValueError:
            pass

    # ── 3. Month + year: "march 2023" / "in 2023 march" ─────────────────
    m = re.search(rf"({MONTH_PAT})\s+(\d{{4}})", q)
    if not m:
        m = re.search(rf"(\d{{4}})\s+({MONTH_PAT})", q)
        if m:
            # swap groups so group(1)=month, group(2)=year
            month_word, year_word = m.group(2), m.group(1)
        else:
            month_word, year_word = None, None
    else:
        month_word, year_word = m.group(1), m.group(2)

    if month_word and year_word:
        month = month_names.get(month_word)
        if month:
            try:
                yr = int(year_word)
                test_date = datetime(yr, int(month), 1).date()
                if test_date < date.today():
                    return f"{yr}-{month}"          # "YYYY-MM"
            except ValueError:
                pass

    # ── 4. Year only: "in 2022" / "during 2019" / "year 2010" ──────────
    m = re.search(
        r"(?:in|during|for|year|of)\s+((?:19|20)\d{2})\b"
        r"|(?:^|\s)((?:19|20)\d{2})\b",
        q
    )
    if m:
        year_str = m.group(1) or m.group(2)
        try:
            yr = int(year_str)
            if yr < date.today().year or (yr == date.today().year and date.today().month > 1):
                return str(yr)                       # "YYYY"
        except ValueError:
            pass

    return None 


def is_historical_weather_question(question: str) -> bool:
    """Returns True if question asks about historical weather for a specific date."""
    q = question.lower()
    has_date = extract_historical_date(question) is not None
    has_weather_intent = any(word in q for word in [
        "weather", "temperature", "temp", "rain", "hot", "cold",
        "climate", "humidity", "wind", "condition", "forecast"
    ])
    return has_date and has_weather_intent


# =========================
# LOCATION EXTRACTION
# =========================

def extract_location(question: str) -> str | None:
    patterns = [
        r"\bin\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from|on|in\s+\d)|\?|$)",
        r"\bfor\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from|on)|\?|$)",
        r"\bat\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from|on)|\?|$)",
        r"\bof\s+([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from|on)|\?|$)",
        r"\b(?:weather|temperature|temp|forecast|condition|conditions)\s+(?:in\s+|for\s+|at\s+|of\s+)?([A-Za-z\s]+?)(?:\s+(?:today|now|with|against|and|based|using|from|on)|\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # remove trailing year or date fragments
            location = re.sub(r"\s+\d{4}.*$", "", location).strip()
            location = re.sub(
                r"\s+(january|february|march|april|may|june|july|august|"
                r"september|october|november|december|jan|feb|mar|apr|"
                r"jun|jul|aug|sep|oct|nov|dec).*$",
                "", location, flags=re.IGNORECASE
            ).strip()
            # NEW: remove trailing "on" and anything after
            location = re.sub(r"\s+on\b.*$", "", location, flags=re.IGNORECASE).strip()
            # remove trailing ordinal day numbers like "24" "12th"
            location = re.sub(r"\s+\d{1,2}(st|nd|rd|th)?\b.*$", "", location).strip()

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
# HISTORICAL WEATHER FETCH
# =========================

def get_historical_weather(location: str, date_str: str) -> dict:
    """
    Fetch historical weather for a location and date/period.

    date_str formats accepted:
        "YYYY-MM-DD"  → single day
        "YYYY-MM"     → monthly average (fetches whole month)
        "YYYY"        → yearly average  (fetches whole year)
    """
    try:
        place = geocode_location(location)
    except Exception as exc:
        raise ValueError(f"Could not find location '{location}': {exc}")

    # ── Resolve start/end dates ──────────────────────────────────────────
    today = date.today()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        mode = "day"
        start_date = date_str
        end_date   = date_str
        label      = date_str

    elif re.match(r"^\d{4}-\d{2}$", date_str):
        mode = "month"
        yr, mo = int(date_str[:4]), int(date_str[5:7])
        import calendar
        last_day = calendar.monthrange(yr, mo)[1]
        start_date = f"{yr}-{mo:02d}-01"
        end_date   = f"{yr}-{mo:02d}-{last_day:02d}"
        # don't exceed today
        if datetime.strptime(end_date, "%Y-%m-%d").date() > today:
            end_date = today.isoformat()
        label = datetime(yr, mo, 1).strftime("%B %Y")

    elif re.match(r"^\d{4}$", date_str):
        mode = "year"
        yr = int(date_str)
        start_date = f"{yr}-01-01"
        end_date   = f"{yr}-12-31"
        if datetime.strptime(end_date, "%Y-%m-%d").date() > today:
            end_date = today.isoformat()
        label = str(yr)

    else:
        raise ValueError(f"Unrecognised date format: {date_str!r}")

    logger.info("Fetching historical weather for %s — %s (%s)", location, label, mode)

    try:
        data = _get_json(
            HISTORICAL_URL,
            params={
                "latitude":  place["latitude"],
                "longitude": place["longitude"],
                "start_date": start_date,
                "end_date":   end_date,
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "temperature_2m_mean",
                    "precipitation_sum",
                    "rain_sum",
                    "wind_speed_10m_max",
                    "weather_code",
                ]),
                "timezone": "auto",
            },
            timeout=(10, 30),
            label="Historical weather",
            attempts=2,
        )
    except Exception as exc:
        raise ValueError(
            f"Historical weather unavailable for '{location}' ({label}): {exc}"
        )

    daily = data.get("daily", {})
    if not daily or not daily.get("time"):
        raise ValueError(f"No historical data found for '{location}' ({label})")

    # ── Aggregate all daily values ───────────────────────────────────────
    def _floats(key):
        return [
            float(v) for v in (daily.get(key) or [])
            if v is not None
        ]

    max_temps  = _floats("temperature_2m_max")
    min_temps  = _floats("temperature_2m_min")
    mean_temps = _floats("temperature_2m_mean")
    rain_vals  = _floats("rain_sum")
    precip_vals= _floats("precipitation_sum")
    wind_vals  = _floats("wind_speed_10m_max")
    codes      = [int(v) for v in (daily.get("weather_code") or []) if v is not None]

    def _avg(lst):  return round(sum(lst) / len(lst), 2) if lst else None
    def _mx(lst):   return round(max(lst), 2) if lst else None
    def _mn(lst):   return round(min(lst), 2) if lst else None
    def _tot(lst):  return round(sum(lst), 2) if lst else None

    avg_mean = _avg(mean_temps) or (
        round((_avg(max_temps) + _avg(min_temps)) / 2, 2)
        if max_temps and min_temps else None
    )
    overall_max  = _mx(max_temps)
    overall_min  = _mn(min_temps)
    total_rain   = _tot(rain_vals)
    total_precip = _tot(precip_vals)
    avg_wind     = _avg(wind_vals)

    # Most common weather code
    dominant_code = max(set(codes), key=codes.count) if codes else 0
    num_days = len(daily.get("time", []))

    # Monthly breakdown for year queries
    monthly_summary = []
    if mode == "year":
        times = daily.get("time", [])
        from collections import defaultdict
        month_temps = defaultdict(list)
        for t, v in zip(times, mean_temps or max_temps):
            mo = t[5:7]  # "YYYY-MM-DD" → "MM"
            month_temps[mo].append(v)
        for mo in sorted(month_temps.keys()):
            vals = month_temps[mo]
            mo_label = datetime(int(date_str), int(mo), 1).strftime("%B")
            monthly_summary.append({
                "month": mo_label,
                "avg_temp": round(sum(vals) / len(vals), 2),
            })

    return {
        "location": {
            "name":      place["name"],
            "country":   place.get("country"),
            "admin1":    place.get("admin1"),
            "latitude":  place["latitude"],
            "longitude": place["longitude"],
        },
        "date":          date_str,
        "label":         label,
        "mode":          mode,          # "day" | "month" | "year"
        "num_days":      num_days,
        "is_historical": True,
        "current": {
            "temperature_c":    avg_mean,
            "feels_like_c":     None,
            "humidity_percent": None,
            "precipitation_mm": total_precip,
            "rain_mm":          total_rain,
            "wind_speed_kmh":   avg_wind,
            "weather_code":     dominant_code,
            "condition":        WEATHER_CODES.get(dominant_code, "variable"),
        },
        "today": {
            "max_temperature_c":      overall_max,
            "min_temperature_c":      overall_min,
            "precipitation_sum_mm":   total_precip,
            "rain_sum_mm":            total_rain,
            "precipitation_probability_percent": None,
            "weather_code":           dominant_code,
            "condition":              WEATHER_CODES.get(dominant_code, "variable"),
            "will_rain":              bool(total_rain and total_rain > 0),
        },
        "monthly_summary": monthly_summary,   # populated for year queries only
    } 


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
    is_historical = weather.get("is_historical", False)
    date_str = weather.get("date", "")

    label = f"Historical weather on {date_str}" if is_historical else "Current conditions"

    return f"""
Location: {location.get('name')}, {location.get('country')}
{label}

Temperature: {current.get('temperature_c')}°C
{"" if is_historical else f"Feels Like: {current.get('feels_like_c')}°C"}
Condition: {current.get('condition')}
{"" if is_historical else f"Humidity: {current.get('humidity_percent')}%"}
Wind Speed: {current.get('wind_speed_kmh')} km/h

Daily Summary:
Min Temp: {today.get('min_temperature_c')}°C
Max Temp: {today.get('max_temperature_c')}°C
Rain: {today.get('rain_sum_mm')} mm
""" 