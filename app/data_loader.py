from pathlib import Path
import re
import threading
import time
import numpy as np
import pandas as pd 
from app.logging_config import logger
from app.dataset_state import get_active_source


PROJECT_ROOT = Path(__file__).resolve().parents[1] 

CANONICAL_REGION_MAP: dict[str, str] = {
    # North America
    "Mexico": "North America", "United States": "North America",
    "Canada": "North America", "Guatemala": "North America",
    "Cuba": "North America", "Haiti": "North America",
    "Dominican Republic": "North America", "Honduras": "North America",
    "Nicaragua": "North America", "Costa Rica": "North America",
    "Panama": "North America", "El Salvador": "North America",
    "Belize": "North America", "Jamaica": "North America",
    "Trinidad and Tobago": "North America",
    # South America
    "Brazil": "South America", "Argentina": "South America",
    "Colombia": "South America", "Peru": "South America",
    "Venezuela": "South America", "Chile": "South America",
    "Ecuador": "South America", "Bolivia": "South America",
    "Paraguay": "South America", "Uruguay": "South America",
    "Guyana": "South America", "Suriname": "South America",
    # Europe
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Poland": "Europe",
    "Netherlands": "Europe", "Belgium": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Denmark": "Europe", "Finland": "Europe",
    "Switzerland": "Europe", "Austria": "Europe", "Portugal": "Europe",
    "Czech Republic": "Europe", "Romania": "Europe", "Hungary": "Europe",
    "Greece": "Europe", "Ukraine": "Europe", "Russia": "Europe",
    # Asia
    "China": "Asia", "India": "Asia", "Japan": "Asia",
    "South Korea": "Asia", "Indonesia": "Asia", "Vietnam": "Asia",
    "Thailand": "Asia", "Malaysia": "Asia", "Philippines": "Asia",
    "Pakistan": "Asia", "Bangladesh": "Asia", "Myanmar": "Asia",
    "Nepal": "Asia", "Sri Lanka": "Asia", "Cambodia": "Asia",
    # Africa
    "Nigeria": "Africa", "Ethiopia": "Africa", "South Africa": "Africa",
    "Kenya": "Africa", "Ghana": "Africa", "Tanzania": "Africa",
    "Egypt": "Africa", "Morocco": "Africa", "Algeria": "Africa",
    "Sudan": "Africa", "Uganda": "Africa", "Mozambique": "Africa",
    "Madagascar": "Africa", "Angola": "Africa", "Cameroon": "Africa",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Papua New Guinea": "Oceania", "Fiji": "Oceania",
    # Middle East
    "Saudi Arabia": "Middle East", "Iran": "Middle East",
    "Iraq": "Middle East", "Israel": "Middle East",
    "Jordan": "Middle East", "Kuwait": "Middle East",
    "United Arab Emirates": "Middle East", "Qatar": "Middle East",
    "Bahrain": "Middle East", "Oman": "Middle East",
    "Syria": "Middle East", "Lebanon": "Middle East", "Yemen": "Middle East",
}


def apply_region_corrections(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Country" not in df.columns:
        return df
    df = df.copy()
    if "Region" not in df.columns:
        df["Region"] = pd.NA
    corrections = df["Country"].map(CANONICAL_REGION_MAP)
    mask = corrections.notna()
    df.loc[mask, "Region"] = corrections[mask]
    return df


# ── DYNAMIC PATHS ─────────────────────────────────────────
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "climate_data.csv"

# default always points to default folder (never changes)
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "default" / "processed_climate_data.csv"


def get_processed_dir() -> Path:
    """Returns the active processed folder — default or upload."""
    source = get_active_source()
    return PROJECT_ROOT / "data" / "processed" / source


def _processed(filename: str) -> Path:
    return get_processed_dir() / filename


def get_country_trends_path() -> Path:
    return _processed("country_warming_trends.csv")


def get_regional_trends_path() -> Path:
    return _processed("regional_warming_trends.csv")


def get_fastest_regions_path() -> Path:
    return _processed("fastest_warming_regions.csv")


def get_hottest_countries_path() -> Path:
    return _processed("hottest_countries.csv")


def get_dataset_paths() -> dict:
    return {
        "raw":              RAW_DATA_PATH,
        "main":             get_processed_dir() / "processed_climate_data.csv",
        "country_trends":   get_country_trends_path(),
        "regional_trends":  get_regional_trends_path(),
        "fastest_regions":  get_fastest_regions_path(),
        "hottest_countries": get_hottest_countries_path(),
    } 



_DATASET_CACHE: dict[str, pd.DataFrame] | None = None
_CACHE_LOCK = threading.RLock() 


# =========================
# ERROR CLASS
# =========================
class DataFileError(Exception):
    pass


# =========================
# CLEAN DATAFRAME
# =========================
def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    temp_keywords = ["temp", "temperature", "warming", "change", "trend"]

    for col in df.columns:
        if any(k in col.lower() for k in temp_keywords):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"C|deg|degree", "", regex=True, flags=re.IGNORECASE)
                .str.replace(r"[^0-9.\-]", "", regex=True)
                .replace("", np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================
# SAFE CSV READER
# =========================
def _read_csv(path, name: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning("%s missing at %s — returning empty DataFrame", name, path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            logger.warning("%s is empty at %s", name, path)
            return pd.DataFrame()
        df = _clean_dataframe(df)
        df = apply_region_corrections(df)
        return df
    except Exception as e:
        logger.error("%s load failed: %s", name, e)
        return pd.DataFrame()


def clear_dataset_cache():
    with _CACHE_LOCK:
        global _DATASET_CACHE
        _DATASET_CACHE = None


def datasets_exist() -> bool:
    return all(path.exists() for path in get_dataset_paths().values())


def datasets_ready() -> bool:
    if not datasets_exist():
        return False

    with _CACHE_LOCK:
        if _DATASET_CACHE is not None:
            return all(df is not None and not df.empty for df in _DATASET_CACHE.values())

    return True


# =========================
# PUBLIC LOADERS
# =========================
def load_raw_data():
    return _read_csv(RAW_DATA_PATH, "Raw Data")


def load_processed_data():
    path = get_processed_dir() / "processed_climate_data.csv"
    return _read_csv(path, "Processed Data") 


def load_country_trends():
    return _read_csv(get_country_trends_path(), "Country Trends")


def load_regional_trends():
    return _read_csv(get_regional_trends_path(), "Regional Trends")


def load_fastest_warming_regions():
    return _read_csv(get_fastest_regions_path(), "Fastest Regions")


def load_hottest_countries():
    return _read_csv(get_hottest_countries_path(), "Hottest Countries")


# =========================
# ALL DATASETS
# =========================
def load_all_datasets(force_reload: bool = False, retries: int = 3, retry_delay: float = 1.0):
    global _DATASET_CACHE

    with _CACHE_LOCK:
        if _DATASET_CACHE is not None and not force_reload:
            return _DATASET_CACHE

    # always force reload to reflect active source changes
    datasets = {
        "raw":               load_raw_data(),
        "main":              load_processed_data(),
        "country_trends":    load_country_trends(),
        "regional_trends":   load_regional_trends(),
        "fastest_regions":   load_fastest_warming_regions(),
        "hottest_countries": load_hottest_countries(),
    }

    with _CACHE_LOCK:
        _DATASET_CACHE = datasets

    empty = [k for k, v in datasets.items() if v is None or v.empty]
    if empty:
        logger.warning("These datasets are empty (ETL may not have run yet): %s", empty)

    return datasets


# =========================
# FORMAT FOR LLM
# =========================
def format_for_llm(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "No data available"

    df = df.head(max_rows)
    return df.to_string(index=False) 