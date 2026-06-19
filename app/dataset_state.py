import json
import os 
import pandas as pd 

PROCESSED_DIR = "data/processed"
ACTIVE_FILE = os.path.join(PROCESSED_DIR, "active.json")

def set_active_source(source: str):
    with open(ACTIVE_FILE, "w") as f:
        json.dump({"source": source}, f)

def get_active_source():
    if not os.path.exists(ACTIVE_FILE):
        return "default"
    with open(ACTIVE_FILE) as f:
        return json.load(f).get("source", "default")

def get_processed_path(filename: str):
    source = get_active_source()
    return os.path.join(PROCESSED_DIR, source, filename)   

def analyze_columns(df: pd.DataFrame) -> dict:
    schema = {
        "columns": list(df.columns),
        "numeric_columns": [],
        "categorical_columns": [],
        "date_columns": [],
    }

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            schema["numeric_columns"].append(col)
        else:
            try:
                pd.to_datetime(df[col], errors="raise")
                schema["date_columns"].append(col)
            except (ValueError, TypeError):
                schema["categorical_columns"].append(col)

    return schema 