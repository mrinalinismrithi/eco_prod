from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Response, Depends, Cookie
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pandas as pd
import traceback 
import os

from app.agent import OpenAIConfigError, ask_ecolens, validate_openai_key
from app.data_loader import (
    DataFileError,
    load_country_trends,
    load_fastest_warming_regions,
    load_hottest_countries,
    load_processed_data,
    load_regional_trends,
)
from app.etl import run_etl
from app.logging_config import logger, setup_logging
from app.weather import get_current_weather
from app.data_loader import load_all_datasets
from app.weather import normalize_location, normalize_weather
from app import auth 
from fastapi import UploadFile, File
import shutil 
from app.upload_etl import run_upload_etl
from app.dataset_state import set_active_source, get_active_source, get_processed_path  
from app.data_loader import clear_dataset_cache 

setup_logging()

app = FastAPI(title="EcoLens Climate Intelligence API")

BASE_DIR     = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE   = FRONTEND_DIR / "index.html"


# =========================
# REQUEST MODELS
# =========================

class Message(BaseModel):
    role: str
    content: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class QuestionRequest(BaseModel):
    question: str
    location: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    analysis_state: Optional[Dict] = None


class WeatherRequest(BaseModel):
    location: str


class RequestBody(BaseModel):
    question: str
    location: Optional[str] = None
    conversation_history: List[ConversationMessage] = []
    analysis_state: Dict[str, Any] = {}


class AuthRequest(BaseModel):
    username: str
    password: str


# =========================
# BASIC ROUTES
# =========================

@app.get("/")
def home():
    return RedirectResponse(url="/login.html")


@app.get("/api")
def api_home():
    return {"message": "EcoLens Climate Intelligence API is running"}


@app.get("/health")
def health():
    status = {
        "system":        "online",
        "system_status": "System: Online",
        "data":          "ok",
        "openai":        "ok",
    }
    try:
        load_processed_data()
    except DataFileError as e:
        status["data"] = str(e)

    try:
        from app.agent import validate_gemini_key
        validate_gemini_key()
    except Exception as e:
        status["openai"] = str(e) 

    return status


# =========================
# AUTH (file-based, no DB)
# =========================

@app.post("/signup")
def signup_route(payload: AuthRequest):
    return auth.signup(payload.username, payload.password)


@app.post("/login")
def login_route(payload: AuthRequest, response: Response):
    return auth.login(payload.username, payload.password, response)


@app.post("/logout")
def logout_route(response: Response, ecolens_session: Optional[str] = Cookie(default=None)):
    return auth.logout(response, ecolens_session)


@app.get("/me")
def me_route(current_user: Optional[str] = Depends(auth.get_current_user_optional)):
    if current_user:
        return {"authenticated": True, "username": current_user}
    return {"authenticated": False}


# =========================
# ETL
# =========================

@app.post("/run-etl")
def etl():
    try:
        analytics = run_etl()
        return {
            "message":     "ETL completed successfully",
            "yearly_rows": len(analytics.get("yearly", [])),
            "countries":   len(analytics.get("country_trends", [])),
        }
    except Exception as e:
        logger.exception("ETL failed")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ANALYTICS
# =========================

def safe_df(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df.astype(object).to_dict(orient="records")


@app.get("/analytics")
def get_analytics():
    clear_dataset_cache() 
    try:
        return {
            "yearly_climate_data":     safe_df(load_processed_data()),
            "country_warming_trends":  safe_df(load_country_trends()),
            "regional_warming_trends": safe_df(load_regional_trends()),
            "fastest_warming_regions": safe_df(load_fastest_warming_regions()),
            "hottest_countries":       safe_df(load_hottest_countries()),
        }
    except Exception as e:
        logger.exception("Analytics failed")
        raise HTTPException(
            status_code=503,
            detail="Climate dataset temporarily unavailable.",
        )


# =========================
# WEATHER
# =========================

@app.get("/weather")
@app.post("/weather")
def weather(location: str | None = None, request: WeatherRequest | None = None):
    loc = request.location if request else location
    if not loc:
        raise HTTPException(status_code=400, detail="Location is required")
    try:
        return get_current_weather(loc)
    except Exception as e:
        logger.exception("Weather error")
        raise HTTPException(status_code=503, detail=str(e))


# =========================
# AI ASSISTANT
# =========================

@app.post("/ask")
def ask_ai(payload: RequestBody, current_user: Optional[str] = Depends(auth.get_current_user_optional)):
    try:
        history_dicts = [
            {"role": m.role, "content": m.content}
            for m in payload.conversation_history
        ]

        response = ask_ecolens(
            question=payload.question,
            history=history_dicts,
            analysis_state=payload.analysis_state,
        )
        return response

    except Exception as e:
        logger.exception("ask_ai failed")
        return {"error": str(e), "success": False} 
    
@app.post("/api/etl/upload")
async def upload_and_run_etl(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"status": "error", "message": "Only CSV files are allowed"}

    upload_path = "data/raw/current_upload.csv"

    
    upload_dir = "data/raw/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = f"{upload_dir}/{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = run_upload_etl(upload_path)    
        clear_dataset_cache() 
        return {"status": "success", "result": result}
    except Exception as e: 
        return {"status": "error", "message": str(e)} 
@app.post("/api/etl/switch-upload")
async def switch_upload(filename: str):
    upload_path = f"data/raw/uploads/{filename}"
    if not os.path.exists(upload_path):
        return {"status": "error", "message": "File not found"}
    try:
        result = run_upload_etl(upload_path)
        clear_dataset_cache()
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}     


@app.post("/api/etl/reset")
async def reset_to_default():
    set_active_source("default")
    clear_dataset_cache()
    return {"status": "success", "message": "Reverted to default climate dataset"} 
@app.get("/api/etl/uploads")
async def list_uploads():
    import json
    upload_dir = "data/raw/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(upload_dir) if f.endswith(".csv")])
    active_file = None
    try:
        if get_active_source() == "upload":
            schema_path = get_processed_path("schema.json")
            with open(schema_path) as f:
                schema = json.load(f)
            active_file = schema.get("filename")
    except Exception:
        pass
    return {"files": files, "active": active_file}


@app.get("/api/etl/active")
async def get_active():
    return {"active": get_active_source()}
@app.get("/api/debug/parse")
async def debug_parse(q: str):
    from app.weather import extract_location, extract_historical_date, is_historical_weather_question
    return {
        "question": q,
        "location": extract_location(q),
        "date": extract_historical_date(q),
        "is_historical": is_historical_weather_question(q),
    } 

@app.get("/api/etl/schema")
async def get_schema():
    import json
    from app.dataset_state import get_processed_path, get_active_source
    try:
        schema_path = get_processed_path("schema.json")
        with open(schema_path) as f:
            schema = json.load(f)
        schema["active_source"] = get_active_source()
        return schema
    except Exception:
        return {
            "active_source": "default",
            "columns": [],
            "row_count": 0,
            "categorical_columns": [],
            "numeric_columns": []
        }

    @app.get("/api/debug/auth")
    def debug_auth():
        from app import auth
        import inspect
        source = inspect.getsource(auth._load_users)
        users = auth._load_users()
        return {
            "users": [u["username"] for u in users],
            "default_username": getattr(auth, "DEFAULT_USERNAME", "NOT FOUND"),
        } 





# ── Static files must be mounted LAST so API routes take priority
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend") 