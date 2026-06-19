"""
EcoLens Authentication
=======================
File-based, no-database auth:
  - Users stored in data/users.json (username + hashed password)
  - Sessions stored in-memory (token -> username), lost on server restart
  - Cookie-based session token

This is intentionally simple — suitable for a demo / academic project.
"""

import json
import secrets
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import Cookie, HTTPException, Response

BASE_DIR = Path(__file__).resolve().parent.parent
USERS_FILE = BASE_DIR / "data" / "users.json"

# token -> username   (in-memory; cleared on restart)
SESSIONS: dict[str, str] = {}

SESSION_COOKIE_NAME = "ecolens_session"


# =========================
# PASSWORD HASHING
# =========================

def _hash_password(password: str) -> str:
    """Simple salted hash using sha256. Good enough for a demo project."""
    salt = "ecolens_static_salt"  # static salt is fine for academic scope
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


# =========================
# USER STORAGE
# =========================

def _load_users() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_users(users: list[dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _find_user(users: list[dict], username: str) -> Optional[dict]:
    username_lower = username.strip().lower()
    for u in users:
        if u["username"].lower() == username_lower:
            return u
    return None


# =========================
# SIGNUP
# =========================

def signup(username: str, password: str) -> dict:
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")

    users = _load_users()
    if _find_user(users, username):
        raise HTTPException(status_code=409, detail="Username already exists.")

    users.append({
        "username": username,
        "password_hash": _hash_password(password),
    })
    _save_users(users)

    return {"message": "Signup successful. You can now log in."}


# =========================
# LOGIN
# =========================

def login(username: str, password: str, response: Response) -> dict:
    username = username.strip()
    users = _load_users()
    user = _find_user(users, username)

    if not user or user["password_hash"] != _hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Create session token
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = user["username"]

    # Set cookie (HTTP-only so JS can't read it, helps a little against XSS)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 7 days
    )

    return {"message": "Login successful.", "username": user["username"]}


# =========================
# LOGOUT
# =========================

def logout(response: Response, session_token: Optional[str]) -> dict:
    if session_token and session_token in SESSIONS:
        del SESSIONS[session_token]
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"message": "Logged out."}


# =========================
# SESSION CHECK (dependency)
# =========================

def get_current_user(
    ecolens_session: Optional[str] = Cookie(default=None),
) -> str:
    """
    FastAPI dependency. Raises 401 if not logged in.
    Use as: current_user: str = Depends(get_current_user)
    """
    if not ecolens_session or ecolens_session not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return SESSIONS[ecolens_session]


def get_current_user_optional(
    ecolens_session: Optional[str] = Cookie(default=None),
) -> Optional[str]:
    """Like get_current_user, but returns None instead of raising."""
    if not ecolens_session or ecolens_session not in SESSIONS:
        return None
    return SESSIONS[ecolens_session] 