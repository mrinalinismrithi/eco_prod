"""
EcoLens Authentication
=======================
File-based auth with hardcoded users that survive Render restarts.
New signups are saved to data/users.json (lost on restart — expected on free tier).
Hardcoded accounts always work regardless of restarts.
"""

import json
import secrets
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import Cookie, HTTPException, Response

BASE_DIR   = Path(__file__).resolve().parent.parent
USERS_FILE = BASE_DIR / "data" / "users.json"

# token -> username (in-memory; cleared on restart)
SESSIONS: dict[str, str] = {}

SESSION_COOKIE_NAME = "ecolens_session"


# =========================
# PASSWORD HASHING
# =========================

def _hash_password(password: str) -> str:
    salt = "ecolens_static_salt"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


# =========================
# HARDCODED USERS (always exist, survive restarts)
# =========================

HARDCODED_USERS = [
    {
        "username": "admin",
        "password_hash": _hash_password("ecolens123"),
    },
    {
        "username": "mrinalini",
        "password_hash": _hash_password("smrithi@123"),
    },
]

HARDCODED_USERNAMES = {u["username"].lower() for u in HARDCODED_USERS}


# =========================
# USER STORAGE
# =========================

def _load_users() -> list[dict]:
    # Start with hardcoded users
    users = list(HARDCODED_USERS)

    # Add any additional users from file (signups)
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                file_users = json.load(f)
                for u in file_users:
                    if u["username"].lower() not in HARDCODED_USERNAMES:
                        users.append(u)
        except (json.JSONDecodeError, OSError):
            pass

    return users


def _save_users(users: list[dict]) -> None:
    # Save only non-hardcoded users to file
    file_users = [
        u for u in users
        if u["username"].lower() not in HARDCODED_USERNAMES
    ]
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(file_users, f, indent=2)


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
    users    = _load_users()
    user     = _find_user(users, username)

    if not user or user["password_hash"] != _hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = secrets.token_urlsafe(32)
    SESSIONS[token] = user["username"]

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
# SESSION CHECK
# =========================

def get_current_user(
    ecolens_session: Optional[str] = Cookie(default=None),
) -> str:
    if not ecolens_session or ecolens_session not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return SESSIONS[ecolens_session]


def get_current_user_optional(
    ecolens_session: Optional[str] = Cookie(default=None),
) -> Optional[str]:
    if not ecolens_session or ecolens_session not in SESSIONS:
        return None
    return SESSIONS[ecolens_session]