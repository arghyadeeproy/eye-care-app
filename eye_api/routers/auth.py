"""
Authentication Router
Handles: Email/Password Sign Up, Sign In, Token Refresh, /me
"""

import logging
import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

from firebase_config import (
    get_firebase_auth,
    FIREBASE_WEB_API_KEY,
    FIREBASE_SIGN_IN_URL, FIREBASE_SIGN_UP_URL, FIREBASE_REFRESH_URL,
)
from auth_middleware import get_current_user
from fastapi import Depends

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class SignUpRequest(BaseModel):
    email:        EmailStr
    password:     str
    display_name: str


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class AuthResponse(BaseModel):
    uid:           str
    email:         str
    display_name:  str
    id_token:      str
    refresh_token: str


# ─────────────────────────────────────────────────────────────
# HELPERS — Firebase REST API calls
# ─────────────────────────────────────────────────────────────
async def _firebase_rest_signup(email: str, password: str) -> dict:
    if not FIREBASE_WEB_API_KEY:
        raise HTTPException(500, "FIREBASE_WEB_API_KEY not configured in .env")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{FIREBASE_SIGN_UP_URL}?key={FIREBASE_WEB_API_KEY}",
            json={"email": email, "password": password, "returnSecureToken": True},
        )
    data = resp.json()
    if "error" in data:
        msg = data["error"].get("message", "Sign-up failed")
        raise HTTPException(status_code=400, detail=msg)
    return data


async def _firebase_rest_signin(email: str, password: str) -> dict:
    if not FIREBASE_WEB_API_KEY:
        raise HTTPException(500, "FIREBASE_WEB_API_KEY not configured in .env")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{FIREBASE_SIGN_IN_URL}?key={FIREBASE_WEB_API_KEY}",
            json={"email": email, "password": password, "returnSecureToken": True},
        )
    data = resp.json()
    if "error" in data:
        msg = data["error"].get("message", "Sign-in failed")
        code = 401 if "INVALID" in msg or "PASSWORD" in msg else 400
        raise HTTPException(status_code=code, detail=msg)
    return data


async def _firebase_rest_refresh(refresh_token: str) -> dict:
    if not FIREBASE_WEB_API_KEY:
        raise HTTPException(500, "FIREBASE_WEB_API_KEY not configured in .env")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{FIREBASE_REFRESH_URL}?key={FIREBASE_WEB_API_KEY}",
            json={"grant_type": "refresh_token", "refresh_token": refresh_token},
        )
    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=401, detail="Could not refresh token. Please log in again.")
    return data


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignUpRequest):
    """Create a new account with email and password."""
    fb_auth = get_firebase_auth()

    # Create user in Firebase Auth
    try:
        fb_user = fb_auth.create_user(
            email=req.email,
            password=req.password,
            display_name=req.display_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Sign in via REST API to get ID + refresh tokens
    tokens = await _firebase_rest_signin(req.email, req.password)

    logger.info(f"New user registered: {req.email} ({fb_user.uid})")
    return AuthResponse(
        uid           = fb_user.uid,
        email         = fb_user.email,
        display_name  = fb_user.display_name or req.display_name,
        id_token      = tokens["idToken"],
        refresh_token = tokens["refreshToken"],
    )


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Sign in with email and password."""
    tokens = await _firebase_rest_signin(req.email, req.password)

    fb_auth = get_firebase_auth()
    fb_user = fb_auth.get_user(tokens["localId"])

    logger.info(f"User logged in: {req.email}")
    return AuthResponse(
        uid           = tokens["localId"],
        email         = tokens["email"],
        display_name  = fb_user.display_name or tokens.get("displayName", ""),
        id_token      = tokens["idToken"],
        refresh_token = tokens["refreshToken"],
    )


@router.post("/refresh")
async def refresh_token(req: RefreshRequest):
    """Exchange a refresh token for a new ID token."""
    data = await _firebase_rest_refresh(req.refresh_token)
    return {
        "id_token":      data["id_token"],
        "refresh_token": data["refresh_token"],
        "expires_in":    data.get("expires_in", "3600"),
    }


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return the current authenticated user's Firebase profile."""
    fb_auth = get_firebase_auth()
    uid     = user["uid"]
    try:
        fb_user = fb_auth.get_user(uid)
        return {
            "uid":          fb_user.uid,
            "email":        fb_user.email,
            "display_name": fb_user.display_name or "",
            "photo_url":    fb_user.photo_url or "",
            "email_verified": fb_user.email_verified,
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
