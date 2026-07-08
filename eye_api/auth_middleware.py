"""
Authentication Middleware — FastAPI dependency for verifying Firebase ID tokens.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from firebase_config import get_firebase_auth

logger = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    Dependency: verifies Bearer token, raises 401 if invalid.
    Returns the decoded Firebase token payload.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
        )
    try:
        fb_auth = get_firebase_auth()
        decoded = fb_auth.verify_id_token(credentials.credentials)
        return decoded
    except Exception as exc:
        logger.warning(f"Token verification failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token. Please log in again.",
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """
    Optional dependency: returns decoded token if present and valid, else None.
    Used on predict endpoints — saves record to Firestore only when authenticated.
    """
    if credentials is None:
        return None
    try:
        fb_auth = get_firebase_auth()
        return fb_auth.verify_id_token(credentials.credentials)
    except Exception:
        return None
