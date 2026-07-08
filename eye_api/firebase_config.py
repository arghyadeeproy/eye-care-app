"""
Firebase Admin SDK — Singleton Initializer
Reads credentials from eye_api/serviceAccountKey.json or FIREBASE_SERVICE_ACCOUNT_JSON env var.
"""

import os
import json
import logging
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore, auth as _firebase_auth
from dotenv import load_dotenv

# Load .env file if it exists
_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(str(_ENV_PATH))

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent


def _init_app() -> firebase_admin.App:
    """Initialize Firebase Admin SDK (runs once)."""
    if firebase_admin._apps:
        return firebase_admin.get_app()

    sa_path = BASE_DIR / "serviceAccountKey.json"
    env_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

    if env_json:
        sa_dict = json.loads(env_json)
        cred = credentials.Certificate(sa_dict)
        logger.info("Firebase Admin: loaded credentials from environment variable.")
    elif sa_path.exists():
        cred = credentials.Certificate(str(sa_path))
        logger.info(f"Firebase Admin: loaded credentials from {sa_path}.")
    else:
        raise RuntimeError(
            "Firebase service account key not found.\n"
            f"  Option A: Place 'serviceAccountKey.json' inside eye_api/\n"
            f"  Option B: Set the FIREBASE_SERVICE_ACCOUNT_JSON environment variable.\n"
            "Download the key from: Firebase Console → Project Settings → Service Accounts."
        )

    return firebase_admin.initialize_app(cred)


def get_db() -> firestore.client:
    """Return a Firestore client (initialises Firebase if needed)."""
    _init_app()
    return firestore.client()


def get_firebase_auth():
    """Return firebase_admin.auth module (initialises Firebase if needed)."""
    _init_app()
    return _firebase_auth


# Config values loaded from env
FIREBASE_WEB_API_KEY: str = os.environ.get("FIREBASE_WEB_API_KEY", "")
FIREBASE_PROJECT_ID:  str = os.environ.get("FIREBASE_PROJECT_ID", "")

# CORS allowed origins — comma-separated list e.g. https://myapp.com,https://app2.com
# Use * to allow all origins (not recommended in production)
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: list = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

FIREBASE_SIGN_IN_URL = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
)
FIREBASE_SIGN_UP_URL = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
)
FIREBASE_REFRESH_URL = "https://securetoken.googleapis.com/v1/token"
