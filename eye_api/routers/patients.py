"""
Patient Profile Router
CRUD operations for patient health profiles stored in Firestore.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from firebase_config import get_db
from auth_middleware import get_current_user

router = APIRouter(prefix="/patients", tags=["Patients"])
logger = logging.getLogger(__name__)

COLLECTION = "patients"


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class PatientProfile(BaseModel):
    name:                    str
    age:                     int
    gender:                  str                    # Male / Female / Other
    location:                str
    optical_power_left:      float = 0.0            # diopters (negative = myopia)
    optical_power_right:     float = 0.0
    has_diabetes:            bool  = False
    diabetes_type:           str   = ""             # Type 1 / Type 2 / Pre-diabetic
    bp_systolic:             int   = 0
    bp_diastolic:            int   = 0
    existing_eye_conditions: List[str] = []         # Glaucoma / Cataract / etc.
    is_smoker:               bool  = False
    notes:                   str   = ""


class PatientProfileResponse(PatientProfile):
    uid:        str
    email:      str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _ts(dt) -> str:
    """Convert Firestore timestamp or datetime to ISO string."""
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    if hasattr(dt, "timestamp"):
        return datetime.fromtimestamp(dt.timestamp(), tz=timezone.utc).isoformat()
    return str(dt)


def _doc_to_response(data: dict) -> PatientProfileResponse:
    return PatientProfileResponse(
        uid                    = data.get("uid", ""),
        email                  = data.get("email", ""),
        name                   = data.get("name", ""),
        age                    = data.get("age", 0),
        gender                 = data.get("gender", ""),
        location               = data.get("location", ""),
        optical_power_left     = data.get("optical_power_left", 0.0),
        optical_power_right    = data.get("optical_power_right", 0.0),
        has_diabetes           = data.get("has_diabetes", False),
        diabetes_type          = data.get("diabetes_type", ""),
        bp_systolic            = data.get("bp_systolic", 0),
        bp_diastolic           = data.get("bp_diastolic", 0),
        existing_eye_conditions= data.get("existing_eye_conditions", []),
        is_smoker              = data.get("is_smoker", False),
        notes                  = data.get("notes", ""),
        created_at             = _ts(data.get("created_at")),
        updated_at             = _ts(data.get("updated_at")),
    )


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@router.post("", response_model=PatientProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(
    profile: PatientProfile,
    user: dict = Depends(get_current_user),
):
    """Create a new patient profile for the authenticated user."""
    db  = get_db()
    uid = user["uid"]
    ref = db.collection(COLLECTION).document(uid)

    if ref.get().exists:
        raise HTTPException(
            status_code=409,
            detail="Patient profile already exists. Use PUT /patients/me to update.",
        )

    now  = datetime.now(timezone.utc)
    data = {
        **profile.model_dump(),
        "uid":        uid,
        "email":      user.get("email", ""),
        "created_at": now,
        "updated_at": now,
    }
    ref.set(data)
    logger.info(f"Patient profile created for uid={uid}")
    return _doc_to_response(data)


@router.get("/me", response_model=PatientProfileResponse)
async def get_my_profile(user: dict = Depends(get_current_user)):
    """Retrieve the authenticated user's patient profile."""
    db  = get_db()
    uid = user["uid"]
    doc = db.collection(COLLECTION).document(uid).get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Patient profile not found.")
    return _doc_to_response(doc.to_dict())


@router.put("/me", response_model=PatientProfileResponse)
async def update_my_profile(
    profile: PatientProfile,
    user: dict = Depends(get_current_user),
):
    """Update the authenticated user's patient profile."""
    db  = get_db()
    uid = user["uid"]
    ref = db.collection(COLLECTION).document(uid)

    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Patient profile not found. POST /patients first.")

    now    = datetime.now(timezone.utc)
    update = {**profile.model_dump(), "updated_at": now}
    ref.update(update)

    updated_doc = ref.get().to_dict()
    logger.info(f"Patient profile updated for uid={uid}")
    return _doc_to_response(updated_doc)
