"""
Test Records Router
Save and retrieve eye disease screening results linked to a patient UID.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from firebase_config import get_db
from auth_middleware import get_current_user

router = APIRouter(prefix="/records", tags=["Test Records"])
logger = logging.getLogger(__name__)

COLLECTION = "test_records"


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class SaveRecordRequest(BaseModel):
    test_type:       str                     # glaucoma / dr / both
    image_name:      str = "unknown"
    glaucoma_result: Optional[dict] = None
    dr_result:       Optional[dict] = None
    notes:           str = ""


class RecordResponse(SaveRecordRequest):
    record_id:   str
    patient_uid: str
    created_at:  str


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _ts(dt) -> str:
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    if hasattr(dt, "timestamp"):
        return datetime.fromtimestamp(dt.timestamp(), tz=timezone.utc).isoformat()
    return str(dt)


def _doc_to_response(record_id: str, data: dict) -> dict:
    return {
        "record_id":       record_id,
        "patient_uid":     data.get("patient_uid", ""),
        "test_type":       data.get("test_type", ""),
        "image_name":      data.get("image_name", ""),
        "glaucoma_result": data.get("glaucoma_result"),
        "dr_result":       data.get("dr_result"),
        "notes":           data.get("notes", ""),
        "created_at":      _ts(data.get("created_at")),
    }


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@router.post("", status_code=status.HTTP_201_CREATED)
async def save_record(
    req: SaveRecordRequest,
    user: dict = Depends(get_current_user),
):
    """Save a screening result to the patient's record history."""
    db  = get_db()
    uid = user["uid"]
    now = datetime.now(timezone.utc)

    data = {
        **req.model_dump(),
        "patient_uid": uid,
        "created_at":  now,
    }
    doc_ref = db.collection(COLLECTION).add(data)
    record_id = doc_ref[1].id
    logger.info(f"Record {record_id} saved for uid={uid} (test_type={req.test_type})")

    return {"record_id": record_id, **_doc_to_response(record_id, data)}


@router.get("")
async def get_my_records(
    user:  dict = Depends(get_current_user),
    limit: int  = Query(default=20, le=100),
    test_type: Optional[str] = Query(default=None),
):
    """
    Return the authenticated user's test records, ordered by newest first.
    Optionally filter by test_type: glaucoma / dr / both
    """
    db    = get_db()
    uid   = user["uid"]
    query = (
        db.collection(COLLECTION)
          .where("patient_uid", "==", uid)
          .order_by("created_at", direction="DESCENDING")
          .limit(limit)
    )
    if test_type:
        query = query.where("test_type", "==", test_type)

    docs = query.stream()
    records = [_doc_to_response(doc.id, doc.to_dict()) for doc in docs]
    return {"records": records, "total": len(records)}


@router.get("/{record_id}")
async def get_record(
    record_id: str,
    user: dict = Depends(get_current_user),
):
    """Retrieve a single test record by ID (must belong to the authenticated user)."""
    db  = get_db()
    doc = db.collection(COLLECTION).document(record_id).get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Record not found.")

    data = doc.to_dict()
    if data.get("patient_uid") != user["uid"]:
        raise HTTPException(status_code=403, detail="Access denied.")

    return _doc_to_response(doc.id, data)


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_record(
    record_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a test record (must belong to the authenticated user)."""
    db  = get_db()
    ref = db.collection(COLLECTION).document(record_id)
    doc = ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Record not found.")
    if doc.to_dict().get("patient_uid") != user["uid"]:
        raise HTTPException(status_code=403, detail="Access denied.")

    ref.delete()
    logger.info(f"Record {record_id} deleted by uid={user['uid']}")
