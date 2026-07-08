"""
Combined Eye Disease Detection FastAPI Backend
Models: Glaucoma (YOLOv8 + CKAN-SE) · Diabetic Retinopathy (CNN_Retino)
Auth:   Firebase Email/Password + Google OAuth
Data:   Firestore patient profiles and test records
"""

import io
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

# ── Ensure eye_api/ is always importable regardless of launch location ────────
# When Render runs `uvicorn eye_api.main:app` from the project root,
# Python treats eye_api as a package and doesn't add its directory to sys.path.
# This line fixes all sibling imports (model_architecture, firebase_config, etc.)
_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import transforms
from ultralytics import YOLO

# ── Model Architectures ──────────────────────────────────────
from model_architecture import VariantD_CKAN_SE, CNN_Retino, findConv2dOutShape

# Fix: inject classes into __main__ so torch.load can unpickle Retino_model.pt
sys.modules["__main__"].CNN_Retino       = CNN_Retino
sys.modules["__main__"].findConv2dOutShape = findConv2dOutShape

# ── Firebase & Auth ──────────────────────────────
from firebase_config import get_db, get_firebase_auth, ALLOWED_ORIGINS
from auth_middleware import get_optional_user

# ── Routers ──────────────────────────────────────────────────
from routers.auth     import router as auth_router
from routers.patients import router as patients_router
from routers.records  import router as records_router


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

YOLO_PATH = MODELS_DIR / "best.pt"
CKAN_PATH = MODELS_DIR / "ckan_se_final.pth"
DR_PATH   = MODELS_DIR / "Retino_model.pt"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CONF_THRESHOLD = 0.25
OD_MARGIN      = 0.20
INFER_SIZE     = 640
GLAUCOMA_SIZE  = (224, 224)
DR_SIZE        = (255, 255)
DR_CLASSES     = ["Diabetic Retinopathy", "No_DR"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GLOBAL MODEL HOLDER
# ─────────────────────────────────────────────────────────────
class Models:
    yolo:      YOLO             = None
    ckan:      VariantD_CKAN_SE = None
    dr:        CNN_Retino       = None
    device:    torch.device     = None
    ckan_meta: dict             = {}

models = Models()


# ─────────────────────────────────────────────────────────────
# LIFESPAN — load ML models + Firebase at startup
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading eye disease models …")
    models.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {models.device}")

    # 1. YOLOv8
    if not YOLO_PATH.exists():
        raise FileNotFoundError(f"YOLOv8 weights not found: {YOLO_PATH}")
    models.yolo = YOLO(str(YOLO_PATH))
    logger.info("YOLOv8 loaded ✓")

    # 2. CKAN-SE
    if not CKAN_PATH.exists():
        raise FileNotFoundError(f"CKAN-SE checkpoint not found: {CKAN_PATH}")
    checkpoint = torch.load(str(CKAN_PATH), map_location=models.device)
    models.ckan_meta = {
        "model_name":        checkpoint.get("model_name", "CKAN-SE"),
        "num_classes":       checkpoint.get("num_classes", 2),
        "class_names":       checkpoint.get("class_names", ["glaucoma", "normal"]),
        "input_size":        checkpoint.get("input_size", (224, 224)),
        "pipeline":          checkpoint.get("pipeline", "YOLOv8_OD_crop + ResNet18 + KAN-SE"),
        "training_datasets": checkpoint.get("training_datasets", "N/A"),
    }
    ckan_model = VariantD_CKAN_SE(num_classes=models.ckan_meta["num_classes"])
    ckan_model.load_state_dict(checkpoint["model_state_dict"])
    ckan_model.to(models.device).eval()
    models.ckan = ckan_model
    logger.info("CKAN-SE loaded ✓")

    # 3. CNN_Retino
    if not DR_PATH.exists():
        raise FileNotFoundError(f"DR model not found: {DR_PATH}")
    dr_model = torch.load(str(DR_PATH), map_location=models.device, weights_only=False)
    dr_model.to(models.device).eval()
    models.dr = dr_model
    logger.info("CNN_Retino loaded ✓")

    # 4. Firebase (non-fatal — app works without Firebase for pure ML inference)
    try:
        get_firebase_auth()   # triggers initialization
        logger.info("Firebase Admin SDK initialised ✓")
    except Exception as exc:
        logger.warning(
            f"Firebase not configured ({exc}). "
            "Auth/patient/records features will be unavailable. "
            "Place serviceAccountKey.json in eye_api/ or set env vars."
        )

    logger.info("Combined API ready 🚀")
    yield
    logger.info("Shutting down …")


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Eye Disease Detection API",
    description=(
        "Unified backend: Glaucoma (YOLOv8 + CKAN-SE) · "
        "Diabetic Retinopathy (CNN_Retino) · "
        "Firebase Auth · Firestore patient records"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS allow_origins: {ALLOWED_ORIGINS}")

# Register routers
app.include_router(auth_router)
app.include_router(patients_router)
app.include_router(records_router)


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
glaucoma_transform = transforms.Compose([
    transforms.Resize(GLAUCOMA_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

dr_transform = transforms.Compose([
    transforms.Resize(DR_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────
def crop_optic_disc(img_rgb: np.ndarray) -> tuple:
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = models.yolo.predict(
        source=img_bgr, conf=CONF_THRESHOLD,
        classes=[0], verbose=False, save=False, imgsz=INFER_SIZE,
    )

    best_box, best_conf = None, 0.0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            c = float(box.conf[0])
            if int(box.cls[0]) == 0 and c > best_conf:
                best_conf, best_box = c, box.xyxy[0].cpu().numpy()

    if best_box is not None and best_conf >= CONF_THRESHOLD:
        x1, y1, x2, y2 = best_box
        bw, bh = x2 - x1, y2 - y1
        x1m = max(0, int(x1 - OD_MARGIN * bw))
        y1m = max(0, int(y1 - OD_MARGIN * bh))
        x2m = min(w, int(x2 + OD_MARGIN * bw))
        y2m = min(h, int(y2 + OD_MARGIN * bh))
        cropped = img_rgb[y1m:y2m, x1m:x2m]
        detection = {
            "detected": True, "confidence": round(float(best_conf), 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "crop_bbox": [x1m, y1m, x2m, y2m],
        }
    else:
        frac = 0.60
        ch, cw = int(h * frac), int(w * frac)
        t, l   = (h - ch) // 2, (w - cw) // 2
        cropped = img_rgb[t:t + ch, l:l + cw]
        detection = {
            "detected": False, "confidence": 0.0, "bbox": None,
            "crop_bbox": [l, t, l + cw, t + ch],
        }
    return cropped, detection


def classify_glaucoma(cropped_rgb: np.ndarray) -> tuple:
    pil_img = Image.fromarray(cropped_rgb)
    tensor  = glaucoma_transform(pil_img).unsqueeze(0).to(models.device)
    with torch.no_grad():
        probs = F.softmax(models.ckan(tensor), dim=1).squeeze(0).cpu().numpy()
    class_names = models.ckan_meta["class_names"]
    pred_idx    = int(probs.argmax())
    return class_names[pred_idx], float(probs[pred_idx]), {c: round(float(p), 6) for c, p in zip(class_names, probs)}


def classify_dr(img_rgb: np.ndarray) -> tuple:
    pil_img = Image.fromarray(img_rgb)
    tensor  = dr_transform(pil_img).unsqueeze(0).to(models.device)
    with torch.no_grad():
        probs = torch.exp(models.dr(tensor)).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    return DR_CLASSES[pred_idx], float(probs[pred_idx]), {c: round(float(p), 6) for c, p in zip(DR_CLASSES, probs)}


def glaucoma_risk(p: str, c: float) -> str:
    if p == "glaucoma":
        return "HIGH" if c >= 0.85 else ("MODERATE" if c >= 0.65 else "LOW-MODERATE")
    return "LOW" if c >= 0.85 else ("LOW-MODERATE" if c >= 0.65 else "UNCERTAIN")


def dr_risk(p: str, c: float) -> str:
    if p == "Diabetic Retinopathy":
        return "HIGH" if c >= 0.85 else ("MODERATE" if c >= 0.65 else "LOW-MODERATE")
    return "LOW" if c >= 0.85 else ("LOW-MODERATE" if c >= 0.65 else "UNCERTAIN")


GLAUCOMA_REC = {
    "HIGH":         "Urgent ophthalmological consultation recommended. High probability of glaucoma detected.",
    "MODERATE":     "Ophthalmological evaluation advised. Moderate glaucoma indicators present.",
    "LOW-MODERATE": "Further screening recommended. Borderline findings detected.",
    "LOW":          "No significant glaucoma indicators detected. Routine annual eye check-up advised.",
    "UNCERTAIN":    "Borderline result. Repeat imaging or clinical evaluation recommended.",
}
DR_REC = {
    "HIGH":         "Urgent ophthalmological consultation recommended. High probability of Diabetic Retinopathy detected.",
    "MODERATE":     "Ophthalmological evaluation advised within 1–3 months. Maintain good glycaemic control.",
    "LOW-MODERATE": "Further screening recommended. Monitor blood sugar and consult a specialist.",
    "LOW":          "No significant DR indicators. Continue regular annual diabetic eye screening.",
    "UNCERTAIN":    "Borderline result. Repeat imaging or ophthalmologist evaluation recommended.",
}


async def _auto_save_record(uid: str, test_type: str, image_name: str, glaucoma_result=None, dr_result=None):
    """Silently save a screening result to Firestore if Firebase is configured."""
    try:
        db  = get_db()
        now = datetime.now(timezone.utc)
        db.collection("test_records").add({
            "patient_uid":     uid,
            "test_type":       test_type,
            "image_name":      image_name,
            "glaucoma_result": glaucoma_result,
            "dr_result":       dr_result,
            "notes":           "",
            "created_at":      now,
        })
    except Exception as exc:
        logger.debug(f"Auto-save record skipped: {exc}")


# ─────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────
class GlaucomaPredictionResponse(BaseModel):
    prediction: str; confidence: float; probabilities: dict
    optic_disc: dict; inference_time_ms: float
    risk_level: str; recommendation: str


class DRPredictionResponse(BaseModel):
    prediction: str; confidence: float; probabilities: dict
    inference_time_ms: float; risk_level: str; recommendation: str


class BothPredictionResponse(BaseModel):
    glaucoma: GlaucomaPredictionResponse
    dr:       DRPredictionResponse


class HealthResponse(BaseModel):
    status: str; models: dict; device: str; firebase: bool


class InfoResponse(BaseModel):
    glaucoma: dict; dr: dict


# ─────────────────────────────────────────────────────────────
# CORE ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Eye Disease Detection API v2", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    firebase_ok = False
    try:
        get_firebase_auth()
        firebase_ok = True
    except Exception:
        pass
    return HealthResponse(
        status  = "healthy",
        models  = {"yolo": models.yolo is not None, "ckan": models.ckan is not None, "dr": models.dr is not None},
        device  = str(models.device),
        firebase= firebase_ok,
    )


@app.get("/info", response_model=InfoResponse, tags=["Info"])
async def info():
    return InfoResponse(
        glaucoma={
            "model_name":  models.ckan_meta.get("model_name", "CKAN-SE"),
            "pipeline":    models.ckan_meta.get("pipeline", "N/A"),
            "num_classes": models.ckan_meta.get("num_classes", 2),
            "class_names": models.ckan_meta.get("class_names", []),
            "input_size":  list(models.ckan_meta.get("input_size", [224, 224])),
        },
        dr={
            "model_name":   "CNN_Retino",
            "architecture": "4-layer CNN + MaxPool + Dropout",
            "num_classes":  2,
            "class_names":  DR_CLASSES,
            "input_size":   list(DR_SIZE),
        }
    )


# ─────────────────────────────────────────────────────────────
# PREDICT ROUTES  (auth optional — saves record if logged in)
# ─────────────────────────────────────────────────────────────
VALID_TYPES = ("image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff")


def _check_file(file: UploadFile):
    if file.content_type not in VALID_TYPES:
        raise HTTPException(400, "Unsupported file type. Please upload JPEG or PNG.")


async def _decode_image(file: UploadFile) -> tuple[np.ndarray, bytes]:
    contents = await file.read()
    try:
        img_rgb = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    except Exception as exc:
        raise HTTPException(400, f"Could not decode image: {exc}")
    return img_rgb, contents


@app.post("/predict/glaucoma", response_model=GlaucomaPredictionResponse, tags=["Predictions"])
async def predict_glaucoma(
    file: UploadFile = File(...),
    user: Optional[dict] = Depends(get_optional_user),
):
    """Glaucoma screening. If authenticated, result is auto-saved to your records."""
    _check_file(file)
    start = time.perf_counter()
    img_rgb, _ = await _decode_image(file)

    cropped, od_info             = crop_optic_disc(img_rgb)
    pred, conf, probs            = classify_glaucoma(cropped)
    elapsed                      = round((time.perf_counter() - start) * 1000, 2)
    risk, rec                    = glaucoma_risk(pred, conf), GLAUCOMA_REC.get(glaucoma_risk(pred, conf), "")

    result = GlaucomaPredictionResponse(
        prediction=pred, confidence=round(conf, 6), probabilities=probs,
        optic_disc=od_info, inference_time_ms=elapsed, risk_level=risk, recommendation=rec,
    )
    if user:
        await _auto_save_record(user["uid"], "glaucoma", file.filename, glaucoma_result=result.model_dump())
    return result


@app.post("/predict/dr", response_model=DRPredictionResponse, tags=["Predictions"])
async def predict_dr(
    file: UploadFile = File(...),
    user: Optional[dict] = Depends(get_optional_user),
):
    """DR screening. If authenticated, result is auto-saved to your records."""
    _check_file(file)
    start = time.perf_counter()
    img_rgb, _ = await _decode_image(file)

    pred, conf, probs = classify_dr(img_rgb)
    elapsed           = round((time.perf_counter() - start) * 1000, 2)
    risk, rec         = dr_risk(pred, conf), DR_REC.get(dr_risk(pred, conf), "")

    result = DRPredictionResponse(
        prediction=pred, confidence=round(conf, 6), probabilities=probs,
        inference_time_ms=elapsed, risk_level=risk, recommendation=rec,
    )
    if user:
        await _auto_save_record(user["uid"], "dr", file.filename, dr_result=result.model_dump())
    return result


@app.post("/predict/both", response_model=BothPredictionResponse, tags=["Predictions"])
async def predict_both(
    file: UploadFile = File(...),
    user: Optional[dict] = Depends(get_optional_user),
):
    """Both Glaucoma + DR screening in one request. Auto-saves if authenticated."""
    _check_file(file)
    img_rgb, _ = await _decode_image(file)

    t0 = time.perf_counter()
    cropped, od_info          = crop_optic_disc(img_rgb)
    g_pred, g_conf, g_probs   = classify_glaucoma(cropped)
    g_ms                      = round((time.perf_counter() - t0) * 1000, 2)
    g_risk, g_rec             = glaucoma_risk(g_pred, g_conf), GLAUCOMA_REC.get(glaucoma_risk(g_pred, g_conf), "")

    t1 = time.perf_counter()
    d_pred, d_conf, d_probs   = classify_dr(img_rgb)
    d_ms                      = round((time.perf_counter() - t1) * 1000, 2)
    d_risk, d_rec             = dr_risk(d_pred, d_conf), DR_REC.get(dr_risk(d_pred, d_conf), "")

    g_result = GlaucomaPredictionResponse(
        prediction=g_pred, confidence=round(g_conf, 6), probabilities=g_probs,
        optic_disc=od_info, inference_time_ms=g_ms, risk_level=g_risk, recommendation=g_rec,
    )
    d_result = DRPredictionResponse(
        prediction=d_pred, confidence=round(d_conf, 6), probabilities=d_probs,
        inference_time_ms=d_ms, risk_level=d_risk, recommendation=d_rec,
    )
    if user:
        await _auto_save_record(
            user["uid"], "both", file.filename,
            glaucoma_result=g_result.model_dump(),
            dr_result=d_result.model_dump(),
        )
    return BothPredictionResponse(glaucoma=g_result, dr=d_result)


# ─────────────────────────────────────────────────────────────
# BATCH ROUTES
# ─────────────────────────────────────────────────────────────
@app.post("/predict/glaucoma/batch", tags=["Predictions"])
async def predict_glaucoma_batch(
    files: list[UploadFile] = File(...),
    user: Optional[dict] = Depends(get_optional_user),
):
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 images per batch.")
    results = []
    for f in files:
        try:
            img_rgb, _ = await _decode_image(f)
            cropped, det = crop_optic_disc(img_rgb)
            pred, conf, probs = classify_glaucoma(cropped)
            risk = glaucoma_risk(pred, conf)
            row = {"filename": f.filename, "prediction": pred, "confidence": round(conf, 6),
                   "probabilities": probs, "risk_level": risk, "optic_disc": det, "error": None}
            if user:
                await _auto_save_record(user["uid"], "glaucoma", f.filename, glaucoma_result=row)
            results.append(row)
        except Exception as exc:
            results.append({"filename": f.filename, "prediction": None, "confidence": None, "error": str(exc)})
    return JSONResponse({"results": results, "total": len(results)})


@app.post("/predict/dr/batch", tags=["Predictions"])
async def predict_dr_batch(
    files: list[UploadFile] = File(...),
    user: Optional[dict] = Depends(get_optional_user),
):
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 images per batch.")
    results = []
    for f in files:
        try:
            img_rgb, _ = await _decode_image(f)
            pred, conf, probs = classify_dr(img_rgb)
            risk = dr_risk(pred, conf)
            row = {"filename": f.filename, "prediction": pred, "confidence": round(conf, 6),
                   "probabilities": probs, "risk_level": risk, "error": None}
            if user:
                await _auto_save_record(user["uid"], "dr", f.filename, dr_result=row)
            results.append(row)
        except Exception as exc:
            results.append({"filename": f.filename, "prediction": None, "confidence": None, "error": str(exc)})
    return JSONResponse({"results": results, "total": len(results)})
