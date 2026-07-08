# Eye Disease Detection API — HTTP Command & Path Guide

This guide describes all API endpoints, HTTP methods, headers, request payloads, response schemas, and `curl` command examples for the deployment-ready backend.

---

## 🌐 Base Configuration

* **Local Base URL**: `http://localhost:8000`
* **Content-Type**: `application/json` (except for `/predict/*` file uploads which use `multipart/form-data`)

### Authentication Header
Endpoints marked with **[Lock]** require the user to be authenticated. You must pass the Firebase ID token in the HTTP headers:
```http
Authorization: Bearer <FIREBASE_ID_TOKEN>
```

---

## 🔒 1. User Authentication (`/auth`)

### A. Sign Up (Create Account)
* **Path**: `/auth/signup`
* **Method**: `POST`
* **Payload**:
```json
{
  "email": "patient@example.com",
  "password": "securepassword123",
  "display_name": "Jane Doe"
}
```
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/auth/signup \
     -H "Content-Type: application/json" \
     -d '{"email":"patient@example.com","password":"securepassword123","display_name":"Jane Doe"}'
```
* **Response (200 OK)**:
```json
{
  "uid": "fb_user_uid_12345",
  "email": "patient@example.com",
  "display_name": "Jane Doe",
  "id_token": "ey...",
  "refresh_token": "AMf..."
}
```

### B. Sign In (Login)
* **Path**: `/auth/login`
* **Method**: `POST`
* **Payload**:
```json
{
  "email": "patient@example.com",
  "password": "securepassword123"
}
```
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"patient@example.com","password":"securepassword123"}'
```
* **Response (200 OK)**: Returns the user's Firebase token payload (`id_token` is valid for 1 hour).

### C. Refresh ID Token
* **Path**: `/auth/refresh`
* **Method**: `POST`
* **Payload**:
```json
{
  "refresh_token": "AMf..."
}
```
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/auth/refresh \
     -H "Content-Type: application/json" \
     -d '{"refresh_token":"AMf..."}'
```
* **Response (200 OK)**:
```json
{
  "id_token": "ey...",
  "refresh_token": "AMf...",
  "expires_in": "3600"
}
```

### D. Get Current User Details [Lock]
* **Path**: `/auth/me`
* **Method**: `GET`
* **cURL Command**:
```bash
curl -X GET http://localhost:8000/auth/me \
     -H "Authorization: Bearer <ID_TOKEN>"
```

---

## 👤 2. Patient Profiles (`/patients`)

### A. Create Profile [Lock]
* **Path**: `/patients`
* **Method**: `POST`
* **Payload**:
```json
{
  "name": "Jane Doe",
  "age": 35,
  "gender": "Female",
  "location": "New York",
  "optical_power_left": -2.50,
  "optical_power_right": -2.25,
  "has_diabetes": true,
  "diabetes_type": "Type 2",
  "bp_systolic": 120,
  "bp_diastolic": 80,
  "existing_eye_conditions": ["Dry Eye"],
  "is_smoker": false,
  "notes": "No known allergies."
}
```
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/patients \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"name":"Jane Doe","age":35,"gender":"Female","location":"New York","optical_power_left":-2.50,"optical_power_right":-2.25,"has_diabetes":true,"diabetes_type":"Type 2","bp_systolic":120,"bp_diastolic":80,"existing_eye_conditions":["Dry Eye"],"is_smoker":false,"notes":""}'
```

### B. Fetch Profile [Lock]
* **Path**: `/patients/me`
* **Method**: `GET`
* **cURL Command**:
```bash
curl -X GET http://localhost:8000/patients/me \
     -H "Authorization: Bearer <ID_TOKEN>"
```

### C. Update Profile [Lock]
* **Path**: `/patients/me`
* **Method**: `PUT`
* **Payload**: Same schema as Create Profile.
* **cURL Command**:
```bash
curl -X PUT http://localhost:8000/patients/me \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"name":"Jane Doe","age":36,"gender":"Female","location":"Boston",...}'
```

---

## 🔬 3. Diagnostics & AI Predictions (`/predict`)

> [!TIP]
> Authentication headers are **optional** on prediction routes. If you provide a valid Bearer token, the backend will silently save the prediction results to the patient's Firestore record history automatically.

### A. Glaucoma Screening (YOLOv8 + CKAN-SE)
* **Path**: `/predict/glaucoma`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `file` (binary fundus image)
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/predict/glaucoma \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -F "file=@/path/to/retina_image.jpg"
```
* **Response (200 OK)**:
```json
{
  "prediction": "glaucoma",
  "confidence": 0.924821,
  "probabilities": {
    "glaucoma": 0.924821,
    "normal": 0.075179
  },
  "optic_disc": {
    "detected": true,
    "confidence": 0.89,
    "bbox": [120, 100, 310, 290],
    "crop_bbox": [96, 75, 334, 315]
  },
  "inference_time_ms": 115.42,
  "risk_level": "HIGH",
  "recommendation": "Urgent ophthalmological consultation recommended."
}
```

### B. Diabetic Retinopathy Screening (CNN_Retino)
* **Path**: `/predict/dr`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `file` (binary fundus image)
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/predict/dr \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -F "file=@/path/to/retina_image.jpg"
```
* **Response (200 OK)**:
```json
{
  "prediction": "No_DR",
  "confidence": 0.984152,
  "probabilities": {
    "Diabetic Retinopathy": 0.015848,
    "No_DR": 0.984152
  },
  "inference_time_ms": 42.15,
  "risk_level": "LOW",
  "recommendation": "No significant DR indicators detected. Continue regular annual diabetic eye screening."
}
```

### C. Combined Test (Both Glaucoma + DR)
* **Path**: `/predict/both`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `file` (binary fundus image)
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/predict/both \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -F "file=@/path/to/retina_image.jpg"
```
* **Response (200 OK)**:
```json
{
  "glaucoma": {
    "prediction": "normal",
    "confidence": 0.971,
    "probabilities": { "glaucoma": 0.029, "normal": 0.971 },
    "optic_disc": { "detected": true, "confidence": 0.92, ... },
    "inference_time_ms": 112.5,
    "risk_level": "LOW",
    "recommendation": "..."
  },
  "dr": {
    "prediction": "No_DR",
    "confidence": 0.992,
    "probabilities": { "Diabetic Retinopathy": 0.008, "No_DR": 0.992 },
    "inference_time_ms": 38.4,
    "risk_level": "LOW",
    "recommendation": "..."
  }
}
```

---

## 📋 4. Screening Records History (`/records`)

### A. Save Custom Record [Lock]
Manually save/log a screening result.
* **Path**: `/records`
* **Method**: `POST`
* **Payload**:
```json
{
  "test_type": "glaucoma",
  "image_name": "retina_left_eye.jpg",
  "glaucoma_result": {
    "prediction": "normal",
    "confidence": 0.985,
    "probabilities": { "glaucoma": 0.015, "normal": 0.985 },
    "risk_level": "LOW",
    "recommendation": "..."
  },
  "dr_result": null,
  "notes": "Patient reports minor blurred vision."
}
```
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/records \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"test_type":"glaucoma","image_name":"eye.jpg","glaucoma_result":{...},"dr_result":null,"notes":""}'
```

### B. List User Records [Lock]
Retrieve user screening logs.
* **Path**: `/records`
* **Method**: `GET`
* **Query Parameters**:
  * `limit` (int, default=20): Number of records to return.
  * `test_type` (string, optional): Filter by `glaucoma`, `dr`, or `both`.
* **cURL Command**:
```bash
curl -X GET "http://localhost:8000/records?limit=10&test_type=both" \
     -H "Authorization: Bearer <ID_TOKEN>"
```

### C. Delete Record [Lock]
* **Path**: `/records/{record_id}`
* **Method**: `DELETE`
* **cURL Command**:
```bash
curl -X DELETE http://localhost:8000/records/some_doc_id_999 \
     -H "Authorization: Bearer <ID_TOKEN>"
```
* **Response**: `244 No Content` on successful deletion.

---

## 🩺 5. Monitoring & Metadata

### A. Health Check
Checks model load status and Firebase configuration.
* **Path**: `/health`
* **Method**: `GET`
* **Response**:
```json
{
  "status": "healthy",
  "models": {
    "yolo": true,
    "ckan": true,
    "dr": true
  },
  "device": "cuda",
  "firebase": true
}
```

### B. Model Info
Returns technical metadata about both loaded models.
* **Path**: `/info`
* **Method**: `GET`
* **cURL Command**:
```bash
curl -X GET http://localhost:8000/info
```
* **Response (200 OK)**:
```json
{
  "glaucoma": {
    "model_name": "CKAN-SE",
    "pipeline": "YOLOv8_OD_crop + ResNet18 + KAN-SE",
    "num_classes": 2,
    "class_names": ["glaucoma", "normal"],
    "input_size": [224, 224]
  },
  "dr": {
    "model_name": "CNN_Retino",
    "architecture": "4-layer CNN + MaxPool + Dropout",
    "num_classes": 2,
    "class_names": ["Diabetic Retinopathy", "No_DR"],
    "input_size": [255, 255]
  }
}
```

---

## 📦 6. Batch Predictions (`/predict/*/batch`)

> [!NOTE]
> Batch endpoints accept up to **10 images per request**. Each image is processed independently and results are returned as an array. Authentication is optional — authenticated requests auto-save all results to Firestore.

### A. Glaucoma Batch Screening
* **Path**: `/predict/glaucoma/batch`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `files` (multiple binary images, up to 10)
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/predict/glaucoma/batch \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```
* **Response (200 OK)**:
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "prediction": "normal",
      "confidence": 0.978,
      "probabilities": { "glaucoma": 0.022, "normal": 0.978 },
      "risk_level": "LOW",
      "optic_disc": { "detected": true, "confidence": 0.91 },
      "error": null
    },
    {
      "filename": "image2.jpg",
      "prediction": "glaucoma",
      "confidence": 0.864,
      "probabilities": { "glaucoma": 0.864, "normal": 0.136 },
      "risk_level": "HIGH",
      "optic_disc": { "detected": true, "confidence": 0.88 },
      "error": null
    }
  ],
  "total": 2
}
```

### B. Diabetic Retinopathy Batch Screening
* **Path**: `/predict/dr/batch`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `files` (multiple binary images, up to 10)
* **cURL Command**:
```bash
curl -X POST http://localhost:8000/predict/dr/batch \
     -H "Authorization: Bearer <ID_TOKEN>" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```
* **Response (200 OK)**:
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "prediction": "No_DR",
      "confidence": 0.991,
      "probabilities": { "Diabetic Retinopathy": 0.009, "No_DR": 0.991 },
      "risk_level": "LOW",
      "error": null
    }
  ],
  "total": 1
}
```

---

## 🚦 7. Risk Levels Reference

All prediction endpoints return a `risk_level` field. Here is the full reference table:

| Risk Level | Condition | Confidence Threshold | Recommended Action |
|---|---|---|---|
| `HIGH` | Positive (disease detected) | ≥ 85% | Urgent specialist consultation |
| `MODERATE` | Positive (disease detected) | 65–84% | Evaluation within 1–3 months |
| `LOW-MODERATE` | Positive (borderline) | < 65% | Further screening recommended |
| `LOW` | Negative (no disease) | ≥ 85% | Routine annual check-up |
| `UNCERTAIN` | Negative (borderline) | < 85% | Repeat imaging advised |

---

## ⚠️ 8. Error Codes Reference

| HTTP Status | Meaning | Common Cause |
|---|---|---|
| `400 Bad Request` | Invalid request payload or credentials | Wrong email/password, bad JSON, unsupported image type |
| `401 Unauthorized` | Authentication token missing or expired | Missing `Authorization` header, expired `id_token` |
| `403 Forbidden` | Access denied to another user's resource | Trying to delete/read another patient's record |
| `404 Not Found` | Resource doesn't exist | Patient profile not yet created, invalid `record_id` |
| `409 Conflict` | Resource already exists | Trying to `POST /patients` when profile already created (use `PUT` instead) |
| `422 Unprocessable Entity` | Request validation error | Missing required field, wrong data type |
| `500 Internal Server Error` | Backend crash or misconfiguration | Firebase not configured, model file not found |
| `503 Service Unavailable` | Feature unavailable | Optional feature not configured |

---

## 🐍 9. Python Client Examples

### Sign Up and Run a Glaucoma Prediction
```python
import requests

BASE = "http://localhost:8000"

# 1. Create account
resp = requests.post(f"{BASE}/auth/signup", json={
    "email": "patient@example.com",
    "password": "pass1234!",
    "display_name": "Arghyadeep Roy"
})
tokens = resp.json()
id_token = tokens["id_token"]
headers = {"Authorization": f"Bearer {id_token}"}

# 2. Create patient profile
requests.post(f"{BASE}/patients", headers=headers, json={
    "name": "Arghyadeep Roy", "age": 23, "gender": "Male",
    "location": "Kolkata", "optical_power_left": -1.75,
    "optical_power_right": -1.5, "has_diabetes": False,
    "bp_systolic": 118, "bp_diastolic": 76,
    "existing_eye_conditions": [], "is_smoker": False, "notes": ""
})

# 3. Run glaucoma prediction (auto-saved to Firestore)
with open("fundus_image.jpg", "rb") as f:
    pred = requests.post(
        f"{BASE}/predict/glaucoma",
        headers=headers,
        files={"file": ("fundus_image.jpg", f, "image/jpeg")}
    )
print(pred.json())

# 4. Fetch saved records
records = requests.get(f"{BASE}/records", headers=headers)
print(records.json())
```

### Refresh Expired Token
```python
import requests

resp = requests.post("http://localhost:8000/auth/refresh", json={
    "refresh_token": "your_refresh_token_here"
})
new_token = resp.json()["id_token"]
```

---

## 🌐 10. JavaScript / Fetch API Examples

### Login and Run DR Prediction
```javascript
const BASE = "http://localhost:8000";

// 1. Login
const loginResp = await fetch(`${BASE}/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ email: "patient@example.com", password: "pass1234!" })
});
const { id_token } = await loginResp.json();

// 2. Run DR prediction with an image file
const formData = new FormData();
formData.append("file", imageFile); // imageFile is a browser File object

const predResp = await fetch(`${BASE}/predict/dr`, {
  method: "POST",
  headers: { "Authorization": `Bearer ${id_token}` },
  body: formData
});
const result = await predResp.json();
console.log(result.prediction, result.risk_level);

// 3. Get screening records
const records = await fetch(`${BASE}/records?limit=10`, {
  headers: { "Authorization": `Bearer ${id_token}` }
});
console.log(await records.json());
```

---

## 📋 11. Complete Endpoint Reference Table

| Method | Path | Auth Required | Description |
|--------|------|:---:|-------------|
| `GET` | `/` | ❌ | API root info |
| `GET` | `/health` | ❌ | Server & model health check |
| `GET` | `/info` | ❌ | Model architecture metadata |
| `POST` | `/auth/signup` | ❌ | Register new account |
| `POST` | `/auth/login` | ❌ | Sign in, returns Firebase tokens |
| `POST` | `/auth/refresh` | ❌ | Refresh expired ID token |
| `GET` | `/auth/me` | ✅ | Get current user info |
| `POST` | `/patients` | ✅ | Create patient health profile |
| `GET` | `/patients/me` | ✅ | Fetch own patient profile |
| `PUT` | `/patients/me` | ✅ | Update patient profile |
| `POST` | `/predict/glaucoma` | ⚠️ Optional | Glaucoma screening (single image) |
| `POST` | `/predict/dr` | ⚠️ Optional | DR screening (single image) |
| `POST` | `/predict/both` | ⚠️ Optional | Combined Glaucoma + DR screening |
| `POST` | `/predict/glaucoma/batch` | ⚠️ Optional | Glaucoma batch (up to 10 images) |
| `POST` | `/predict/dr/batch` | ⚠️ Optional | DR batch (up to 10 images) |
| `POST` | `/records` | ✅ | Manually save a screening record |
| `GET` | `/records` | ✅ | List screening history (paginated) |
| `GET` | `/records/{record_id}` | ✅ | Get single screening record |
| `DELETE` | `/records/{record_id}` | ✅ | Delete a screening record |

> ⚠️ **Optional Auth**: Predictions work without a token but results are not saved. With a valid Bearer token, results are auto-saved to Firestore.

---

## 🚀 12. Deployment Notes

### Environment Variables Required on Server
```
FIREBASE_WEB_API_KEY=AIzaSy...
FIREBASE_PROJECT_ID=your-project-id
```

### Model Files Required
Place these in the `models/` directory at the project root:
```
models/
├── best.pt            ← YOLOv8 optic disc weights
├── ckan_se_final.pth  ← CKAN-SE glaucoma classifier
└── Retino_model.pt    ← CNN_Retino DR classifier
```

### Starting the Server (Uvicorn)
```bash
# From the eye_api/ directory:
uvicorn main:app --host 0.0.0.0 --port 8000

# With auto-reload for development:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# With multiple workers for production:
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "eye_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
