# Eye Disease Detection API — Deployable Backend Server

A high-performance FastAPI backend server providing deep learning eye screening endpoints for **Glaucoma** and **Diabetic Retinopathy** from fundus retinal images. It integrates **Firebase Authentication** and **Firestore** for secure patient profile and screening records management.

---

## 🏗️ Project Structure

```
Final Year Project Application/
├── models/
│   ├── best.pt              ← YOLOv8 optic disc detector weights
│   ├── ckan_se_final.pth    ← CKAN-SE classifier checkpoint (Glaucoma)
│   └── Retino_model.pt      ← CNN_Retino model (Diabetic Retinopathy)
│
├── eye_api/
│   ├── main.py              ← FastAPI main application
│   ├── model_architecture.py← CKAN-SE & CNN_Retino model class definitions
│   ├── firebase_config.py   ← Firebase SDK client initializer
│   ├── auth_middleware.py   ← Bearer ID token auth dependencies
│   ├── routers/
│   │   ├── auth.py          ← Login, Registration, Token Refresh (/auth)
│   │   ├── patients.py      ← Patient profile CRUD (/patients)
│   │   └── records.py       ← Screening records management (/records)
│   ├── .env.example         ← Template for configuration variables
│   └── requirements.txt     ← API-specific dependency list
│
├── requirements.txt         ← Deployable root requirements (install this one)
└── start_api.bat            ← Launch local script (port 8000)
```

---

## ⚙️ Deployment & Configuration

### 1. Firebase Credentials Setup
To deploy this backend, you must configure a Firebase project:
1. Create a Firebase project in the [Firebase Console](https://console.firebase.google.com/).
2. Enable **Email/Password** provider in the **Authentication** tab.
3. Enable **Cloud Firestore** in your project database settings.
4. Go to **Project Settings** → **Service Accounts** → Click **Generate new private key** and download the credentials JSON.
5. Save this credentials file as **`eye_api/serviceAccountKey.json`**.

### 2. Configure Environment Variables
Copy `eye_api/.env.example` to create `eye_api/.env` and supply:
* `FIREBASE_WEB_API_KEY`: Found in Firebase Console under Project Settings → General (Web API Key). Used by backend to make REST sign-in/up calls.
* `FIREBASE_PROJECT_ID`: Your Firebase project identifier.

---

## 🚀 Running Locally

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install deployment requirements
pip install -r requirements.txt

# 3. Start the server
cd eye_api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 API Documentation & Endpoints

Once the server is running, navigate to:
* **Interactive Swagger UI Docs**: `http://localhost:8000/docs`
* **Alternative Redoc Documentation**: `http://localhost:8000/redoc`

### Endpoints Overview

#### 🔒 User Authentication (`/auth`)
* `POST /auth/signup` - Register a new user using email, password, and full name.
* `POST /auth/login` - Sign in using email and password; returns Firebase ID token and refresh token.
* `POST /auth/refresh` - Exchange a refresh token for a new ID token.
* `GET /auth/me` - Retrieve current user profile metadata.

#### 👤 Patient Health Profiles (`/patients`)
* `POST /patients` - Create patient profile (requires Bearer token).
* `GET /patients/me` - Fetch patient profile (requires Bearer token).
* `PUT /patients/me` - Edit patient profile (requires Bearer token).

#### 📋 Screening Records (`/records`)
* `POST /records` - Add a screening test result to the history records.
* `GET /records` - Fetch patient screening history logs.
* `GET /records/{record_id}` - Get detailed view of a single screening record.
* `DELETE /records/{record_id}` - Delete a record.

#### 🔬 AI Inference & Predictions (`/predict`)
All prediction requests accept standard multipart files (`file`). Authentication headers (`Authorization: Bearer <ID_TOKEN>`) are optional. If authenticated, the diagnostic output will be **automatically saved** to the patient's Firestore record.
* `POST /predict/glaucoma` - Runs YOLOv8 Optic Disc localization followed by CKAN-SE classification.
* `POST /predict/dr` - Runs CNN_Retino classification over the input fundus image.
* `POST /predict/both` - Executes both Glaucoma & Diabetic Retinopathy models in parallel, return combined results.
* `POST /predict/glaucoma/batch` - Run batch screening for up to 10 images at once.
* `POST /predict/dr/batch` - Run batch DR screening for up to 10 images at once.
