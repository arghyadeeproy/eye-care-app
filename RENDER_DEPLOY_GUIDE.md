# Deploying to Render — Eye Disease Detection API

A step-by-step guide to deploying the FastAPI backend to **[Render](https://render.com)**, a modern cloud platform with a generous free tier.

---

## 📋 Prerequisites

Before starting, make sure you have:
- A **[Render account](https://dashboard.render.com/register)** (free)
- Your project pushed to a **GitHub repository**
- Your **Firebase project** already configured (serviceAccountKey.json downloaded)

---

## 🔑 Step 1 — Prepare the Firebase Service Account for the Cloud

> [!IMPORTANT]
> Never commit `serviceAccountKey.json` to your git repository. On Render, you pass it as an environment variable instead.

1. Open your `eye_api/serviceAccountKey.json` in a text editor.
2. Copy the **entire file contents** (the full JSON object).
3. You will paste this as the value for the `FIREBASE_SERVICE_ACCOUNT_JSON` environment variable in Render (Step 4 below).

---

## 🚀 Step 2 — Push Your Project to GitHub

```bash
cd "d:\Final Year Project Application"
git init
git add .
git commit -m "Initial backend deployment"
git remote add origin https://github.com/YOUR_USERNAME/eye-disease-api.git
git push -u origin main
```

> [!WARNING]
> Make sure `eye_api/serviceAccountKey.json` is listed in your `.gitignore` before pushing:
> ```
> # .gitignore
> eye_api/serviceAccountKey.json
> eye_api/.env
> venv/
> __pycache__/
> *.pyc
> ```

---

## ⚙️ Step 3 — Create a New Web Service on Render

1. Go to your [Render Dashboard](https://dashboard.render.com).
2. Click **+ New** → **Web Service**.
3. Connect your **GitHub account** and select your repository.
4. Configure the service:

| Setting | Value |
|---|---|
| **Name** | `eye-disease-api` |
| **Region** | Your preferred region (e.g. Frankfurt for India latency) |
| **Branch** | `main` |
| **Root Directory** | *(leave blank — it will use the project root)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn eye_api.main:app --host 0.0.0.0 --port $PORT` |
| **Plan** | Free (for testing) or Starter ($7/mo for always-on) |

5. Click **Create Web Service**.

---

## 🔐 Step 4 — Configure Environment Variables

In the Render service dashboard, go to **Environment** → **Add Environment Variable** and set the following:

| Key | Value |
|---|---|
| `FIREBASE_WEB_API_KEY` | `AIzaSyBEiLbRAexvwj9rF1IQKTS5zIvLppHnGQo` |
| `FIREBASE_PROJECT_ID` | `eye-disease-detection-dfc71` |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | *(Paste the full JSON from serviceAccountKey.json as a single-line string)* |
| `ALLOWED_ORIGINS` | `*` *(or your frontend URL e.g. `https://my-frontend.vercel.app`)* |

> [!TIP]
> To convert your `serviceAccountKey.json` to a single-line string for pasting, run this in PowerShell:
> ```powershell
> (Get-Content "eye_api\serviceAccountKey.json" -Raw) -replace "`r`n", "" -replace "`n", ""
> ```
> Then copy and paste the output as the value of `FIREBASE_SERVICE_ACCOUNT_JSON`.

---

## 🩺 Step 5 — Set the Health Check Path

1. In the Render service settings, scroll to **Health & Alerts**.
2. Set **Health Check Path** to: `/health`

This allows Render to verify the service is running by calling `GET /health` and checking for a `200 OK` response.

---

## ✅ Step 6 — Verify Deployment

Once deployment completes (typically 5–10 minutes on first deploy due to ML library installation):

1. Your API URL will be: `https://eye-disease-api.onrender.com`
2. Visit the interactive docs at: `https://eye-disease-api.onrender.com/docs`
3. Test the health endpoint:
```bash
curl https://eye-disease-api.onrender.com/health
```
Expected response:
```json
{ "status": "healthy", "models": { "yolo": true, "ckan": true, "dr": true }, "firebase": true }
```

---

## 🌐 Step 7 — Configure CORS for Your Frontend

Once you know your deployed frontend URL (e.g. on Vercel or Netlify), update the `ALLOWED_ORIGINS` environment variable in Render:

**Example (multiple origins):**
```
https://my-eye-app.vercel.app,https://myclinic.com
```

After saving the env var, **Render will automatically redeploy** the service with the new CORS settings.

---

## 📁 Using render.yaml (Blueprint — Optional)

The project includes a `render.yaml` file which allows you to deploy via a **Render Blueprint** (one-click infrastructure-as-code):

1. In your Render dashboard, click **+ New** → **Blueprint**.
2. Connect the GitHub repo.
3. Render will detect `render.yaml` and auto-configure the service.
4. You still need to manually fill in the secret env vars (`FIREBASE_WEB_API_KEY`, `FIREBASE_PROJECT_ID`, `FIREBASE_SERVICE_ACCOUNT_JSON`) in the dashboard after creation.

---

## ⚠️ Important Notes & Troubleshooting

| Issue | Solution |
|---|---|
| **Free tier spins down** after 15 mins of inactivity | Upgrade to Starter ($7/mo) for always-on, or use UptimeRobot to ping `/health` every 10 mins |
| **Build times out** | ML libraries (torch, ultralytics) are large (~2GB). Render's free build timeout is 15 mins. Upgrade if needed. |
| **Model files not found** | Render's free tier has ephemeral disk. Upload model files to a persistent storage (e.g. Render Disk, S3, or Google Drive) and download them at startup |
| **Firebase 500 error** | Make sure `FIREBASE_SERVICE_ACCOUNT_JSON` is the full, valid JSON string |
| **CORS error from frontend** | Add your exact frontend URL (with `https://`, no trailing slash) to `ALLOWED_ORIGINS` |
| **`torch` out of memory`** | Use a Render plan with more RAM (Standard = 2GB, Pro = 4GB) |

---

## 🗄️ Serving Model Files on Render

> [!CAUTION]
> Render's **Free and Starter** plans use **ephemeral disks** — files written at runtime are lost on restart. Your model `.pt` files must be either:
> - Committed to the git repo (only feasible if they are < 100MB; GitHub's limit is 100MB per file)
> - **Recommended**: Stored on **Render Disk** (a persistent mounted volume, $0.25/GB/month)

### Option A: Add a Render Disk (Recommended)

Add this to your `render.yaml`:
```yaml
  disk:
    name: model-storage
    mountPath: /opt/models
    sizeGB: 5
```
Then update your `eye_api/main.py` to look for models in `/opt/models` when running on Render.

### Option B: Download from URL at Startup

Add a `startup.sh` script that downloads models from a public URL (e.g. Google Drive or Hugging Face Hub) before starting the server:

```bash
#!/bin/bash
# startup.sh
mkdir -p models
# Download model weights if not present
[ ! -f models/best.pt ]           && wget -q -O models/best.pt           "$YOLO_MODEL_URL"
[ ! -f models/ckan_se_final.pth ] && wget -q -O models/ckan_se_final.pth "$CKAN_MODEL_URL"
[ ! -f models/Retino_model.pt ]   && wget -q -O models/Retino_model.pt   "$RETINO_MODEL_URL"
# Start the API
uvicorn eye_api.main:app --host 0.0.0.0 --port $PORT
```

Then change your **Start Command** in Render to: `bash startup.sh`

Set `YOLO_MODEL_URL`, `CKAN_MODEL_URL`, and `RETINO_MODEL_URL` as environment variables pointing to your model file download links.
