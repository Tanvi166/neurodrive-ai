# NeuroDrive AI

Production-style full stack web application for cognitive fatigue and driver attention monitoring.

The system uses a browser webcam feed, sends compressed frames to a FastAPI backend, runs OpenCV, MediaPipe, YOLOv8, and face recognition inference, then returns live driver attention analytics to a React dashboard.

## Project Structure

```text
backend/
  app.py
  model_utils.py
  requirements.txt
  Dockerfile
frontend/
  src/
  package.json
  vite.config.js
  tailwind.config.js
yolov8n.pt
render.yaml
```

## Backend API

`GET /health`

Returns service health and model loading status.

`POST /analyze-frame`

Accepts JSON:

```json
{
  "image_base64": "data:image/jpeg;base64,..."
}
```

Returns:

```json
{
  "status": "FOCUSED",
  "mode": "FOCUSED MODE",
  "fatigue_score": 72,
  "attention_score": 88,
  "phone_detected": false,
  "driver_found": true,
  "ear": 0.31,
  "baseline_ear": 0.34,
  "calibrated": true,
  "alert": null
}
```

`GET /session-stats`

Returns session duration, processed frame count, event ratios, calibration state, and recent prediction history.

## Local Development

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Vercel Deployment

1. Create a Vercel project from the repository.
2. Set the project root to `frontend`.
3. Add environment variable:

```text
VITE_API_BASE_URL=https://your-backend.onrender.com
```

4. Build command:

```text
npm run build
```

5. Output directory:

```text
dist
```

## Render Deployment

Use the included `render.yaml`, or create a Docker web service manually.

Manual setup:

1. Create a new Render Web Service.
2. Environment: Docker.
3. Dockerfile path: `backend/Dockerfile`.
4. Health check path: `/health`.
5. Add environment variables:

```text
CORS_ORIGINS=https://your-vercel-app.vercel.app
MODEL_PATH=/app/yolov8n.pt
```

The Dockerfile installs the native build dependencies needed by `face_recognition` and `dlib`.

## Railway Deployment

Create a Docker service using `backend/Dockerfile`, expose the generated `$PORT`, and set:

```text
MODEL_PATH=/app/yolov8n.pt
CORS_ORIGINS=https://your-frontend-domain
```

## Notes

The backend stores driver enrollment and session analytics in memory. For multi-user production use, split session state by user/session id and store long-running analytics in Redis or a database.
