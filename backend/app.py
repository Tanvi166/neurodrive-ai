

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

try:
    from model_utils import WebDrowsinessDetector, decode_base64_image, decode_uploaded_file
except ImportError:
    from .model_utils import WebDrowsinessDetector, decode_base64_image, decode_uploaded_file


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIST_DIR = PROJECT_DIR / "frontend" / "dist"
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "yolov8n.pt")).resolve()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

detector: WebDrowsinessDetector | None = None


@asynccontextmanager
async def lifespan(api: FastAPI):
    global detector
    if not MODEL_PATH.exists():
        raise RuntimeError(f"YOLO model file not found: {MODEL_PATH}")
    detector = WebDrowsinessDetector(MODEL_PATH)
    yield


api = FastAPI(
    title="Cognitive Fatigue & Driver Attention API",
    version="1.0.0",
    lifespan=lifespan,
)
app = api

api.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIST_DIR / "assets").exists():
    api.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")




# NEW CODE (FASTAPI REST API)
def process_frame(frame):
    if detector is None:
        raise RuntimeError("Detector is not initialized.")
    return detector.process_frame(frame)


def get_detector() -> WebDrowsinessDetector:
    if detector is None:
        raise RuntimeError("Detector is not initialized.")
    return detector


@api.get("/health")
async def health():
    return {
        "ok": True,
        "service": "cognitive-fatigue-driver-attention-api",
        "model_loaded": detector is not None,
    }


@api.get("/session-stats")
async def session_stats():
    try:
        return get_detector().get_session_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.post("/reset-session")
async def reset_session():
    try:
        return get_detector().reset_session()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.post("/analyze-frame")
async def analyze_frame(
    request: Request,
    image_base64: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    content_type = request.headers.get("content-type", "")
    if image_base64 is None and "application/json" in content_type:
        body = await request.json()
        image_base64 = body.get("image_base64")

    if image_base64 is None and file is None:
        raise HTTPException(status_code=400, detail="Send image_base64 JSON/form data or file.")

    try:
        if image_base64 is not None:
            frame = decode_base64_image(image_base64)
        else:
            frame = decode_uploaded_file(await file.read())

        return await run_in_threadpool(process_frame, frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api.post("/predict")
async def predict_compat(
    request: Request,
    image_base64: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    return await analyze_frame(request=request, image_base64=image_base64, file=file)


@api.get("/")
async def root():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return {
        "service": "Cognitive Fatigue & Driver Attention API",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze-frame",
    }
