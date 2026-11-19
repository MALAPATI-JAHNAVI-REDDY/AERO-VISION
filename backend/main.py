
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_TITLE, API_VERSION
from backend.routers import health, detections, live, video, mode
from backend.services.camera_service import camera_service


# Camera configuration
CAMERA_ENABLED = (
    os.getenv("AEROVISION_START_CAMERA", "true").lower()
    in ("1", "true", "yes")
)


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):

    if CAMERA_ENABLED:
        camera_service.start()
        print("[SYSTEM] AeroVision started in Live Monitoring mode")
    else:
        print("[SYSTEM] Camera startup disabled; API running in cloud mode")

    yield

    if CAMERA_ENABLED:
        camera_service.stop()
        print("[SYSTEM] AeroVision camera service stopped")


# FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://aero-vision-one.vercel.app",
    ],

    # Allow Vercel preview deployments
    allow_origin_regex=r"https://.*\.vercel\.app",

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API routers
app.include_router(health.router)
app.include_router(detections.router)
app.include_router(live.router)
app.include_router(video.router)
app.include_router(mode.router)


# Root endpoint
@app.get("/")
def root():
    return {
        "service": "AeroVision API",
        "status": "online"
    }
