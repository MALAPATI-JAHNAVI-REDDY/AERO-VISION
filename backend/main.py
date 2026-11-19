
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_TITLE, API_VERSION
from backend.routers import health, detections, live, video, mode
from backend.services.camera_service import camera_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Default operating mode is Live Monitoring.
    camera_service.start()
    print("[SYSTEM] AeroVision started in Live Monitoring mode")

    yield

    camera_service.stop()
    print("[SYSTEM] AeroVision camera service stopped")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router)
app.include_router(detections.router)
app.include_router(live.router)
app.include_router(video.router)
app.include_router(mode.router)


@app.get("/")
def root():
    return {
        "service": "AeroVision API",
        "status": "online"
    }
