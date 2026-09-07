"""VoiceGuard FastAPI application.  [Day 5]

    uvicorn backend.main:app --reload

App-scoped singletons (pipeline, webhook dispatcher) are built once on startup so model
weights load a single time. Per-connection state (RiskEngine, AudioChunker) is created in the
WS handler.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core import get_config

_pipeline = None
_webhook = None

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend"


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from backend.inference import DetectionPipeline

        _pipeline = DetectionPipeline.from_config(get_config())
    return _pipeline


def get_webhook():
    global _webhook
    if _webhook is None:
        from backend.alerts import WebhookDispatcher

        _webhook = WebhookDispatcher.from_config(get_config().webhook)
    return _webhook


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # warm the model
    get_webhook()
    yield


app = FastAPI(title="VoiceGuard", version="0.1.0", lifespan=lifespan)

_cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.server.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api import rest_router, ws_router  # noqa: E402

app.include_router(rest_router)
app.include_router(ws_router)

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
