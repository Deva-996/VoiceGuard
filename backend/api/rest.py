"""REST endpoints.  [Day 5]

  GET  /health            liveness + model/backend info
  GET  /config            effective runtime config (thresholds etc. for the dashboard)
  POST /score             score a single uploaded audio clip (multipart 'file') -> fake_prob

The streaming path is in ``websocket.py``.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from backend.core import get_config

router = APIRouter()


@router.get("/health")
def health() -> dict:
    cfg = get_config()
    return {
        "status": "ok",
        "feature_extractor": cfg.feature_extractor.backend,
        "device": cfg.classifier.device,
        "sample_rate": cfg.audio.sample_rate,
    }


@router.get("/config")
def config() -> dict:
    cfg = get_config()
    return {
        "audio": {
            "sample_rate": cfg.audio.sample_rate,
            "chunk_seconds": cfg.audio.chunk_seconds,
            "hop_seconds": cfg.audio.hop_seconds,
            "pcm_format": cfg.audio.pcm_format,
        },
        "risk": {
            "window": cfg.risk.window,
            "low": cfg.risk.low_threshold,
            "medium": cfg.risk.medium_threshold,
            "high": cfg.risk.high_threshold,
            "high_consecutive": cfg.risk.high_consecutive,
        },
    }


@router.post("/score")
async def score(file: UploadFile = File(...)) -> JSONResponse:
    from backend.main import get_pipeline  # lazy: pipeline is app-scoped

    raw = await file.read()
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from backend.audio import resample_to_16k

        wav = resample_to_16k(np.asarray(wav), sr)

    result = get_pipeline().infer_chunk(torch.from_numpy(np.asarray(wav, dtype="float32")))
    return JSONResponse(
        {
            "fake_prob": round(result.fake_prob, 4),
            "n_samples": result.n_samples,
            "n_frames": result.n_frames,
            "latency_ms": round(result.latency_ms, 1),
        }
    )
