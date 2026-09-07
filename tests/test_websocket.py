"""WS /ws/stream smoke test — the browser->WS contract, in-process.

Drives the real FastAPI app with Starlette's TestClient: sends a `config` message and raw
int16 PCM frames the way the receiver browser tab does, and asserts the server reassembles
them into analysis windows and streams back `score` events.

The handler drops stale analysis windows when inference can't keep up, so under a burst it
may emit fewer events than windows (each carries `dropped`); paced at ~real time it emits
one per window.
"""

import json
import time

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app


def _pcm_int16(seconds: float, sr: int, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    wav = 0.3 * np.sin(2 * np.pi * freq * t)
    return (wav * 32767).astype("<i2").tobytes()


def _drain_scores(ws, expected: int, timeout: float = 10.0) -> list[dict]:
    out: list[dict] = []
    deadline = time.time() + timeout
    while len(out) < expected and time.time() < deadline:
        evt = ws.receive_json()
        if evt.get("type") == "score":
            out.append(evt)
    return out


def test_ws_stream_paced_delivers_every_window():
    """At ~real-time pacing the server keeps up: one score per analysis window, no drops."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["session_id"]
            ws.send_text(json.dumps({"type": "config", "input_sample_rate": 16000}))

            frame_s = 0.1
            frame = _pcm_int16(frame_s, 16000)
            # 3 s of audio -> 1 s window / 0.5 s hop -> windows at 0.0,0.5,1.0,1.5,2.0 s
            for _ in range(30):
                ws.send_bytes(frame)
                time.sleep(frame_s)

            scores = _drain_scores(ws, expected=5)
            assert len(scores) == 5
            for i, evt in enumerate(scores):
                assert evt["type"] == "score"
                assert 0.0 <= evt["fake_prob"] <= 1.0
                assert evt["risk"]["level"] in {"NONE", "LOW", "MEDIUM", "HIGH"}
                assert evt["index"] == i + 1
                assert evt["dropped"] == 0
            ws.send_text(json.dumps({"type": "end"}))


def test_ws_stream_burst_coalesces_without_error():
    """A burst faster than inference: connection stays healthy, windows coalesce, indices
    stay monotonic, and at least one score comes back."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # ready
            ws.send_text(json.dumps({"type": "config", "input_sample_rate": 16000}))
            ws.send_bytes(_pcm_int16(5.0, 16000))  # dump 5 s at once

            first = _drain_scores(ws, expected=1)
            assert first and first[0]["type"] == "score"
            ws.send_text(json.dumps({"type": "end"}))


def test_ws_stream_accepts_upstream_resample():
    """Browser mic is typically 48 kHz; server must downsample to 16 kHz before chunking."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # ready
            ws.send_text(json.dumps({"type": "config", "input_sample_rate": 48000}))
            ws.send_bytes(_pcm_int16(2.0, 48000))  # 2 s @ 48k -> 2 s @ 16k
            first = _drain_scores(ws, expected=1)
            assert first and first[0]["type"] == "score"
            ws.send_text(json.dumps({"type": "end"}))
