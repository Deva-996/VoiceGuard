"""WebSocket streaming endpoint.  [Day 2 -> Day 5 full loop, Day 3 real-time hardening]

  WS /ws/stream

Client (receiver browser tab) sends:
  * binary frames  -> raw PCM (int16 or float32 LE, mono, sample_rate from /config)
  * text  {"type":"config", "input_sample_rate": 48000}      (optional, first message)
  * text  {"type":"end"}                                     (graceful close)

Server sends, per analysis window it actually scores:
  {"type":"score", "index":N, "fake_prob":0.xx, "risk":{...}, "latency_ms":..,
   "dropped":K, "alert":false}
  ``dropped`` = analysis windows skipped since the last score because inference could not
  keep up with the incoming audio (see below).

Real-time strategy
------------------
SSL inference on CPU (~0.4-1.0 s per 1 s chunk) can be slower than audio arrives. To keep
latency bounded instead of growing without limit:
  * a receiver task drains the socket continuously and feeds the chunker;
  * only the *most recent* pending analysis window is kept — older ones are dropped and
    counted;
  * inference runs in a worker thread (``asyncio.to_thread``) so receiving never blocks.
Under load the risk score simply updates less often; it never falls behind real time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.audio import AudioChunker
from backend.core import get_config

router = APIRouter()


@router.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    cfg = get_config()
    session_id = uuid.uuid4().hex[:12]

    # lazy import so Day 1/2 work before the pipeline is ready
    from backend.main import get_pipeline, get_webhook
    from backend.scoring import RiskEngine

    pipeline = get_pipeline()
    risk = RiskEngine.from_config(cfg.risk)
    webhook = get_webhook()

    chunker = AudioChunker(
        sample_rate=cfg.audio.sample_rate,
        chunk_seconds=cfg.audio.chunk_seconds,
        hop_seconds=cfg.audio.hop_seconds,
        input_sample_rate=cfg.audio.sample_rate,
        pcm_format=cfg.audio.pcm_format,
    )

    # shared state between the receiver task and the inference loop
    pending: dict = {"window": None, "dropped": 0}
    got_window = asyncio.Event()
    stop = asyncio.Event()

    async def receiver() -> None:
        try:
            while not stop.is_set():
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if (text := msg.get("text")) is not None:
                    data = json.loads(text)
                    if data.get("type") == "config" and data.get("input_sample_rate"):
                        chunker.input_sample_rate = int(data["input_sample_rate"])
                    elif data.get("type") == "end":
                        break
                    continue

                pcm = msg.get("bytes")
                if not pcm:
                    continue

                for window in chunker.push(pcm):
                    if pending["window"] is not None:
                        pending["dropped"] += 1  # inference is behind; skip the older window
                    pending["window"] = window
                    got_window.set()
        finally:
            stop.set()
            got_window.set()  # wake the inference loop so it can exit

    await ws.send_json({"type": "ready", "session_id": session_id})
    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            await got_window.wait()
            got_window.clear()

            window = pending["window"]
            if window is None:
                if stop.is_set():
                    break
                continue
            pending["window"] = None
            dropped = pending["dropped"]
            pending["dropped"] = 0

            result = await asyncio.to_thread(pipeline.infer_chunk, window)
            state = risk.update(result.fake_prob)

            await ws.send_json(
                {
                    "type": "score",
                    "index": result.index,
                    "fake_prob": round(result.fake_prob, 4),
                    "latency_ms": round(result.latency_ms, 1),
                    "dropped": dropped,
                    "risk": {
                        "score": state.score,
                        "level": state.level.value,
                        "consecutive_high": state.consecutive_high,
                    },
                    "alert": state.alert,
                }
            )
            if state.alert:
                from backend.alerts import AlertPayload

                await webhook.fire(
                    AlertPayload(
                        session_id=session_id,
                        score=state.score,
                        raw_prob=state.raw_prob,
                        consecutive_high=state.consecutive_high,
                        n_chunks=state.n_chunks,
                    )
                )
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await recv_task
        with contextlib.suppress(RuntimeError):
            await ws.close()
