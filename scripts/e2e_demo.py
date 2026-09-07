"""End-to-end demo: play an audio clip through the live backend and watch the risk verdict.

    # terminal 1  — backend with the webhook pointed at this script's catcher
    VG_WEBHOOK__ENABLED=true VG_WEBHOOK__URL=http://localhost:9099/hook uvicorn backend.main:app

    # terminal 2
    python scripts/e2e_demo.py --wav path/to/deepfake.wav

Streams the clip to /ws/stream exactly like the receiver tab, prints the per-window verdict
timeline, and runs a tiny HTTP server on :9099 to catch the HIGH webhook. Exits 0 if a HIGH
alert (and webhook) fired; useful as the P7 acceptance check once a trained checkpoint is in
backend/models/.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_webhooks: list = []


class _Hook(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _webhooks.append(json.loads(self.rfile.read(n) or b"{}"))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_):
        pass


def _frames(wav_path: Path, input_sr: int, frame_ms: float) -> list[bytes]:
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    if sr != input_sr:
        n = int(round(len(wav) / sr * input_sr))
        wav = np.interp(np.linspace(0, 1, n, endpoint=False),
                        np.linspace(0, 1, len(wav), endpoint=False), wav).astype("float32")
    i16 = np.clip(wav, -1, 1)
    i16 = np.where(i16 < 0, i16 * 0x8000, i16 * 0x7FFF).astype("<i2")
    step = max(1, int(input_sr * frame_ms / 1000))
    return [i16[i:i + step].tobytes() for i in range(0, len(i16), step)]


async def run(url: str, wav: Path, input_sr: int, frame_ms: float, realtime: bool) -> int:
    import websockets

    frames = _frames(wav, input_sr, frame_ms)
    print(f"{wav.name}: {len(frames)} frames @ {input_sr} Hz -> {url}")
    high = alert = False
    peak = 0.0
    async with websockets.connect(url, max_size=None) as ws:
        print("server:", json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "config", "input_sample_rate": input_sr}))

        async def rx():
            nonlocal high, alert, peak
            with contextlib.suppress(Exception):
                async for raw in ws:
                    e = json.loads(raw)
                    if e.get("type") != "score":
                        continue
                    r = e["risk"]
                    peak = max(peak, r["score"])
                    high |= r["level"] == "HIGH"
                    alert |= e["alert"]
                    print(f"#{e['index']:>3} chunk={e['fake_prob']:.3f} roll={r['score']:.3f} "
                          f"{r['level']:<6} {'ALERT' if e['alert'] else ''}")

        task = asyncio.create_task(rx())
        with contextlib.suppress(Exception):
            for f in frames:
                await ws.send(f)
                if realtime:
                    await asyncio.sleep(frame_ms / 1000)
            await ws.send(json.dumps({"type": "end"}))
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=20)

    await asyncio.sleep(0.5)  # let a webhook land
    print(f"\npeak rolling score: {peak:.3f}   HIGH: {high}   alert event: {alert}   "
          f"webhooks received: {len(_webhooks)}")
    if _webhooks:
        print("webhook payload:", json.dumps(_webhooks[-1], indent=2)[:400])
    return 0 if (alert and _webhooks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", type=Path, default=ROOT / "tests" / "fixtures" / "sample_spoof.wav")
    ap.add_argument("--url", default="ws://localhost:8000/ws/stream")
    ap.add_argument("--input-sr", type=int, default=48000)
    ap.add_argument("--frame-ms", type=float, default=85.0)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--hook-port", type=int, default=9099)
    args = ap.parse_args()
    if not args.wav.exists():
        print(f"no such wav: {args.wav}")
        return 2

    srv = HTTPServer(("localhost", args.hook_port), _Hook)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"webhook catcher on :{args.hook_port}/hook")
    try:
        return asyncio.run(run(args.url, args.wav, args.input_sr, args.frame_ms, args.realtime))
    finally:
        srv.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
