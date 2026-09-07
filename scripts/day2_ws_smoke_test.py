"""Day 2 deliverable: browser -> WS smoke test.

Streams a wav file to a running backend over WS /ws/stream exactly the way the receiver
browser tab will (int16 mono PCM frames + a `config` message), and prints the `score`
events that come back.

    # terminal 1
    uvicorn backend.main:app

    # terminal 2
    python scripts/day2_ws_smoke_test.py
    python scripts/day2_ws_smoke_test.py --wav path/to/clip.wav --input-sr 48000

Needs the `websockets` package (already in requirements.txt). Exit code 0 if at least one
score event was received.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_WAV = ROOT / "tests" / "fixtures" / "sample_spoof.wav"


def _load_frames(wav_path: Path, input_sr: int, frame_ms: float) -> list[bytes]:
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    # simulate the browser capturing at `input_sr`: resample 16k fixture up to input_sr
    if sr != input_sr:
        # linear resample up to the simulated capture rate (fine for a smoke test)
        duration = len(wav) / sr
        n_out = int(round(duration * input_sr))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
    i16 = np.clip(wav, -1.0, 1.0)
    i16 = np.where(i16 < 0, i16 * 0x8000, i16 * 0x7FFF).astype("<i2")
    step = max(1, int(input_sr * frame_ms / 1000.0))
    return [i16[i : i + step].tobytes() for i in range(0, len(i16), step)]


async def run(url: str, wav_path: Path, input_sr: int, frame_ms: float, realtime: bool) -> int:
    import websockets

    frames = _load_frames(wav_path, input_sr, frame_ms)
    print(f"connecting to {url}")
    print(f"streaming {wav_path.name}: {len(frames)} frames @ {input_sr} Hz, ~{frame_ms:.0f} ms each")

    try:
        ws = await websockets.connect(url, max_size=None)
    except (OSError, websockets.InvalidURI, websockets.InvalidHandshake) as exc:
        print(f"could not connect to {url}: {exc or type(exc).__name__}")
        print("is the backend running?  ->  uvicorn backend.main:app")
        return 2

    # connected — from here on a dropped connection just ends the run, it is not an error
    n_scores = 0

    async def receiver() -> None:
        nonlocal n_scores
        with contextlib.suppress(websockets.ConnectionClosed, OSError):
            async for raw in ws:
                evt = json.loads(raw)
                if evt.get("type") != "score":
                    continue
                n_scores += 1
                r = evt["risk"]
                flag = "  <-- ALERT" if evt.get("alert") else ""
                drop = f"  drop={evt['dropped']}" if evt.get("dropped") else ""
                print(
                    f"#{evt['index']:>3}  chunk={evt['fake_prob']:.3f}  "
                    f"roll={r['score']:.3f}  {r['level']:<6}  "
                    f"cons_high={r['consecutive_high']}  {evt['latency_ms']:.0f} ms{drop}{flag}"
                )

    recv_task = asyncio.create_task(receiver())
    with contextlib.suppress(websockets.ConnectionClosed, OSError):
        await ws.send(json.dumps({"type": "config", "input_sample_rate": input_sr}))
        for frame in frames:
            await ws.send(frame)
            if realtime:
                await asyncio.sleep(frame_ms / 1000.0)
        await ws.send(json.dumps({"type": "end"}))

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recv_task, timeout=15)
    recv_task.cancel()
    with contextlib.suppress(Exception):
        await ws.close()

    print(f"\nreceived {n_scores} score event(s)")
    return 0 if n_scores > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="ws://localhost:8000/ws/stream")
    ap.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    ap.add_argument("--input-sr", type=int, default=48000, help="simulated browser capture rate")
    ap.add_argument("--frame-ms", type=float, default=85.0, help="PCM frame size sent per message")
    ap.add_argument("--realtime", action="store_true", help="pace frames at wall-clock speed")
    args = ap.parse_args()

    if not args.wav.exists():
        print(f"wav not found: {args.wav}  (run scripts/make_sample_audio.py first)")
        return 2
    return asyncio.run(run(args.url, args.wav, args.input_sr, args.frame_ms, args.realtime))


if __name__ == "__main__":
    raise SystemExit(main())
