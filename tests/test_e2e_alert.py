"""End-to-end alerting path: PCM in -> score events -> HIGH latch -> webhook fired.

Deterministic (no model): the detection pipeline is stubbed to return a rising fake_prob so
the risk engine crosses HIGH; asserts the score stream, the `alert` event, and that the
webhook dispatcher was invoked with a sane payload.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient


class _RisingPipeline:
    """fake_prob climbs 0.2 -> 0.95 over the first ~8 chunks, then stays high."""

    def __init__(self):
        self._i = 0

    def infer_chunk(self, waveform):
        from backend.inference.pipeline import ChunkResult

        self._i += 1
        prob = min(0.2 + 0.1 * self._i, 0.95)
        return ChunkResult(fake_prob=prob, n_samples=int(waveform.numel()), n_frames=50,
                           latency_ms=1.0, index=self._i)


class _RecordingWebhook:
    def __init__(self):
        self.calls = []

    async def fire(self, payload):
        self.calls.append(payload)
        return True


@pytest.fixture
def app_with_stubs(monkeypatch):
    import backend.main as m

    pipe, hook = _RisingPipeline(), _RecordingWebhook()
    monkeypatch.setattr(m, "get_pipeline", lambda: pipe)
    monkeypatch.setattr(m, "get_webhook", lambda: hook)
    return m.app, hook


def _pcm(seconds, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2").tobytes()


def test_sustained_spoof_raises_high_and_fires_webhook(app_with_stubs):
    app, hook = app_with_stubs
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_text(json.dumps({"type": "config", "input_sample_rate": 16000}))

            frame = _pcm(0.1)
            saw_high = saw_alert = False
            levels = []
            for _ in range(140):  # ~14 s of audio
                ws.send_bytes(frame)
            for _ in range(20):
                evt = ws.receive_json()
                if evt.get("type") != "score":
                    continue
                levels.append(evt["risk"]["level"])
                saw_high |= evt["risk"]["level"] == "HIGH"
                saw_alert |= evt["alert"]
                if saw_alert:
                    break
            ws.send_text(json.dumps({"type": "end"}))

    assert saw_high, f"never reached HIGH; levels={levels}"
    assert saw_alert, "HIGH reached but no alert event"
    assert len(hook.calls) == 1
    p = hook.calls[0]
    assert p.score > 0.75 and p.consecutive_high >= 3 and p.session_id
