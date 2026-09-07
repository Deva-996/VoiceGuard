# VoiceGuard architecture

## Data flow (streaming)

```
caller mic ──WebRTC──▶ receiver tab ──WS binary PCM──▶ FastAPI /ws/stream
                            ▲                               │
                            │                               ▼
                    dashboard events ◀──WS JSON────  AudioChunker (1s / 0.5s hop)
                                                            │
                                                            ▼
                                              DetectionPipeline.infer_chunk
                                          IndicWav2Vec ─▶ AASIST ─▶ fake_prob
                                                            │
                                                            ▼
                                                RiskEngine.update(fake_prob)
                                        rolling weighted avg (10) + LOW/MED/HIGH
                                                            │
                                              alert on HIGH ─┴─▶ WebhookDispatcher
```

## Components

| Module | Responsibility | Day |
|---|---|---|
| `backend/audio/chunker.py` | PCM decode, resample→16k, fixed windows + overlap | 2 |
| `backend/inference/feature_extractor.py` | `DummyFeatureExtractor` / `IndicWav2VecExtractor` → `(T,1024)` | 1 / 3 |
| `backend/inference/aasist.py` | vendored `SSL_BACKEND_aasist` (graph attention backend) | 1 |
| `backend/inference/classifier.py` | AASIST + linear head → `[bonafide, spoof]` logits → `fake_prob` | 1 |
| `backend/inference/pipeline.py` | glue: waveform → `ChunkResult` | 3 |
| `backend/scoring/risk_engine.py` | rolling weighted avg, thresholds, HIGH latch, alert flag | 4 |
| `backend/alerts/webhook.py` | POST payload on HIGH transition, cooldown | 4 |
| `backend/api/websocket.py` | per-connection loop: chunk → score event | 2 / 5 |
| `backend/api/rest.py` | `/health`, `/config`, `/score` | 5 |
| `frontend/` | WebRTC signaling, PCM capture, gauge + alert | 6 |

## Key decisions

- **Vendor only `aasist.py`** from WeDefense — the full toolkit pulls s3prl, kaldiio, fire and
  a Kaldi-style data layout we don't need for a real-time service.
- **16 kHz mono everywhere.** The browser captures at 44.1/48 kHz; the receiver downsamples
  before sending, and `AudioChunker` resamples again defensively.
- **Pipeline is stateless; risk is per-connection.** One warm model in the app; a fresh
  `RiskEngine` + `AudioChunker` per WebSocket.
- **`fake_prob` = P(spoof).** Class order `[bonafide, spoof]`, spoof index 1, fixed in
  `classifier.SPOOF_INDEX`.
- **Frontend feature extractor is swappable via config** (`feature_extractor.backend`), so
  the streaming/scoring/UI work (Days 2, 4–6) proceeds before IndicWav2Vec weights are in.

## Production note

The browser demo substitutes for a VoIP media feed. In production, `/ws/stream` receives PCM
from a gateway (Twilio Media Streams, Genesys AudioHook, SIPREC recorder, FreeSWITCH
`mod_audio_fork`, …). The wire contract — 16 kHz mono little-endian PCM frames + a JSON
`config`/`end` control channel — is the integration seam.
