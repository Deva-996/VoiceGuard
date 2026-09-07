# VoiceGuard

Real-time AI voice-cloning detection for live calls. Streams call audio to a backend that
runs **IndicWav2Vec → AASIST** inference per chunk and returns a rolling **risk score**, with
a **HIGH alert** (and webhook) when spoofing is sustained.

Architecture, dataset plan, and the 7-day build schedule live in **[CLAUDE.md](./CLAUDE.md)**.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate        # Windows: .venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# reference repo (AASIST source; not packaged)
git clone --depth 1 https://github.com/zlin0/wedefense third_party/wedefense
```

## Day 1 — verify AASIST runs on sample audio

```bash
python scripts/make_sample_audio.py        # writes tests/fixtures/sample_*.wav
python scripts/day1_sample_inference.py     # waveform -> dummy features -> AASIST -> fake_prob
pytest -q                                   # chunker + risk engine + pipeline tests
```

`day1_sample_inference.py` uses random-init AASIST weights and a placeholder feature
extractor — the scores are not meaningful yet; it proves the forward path works. Real
features (IndicWav2Vec) land on Day 3, trained weights after that.

## Run the backend (Day 5+)

```bash
uvicorn backend.main:app --reload
#  GET  /health   GET /config   POST /score (multipart file)   WS /ws/stream
#  frontend served at /  (caller.html / receiver.html)
```

## Layout

| Path | What |
|---|---|
| `backend/inference/` | feature extractor, vendored AASIST, classifier, pipeline |
| `backend/audio/` | PCM decode + chunker (streaming) |
| `backend/scoring/` | rolling risk engine + thresholds |
| `backend/alerts/` | webhook dispatch |
| `backend/api/` + `backend/main.py` | FastAPI REST + WebSocket |
| `frontend/` | WebRTC caller/receiver tabs + live dashboard |
| `training/` | dataset / train / evaluate (post-demo) |
| `scripts/` | sample audio, dataset download, Indian-fake generation, manifests |
| `config/config.yaml` | thresholds, model paths, sample rate |

## Constraints

iOS cannot intercept native calls (Apple sandbox), so VoiceGuard operates at the
WebRTC/browser layer — works everywhere including iOS Safari. In production the audio feed
comes from a VoIP gateway / PBX; the browser demo stands in for that.
