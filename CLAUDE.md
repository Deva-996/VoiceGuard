# VoiceGuard — Real-time AI Voice Cloning Detection

Detect synthetic / cloned voices during live calls and return a real-time risk score.

This file is the source of truth for the architecture and the 7-day build plan. Update it
whenever the design changes. `currentDate` when scaffolded: 2026-09-07.

---

## 1. Problem

During a live phone/VoIP call, decide — continuously, per short audio chunk — whether the
incoming voice is a genuine human or an AI-generated / cloned voice, and surface a rolling
**risk score** plus a **HIGH alert** when confidence of spoofing is sustained.

## 2. Detection pipeline

```
raw audio chunk (16 kHz mono PCM)
        │
        ▼
┌─────────────────────────┐
│ IndicWav2Vec (AI4Bharat)│  SSL feature extractor
│ wav2vec2-large, 1024-d  │  covers English + 40 Indian languages
│ output: (T, 1024) frames│
└─────────────┬───────────┘
              ▼
┌─────────────────────────┐
│ AASIST backend          │  from WeDefense (SSL_BACKEND_aasist)
│ spectro-temporal graph  │  feat_dim=1024 → embed_dim=256
│ attention               │
└─────────────┬───────────┘
              ▼
┌─────────────────────────┐
│ Linear head → 2 logits  │  [bonafide, spoof]
│ softmax → fake_prob     │
└─────────────┬───────────┘
              ▼
   fake_prob ∈ [0, 1] per chunk
```

- **Feature extractor:** a HF wav2vec2-large checkpoint → 1024-d frame features at 20 ms
  frame shift, frozen for the baseline (fine-tuned in Day 3+). `Wav2Vec2Extractor` loads any
  wav2vec2 model via `transformers.AutoModel`.
  - **Default: `facebook/wav2vec2-xls-r-300m`** — public, SSL-pretrained, multilingual
    (128 langs incl. Hindi/Tamil/Telugu/Bengali/…).
  - **Planned: `ai4bharat/indicwav2vec-hindi`** — now *gated* on HF (free, auto-approve, but
    needs an account that accepted the license + a token). Switch with
    `feature_extractor.backend: indicwav2vec` after `huggingface-cli login`.
- **Classifier:** AASIST as backend, taken from the WeDefense repo
  (`https://github.com/zlin0/wedefense`, module
  `wedefense/models/ssl_backend/aasist.py::SSL_BACKEND_aasist`). Vendored into
  `backend/inference/aasist.py` (MIT, NAVER Corp / Hemlata Tak) so we don't depend on the
  full WeDefense training stack (s3prl, kaldiio, fire).
- **Input:** raw audio chunks. **Output:** `fake_prob` score per chunk.

## 3. Real-time input — WebRTC web app

```
┌────────────┐   WebRTC     ┌────────────┐   PCM chunks / WS    ┌──────────────┐
│ caller tab │ ───audio───► │ receiver   │ ──────────────────► │ FastAPI       │
│ (mic)      │              │ tab        │                     │ backend       │
└────────────┘              │ - captures │ ◄────────────────── │ IndicWav2Vec  │
                            │   remote   │   risk score events │ + AASIST      │
                            │   stream   │                     │ + risk engine │
                            │ - WS client│                     └──────┬───────┘
                            │ - dashboard│                            │ webhook on HIGH
                            └────────────┘                            ▼
                                                              external endpoint
```

1. Two browser tabs: **caller** and **receiver**, connected by WebRTC (manual SDP copy-paste
   for the demo, or a tiny signaling endpoint on the backend).
2. The **receiver** tab captures the incoming `MediaStream`, downsamples to 16 kHz mono, and
   streams raw PCM (`Int16`/`Float32`) frames over a **WebSocket** to the backend.
3. Backend buffers PCM into analysis windows, runs wav2vec2 + AASIST inference (in a worker
   thread; if inference falls behind, only the newest pending window is kept).
4. Backend returns a **risk score event per scored window** over the same WebSocket
   (`dropped` = windows skipped since the last event).
5. The receiver **dashboard** shows a live risk gauge and raises an **alert** when the HIGH
   threshold is crossed.

## 4. Risk scoring engine

- **Rolling weighted average** of the last **10** chunk `fake_prob` scores (more recent
  chunks weighted higher — linear or exponential weights).
- **Thresholds** on the rolling score:
  - `LOW`    > 0.40
  - `MEDIUM` > 0.60
  - `HIGH`   > 0.75 **sustained for 3+ consecutive chunks**
- **Webhook dispatch** fires once when the state transitions into `HIGH` (with cooldown to
  avoid repeats).

## 5. Datasets

All acquired by `scripts/download_datasets.py` — open / non-gated only, no registration or HF
token. Raw data lives under `data/raw/` (a junction to a roomy volume; see §9).

| Dataset | Role | Language | Source |
|---|---|---|---|
| **ASVspoof 2019 LA** | base training + dev | English | HF `Bisher/ASVspoof_2019_LA` (parquet: train/val/test, `key` 0=bonafide/1=spoof) |
| **ASVspoof 2021 LA eval** | eval (codec/telephone) | English | Zenodo (eval + keys) |
| **In-the-Wild** | eval (unseen, domain shift) | English | deepfake-total.com (`release_in_the_wild.zip`, `meta.csv`) |
| **Indian fine-tuning set** (self-generated) | fine-tune + eval | hi, ta, te, bn, mr, gu | see below |

**Indian fine-tuning set:**
- **Genuine:** **SPRINGLab/IndicTTS_\*** on HF — the IIT-Madras IndicTTS corpus itself,
  non-gated (no email request). ~1600 utts/lang streamed. Plus **google/fleurs** (hi, bn, ta,
  te, mr, gu) for speaker variety + a separate genuine-domain eval slice.
- **Fake:** `scripts/generate_indian_fakes.py` re-synthesises the **same IndicTTS sentences**
  with **Meta MMS-TTS** (`facebook/mms-tts-<iso3>`, 16 kHz, local, CPU) → content-matched
  genuine/fake pairs. Google Cloud TTS / Bhashini can be added as extra `engine`s later.
- Deterministic 80/10/10 per-language split (`scripts/prepare_manifests.py::split_indic`).
- Manifest schema: `utt_id  path  label(bonafide|spoof)  language  source  dataset`.

## 6. Tech stack

Python · FastAPI · PyTorch · torchaudio · librosa · WebSockets · WebRTC (browser) ·
transformers/huggingface_hub (IndicWav2Vec) · soundfile.

## 7. Repository layout

```
voiceguard/
├── CLAUDE.md                  # this file
├── config/config.yaml         # runtime config (thresholds, model paths, sample rate)
├── backend/
│   ├── core/                  # config loading, logging
│   ├── audio/                 # chunker, resampling, PCM framing        (Day 2)
│   ├── inference/
│   │   ├── aasist.py          # vendored AASIST backend (MIT)
│   │   ├── feature_extractor.py  # IndicWav2Vec wrapper + dummy         (Day 3)
│   │   ├── classifier.py      # AASIST + linear head → fake_prob        (Day 1/3)
│   │   └── pipeline.py        # extractor → classifier end to end       (Day 3)
│   ├── scoring/risk_engine.py # rolling avg + thresholds + alerts       (Day 4)
│   ├── alerts/webhook.py      # webhook dispatch on HIGH                (Day 4)
│   ├── api/
│   │   ├── rest.py            # REST endpoints                          (Day 5)
│   │   └── websocket.py       # WS streaming endpoint                   (Day 2/5)
│   └── main.py                # FastAPI app entrypoint                  (Day 5)
├── frontend/                  # caller/receiver tabs + dashboard        (Day 6)
├── training/                  # dataset, features (SSL cache), losses, metrics, train, evaluate
├── scripts/                   # download_datasets, generate_indian_fakes, prepare_manifests,
│                              #   bench_frontends, day{1,2,3} checks / smoke tests
├── tests/
├── third_party/wedefense/     # cloned reference repo (not packaged)
└── data/                      # raw / processed / generated / manifests (gitignored)
```

## 8. 7-day plan

| Day | Goal | Key deliverables | Status |
|---|---|---|---|
| **1** | Clone WeDefense, get AASIST running on sample audio, scaffold project | repo skeleton; vendored `aasist.py`; `scripts/day1_sample_inference.py` runs a forward pass on a generated sample wav and prints `fake_prob` | **DONE** |
| **2** | Audio chunker + WebSocket streaming pipeline | `backend/audio/chunker.py` (PCM framing, 16 kHz, overlap); `backend/api/websocket.py` full chunk→score loop; `tests/test_websocket.py` in-process WS smoke test; `scripts/day2_ws_smoke_test.py` browser→WS smoke test against a live server | **DONE** |
| **3** | Real SSL frontend + AASIST, real-time on CPU | `Wav2Vec2Extractor` loads any HF wav2vec2 checkpoint; streaming default `facebook/wav2vec2-base` (~0.45 s / 1 s chunk on CPU — keeps up at the 0.5 s hop); XLS-R / IndicWav2Vec swappable for offline eval; WS handler drops stale windows + runs inference in a worker thread so lag stays bounded; `scripts/day3_pipeline_check.py`, `scripts/bench_frontends.py` | **DONE** |
| **4** | Risk scoring engine + alert thresholds | `risk_engine.py` rolling weighted avg over 10, LOW/MED/HIGH, 3-consecutive HIGH latch; `webhook.py` dispatch + cooldown; unit tests | TODO |
| **5** | FastAPI fully wired — REST + WebSocket endpoints | `/health`, `/score` (single clip), `/config`; `/ws/stream` full loop chunk→score event; `main.py` wires pipeline + risk engine per connection | TODO |
| **6** | WebRTC web app — caller/receiver tabs, audio capture, live risk dashboard | `caller.html`, `receiver.html`, WebRTC connect, receiver captures remote stream, downsample + WS send, live gauge + HIGH alert banner | TODO |
| **7** | End to end test with real deepfake audio, polish demo | run a known deepfake clip through caller→receiver→backend, verify HIGH alert + webhook; README demo script; screenshots | TODO |

### 8b. Full build (post-demo — "all features")

The 7-day plan builds the *system*; this track makes detection real.

| Phase | Work | Status |
|---|---|---|
| **P1** | Training + eval code — `features.py` (SSL cache), `losses.py` (weighted-CE + OC-Softmax), `metrics.py` (EER, min-tDCF), `augment.py` (RawBoost), `model.py` (`EndToEndDetector`), `dataset.py`, `train.py` (frozen **and** e2e modes, staged unfreeze, balanced sampler), `evaluate.py`, `pipeline_run.py`. Serving loads fine-tuned frontend weights from the checkpoint. Unit-tested offline. | **DONE** |
| **P2** | Data (`scripts/download_datasets.py`, all non-gated) — ASVspoof 2019 LA ✅, IndicTTS 6 langs ✅, In-the-Wild ✅; ASVspoof 2021 LA eval + FLEURS downloading | ~90% |
| **P3** | `scripts/generate_indian_fakes.py` — MMS-TTS fakes, content-matched to IndicTTS. Code done + smoke-tested; full run pending genuine data. | code done |
| **P4** | `scripts/prepare_manifests.py` — per-dataset parsers → train/dev/eval TSVs. Code done; ASVspoof2019 parser verified. | code done |
| **P5** | **Train on GPU (Kaggle P100)** — `notebooks/train_kaggle.ipynb`. Fine-tune XLS-R **and** IndicWav2Vec + RawBoost + OC-Softmax, A/B on the Indian slices → `backend/models/aasist_indicw2v.pt`. Target ~<1% EER (ASVspoof), meaningful cross-lingual. | ready to launch |
| **P6** | Frontend: real WebRTC caller/receiver, AudioWorklet capture, live dashboard. | TODO |
| **P7** | Integration + polish: real deepfake clip end-to-end, README, screenshots. | TODO |

## 9. Known constraints

- **iOS native calls are not interceptable** (Apple sandbox). VoiceGuard therefore works at
  the **WebRTC / browser** layer, which runs on every platform including iOS Safari.
- **Inference latency (CPU, i7-1360P, measured on 1 s chunks via `scripts/bench_frontends.py`):**
  `wav2vec2-base` ≈ 0.45 s · `wav2vec2-base` L7-truncated ≈ 0.37 s · `xls-r-300m` ≈ 0.97 s ·
  `xls-r-300m` int8 ≈ 0.74 s. So **`wav2vec2-base` is real-time at the 0.5 s hop**; XLS-R
  needs `hop_seconds: 1.0`. Trade-off: `wav2vec2-base` is English-only pretrained → weaker on
  Indian-language calls until the pipeline is fine-tuned; XLS-R / IndicWav2Vec are multilingual.
  Regardless of frontend, the WS handler keeps only the newest pending analysis window
  (drops + counts the rest) and runs inference off the receive loop, so latency is bounded —
  under load the score just updates less often. No GPU on the target machine (Intel Iris Xe;
  OpenVINO on the iGPU is a possible later optimisation).
- In production the backend would receive call audio from a **VoIP gateway / enterprise PBX**
  (SIPREC, Twilio Media Streams, Genesys AudioHook, …) rather than a browser tab; the
  browser demo stands in for that feed. The WebSocket PCM contract is the integration point.
- **Data location:** C: had ~13 GB free, so `data/` is a **Windows junction to
  `E:\voiceguard-data`** (~300 GB). All code paths use `data/…` unchanged. The HF cache is at
  the default `~/.cache/huggingface`. If E: is ever detached the junction breaks — recreate
  with `New-Item -ItemType Junction`.
- **Training compute:** no GPU. The frozen-frontend baseline (features cached once, AASIST
  head trained off the cache) is CPU-feasible but slow (~overnight per stage). Stage-2
  frontend fine-tuning needs a GPU (Colab/Kaggle/cloud).

## 10. Conventions

- Sample rate everywhere: **16 kHz mono**. Chunk/analysis window: **~1 s** (configurable in
  `config/config.yaml`), hop ~0.5 s.
- `label` convention: `bonafide` = genuine human, `spoof` = synthetic/cloned. Model output
  `fake_prob` = P(spoof).
- Never commit datasets, checkpoints, or generated audio — see `.gitignore`.
- WeDefense is a *reference*; only `aasist.py` is vendored. Don't `pip install -e` it.

## 11. Attribution for commits / PRs

```
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013dzEgt2HWwaKtCRCsCejx4
```
PR descriptions end with: 🤖 Generated with [Claude Code](https://claude.com/claude-code)
