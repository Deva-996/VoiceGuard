"""Day 3 deliverable: real SSL frontend -> AASIST -> fake_prob, end to end.

    python scripts/day3_pipeline_check.py
    python scripts/day3_pipeline_check.py --backend indicwav2vec   # needs an HF token

Loads the wav2vec2 frontend from HuggingFace (default: facebook/wav2vec2-xls-r-300m),
builds the DetectionPipeline, and runs each sample clip through
waveform -> SSL features -> AASIST -> fake_prob, printing shapes, scores, and latency.

The AASIST head is still randomly initialised (training is post-demo), so the scores are
not meaningful yet — the point is that the real feature frontend loads and the shapes line
up through the whole pipeline. Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core import get_config  # noqa: E402
from backend.inference import DetectionPipeline  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def _ensure_fixtures() -> list[Path]:
    wavs = sorted(FIXTURES.glob("sample_*.wav"))
    if not wavs:
        from scripts.make_sample_audio import main as make_samples

        make_samples()
        wavs = sorted(FIXTURES.glob("sample_*.wav"))
    return wavs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=None, help="override feature_extractor.backend")
    ap.add_argument("--model-id", default=None, help="override feature_extractor.model_id")
    ap.add_argument("--layer", type=int, default=None, help="hidden layer for frame features")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = get_config()
    if args.backend:
        cfg.feature_extractor.backend = args.backend
    if args.model_id:
        cfg.feature_extractor.model_id = args.model_id
    if args.layer is not None:
        cfg.feature_extractor.layer = args.layer

    print(f"feature_extractor backend : {cfg.feature_extractor.backend}")
    print(f"model_id                  : {cfg.feature_extractor.model_id}")
    print(f"layer                     : {cfg.feature_extractor.layer}")

    t0 = time.perf_counter()
    pipeline = DetectionPipeline.from_config(cfg)
    load_s = time.perf_counter() - t0
    print(f"pipeline loaded in {load_s:.1f}s")
    print(f"SSL feat_dim              : {pipeline.extractor.feat_dim}")
    n_params = sum(p.numel() for p in pipeline.classifier.parameters())
    print(f"AASIST classifier params  : {n_params:,}\n")

    wavs = _ensure_fixtures()
    for wav_path in wavs:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        assert sr == cfg.audio.sample_rate, f"{wav_path.name}: expected 16 kHz, got {sr}"
        result = pipeline.infer_chunk(torch.from_numpy(wav))
        print(
            f"{wav_path.name:<24}  fake_prob={result.fake_prob:6.4f}  "
            f"frames={result.n_frames:<4d} samples={result.n_samples:<6d} "
            f"latency={result.latency_ms:7.1f} ms"
        )

    print("\nOK - real SSL frontend + AASIST forward pass works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
