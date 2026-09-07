"""Day 1 deliverable: prove the AASIST backend runs end to end on sample audio.

    python scripts/day1_sample_inference.py

What it does:
  1. ensures tests/fixtures/sample_*.wav exist (generates them if not);
  2. builds the DetectionPipeline with the *dummy* feature extractor
     (IndicWav2Vec integration is Day 3);
  3. runs each sample clip through  waveform -> features -> AASIST -> fake_prob
     and prints the score, frame count, and latency.

Random-init weights => scores are meaningless numbers; the point is that shapes line up and
the forward pass works. Exit code 0 on success.
"""

from __future__ import annotations

import sys
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
    torch.manual_seed(0)
    cfg = get_config()
    print(f"feature_extractor backend : {cfg.feature_extractor.backend}")
    print(f"classifier                : AASIST (feat_dim={cfg.feature_extractor.feat_dim}, "
          f"embed_dim={cfg.classifier.embed_dim})")

    pipeline = DetectionPipeline.from_config(cfg)
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
            f"latency={result.latency_ms:6.1f} ms"
        )

    print("\nOK - AASIST forward pass works on sample audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
