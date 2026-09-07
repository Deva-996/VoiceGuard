"""Generate throwaway sample WAVs for smoke-testing the pipeline (no real data needed).

    python scripts/make_sample_audio.py

Writes to tests/fixtures/:
  * sample_bonafide.wav  — voiced-ish: summed harmonics + vibrato + breath noise
  * sample_spoof.wav     — 'synthetic-ish': pure tones, no jitter, faint 8 kHz whine

These are NOT real bonafide/spoof audio — they only give the model plausibly-shaped input.
Real evaluation uses the datasets in CLAUDE.md section 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DUR = 3.0
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _bonafide(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f0 = 130.0 + 4.0 * np.sin(2 * np.pi * 5.0 * t)  # vibrato
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 6))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t)) * 0.6 + 0.4
    sig = sig * env + 0.02 * rng.standard_normal(t.size)
    return sig


def _spoof(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f0 = 155.0  # dead-steady pitch
    phase = 2 * np.pi * f0 * t
    sig = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 6))
    sig += 0.03 * np.sin(2 * np.pi * 8000.0 * t)  # vocoder-ish whine
    sig += 0.003 * rng.standard_normal(t.size)
    return sig


def _write(name: str, sig: np.ndarray) -> None:
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
    path = OUT / name
    sf.write(path, sig.astype(np.float32), SR)
    print(f"wrote {path}  ({sig.size / SR:.1f}s)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    t = np.arange(int(SR * DUR)) / SR
    _write("sample_bonafide.wav", _bonafide(t, rng))
    _write("sample_spoof.wav", _spoof(t, rng))


if __name__ == "__main__":
    main()
