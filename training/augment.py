"""RawBoost data augmentation (Tak et al., ICASSP 2022) for anti-spoofing.

Three perturbations applied to the raw 16 kHz waveform:
  1. linear + non-linear convolutive noise   (channel / device colouration + harmonic distortion)
  2. impulsive, signal-dependent additive noise
  3. stationary, signal-independent additive coloured noise

The paper's recipes: algo 1+2 series for LA, algo 3 (or 1+2+3) for codec/telephone (2021 LA /
In-the-Wild). ``RawBoost(mode=...)`` picks the combination; ``mode=0`` is a no-op.

Ported to NumPy from the reference implementation (github.com/TakHemlata/RawBoost-antispoofing,
MIT). Operates per-utterance on a 1-D float array.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


# ------------------------------------------------------------------ building blocks
def _randfloat(lo: float, hi: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(lo, hi))


def _norm_wav(x: np.ndarray, always: bool = False) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 1.0 or always:
        x = x / peak
    return x


def _gen_notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
                      minG, maxG, fs, rng) -> np.ndarray:
    b = np.array([1.0])
    for _ in range(nBands):
        fc = _randfloat(minF, maxF, rng)
        bw = _randfloat(minBW, maxBW, rng)
        c = int(_randfloat(minCoeff, maxCoeff, rng))
        c = c + 1 if c % 2 == 0 else c
        f1 = max(fc - bw / 2, 1.0)
        f2 = min(fc + bw / 2, fs / 2 - 1.0)
        cutoff = [f1 / (fs / 2), f2 / (fs / 2)]
        try:
            fir = signal.firwin(c, cutoff, window="hamming", pass_zero="bandstop")
        except Exception:
            continue
        g = 10 ** (_randfloat(minG, maxG, rng) / 20.0)
        fir = fir * g
        b = np.convolve(b, fir)
    b = b / (np.sum(np.abs(b)) + 1e-9)
    return b


def _filter_fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return signal.lfilter(b, [1.0], x).astype(np.float32)


# ------------------------------------------------------------------ the 3 algorithms
def lnl_convolutive_noise(x, fs, rng, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                          minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLin=5,
                          maxBiasLin=20, N_f=5) -> np.ndarray:
    y = np.zeros_like(x)
    for order in range(1, N_f + 1):
        b = _gen_notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
                              minG, maxG, fs, rng)
        power = _randfloat(minBiasLin, maxBiasLin, rng) if order > 1 else 0.0
        contrib = _filter_fir(np.sign(x) * (np.abs(x) ** order), b)
        y = y + contrib / (10 ** (power / 20.0))
    return _norm_wav(y)


def isd_additive_noise(x, rng, P=10, g_sd=2) -> np.ndarray:
    n = x.shape[0]
    k = max(int(P * n / 100), 1)
    idx = rng.choice(n, k, replace=False)
    mask = np.zeros(n, dtype=np.float32)
    mask[idx] = 1.0
    noise = rng.standard_normal(n).astype(np.float32) * mask
    noise = noise * (np.abs(x) ** 1) * g_sd
    return _norm_wav(x + noise)


def ssi_additive_noise(x, fs, rng, SNRmin=10, SNRmax=40, nBands=5, minF=20, maxF=8000,
                       minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100,
                       minG=0, maxG=0) -> np.ndarray:
    noise = rng.standard_normal(x.shape[0]).astype(np.float32)
    b = _gen_notch_coeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs, rng)
    noise = _filter_fir(noise, b)
    snr = _randfloat(SNRmin, SNRmax, rng)
    ps, pn = np.mean(x ** 2) + 1e-9, np.mean(noise ** 2) + 1e-9
    noise = noise * np.sqrt(ps / (pn * (10 ** (snr / 10.0))))
    return _norm_wav(x + noise)


# ------------------------------------------------------------------ front end
class RawBoost:
    """mode: 0 none · 1 conv · 2 impulsive · 3 stationary · 4 (1->2 series) · 5 (1+2+3) ·
    6 (1+2 parallel).  Default 5 is the most general (covers codec/telephone)."""

    def __init__(self, mode: int = 5, fs: int = 16000, p: float = 0.5, seed: int | None = None) -> None:
        self.mode = mode
        self.fs = fs
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        if self.mode == 0 or self.rng.random() > self.p:
            return wav.astype(np.float32)
        x = wav.astype(np.float32)
        r = self.rng
        if self.mode == 1:
            return lnl_convolutive_noise(x, self.fs, r)
        if self.mode == 2:
            return isd_additive_noise(x, r)
        if self.mode == 3:
            return ssi_additive_noise(x, self.fs, r)
        if self.mode == 4:
            return isd_additive_noise(lnl_convolutive_noise(x, self.fs, r), r)
        if self.mode == 6:
            return _norm_wav(lnl_convolutive_noise(x, self.fs, r) + isd_additive_noise(x, r))
        # mode 5: 1+2 series, then +3
        y = isd_additive_noise(lnl_convolutive_noise(x, self.fs, r), r)
        return ssi_additive_noise(y, self.fs, r)
