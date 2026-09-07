"""Scoring metrics for spoof detection: EER and a CM-only min t-DCF.

Score convention throughout VoiceGuard: **higher score = more likely spoof** (``fake_prob``).

Anti-spoofing decision at threshold ``t``:  predict *spoof* iff ``score >= t``.
  * FAR(t) = P(spoof scores below t)   — spoofed audio that slips through as bonafide
  * FRR(t) = P(bonafide scores >= t)   — genuine audio wrongly flagged
FAR rises and FRR falls as ``t`` increases; the crossing point is the EER.
"""

from __future__ import annotations

import numpy as np


def _far_frr(bona: np.ndarray, spoof: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bona = np.sort(np.asarray(bona, dtype=np.float64))
    spoof = np.sort(np.asarray(spoof, dtype=np.float64))
    thr = np.unique(np.concatenate([bona, spoof, [np.inf]]))
    far = np.searchsorted(spoof, thr, side="left") / max(len(spoof), 1)      # spoof < t
    frr = 1.0 - np.searchsorted(bona, thr, side="left") / max(len(bona), 1)  # bona >= t
    return thr, far, frr


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate. Returns ``(eer, threshold)``."""
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return float("nan"), float("nan")
    thr, far, frr = _far_frr(bonafide_scores, spoof_scores)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0), float(thr[idx])


def compute_min_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    *,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """Normalised CM-only detection-cost floor (not the full ASVspoof2019 t-DCF, which also
    needs ASV scores/priors — that is computed by the official toolkit at final eval).
    Useful as a training-progress signal alongside EER.
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return float("nan")
    _, far, frr = _far_frr(bonafide_scores, spoof_scores)
    p_miss = far   # CM fails to catch a spoof (spoof scored below threshold)
    p_fa = frr     # CM false-alarms on genuine audio
    cost = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa
    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(np.min(cost) / norm)
