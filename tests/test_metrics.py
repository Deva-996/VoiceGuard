import numpy as np

from training.metrics import compute_eer, compute_min_tdcf


def test_eer_perfectly_separable_is_zero():
    bona = np.array([0.0, 0.1, 0.2])
    spoof = np.array([0.8, 0.9, 1.0])
    eer, thr = compute_eer(bona, spoof)
    assert eer == 0.0
    assert 0.2 <= thr <= 0.8


def test_eer_fully_overlapping_is_about_half():
    rng = np.random.default_rng(0)
    bona = rng.normal(0.5, 0.1, 2000)
    spoof = rng.normal(0.5, 0.1, 2000)
    eer, _ = compute_eer(bona, spoof)
    assert 0.4 < eer < 0.6


def test_eer_direction_higher_score_is_spoof():
    # if we accidentally flipped direction, a clean split would give EER ~1.0
    bona = np.linspace(0.0, 0.4, 100)
    spoof = np.linspace(0.6, 1.0, 100)
    eer, _ = compute_eer(bona, spoof)
    assert eer < 0.01


def test_min_tdcf_lower_for_better_separation():
    rng = np.random.default_rng(1)
    good = compute_min_tdcf(rng.normal(0.2, 0.1, 500), rng.normal(0.8, 0.1, 500))
    bad = compute_min_tdcf(rng.normal(0.5, 0.2, 500), rng.normal(0.5, 0.2, 500))
    assert good < bad
    assert 0.0 <= good <= 1.5


def test_eer_handles_empty():
    eer, thr = compute_eer(np.array([]), np.array([1.0]))
    assert np.isnan(eer)
