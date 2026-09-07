import numpy as np
import pytest

from training.augment import RawBoost


@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5, 6])
def test_rawboost_preserves_shape_and_finiteness(mode):
    rng = np.random.default_rng(0)
    wav = (0.3 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
           + 0.01 * rng.standard_normal(16000)).astype("float32")
    rb = RawBoost(mode=mode, p=1.0, seed=1)
    out = rb(wav)
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0 + 1e-4  # normalised


def test_rawboost_mode0_is_noop():
    wav = np.linspace(-0.5, 0.5, 16000, dtype="float32")
    assert np.array_equal(RawBoost(mode=0)(wav), wav)


def test_rawboost_probability_gate():
    wav = np.ones(4000, dtype="float32") * 0.2
    untouched = sum(np.array_equal(RawBoost(mode=5, p=0.0, seed=s)(wav), wav) for s in range(10))
    assert untouched == 10  # p=0 -> never augments


def test_rawboost_actually_changes_signal():
    rng = np.random.default_rng(2)
    wav = rng.standard_normal(16000).astype("float32") * 0.3
    out = RawBoost(mode=5, p=1.0, seed=3)(wav)
    assert not np.allclose(out, wav)
