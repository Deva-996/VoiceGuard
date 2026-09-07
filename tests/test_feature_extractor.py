"""Feature-extractor tests.

The dummy path runs everywhere. The real wav2vec2 (XLS-R) path is opt-in — it loads a
~1.2 GB model into memory, so it stays out of the default run. Enable it with:
    VG_TEST_HF=1 python -m pytest tests/test_feature_extractor.py
"""

import os

import pytest
import torch

from backend.core import load_config
from backend.inference import (
    DetectionPipeline,
    DummyFeatureExtractor,
    build_feature_extractor,
)

XLSR_ID = "facebook/wav2vec2-xls-r-300m"

hf_available = pytest.mark.skipif(
    not os.environ.get("VG_TEST_HF"),
    reason="real SSL frontend is opt-in; set VG_TEST_HF=1 to run",
)


def test_dummy_extractor_frame_shape():
    ext = DummyFeatureExtractor(feat_dim=1024)
    feats = ext.extract(torch.zeros(16000))
    assert feats.shape == (50, 1024)


def test_build_feature_extractor_rejects_unknown_backend():
    cfg = load_config().feature_extractor
    cfg.backend = "bogus"
    with pytest.raises(ValueError):
        build_feature_extractor(cfg)


@hf_available
def test_xlsr_extractor_shapes_and_range():
    cfg = load_config()
    cfg.feature_extractor.backend = "wav2vec2"
    cfg.feature_extractor.model_id = XLSR_ID
    ext = build_feature_extractor(cfg.feature_extractor)
    assert ext.feat_dim == 1024
    feats = ext.extract(torch.randn(16000))
    assert feats.ndim == 2 and feats.shape[1] == 1024
    assert 45 <= feats.shape[0] <= 55  # ~50 frames for 1 s at 20 ms shift
    assert torch.isfinite(feats).all()


@hf_available
def test_pipeline_end_to_end_with_xlsr():
    cfg = load_config()
    cfg.feature_extractor.backend = "wav2vec2"
    cfg.feature_extractor.model_id = XLSR_ID
    pipe = DetectionPipeline.from_config(cfg)
    res = pipe.infer_chunk(torch.randn(16000))
    assert 0.0 <= res.fake_prob <= 1.0
    assert res.n_frames >= 1
