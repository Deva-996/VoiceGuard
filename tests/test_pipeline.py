import torch

from backend.core import load_config
from backend.inference import (
    AASISTClassifier,
    DetectionPipeline,
    DummyFeatureExtractor,
)


def test_dummy_extractor_shape():
    ext = DummyFeatureExtractor(feat_dim=1024)
    feats = ext.extract(torch.zeros(16000))
    assert feats.shape[1] == 1024
    assert feats.shape[0] == 50  # 1 s / 20 ms frames


def test_aasist_forward_produces_prob_in_unit_interval():
    torch.manual_seed(0)
    ext = DummyFeatureExtractor(feat_dim=1024)
    clf = AASISTClassifier(feat_dim=1024, embed_dim=256, num_classes=2).eval()
    feats = ext.extract(torch.randn(16000 * 2))
    prob = float(clf.fake_prob(feats))
    assert 0.0 <= prob <= 1.0


def test_pipeline_from_config_runs_on_short_and_long_audio():
    cfg = load_config()
    cfg.feature_extractor.backend = "dummy"
    pipe = DetectionPipeline.from_config(cfg)
    for n in (4000, 16000, 16000 * 3):
        res = pipe.infer_chunk(torch.randn(n))
        assert 0.0 <= res.fake_prob <= 1.0
        assert res.n_frames >= 1


def test_pipeline_batch_consistency():
    torch.manual_seed(0)
    clf = AASISTClassifier(feat_dim=1024, embed_dim=256).eval()
    feats = torch.randn(2, 60, 1024)
    probs = clf.fake_prob(feats)
    assert probs.shape == (2,)
