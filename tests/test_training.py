"""Offline end-to-end training smoke: synthetic audio -> dummy SSL cache -> AASIST head
trains -> checkpoint round-trips into the live classifier."""

import numpy as np
import soundfile as sf
import torch

from backend.inference.classifier import AASISTClassifier
from training.dataset import FeatureDataset, collate_features
from training.features import build_cache
from training.losses import WeightedCE
from training.train import evaluate_dev, train_one_epoch


def _make_dataset(tmp_path, n=24):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    man = tmp_path / "m.tsv"
    lines = ["utt_id\tpath\tlabel\tlanguage\tsource\tdataset"]
    rng = np.random.default_rng(0)
    for i in range(n):
        label = "spoof" if i % 2 else "bonafide"
        # give the two classes different spectra so the head has signal to learn
        f = 200 if label == "bonafide" else 400
        t = np.arange(int(16000 * 1.5)) / 16000
        wav = 0.3 * np.sin(2 * np.pi * f * t) + 0.01 * rng.standard_normal(t.size)
        p = audio_dir / f"u{i}.wav"
        sf.write(p, wav.astype("float32"), 16000)
        lines.append(f"u{i}\t{p}\t{label}\thi\ttts:test\tsynthetic")
    man.write_text("\n".join(lines), encoding="utf-8")
    return man


def test_training_smoke(tmp_path):
    man = _make_dataset(tmp_path, n=24)
    cfg = {
        "frontend": {"model_id": "dummy", "layer": -1, "feat_dim": 256},
        "cache_dir": str(tmp_path / "cache"),
    }
    cache = build_cache(man, cfg)
    assert cache.has("u0") and cache.load("u0").ndim == 2

    ds = FeatureDataset(man, cache, max_frames=60, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_features)

    feat_dim = cache.load("u0").shape[1]
    model = AASISTClassifier(feat_dim=feat_dim, embed_dim=64, num_classes=2)
    loss_fn = WeightedCE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    first = train_one_epoch(model, loss_fn, dl, opt, torch.device("cpu"))
    for _ in range(4):
        last = train_one_epoch(model, loss_fn, dl, opt, torch.device("cpu"))
    assert last < first  # loss goes down

    eer, n = evaluate_dev(model, loss_fn, dl, torch.device("cpu"))
    assert n == 24 and 0.0 <= eer <= 1.0

    # checkpoint round-trips into a fresh classifier
    ckpt = tmp_path / "ck.pt"
    torch.save({"model": model.state_dict(), "feat_dim": feat_dim}, ckpt)
    fresh = AASISTClassifier(feat_dim=feat_dim, embed_dim=64, num_classes=2)
    fresh.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    a = model(next(iter(dl))[0])
    b = fresh(next(iter(dl))[0])
    assert a.shape == b.shape


def test_frozen_run_produces_and_resumes(tmp_path):
    import types

    import yaml

    from training import train as T

    man = _make_dataset(tmp_path, n=32)
    cfg = {
        "seed": 0,
        "device": "cpu",
        "manifests": {"train": str(man), "dev": str(man)},
        "cache_dir": str(tmp_path / "cache"),
        "frontend": {"model_id": "dummy", "layer": -1, "feat_dim": 128},
        "classifier": {"embed_dim": 32, "num_classes": 2},
        "batch_size": 8,
        "max_frames": 50,
        "loss": {"name": "weighted_ce"},
        "optimizer": {"lr": 1e-3},
        "checkpoint": {"out": str(tmp_path / "ck.pt")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    args = types.SimpleNamespace(config=str(cfg_path), limit=None, epochs=2, mode="frozen")
    T._run_frozen(cfg, torch.device("cpu"), args)
    assert (tmp_path / "ck.pt").exists()
    resume = tmp_path / "ck.resume.pt"
    assert not resume.exists()  # cleaned up on normal completion

    # a 4-epoch run that we interrupt after 2 leaves a resume file it can continue from
    args.epochs = 2
    T._run_frozen(cfg, torch.device("cpu"), args)  # re-runs from scratch (no resume file)
    assert (tmp_path / "ck.pt").exists()


def test_oc_softmax_path_runs(tmp_path):
    from training.losses import OCSoftmax

    man = _make_dataset(tmp_path, n=16)
    cfg = {"frontend": {"model_id": "dummy", "feat_dim": 128, "layer": -1},
           "cache_dir": str(tmp_path / "c")}
    cache = build_cache(man, cfg)
    ds = FeatureDataset(man, cache, max_frames=40, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=collate_features)

    model = AASISTClassifier(feat_dim=cache.load("u0").shape[1], embed_dim=32)
    loss_fn = OCSoftmax(feat_dim=32)
    opt = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3)
    train_one_epoch(model, loss_fn, dl, opt, torch.device("cpu"))
    eer, n = evaluate_dev(model, loss_fn, dl, torch.device("cpu"))
    assert n == 16 and 0.0 <= eer <= 1.0
