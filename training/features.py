"""SSL feature cache.

Extracting wav2vec2 features is the expensive part of training. With the frontend frozen we
only ever need each utterance's features once, so we cache them to disk and train the AASIST
head off the cache — fast enough on CPU.

    python -m training.features --manifest data/manifests/train.tsv --config training/config_train.yaml

Layout:  <cache_dir>/<model_tag>/<layer>/<utt_id>.npy   (float16, shape [T, feat_dim])
``model_tag`` folds in the model id + layer so switching frontends doesn't collide.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _tag(model_id: str, layer: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()
    return f"{slug}_L{layer}"


class FeatureCache:
    def __init__(self, cache_dir: str | Path, model_id: str, layer: int) -> None:
        self.root = Path(cache_dir) / _tag(model_id, layer)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, utt_id: str) -> Path:
        return self.root / f"{utt_id}.npy"

    def has(self, utt_id: str) -> bool:
        return self.path(utt_id).exists()

    def load(self, utt_id: str) -> np.ndarray:
        return np.load(self.path(utt_id)).astype(np.float32)

    def save(self, utt_id: str, feats: np.ndarray) -> None:
        np.save(self.path(utt_id), feats.astype(np.float16))


def _make_extractor(fe: dict, device: str):
    """Build an SSL extractor from the training-config ``frontend`` block. ``model_id: dummy``
    (or ``backend: dummy``) uses the deterministic stand-in so tests / smoke runs need no
    download."""
    from backend.inference.feature_extractor import DummyFeatureExtractor, Wav2Vec2Extractor

    if fe.get("backend") == "dummy" or fe["model_id"] == "dummy":
        return DummyFeatureExtractor(feat_dim=int(fe.get("feat_dim", 768)))
    return Wav2Vec2Extractor(
        model_id=fe["model_id"], layer=fe["layer"], frozen=True, device=device
    )


def build_cache(manifest: str | Path, cfg: dict, limit: int | None = None) -> FeatureCache:
    """Extract + cache features for every utt in ``manifest`` that isn't cached yet."""
    from training.dataset import read_manifest

    fe = cfg["frontend"]
    cache = FeatureCache(cfg.get("cache_dir", str(ROOT / "data" / "processed" / "feat_cache")),
                         fe["model_id"], fe["layer"])
    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = _make_extractor(fe, device)

    samples = read_manifest(manifest)
    todo = [s for s in samples if not cache.has(s.utt_id)]
    if limit:
        todo = todo[:limit]
    print(f"{manifest}: {len(samples)} utts, {len(todo)} to extract  ({cache.root})")

    import soundfile as sf

    from training.hf_audio import HFAudioStore, is_ref

    for i, s in enumerate(todo, 1):
        if is_ref(s.path):
            wav, sr = HFAudioStore.load(s.path)
        else:
            wav, sr = sf.read(s.path, dtype="float32", always_2d=False)
            if getattr(wav, "ndim", 1) == 2:
                wav = wav.mean(axis=1)
        if sr != 16000:
            from backend.audio import resample_to_16k

            wav = resample_to_16k(np.asarray(wav), sr)
        feats = extractor.extract(torch.from_numpy(np.asarray(wav, dtype="float32")))
        cache.save(s.utt_id, feats.numpy())
        if i % 200 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
    return cache


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=str(ROOT / "training" / "config_train.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    build_cache(args.manifest, cfg, limit=args.limit)


if __name__ == "__main__":
    main()
