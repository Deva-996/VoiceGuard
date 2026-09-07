"""Evaluate a checkpoint: EER (+ CM-only min t-DCF) pooled and per slice.

    python -m training.evaluate --checkpoint backend/models/aasist_indicw2v.pt \
        --manifest data/manifests/eval.tsv --by dataset,language

Per-slice numbers matter here: ASVspoof2021 LA / In-the-Wild / the Indic set are separate
domains and domain shift is the whole point of holding them out.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _build_model(ckpt: dict):
    from backend.inference.classifier import AASISTClassifier

    cfg = ckpt.get("config", {})
    clf = cfg.get("classifier", {"embed_dim": 256, "num_classes": 2})
    model = AASISTClassifier(
        feat_dim=int(ckpt.get("feat_dim", 768)),
        embed_dim=clf.get("embed_dim", 256),
        num_classes=clf.get("num_classes", 2),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def score_manifest(checkpoint: str | Path, manifest: str | Path, device: str = "cpu",
                   limit: int | None = None):
    from backend.inference.feature_extractor import Wav2Vec2Extractor
    from training.dataset import read_manifest
    from training.features import FeatureCache

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, cfg = _build_model(ckpt)
    model.to(device)

    fe = cfg.get("frontend", {"model_id": "facebook/wav2vec2-base", "layer": -1})
    cache = FeatureCache(
        cfg.get("cache_dir", str(ROOT / "data" / "processed" / "feat_cache")),
        fe["model_id"], fe["layer"],
    )
    extractor = None  # lazy — only build if some utt isn't cached

    rows = read_manifest(manifest)
    if limit:
        rows = rows[:limit]

    import soundfile as sf

    out = []
    for i, s in enumerate(rows, 1):
        if cache.has(s.utt_id):
            feats = torch.from_numpy(cache.load(s.utt_id))
        else:
            if extractor is None:
                extractor = Wav2Vec2Extractor(model_id=fe["model_id"], layer=fe["layer"],
                                              frozen=True, device=device)
            wav, sr = sf.read(s.path, dtype="float32", always_2d=False)
            if getattr(wav, "ndim", 1) == 2:
                wav = wav.mean(axis=1)
            if sr != 16000:
                from backend.audio import resample_to_16k

                wav = resample_to_16k(np.asarray(wav), sr)
            feats = extractor.extract(torch.from_numpy(np.asarray(wav, dtype="float32")))
            cache.save(s.utt_id, feats.numpy())
        logits = model(feats.unsqueeze(0).to(device))
        fake_prob = float(torch.softmax(logits, dim=-1)[0, 1])
        out.append((s, fake_prob))
        if i % 500 == 0:
            print(f"  scored {i}/{len(rows)}")
    return out


def report(scored, by: list[str]) -> None:
    from training.metrics import compute_eer, compute_min_tdcf

    def eer_line(name, pairs):
        bona = np.array([p for s, p in pairs if s.label == 0])
        spoof = np.array([p for s, p in pairs if s.label == 1])
        eer, thr = compute_eer(bona, spoof)
        tdcf = compute_min_tdcf(bona, spoof)
        print(f"  {name:28s}  EER={eer*100:6.2f}%   min-tDCF={tdcf:.4f}   "
              f"(n={len(pairs)}, {len(spoof)} spoof / {len(bona)} bona)")

    print("\npooled:")
    eer_line("ALL", scored)
    for col in by:
        print(f"\nby {col}:")
        groups: dict[str, list] = defaultdict(list)
        for s, p in scored:
            groups[getattr(s, col, "?")].append((s, p))
        for k in sorted(groups):
            eer_line(f"{col}={k}", groups[k])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--by", default="dataset", help="comma-separated slice columns")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump", default=None, help="write per-utt scores TSV here")
    args = ap.parse_args()

    scored = score_manifest(args.checkpoint, args.manifest, args.device, args.limit)
    if args.dump:
        Path(args.dump).write_text(
            "utt_id\tlabel\tdataset\tlanguage\tfake_prob\n"
            + "\n".join(f"{s.utt_id}\t{s.label}\t{s.dataset}\t{s.language}\t{p:.6f}"
                        for s, p in scored),
            encoding="utf-8",
        )
        print(f"wrote {args.dump}")
    report(scored, [c.strip() for c in args.by.split(",") if c.strip()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
