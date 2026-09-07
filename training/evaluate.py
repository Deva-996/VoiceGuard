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


def _per_domain_sample(rows, per_domain: int):
    """Deterministic ~balanced subset per dataset — a representative eval that runs in
    minutes instead of hours. Keeps all rows of a domain that has fewer than the cap."""
    import random
    from collections import defaultdict

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r.dataset].append(r)
    rng = random.Random(0)
    out = []
    for ds, items in by_ds.items():
        if len(items) <= per_domain:
            out += items
            continue
        bona = [r for r in items if r.label == 0]
        spoof = [r for r in items if r.label == 1]
        rng.shuffle(bona); rng.shuffle(spoof)
        half = per_domain // 2
        out += bona[:half] + spoof[:half]
    return out


@torch.no_grad()
def score_manifest(checkpoint: str | Path, manifest: str | Path, device: str = "cpu",
                   limit: int | None = None, per_domain: int | None = None,
                   batch_size: int = 16, crop_seconds: float = 6.0):
    import soundfile as sf

    from backend.audio import resample_to_16k
    from training.dataset import read_manifest
    from training.model import EndToEndDetector

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    fe = cfg.get("frontend") or {"model_id": ckpt.get("frontend_model_id", "facebook/wav2vec2-base"),
                                 "layer": ckpt.get("layer", -1)}
    clf = cfg.get("classifier", {})
    embed_dim = ckpt.get("embed_dim", clf.get("embed_dim", 256))
    num_classes = ckpt.get("num_classes", clf.get("num_classes", 2))

    det = EndToEndDetector(model_id=fe["model_id"], layer=fe["layer"],
                           embed_dim=embed_dim, num_classes=num_classes)
    if ckpt.get("frontend"):
        det.frontend.load_state_dict(ckpt["frontend"], strict=False)
    det.classifier.load_state_dict(ckpt["model"], strict=False)
    det.set_frontend_trainable(False)
    det.to(device).eval()

    rows = read_manifest(manifest)
    if per_domain:
        rows = _per_domain_sample(rows, per_domain)
    if limit:
        rows = rows[:limit]
    crop = int(16000 * crop_seconds)
    print(f"scoring {len(rows)} utts (batch {batch_size})")

    def load(path):
        w, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(w, "ndim", 1) == 2:
            w = w.mean(axis=1)
        if sr != 16000:
            w = resample_to_16k(np.asarray(w), sr)
        w = np.asarray(w, dtype="float32")
        if len(w) >= crop:
            w = w[:crop]
        else:
            w = np.tile(w, int(np.ceil(crop / max(len(w), 1))))[:crop]
        return w

    out = []
    for b in range(0, len(rows), batch_size):
        chunk = rows[b:b + batch_size]
        wavs = torch.from_numpy(np.stack([load(s.path) for s in chunk])).to(device)
        probs = torch.softmax(det(wavs), dim=-1)[:, 1].cpu().numpy()
        out += list(zip(chunk, (float(p) for p in probs)))
        if (b // batch_size) % 25 == 0:
            print(f"  {len(out)}/{len(rows)}")
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
    ap.add_argument("--per-domain", type=int, default=6000,
                    help="cap utts per dataset (balanced); 0 = use all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dump", default=None, help="write per-utt scores TSV here")
    args = ap.parse_args()

    scored = score_manifest(args.checkpoint, args.manifest, args.device, args.limit,
                            per_domain=args.per_domain or None, batch_size=args.batch_size)
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
