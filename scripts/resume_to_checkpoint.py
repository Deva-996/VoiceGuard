"""Build a serving checkpoint from a training `.resume.pt`.

A run trained with OC-Softmax saves its best checkpoint via ``EndToEndDetector.state_for_checkpoint``
— which historically dropped the OC-Softmax centre, leaving the checkpoint scorable only by the
*untrained* logit head (garbage). The per-epoch ``<out>.resume.pt`` does carry the centre
(``loss`` state) alongside the full model state, so it can be rebuilt into a correct,
self-consistent serving checkpoint.

    python scripts/resume_to_checkpoint.py <resume.pt> [-o backend/models/aasist_indicw2v.pt] \
        [--meta-from backend/models/aasist_indicw2v.pt]

The output matches what ``training/train.py`` now writes directly (so a future finished run
overwrites it in the same format).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _split_prefix(sd: dict, prefix: str) -> dict:
    p = prefix if prefix.endswith(".") else prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("resume", type=Path)
    ap.add_argument("-o", "--out", type=Path,
                    default=ROOT / "backend" / "models" / "aasist_indicw2v.pt")
    ap.add_argument("--meta-from", type=Path,
                    default=ROOT / "backend" / "models" / "aasist_indicw2v.pt",
                    help="existing checkpoint to copy frontend_model_id/layer/dims from")
    args = ap.parse_args()

    r = torch.load(args.resume, map_location="cpu", weights_only=False)
    if "model" not in r or "loss" not in r:
        raise SystemExit(f"{args.resume} is not a training resume file (need 'model' + 'loss')")

    full = r["model"]                       # EndToEndDetector.state_dict()
    frontend = _split_prefix(full, "frontend")
    classifier = _split_prefix(full, "classifier")
    if not frontend or not classifier:
        raise SystemExit(f"unexpected key layout in resume['model']: {list(full)[:4]}…")

    center = r["loss"].get("center")
    if center is None:
        raise SystemExit(f"resume['loss'] has no 'center' (keys: {list(r['loss'])})")

    meta = {}
    if args.meta_from.exists():
        m = torch.load(args.meta_from, map_location="cpu", weights_only=False)
        meta = {k: m.get(k) for k in
                ("frontend_model_id", "layer", "feat_dim", "embed_dim", "num_classes")}

    out = {
        "model": classifier,
        "frontend": frontend,
        "frontend_model_id": meta.get("frontend_model_id", "facebook/wav2vec2-xls-r-300m"),
        "layer": meta.get("layer", -1),
        "feat_dim": meta.get("feat_dim", 1024),
        "embed_dim": meta.get("embed_dim", center.shape[-1]),
        "num_classes": meta.get("num_classes", 2),
        "dev_eer": r.get("best"),
        "epoch": r.get("epoch"),
        "oc_softmax": {"state": {"center": center.cpu()},
                       "m_real": 0.9, "m_fake": 0.2, "alpha": 20.0},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}")
    print(f"  epoch={out['epoch']}  dev_eer={out['dev_eer']}  frontend={out['frontend_model_id']}")
    print(f"  centre {tuple(center.shape)}  |c|={center.norm().item():.3f}")
    print(f"  frontend tensors={len(frontend)}  classifier tensors={len(classifier)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
