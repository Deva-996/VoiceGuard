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

    # mmap=True keeps the 1.2 GB checkpoint tensors on disk instead of copying them into RAM;
    # load_state_dict then reads them through. Halves peak memory on the load spike, which
    # otherwise holds two full copies of the XLS-R weights at once.
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
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

    # OC-Softmax checkpoint: the logit head is untrained — score by cosine distance to the
    # learned centre (same path training used for its dev-EER), mapped to P(spoof).
    oc = ckpt.get("oc_softmax")
    if oc:
        st = oc.get("state", oc)
        center = st["center"] if isinstance(st, dict) and "center" in st else st
        det.classifier.set_oc_softmax(center, m_real=oc.get("m_real", 0.9),
                                      m_fake=oc.get("m_fake", 0.2), alpha=oc.get("alpha", 20.0))
        print("scoring via OC-Softmax centre distance")

    del ckpt, cfg, clf  # release the checkpoint dict before the scoring loop
    import gc
    gc.collect()

    rows = read_manifest(manifest)
    if per_domain:
        rows = _per_domain_sample(rows, per_domain)
    if limit:
        rows = rows[:limit]
    crop = int(16000 * crop_seconds)
    print(f"scoring {len(rows)} utts (batch {batch_size})")

    from training.hf_audio import HFAudioStore, is_ref

    def load(path):
        if is_ref(path):
            w, sr = HFAudioStore.load(path)
        else:
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

    out, bad = [], 0
    for b in range(0, len(rows), batch_size):
        keep, wavs = [], []
        for s in rows[b:b + batch_size]:
            try:
                wavs.append(load(s.path))
                keep.append(s)
            except Exception as exc:  # noqa: BLE001 - one unreadable clip must not kill the eval
                bad += 1
                print(f"  skip {s.utt_id} ({s.dataset}): {type(exc).__name__}: {exc}", flush=True)
        if not keep:
            continue
        batch = torch.from_numpy(np.stack(wavs)).to(device)
        probs = det.fake_prob(batch).cpu().numpy()
        out += list(zip(keep, (float(p) for p in probs)))
        del batch
        if (b // batch_size) % 25 == 0:
            import gc
            gc.collect()
            print(f"  {len(out)}/{len(rows)}", flush=True)
    if bad:
        print(f"\n{bad} clips skipped (unreadable)")
    return out


def _eer_row(pairs):
    from training.metrics import compute_eer, compute_min_tdcf

    bona = np.array([p for s, p in pairs if s.label == 0])
    spoof = np.array([p for s, p in pairs if s.label == 1])
    eer, _ = compute_eer(bona, spoof)
    return eer, compute_min_tdcf(bona, spoof), len(spoof), len(bona)


def report(scored, by: list[str], md_path: str | None = None) -> None:
    lines = []

    def emit(name, pairs):
        bona = np.array([p for s, p in pairs if s.label == 0])
        spoof = np.array([p for s, p in pairs if s.label == 1])
        if len(bona) and len(spoof):
            eer, tdcf, ns, nb = _eer_row(pairs)
            eer_s, tdcf_s = f"{eer*100:.2f}%", f"{tdcf:.4f}"
            print(f"  {name:28s}  EER={eer*100:6.2f}%   min-tDCF={tdcf:.4f}   (n={len(pairs)}, {ns}s/{nb}b)")
        else:
            # single-class slice (genuine-only or fake-only domain): EER undefined; report the
            # mean P(spoof) and the detection rate at a fixed 0.5 threshold instead.
            eer_s = tdcf_s = "—"
            only = spoof if len(spoof) else bona
            kind = "spoof" if len(spoof) else "bona"
            det_rate = float((only > 0.5).mean()) if kind == "spoof" else float((only <= 0.5).mean())
            ns, nb = len(spoof), len(bona)
            print(f"  {name:28s}  {kind}-only  mean P(spoof)={only.mean():.3f}   "
                  f"acc@0.5={det_rate*100:.1f}%   (n={len(pairs)})")
            lines.append((name, eer_s, tdcf_s, ns, nb, f"mean {only.mean():.3f}, acc@0.5 {det_rate*100:.0f}%"))
            return
        lines.append((name, eer_s, tdcf_s, ns, nb, ""))

    print("\npooled:")
    emit("ALL", scored)
    for col in by:
        print(f"\nby {col}:")
        groups: dict[str, list] = defaultdict(list)
        for s, p in scored:
            groups[getattr(s, col, "?")].append((s, p))
        for k in sorted(groups):
            emit(f"{col}={k}", groups[k])

    if md_path:
        md = ["| slice | EER | min-tDCF | spoof | bona | note |", "|---|---|---|---|---|---|"]
        md += [f"| {n} | {e} | {t} | {s} | {b} | {note} |" for n, e, t, s, b, note in lines]
        Path(md_path).write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nwrote {md_path}")


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
    ap.add_argument("--crop-seconds", type=float, default=6.0,
                    help="fixed segment length scored per utt (pad/truncate)")
    ap.add_argument("--dump", default=None, help="write per-utt scores TSV here")
    ap.add_argument("--md", default=None, help="write the results table as markdown here")
    args = ap.parse_args()

    scored = score_manifest(args.checkpoint, args.manifest, args.device, args.limit,
                            per_domain=args.per_domain or None, batch_size=args.batch_size,
                            crop_seconds=args.crop_seconds)
    if args.dump:
        Path(args.dump).write_text(
            "utt_id\tlabel\tdataset\tlanguage\tfake_prob\n"
            + "\n".join(f"{s.utt_id}\t{s.label}\t{s.dataset}\t{s.language}\t{p:.6f}"
                        for s, p in scored),
            encoding="utf-8",
        )
        print(f"wrote {args.dump}")
    report(scored, [c.strip() for c in args.by.split(",") if c.strip()], md_path=args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
