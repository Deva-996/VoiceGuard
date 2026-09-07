"""Train the wav2vec2 -> AASIST spoof detector.

    python -m training.train --config training/config_train.yaml
    python -m training.train --config training/config_train.yaml --limit 400   # smoke run

Baseline path (CPU-viable):
  1. frontend frozen -> features for train/dev are extracted once into a disk cache
     (training.features);
  2. the AASIST head + backend train off the cache;
  3. best dev-EER checkpoint is written to backend/models/aasist_indicw2v.pt, which the live
     pipeline picks up automatically (classifier.checkpoint in config/config.yaml).

Stage-2 (unfreeze the frontend, lr*0.1) needs a GPU and is skipped with a warning on CPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


def _device(cfg: dict) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


def _loaders(cfg: dict, cache, limit: int | None):
    from training.dataset import FeatureDataset, collate_features

    mans = cfg["manifests"]
    max_frames = int(cfg.get("max_frames", 400))
    dev_limit = max(limit // 4, 8) if limit else None
    train_ds = FeatureDataset(mans["train"], cache, max_frames=max_frames, train=True, limit=limit)
    dev_ds = FeatureDataset(mans["dev"], cache, max_frames=max_frames, train=False, limit=dev_limit)
    bs = int(cfg.get("batch_size", 24))
    nw = int(cfg.get("num_workers", 0))
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_features),
        DataLoader(dev_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_features),
    )


@torch.no_grad()
def evaluate_dev(model, loss_fn, loader, device) -> tuple[float, float]:
    from training.metrics import compute_eer

    model.eval()
    bona, spoof = [], []
    for feats, labels, _ in loader:
        feats = feats.to(device)
        if getattr(loss_fn, "uses_embedding", False):
            _, emb = model(feats, return_embedding=True)
            score = -loss_fn.score(emb).cpu().numpy()  # higher = more spoof
        else:
            logits = model(feats)
            score = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        for s, y in zip(score, labels.numpy()):
            (spoof if y == 1 else bona).append(float(s))
    eer, _ = compute_eer(np.array(bona), np.array(spoof))
    return eer, len(bona) + len(spoof)


def train_one_epoch(model, loss_fn, loader, opt, device) -> float:
    model.train()
    total, n = 0.0, 0
    for feats, labels, _ in loader:
        feats, labels = feats.to(device), labels.to(device)
        if getattr(loss_fn, "uses_embedding", False):
            logits, emb = model(feats, return_embedding=True)
            loss = loss_fn(emb, labels)
        else:
            loss = loss_fn(model(feats), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(labels)
        n += len(labels)
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "training" / "config_train.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="cap utts (smoke test)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(cfg.get("seed", 42))
    device = _device(cfg)
    fe = cfg["frontend"]
    print(f"device={device}  frontend={fe['model_id']}  layer={fe['layer']}")

    # 1. feature cache (frozen frontend)
    from training.features import build_cache

    cache = build_cache(cfg["manifests"]["train"], cfg, limit=args.limit)
    build_cache(cfg["manifests"]["dev"], cfg, limit=(args.limit // 4 if args.limit else None))

    # 2. model + loss + opt
    from backend.inference.classifier import AASISTClassifier
    from training.losses import make_loss

    feat_dim = int(np.load(next(cache.root.glob("*.npy"))).shape[1])
    model = AASISTClassifier(
        feat_dim=feat_dim,
        embed_dim=cfg["classifier"]["embed_dim"],
        num_classes=cfg["classifier"]["num_classes"],
    ).to(device)
    loss_fn = make_loss(cfg.get("loss", {}), feat_dim=cfg["classifier"]["embed_dim"]).to(device)

    o = cfg.get("optimizer", {})
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.Adam(params, lr=o.get("lr", 1e-4), weight_decay=o.get("weight_decay", 1e-4))

    train_dl, dev_dl = _loaders(cfg, cache, args.limit)
    print(f"train={len(train_dl.dataset)}  dev={len(dev_dl.dataset)}  feat_dim={feat_dim}")

    if cfg.get("frontend", {}).get("stage2_unfreeze_epoch") and device.type != "cuda":
        print("!! stage-2 frontend fine-tuning needs a GPU — running frozen-frontend baseline only.")

    # 3. train
    epochs = args.epochs or int(cfg.get("epochs", 30))
    out = Path(cfg["checkpoint"]["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    best_eer = float("inf")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, loss_fn, train_dl, opt, device)
        eer, n_dev = evaluate_dev(model, loss_fn, dev_dl, device)
        flag = ""
        if eer < best_eer:
            best_eer = eer
            torch.save(
                {
                    "model": model.state_dict(),
                    "loss_state": loss_fn.state_dict(),
                    "config": cfg,
                    "feat_dim": feat_dim,
                    "dev_eer": eer,
                    "epoch": ep,
                },
                out,
            )
            flag = "  <- best, saved"
        print(
            f"epoch {ep:2d}/{epochs}  loss={tr_loss:.4f}  dev_EER={eer*100:.2f}%  "
            f"({n_dev} dev, {time.time()-t0:.0f}s){flag}"
        )

    print(f"\nbest dev EER {best_eer*100:.2f}%  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
