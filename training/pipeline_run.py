"""One-shot: data -> fakes -> manifests -> feature cache -> train -> evaluate.

    python -m training.pipeline_run --config training/config_train.yaml

Designed for a GPU box / Colab (see notebooks/train_colab.ipynb). Each step is skippable and
resumable (feature cache and materialised audio are keyed by id), so a killed run can be
re-launched. Produces backend/models/aasist_indicw2v.pt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sh(*cmd: str) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable, *cmd], check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "training" / "config_train.yaml"))
    ap.add_argument("--fake-n", type=int, default=1200, help="MMS-TTS clips per language")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke run: cap utts")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-fakes", action="store_true")
    ap.add_argument("--eval-by", default="dataset,language")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpt = ROOT / cfg["checkpoint"]["out"]

    if not args.skip_download:
        # ASVspoof2019 (train/dev/eval) + Indic genuine; big eval sets optional
        sh("scripts/download_datasets.py", "--only", "asvspoof2019,indictts,fleurs")
    if not args.skip_fakes:
        sh("scripts/generate_indian_fakes.py", "--n", str(args.fake_n))

    sh("scripts/prepare_manifests.py")

    # the ASVspoof2019 audio is now materialised under data/processed/ — drop the ~7.5 GB of
    # source parquet so a persisted data dir (Kaggle /kaggle/working, 20 GB) stays under quota
    pq = ROOT / "data" / "raw" / "asvspoof2019_LA" / "data"
    if pq.exists() and (ROOT / "data" / "processed" / "asvspoof2019_LA").exists():
        for f in pq.glob("*.parquet"):
            f.unlink()
        print(f"cleaned {pq}/*.parquet")

    train_cmd = ["-m", "training.train", "--config", args.config]
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.limit:
        train_cmd += ["--limit", str(args.limit)]
    sh(*train_cmd)  # train.py builds the feature cache itself, then trains

    if ckpt.exists():
        dev = "cuda" if cfg.get("device") in (None, "auto", "cuda") else "cpu"
        sh(
            "-m", "training.evaluate",
            "--checkpoint", str(ckpt),
            "--manifest", str(ROOT / cfg["manifests"]["eval"]),
            "--by", args.eval_by,
            "--device", dev,
            "--per-domain", "6000",
            "--batch-size", "32" if dev == "cuda" else "8",
            "--dump", str(ckpt.with_suffix(".eval_scores.tsv")),
        )
    print(f"\nDONE. checkpoint -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
