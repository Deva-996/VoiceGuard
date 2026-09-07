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


def sh(*cmd: str, ok_if=None) -> None:
    """Run a step. If it exits non-zero but ``ok_if()`` is true, warn and continue —
    HF streaming can crash on interpreter teardown *after* doing its job."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode
    if rc != 0:
        if ok_if and ok_if():
            print(f"!! step exited {rc} but its output is present — continuing")
            return
        raise SystemExit(f"step failed ({rc}): {' '.join(cmd)}")


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
    dev = cfg.get("device", "auto")
    if dev == "auto":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _have(*parts):
        return (ROOT.joinpath("data", "raw", *parts)).exists()

    if not args.skip_download:
        # ASVspoof2019 (train/dev/eval) + IndicTTS genuine. FLEURS / the big eval sets are
        # extra coverage — add "--only ...,fleurs,in_the_wild" locally when disk allows.
        sh("scripts/download_datasets.py", "--only", "asvspoof2019,indictts",
           ok_if=lambda: _have("asvspoof2019_LA", "data") and _have("indictts", "hindi"))
    if not args.skip_fakes:
        sh("scripts/generate_indian_fakes.py", "--n", str(args.fake_n), "--device", dev,
           ok_if=lambda: ROOT.joinpath("data", "generated", "indic_fake", "gu", "manifest.tsv").exists())

    sh("scripts/prepare_manifests.py")  # ASVspoof2019 stays as parquet, read in place

    train_cmd = ["-m", "training.train", "--config", args.config]
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]
    if args.limit:
        train_cmd += ["--limit", str(args.limit)]
    sh(*train_cmd)  # train.py builds the feature cache itself, then trains

    if ckpt.exists():
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
