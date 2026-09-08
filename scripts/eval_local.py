"""Run after downloading a trained checkpoint into backend/models/aasist_indicw2v.pt.

    python scripts/eval_local.py

- prints the checkpoint's metadata
- evaluates it on the full local eval manifest (ASVspoof 2019 + 2021 LA + In-the-Wild +
  Indic), sliced by dataset and language, capped per domain so it finishes in a few minutes
  on CPU
- writes the results table to docs/RESULTS.md
- checks the serving pipeline can load it
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/eval_local.py` (not just -m)
CKPT = ROOT / "backend" / "models" / "aasist_indicw2v.pt"
EVAL = ROOT / "data" / "manifests" / "eval.tsv"


def main() -> int:
    if not CKPT.exists():
        print(f"no checkpoint at {CKPT}\n"
              f"download aasist_indicw2v.pt from the Kaggle run's Output tab and put it there.")
        return 2
    if not EVAL.exists():
        print(f"no eval manifest — run: python scripts/prepare_manifests.py")
        return 2

    import torch

    c = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"checkpoint: dev_EER={c.get('dev_eer')} epoch={c.get('epoch')} "
          f"frontend={c.get('frontend_model_id')} finetuned={c.get('frontend') is not None}\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = dev == "cuda"
    subprocess.run(
        [sys.executable, "-m", "training.evaluate",
         "--checkpoint", str(CKPT), "--manifest", str(EVAL),
         "--by", "dataset,language", "--device", dev,
         # XLS-R on CPU is ~3-4 s/clip and memory-tight on a 16 GB box: a balanced
         # 400/domain subset on 4 s segments at batch 2 is a stable EER estimate that
         # fits in RAM and finishes in ~1.5 h instead of ~20 h.
         "--per-domain", "6000" if gpu else "400",
         "--crop-seconds", "6.0" if gpu else "4.0",
         "--batch-size", "32" if gpu else "2",
         "--md", str(ROOT / "docs" / "RESULTS.md"),
         "--dump", str(ROOT / "data" / "eval_scores.tsv")],
        check=True, cwd=ROOT,
    )

    print("\n--- serving pipeline load check ---")
    from backend.core import get_config
    from backend.inference import DetectionPipeline

    pipe = DetectionPipeline.from_config(get_config())
    r = pipe.infer_chunk(torch.randn(16000))
    print(f"infer_chunk OK — fake_prob={r.fake_prob:.4f}, latency={r.latency_ms:.0f} ms")
    print("\nStart the demo:  uvicorn backend.main:app   then open http://localhost:8000/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
