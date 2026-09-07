"""Benchmark SSL frontend + AASIST latency for the *streaming* case (1 s chunks).

    python scripts/bench_frontends.py

Loads one config at a time, warms up, then times infer_chunk on a 1 s / 16 kHz chunk.
Target: median latency < hop_seconds (0.5 s) to keep up without dropping, or < chunk_seconds
(1.0 s) with the drop-stale-window strategy.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core import load_config  # noqa: E402
from backend.inference import DetectionPipeline  # noqa: E402

CONFIGS = [
    ("xls-r-300m  L-1 (24 layers)", "facebook/wav2vec2-xls-r-300m", -1, 1024, False),
    ("wav2vec2-base  L-1 (12)", "facebook/wav2vec2-base", -1, 768, False),
    ("wav2vec2-base  L7 (trunc)", "facebook/wav2vec2-base", 7, 768, True),
    ("wav2vec2-base  L5 (trunc)", "facebook/wav2vec2-base", 5, 768, True),
]

N_ITERS = 15


def _truncate(model, keep_layers: int) -> None:
    enc = model.encoder.layers
    if keep_layers < len(enc):
        model.encoder.layers = torch.nn.ModuleList(list(enc[:keep_layers]))
        model.config.num_hidden_layers = keep_layers


def bench(label, model_id, layer, feat_dim, truncate) -> None:
    torch.manual_seed(0)
    cfg = load_config()
    cfg.feature_extractor.backend = "wav2vec2"
    cfg.feature_extractor.model_id = model_id
    cfg.feature_extractor.layer = layer if not truncate else -1
    cfg.feature_extractor.feat_dim = feat_dim
    cfg.classifier.device = "cpu"

    t0 = time.perf_counter()
    try:
        pipe = DetectionPipeline.from_config(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"{label:<32} LOAD FAILED: {repr(e)[:120]}")
        return
    if truncate:
        _truncate(pipe.extractor.model, layer)
    load_s = time.perf_counter() - t0

    chunk = torch.randn(16000)
    for _ in range(3):  # warmup
        pipe.infer_chunk(chunk)

    times = []
    for _ in range(N_ITERS):
        t = time.perf_counter()
        pipe.infer_chunk(chunk)
        times.append((time.perf_counter() - t) * 1000)

    med = statistics.median(times)
    p90 = sorted(times)[int(0.9 * len(times)) - 1]
    verdict = "OK <0.5s" if med < 500 else ("OK <1.0s" if med < 1000 else "TOO SLOW")
    print(f"{label:<32} load={load_s:5.1f}s  median={med:7.1f}ms  p90={p90:7.1f}ms   {verdict}")

    del pipe
    gc.collect()


def main() -> int:
    print(f"threads: torch={torch.get_num_threads()}  |  1 s chunk @ 16 kHz  |  {N_ITERS} iters\n")
    for c in CONFIGS:
        bench(*c)
    print("\n(quantization + ONNX not tested here — try after picking a base model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
