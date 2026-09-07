"""Read audio straight from HuggingFace parquet shards — no per-utterance file materialisation.

ASVspoof 2019 LA arrives as ~7.5 GB of parquet (audio bytes inline). Materialising 121k FLAC
files costs ~5 GB of disk and ~10 min of I/O; instead we memory-map the parquet with
``datasets`` and index into it at load time (low RAM, fast random access).

Manifest rows that live in parquet carry a ``path`` of the form

    hf::<parquet_dir>::<split>::<row_index>

``ManifestDataset`` recognises the ``hf::`` prefix and routes through ``HFAudioStore``.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

HF_PREFIX = "hf::"


def make_ref(parquet_dir: str | Path, split: str, idx: int) -> str:
    return f"{HF_PREFIX}{parquet_dir}::{split}::{idx}"


def is_ref(path: str) -> bool:
    return path.startswith(HF_PREFIX)


def parse_ref(path: str) -> tuple[str, str, int]:
    body = path[len(HF_PREFIX):]
    pdir, split, idx = body.rsplit("::", 2)
    return pdir, split, int(idx)


@lru_cache(maxsize=8)
def _load_split(parquet_dir: str, split: str):
    from datasets import Audio, load_dataset

    files = sorted(str(p) for p in Path(parquet_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet in {parquet_dir}")
    ds = load_dataset("parquet", data_files={split: files}, split=split)
    return ds.cast_column("audio", Audio(decode=False))  # raw bytes, decode with soundfile


class HFAudioStore:
    """Resolve ``hf::`` manifest refs to ``(waveform_float32_mono, sr)``."""

    @staticmethod
    def load(ref: str) -> tuple[np.ndarray, int]:
        pdir, split, idx = parse_ref(ref)
        ex = _load_split(pdir, split)[idx]
        a = ex["audio"]
        raw = a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes()
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) == 2:
            wav = wav.mean(axis=1)
        return np.asarray(wav, dtype="float32"), int(sr)
