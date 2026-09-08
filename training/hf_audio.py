"""Read audio straight from HuggingFace parquet shards — no per-utterance file materialisation
and no ``datasets`` Arrow cache.

ASVspoof 2019 LA arrives as ~7.5 GB of parquet (audio bytes inline). Materialising 121k FLAC
files costs ~5 GB of disk; routing through ``datasets.load_dataset`` is worse — because the
shards carry an HF ``Audio`` feature in their schema, generation *decodes every clip to
float32* and the Arrow cache blows past 20 GB on a small system drive.

Instead we memory-map the parquet with ``pyarrow`` and read a single row group at a time
(shards use 100-row groups), keeping an LRU of the most-recently-touched groups. Disk cost:
zero. RAM cost: a handful of row groups (~MB).

Manifest rows that live in parquet carry a ``path`` of the form

    hf::<parquet_dir>::<split>::<row_index>

``ManifestDataset`` recognises the ``hf::`` prefix and routes through ``HFAudioStore``.
"""

from __future__ import annotations

import bisect
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


class _ParquetSplit:
    """Random-access reader over one split's parquet shards, one row group at a time."""

    _CACHE = 24  # row groups kept resident (~a few MB each)

    def __init__(self, files: list[str]) -> None:
        import pyarrow.parquet as pq

        self._pf = [pq.ParquetFile(f, memory_map=True) for f in files]
        # flat list of (file_index, row_group_index) with a cumulative first-row offset
        self._groups: list[tuple[int, int]] = []
        self._starts: list[int] = []
        running = 0
        for fi, pf in enumerate(self._pf):
            for gi in range(pf.metadata.num_row_groups):
                self._starts.append(running)
                self._groups.append((fi, gi))
                running += pf.metadata.row_group(gi).num_rows
        self._n = running
        self._rg_cache: dict[int, list] = {}
        self._rg_order: list[int] = []

    def __len__(self) -> int:
        return self._n

    def _row_group_rows(self, g: int) -> list:
        hit = self._rg_cache.get(g)
        if hit is not None:
            return hit
        fi, gi = self._groups[g]
        tbl = self._pf[fi].read_row_group(gi, columns=["audio"])
        rows = tbl.column("audio").to_pylist()
        self._rg_cache[g] = rows
        self._rg_order.append(g)
        if len(self._rg_order) > self._CACHE:
            self._rg_cache.pop(self._rg_order.pop(0), None)
        return rows

    def audio_bytes(self, idx: int) -> bytes:
        if not 0 <= idx < self._n:
            raise IndexError(f"row {idx} out of range (0..{self._n - 1})")
        g = bisect.bisect_right(self._starts, idx) - 1
        row = self._row_group_rows(g)[idx - self._starts[g]]
        raw = row.get("bytes")
        if raw:
            return raw
        return Path(row["path"]).read_bytes()


@lru_cache(maxsize=8)
def _load_split(parquet_dir: str, split: str) -> _ParquetSplit:
    files = sorted(str(p) for p in Path(parquet_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet in {parquet_dir}")
    return _ParquetSplit(files)


class HFAudioStore:
    """Resolve ``hf::`` manifest refs to ``(waveform_float32_mono, sr)``."""

    @staticmethod
    def load(ref: str) -> tuple[np.ndarray, int]:
        pdir, split, idx = parse_ref(ref)
        raw = _load_split(pdir, split).audio_bytes(idx)
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) == 2:
            wav = wav.mean(axis=1)
        return np.asarray(wav, dtype="float32"), int(sr)
