"""Datasets for spoof-detection training.

Two views over the same manifests (TSV from scripts/prepare_manifests.py, columns:
``utt_id  path  label  language  source  dataset``):

  * ``ManifestDataset``  -> ``(waveform_16k, label, utt_id)`` — raw audio, used to build the
    feature cache and for stage-2 fine-tuning (frontend in the loop).
  * ``FeatureDataset``   -> ``(features[T,D], label, utt_id)`` — pre-extracted SSL features
    from ``training.features.FeatureCache``; this is the fast, CPU-viable training path with
    the frontend frozen.

``label``: 0 = bonafide, 1 = spoof  (model output ``fake_prob`` = P(label==1)).
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"bonafide": 0, "bona-fide": 0, "bona fide": 0, "genuine": 0, "spoof": 1, "fake": 1}


@dataclass
class Sample:
    utt_id: str
    path: str
    label: int
    language: str
    source: str
    dataset: str


def read_manifest(path: str | Path) -> list[Sample]:
    rows: list[Sample] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(
                Sample(
                    utt_id=r["utt_id"],
                    path=r["path"],
                    label=LABEL_MAP[r["label"].strip().lower()],
                    language=r.get("language", "und"),
                    source=r.get("source", "unknown"),
                    dataset=r.get("dataset", "unknown"),
                )
            )
    if not rows:
        raise ValueError(f"no rows in manifest: {path}")
    return rows


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        sample_rate: int = 16000,
        crop_seconds: float = 4.0,
        train: bool = True,
        augment=None,
        limit: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.crop_samples = int(sample_rate * crop_seconds)
        self.train = train
        self.augment = augment  # callable(np.ndarray)->np.ndarray, applied when train=True
        self.samples = read_manifest(manifest)
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, wav: np.ndarray) -> np.ndarray:
        n = wav.shape[0]
        if n >= self.crop_samples:
            start = random.randint(0, n - self.crop_samples) if self.train else 0
            return wav[start : start + self.crop_samples]
        # tile-pad short clips (ASVspoof convention) rather than zero-pad
        reps = int(np.ceil(self.crop_samples / max(n, 1)))
        return np.tile(wav, reps)[: self.crop_samples]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        s = self.samples[idx]
        wav, sr = sf.read(s.path, dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) == 2:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            from backend.audio import resample_to_16k

            wav = resample_to_16k(np.asarray(wav), sr)
        wav = self._crop_or_pad(np.asarray(wav, dtype="float32"))
        if self.train and self.augment is not None:
            wav = np.asarray(self.augment(wav), dtype="float32")
            wav = self._crop_or_pad(wav)  # augment may change length
        return torch.from_numpy(np.ascontiguousarray(wav)), s.label, s.utt_id


class FeatureDataset(Dataset):
    """Reads cached SSL features. ``cache`` is a ``training.features.FeatureCache``."""

    def __init__(
        self,
        manifest: str | Path,
        cache,
        max_frames: int = 400,
        train: bool = True,
        limit: int | None = None,
    ) -> None:
        self.samples = read_manifest(manifest)
        if limit:
            self.samples = self.samples[:limit]
        self.cache = cache
        self.max_frames = max_frames
        self.train = train
        missing = [s.utt_id for s in self.samples if not cache.has(s.utt_id)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} utts not in feature cache (e.g. {missing[:3]}). "
                "Run training.features first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _fit(self, feats: np.ndarray) -> np.ndarray:
        t = feats.shape[0]
        if t == self.max_frames:
            return feats
        if t > self.max_frames:
            start = random.randint(0, t - self.max_frames) if self.train else 0
            return feats[start : start + self.max_frames]
        reps = int(np.ceil(self.max_frames / max(t, 1)))
        return np.tile(feats, (reps, 1))[: self.max_frames]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        s = self.samples[idx]
        feats = self._fit(self.cache.load(s.utt_id))
        return torch.from_numpy(np.ascontiguousarray(feats)), s.label, s.utt_id


def collate_features(batch):
    feats, labels, uids = zip(*batch)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long), list(uids)


def collate_waveforms(batch):
    wavs, labels, uids = zip(*batch)
    return torch.stack(wavs), torch.tensor(labels, dtype=torch.long), list(uids)
