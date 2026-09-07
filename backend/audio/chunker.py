"""Audio framing for the streaming path.  [Day 2]

The receiver browser tab sends raw PCM frames of arbitrary size over the WebSocket. The
backend must reassemble them into fixed analysis windows (``chunk_seconds`` long, advancing
by ``hop_seconds``) before inference.

``AudioChunker`` is a stateful accumulator: ``push(pcm_bytes_or_array)`` -> yields any
complete windows now available as float32 tensors at 16 kHz mono in [-1, 1].

Day 2 scope: PCM decode, resample, ring buffer, overlap. Day 5 wires this into the WS route.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

TARGET_SR = 16000


def decode_pcm(data: bytes | np.ndarray, pcm_format: str = "int16") -> np.ndarray:
    """Raw wire bytes -> float32 mono array in [-1, 1]."""
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float32)
    elif pcm_format == "int16":
        usable = len(data) - (len(data) % 2)  # drop a stray trailing byte
        arr = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
    elif pcm_format == "float32":
        usable = len(data) - (len(data) % 4)
        arr = np.frombuffer(data[:usable], dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported pcm_format: {pcm_format!r}")
    return arr


def resample_to_16k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    """Linear resample to 16 kHz. Good enough for a mic feed; swap for a polyphase filter
    (``torchaudio.functional.resample``) if artefacts matter."""
    if orig_sr == TARGET_SR:
        return wav.astype(np.float32)
    duration = wav.shape[-1] / float(orig_sr)
    n_out = int(round(duration * TARGET_SR))
    if n_out <= 1:
        return wav.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=wav.shape[-1], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


class AudioChunker:
    def __init__(
        self,
        sample_rate: int = TARGET_SR,
        chunk_seconds: float = 1.0,
        hop_seconds: float = 0.5,
        input_sample_rate: int | None = None,
        pcm_format: str = "int16",
    ) -> None:
        self.sample_rate = sample_rate
        self.pcm_format = pcm_format
        self.input_sample_rate = input_sample_rate or sample_rate
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.hop_samples = int(sample_rate * hop_seconds)
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, data: bytes | np.ndarray) -> Iterator[torch.Tensor]:
        wav = decode_pcm(data, self.pcm_format)
        wav = resample_to_16k(wav, self.input_sample_rate)
        self._buf = np.concatenate([self._buf, wav])
        while self._buf.shape[0] >= self.chunk_samples:
            window = self._buf[: self.chunk_samples].copy()
            self._buf = self._buf[self.hop_samples :]
            yield torch.from_numpy(window)

    def flush(self) -> torch.Tensor | None:
        """Return whatever remains, zero-padded to one window (call on stream close)."""
        if self._buf.size == 0:
            return None
        window = np.zeros(self.chunk_samples, dtype=np.float32)
        window[: min(self._buf.size, self.chunk_samples)] = self._buf[: self.chunk_samples]
        self._buf = np.zeros(0, dtype=np.float32)
        return torch.from_numpy(window)
