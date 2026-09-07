"""End-to-end detection pipeline: waveform chunk -> fake_prob.

    waveform (16 kHz mono) ->  feature extractor  ->  AASIST classifier  ->  fake_prob

Used by the WebSocket handler (one pipeline per connection is fine; it is stateless) and by
offline scripts / evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .classifier import AASISTClassifier
from .feature_extractor import BaseFeatureExtractor, build_feature_extractor


@dataclass
class ChunkResult:
    fake_prob: float
    n_samples: int
    n_frames: int
    latency_ms: float
    index: int = 0
    extra: dict = field(default_factory=dict)


class DetectionPipeline:
    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        classifier: AASISTClassifier,
        min_samples: int = 16000,
        device: str = "cpu",
    ) -> None:
        self.extractor = extractor
        self.classifier = classifier
        self.min_samples = min_samples
        self.device = torch.device(device)
        self._counter = 0

    @classmethod
    def from_config(cls, cfg) -> "DetectionPipeline":
        import copy

        # A training checkpoint may bundle fine-tuned frontend weights + the frontend id it
        # was trained with (training/model.py). Honour those over config so serving == training.
        bundle = {}
        ckpt_path = cfg.classifier.checkpoint_path
        if ckpt_path.exists():
            try:
                loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict):
                    bundle = loaded
            except Exception as exc:  # noqa: BLE001
                print(f"[DetectionPipeline] could not read checkpoint {ckpt_path}: {exc}")

        fe_cfg = cfg.feature_extractor
        if bundle.get("frontend_model_id"):
            fe_cfg = copy.copy(fe_cfg)
            fe_cfg.backend = "wav2vec2"
            fe_cfg.model_id = bundle["frontend_model_id"]
            fe_cfg.layer = bundle.get("layer", fe_cfg.layer)

        extractor = build_feature_extractor(fe_cfg, finetuned_state=bundle.get("frontend"))
        classifier = AASISTClassifier.from_config(cfg.classifier, feat_dim=extractor.feat_dim)
        return cls(
            extractor=extractor,
            classifier=classifier,
            min_samples=cfg.audio.chunk_samples,
            device=cfg.classifier.device,
        )

    def infer_chunk(self, waveform: torch.Tensor) -> ChunkResult:
        t0 = time.perf_counter()
        wav = torch.as_tensor(waveform, dtype=torch.float32).reshape(-1)
        n_samples = wav.numel()
        if n_samples < self.min_samples:
            wav = torch.nn.functional.pad(wav, (0, self.min_samples - n_samples))

        features = self.extractor.extract(wav)
        prob = float(self.classifier.fake_prob(features.to(self.device)))

        self._counter += 1
        return ChunkResult(
            fake_prob=prob,
            n_samples=n_samples,
            n_frames=int(features.shape[0]),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            index=self._counter,
        )
