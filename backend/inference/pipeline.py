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
                try:  # mmap keeps the ~1.2 GB of weights off the heap until read through
                    loaded = torch.load(ckpt_path, map_location="cpu",
                                        weights_only=False, mmap=True)
                except (TypeError, RuntimeError):
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

        if bundle.get("model"):
            # self-describing bundle: build the head to the trained dims, load it
            embed_dim = bundle.get("embed_dim", cfg.classifier.embed_dim)
            classifier = AASISTClassifier(
                feat_dim=extractor.feat_dim, embed_dim=embed_dim,
                num_classes=bundle.get("num_classes", cfg.classifier.num_classes),
            )
            missing, unexpected = classifier.load_state_dict(bundle["model"], strict=False)
            if missing or unexpected:
                print(f"[DetectionPipeline] head load: {len(missing)} missing, {len(unexpected)} unexpected")

            # OC-Softmax checkpoint: the logit head is untrained; score via the centre instead.
            oc = bundle.get("oc_softmax")
            if oc:
                st = oc.get("state", oc)
                center = st["center"] if isinstance(st, dict) and "center" in st else st
                classifier.set_oc_softmax(
                    center,
                    m_real=oc.get("m_real", 0.9),
                    m_fake=oc.get("m_fake", 0.2),
                    alpha=oc.get("alpha", 20.0),
                )
                print("[DetectionPipeline] OC-Softmax scoring (centre distance)")
            classifier.to(cfg.classifier.device).eval()
        else:
            classifier = AASISTClassifier.from_config(cfg.classifier, feat_dim=extractor.feat_dim)

        bundle = loaded = None  # release the checkpoint dict (~1.2 GB) before serving
        import gc
        gc.collect()

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
