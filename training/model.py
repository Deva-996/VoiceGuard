"""End-to-end detector: waveform -> SSL frontend -> AASIST -> logits.

Used for GPU training where the frontend is fine-tuned (the big quality lever). The frontend
can be frozen for a warm-up phase and unfrozen later (``set_frontend_trainable``).

For frozen-only / CPU training the fast path is still cached features + ``AASISTClassifier``
directly (see ``training/train.py``); this module is the joint-optimisation path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backend.inference.classifier import AASISTClassifier


class EndToEndDetector(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xls-r-300m",
        layer: int = -1,
        embed_dim: int = 256,
        num_classes: int = 2,
        frontend_trainable: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        self._fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.sampling_rate = int(getattr(self._fe, "sampling_rate", 16000))
        self.frontend = AutoModel.from_pretrained(model_id)
        self.model_id = model_id
        self.layer = layer
        feat_dim = int(self.frontend.config.hidden_size)
        self.classifier = AASISTClassifier(
            feat_dim=feat_dim, embed_dim=embed_dim, num_classes=num_classes
        )
        self.feat_dim = feat_dim
        self.set_frontend_trainable(frontend_trainable)

    def set_frontend_trainable(self, flag: bool) -> None:
        self.frontend_trainable = flag
        for p in self.frontend.parameters():
            p.requires_grad_(flag)
        self.frontend.train(flag)

    def _normalize(self, wav_batch: torch.Tensor) -> torch.Tensor:
        # wav_batch: (B, T) float in [-1, 1] at 16 kHz. wav2vec2 wants zero-mean unit-var.
        if getattr(self._fe, "do_normalize", True):
            m = wav_batch.mean(dim=1, keepdim=True)
            s = wav_batch.std(dim=1, keepdim=True) + 1e-7
            wav_batch = (wav_batch - m) / s
        return wav_batch

    def forward(self, wav_batch: torch.Tensor, return_embedding: bool = False):
        x = self._normalize(wav_batch)
        ctx = torch.enable_grad() if self.frontend_trainable else torch.no_grad()
        with ctx:
            out = self.frontend(x, output_hidden_states=True)
            feats = out.hidden_states[self.layer]  # (B, T', H)
        if not self.frontend_trainable:
            feats = feats.detach()
        return self.classifier(feats, return_embedding=return_embedding)

    # --- checkpoints: same shape the live pipeline (DetectionPipeline.from_config) loads ---
    def state_for_checkpoint(self, cpu: bool = True) -> dict:
        def move(sd):
            return {k: v.cpu() for k, v in sd.items()} if (sd and cpu) else sd

        return {
            "model": move(self.classifier.state_dict()),       # AASIST backend + head
            "frontend": move(self.frontend.state_dict()) if self.frontend_trainable else None,
            "frontend_model_id": self.model_id,
            "layer": self.layer,
            "feat_dim": self.feat_dim,
        }
