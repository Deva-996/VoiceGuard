"""AASIST classifier head for VoiceGuard.

Wraps the vendored ``SSL_BACKEND_aasist`` (spectro-temporal graph-attention backend) with a
linear classification head producing ``[bonafide, spoof]`` logits. ``fake_prob`` is the
softmax probability of the ``spoof`` class.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .aasist import SSL_BACKEND_aasist

SPOOF_INDEX = 1  # class order: [bonafide, spoof]


class AASISTClassifier(nn.Module):
    def __init__(self, feat_dim: int = 1024, embed_dim: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.backend = SSL_BACKEND_aasist(feat_dim=feat_dim, embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, features: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """features: (B, T, feat_dim) -> logits (B, num_classes); optionally also the
        pre-head embedding (B, embed_dim) for embedding-space losses (OC-Softmax)."""
        if features.ndim == 2:
            features = features.unsqueeze(0)
        embedding = self.backend(features)
        logits = self.head(embedding)
        return (logits, embedding) if return_embedding else logits

    @torch.inference_mode()
    def fake_prob(self, features: torch.Tensor) -> torch.Tensor:
        """features: (T, feat_dim) or (B, T, feat_dim) -> P(spoof), shape () or (B,)."""
        was_unbatched = features.ndim == 2
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)[..., SPOOF_INDEX]
        return probs.squeeze(0) if was_unbatched else probs

    # ------------------------------------------------------------------ checkpoints
    def load_checkpoint(self, path: str | Path, strict: bool = False) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            print(f"[AASISTClassifier] {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[AASISTClassifier] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    @classmethod
    def from_config(cls, classifier_cfg, feat_dim: int) -> "AASISTClassifier":
        model = cls(
            feat_dim=feat_dim,
            embed_dim=classifier_cfg.embed_dim,
            num_classes=classifier_cfg.num_classes,
        )
        ckpt = classifier_cfg.checkpoint_path
        if ckpt.exists():
            model.load_checkpoint(ckpt)
        else:
            print(
                f"[AASISTClassifier] no checkpoint at {ckpt} - using randomly initialised "
                "weights (expected until training produces one)."
            )
        model.to(classifier_cfg.device).eval()
        return model
