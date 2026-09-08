"""AASIST classifier head for VoiceGuard.

Wraps the vendored ``SSL_BACKEND_aasist`` (spectro-temporal graph-attention backend) with a
linear classification head producing ``[bonafide, spoof]`` logits. ``fake_prob`` is the
softmax probability of the ``spoof`` class.

If the model was trained with **OC-Softmax** (an embedding-space loss), the 2-logit head is
never trained — scoring must instead use cosine distance from the pre-head embedding to the
learned OC-Softmax centre. ``set_oc_softmax()`` switches ``fake_prob`` onto that path; the
serving pipeline calls it when the checkpoint carries an ``oc_softmax`` block.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self._oc_center: torch.Tensor | None = None  # (1, embed_dim), L2-normalised
        self._oc_mid = 0.55        # midpoint of the OC-Softmax margins (m_real+m_fake)/2
        self._oc_slope = 8.0       # cos -> P(spoof) sigmoid steepness (display/threshold calib,
                                   # deliberately gentler than the loss's gradient-shaping alpha)

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

    def set_oc_softmax(self, center: torch.Tensor, m_real: float = 0.9,
                       m_fake: float = 0.2, alpha: float = 20.0) -> None:
        """Score via distance to the OC-Softmax centre instead of the (untrained) logit head.
        ``center``: the learned centre vector, shape (embed_dim,) or (1, embed_dim).
        ``alpha`` is accepted for call-site symmetry but not used for the probability map."""
        c = F.normalize(torch.as_tensor(center, dtype=torch.float32).reshape(1, -1), dim=1)
        self._oc_center = c
        self._oc_mid = (float(m_real) + float(m_fake)) / 2.0

    def _oc_fake_prob(self, features: torch.Tensor) -> torch.Tensor:
        _, emb = self.forward(features, return_embedding=True)   # (B, embed_dim)
        cos = F.normalize(emb, dim=1) @ self._oc_center.to(emb.device).t()  # (B, 1)
        # cos high => bonafide; map to P(spoof), monotone decreasing in cos
        return torch.sigmoid(self._oc_slope * (self._oc_mid - cos)).squeeze(-1)

    @torch.inference_mode()
    def fake_prob(self, features: torch.Tensor) -> torch.Tensor:
        """features: (T, feat_dim) or (B, T, feat_dim) -> P(spoof), shape () or (B,)."""
        was_unbatched = features.ndim == 2
        if self._oc_center is not None:
            probs = self._oc_fake_prob(features)
        else:
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
