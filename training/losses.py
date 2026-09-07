"""Losses for spoof detection.

  * ``weighted_ce``  — cross-entropy on the 2 logits with a class weight for the
    bonafide:spoof imbalance. Simple, strong baseline for ASVspoof / AASIST.
  * ``oc_softmax``   — one-class softmax (Zhang et al. 2021): pulls bonafide embeddings
    toward a learned centre, pushes spoof past a margin. Better generalisation to unseen
    attacks. Operates on the pre-head embedding, so the classifier is used with
    ``return_embedding=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCE(nn.Module):
    uses_embedding = False

    def __init__(self, bonafide_weight: float = 1.0, spoof_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("weight", torch.tensor([bonafide_weight, spoof_weight]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight.to(logits.dtype))


class OCSoftmax(nn.Module):
    """One-class softmax. ``embedding`` -> scalar loss; score = cos sim to the centre."""

    uses_embedding = True

    def __init__(self, feat_dim: int, m_real: float = 0.9, m_fake: float = 0.2, alpha: float = 20.0) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.randn(1, feat_dim))
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha

    def forward(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.center, dim=1)
        x = F.normalize(embedding, dim=1)
        cos = x @ w.t()  # (B, 1)
        cos = cos.squeeze(1)
        margins = torch.where(target == 0, self.m_real - cos, cos - self.m_fake)
        return F.softplus(self.alpha * margins).mean()

    @torch.no_grad()
    def score(self, embedding: torch.Tensor) -> torch.Tensor:
        """Higher = more bonafide. fake_prob = 1 - sigmoid(score) style handled by caller."""
        w = F.normalize(self.center, dim=1)
        x = F.normalize(embedding, dim=1)
        return (x @ w.t()).squeeze(1)


def make_loss(cfg: dict, feat_dim: int) -> nn.Module:
    name = cfg.get("name", "weighted_ce")
    if name == "weighted_ce":
        return WeightedCE(cfg.get("bonafide_weight", 1.0), cfg.get("spoof_weight", 1.0))
    if name == "oc_softmax":
        return OCSoftmax(feat_dim)
    raise ValueError(f"unknown loss: {name}")
