"""Rolling risk-scoring engine.  [Day 4]

Consumes per-chunk ``fake_prob`` values and produces a smoothed risk score + level:

  * rolling **weighted** average over the last ``window`` (default 10) chunk scores,
    recent chunks weighted higher;
  * levels: ``LOW`` > 0.40, ``MEDIUM`` > 0.60, ``HIGH`` > 0.75 sustained for
    ``high_consecutive`` (default 3) consecutive chunks;
  * emits an ``alert`` flag exactly once on the transition into ``HIGH`` (with cooldown),
    which the caller uses to fire the webhook.

This module is pure / synchronous and unit-tested. Webhook I/O lives in
``backend/alerts/webhook.py``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RiskState:
    score: float
    level: RiskLevel
    raw_prob: float
    consecutive_high: int
    alert: bool          # True only on the chunk that transitions into HIGH
    n_chunks: int


def _weights(kind: str, n: int) -> list[float]:
    if n <= 0:
        return []
    if kind == "uniform":
        w = [1.0] * n
    elif kind == "exponential":
        alpha = 0.6
        w = [alpha ** (n - 1 - i) for i in range(n)]  # newest gets weight 1
    else:  # linear (default): oldest=1 ... newest=n
        w = [float(i + 1) for i in range(n)]
    total = sum(w)
    return [x / total for x in w]


class RiskEngine:
    def __init__(
        self,
        window: int = 10,
        weighting: str = "linear",
        low_threshold: float = 0.40,
        medium_threshold: float = 0.60,
        high_threshold: float = 0.75,
        high_consecutive: int = 3,
        alert_cooldown_seconds: float = 30.0,
    ) -> None:
        self.window = window
        self.weighting = weighting
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.high_consecutive = high_consecutive
        self.alert_cooldown_seconds = alert_cooldown_seconds

        self._scores: deque[float] = deque(maxlen=window)
        self._consecutive_high = 0
        self._in_high = False
        self._last_alert_ts = float("-inf")
        self._n = 0

    @classmethod
    def from_config(cls, risk_cfg) -> "RiskEngine":
        return cls(
            window=risk_cfg.window,
            weighting=risk_cfg.weighting,
            low_threshold=risk_cfg.low_threshold,
            medium_threshold=risk_cfg.medium_threshold,
            high_threshold=risk_cfg.high_threshold,
            high_consecutive=risk_cfg.high_consecutive,
            alert_cooldown_seconds=risk_cfg.alert_cooldown_seconds,
        )

    def reset(self) -> None:
        self._scores.clear()
        self._consecutive_high = 0
        self._in_high = False
        self._last_alert_ts = float("-inf")
        self._n = 0

    def update(self, fake_prob: float, now: float | None = None) -> RiskState:
        now = time.monotonic() if now is None else now
        fake_prob = float(max(0.0, min(1.0, fake_prob)))
        self._scores.append(fake_prob)
        self._n += 1

        w = _weights(self.weighting, len(self._scores))
        score = sum(p * wi for p, wi in zip(self._scores, w))

        if fake_prob > self.high_threshold:
            self._consecutive_high += 1
        else:
            self._consecutive_high = 0

        level = self._level(score)
        alert = False
        if level is RiskLevel.HIGH and not self._in_high:
            if now - self._last_alert_ts >= self.alert_cooldown_seconds:
                alert = True
                self._last_alert_ts = now
            self._in_high = True
        elif level is not RiskLevel.HIGH:
            self._in_high = False

        return RiskState(
            score=round(score, 4),
            level=level,
            raw_prob=fake_prob,
            consecutive_high=self._consecutive_high,
            alert=alert,
            n_chunks=self._n,
        )

    def _level(self, score: float) -> RiskLevel:
        high_ready = (
            score > self.high_threshold and self._consecutive_high >= self.high_consecutive
        )
        if high_ready:
            return RiskLevel.HIGH
        if score > self.medium_threshold:
            return RiskLevel.MEDIUM
        if score > self.low_threshold:
            return RiskLevel.LOW
        return RiskLevel.NONE
