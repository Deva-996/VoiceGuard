"""Webhook dispatch on HIGH-risk transition.  [Day 4]

``RiskEngine`` sets ``RiskState.alert = True`` on the chunk that latches HIGH. The WS handler
passes that to ``WebhookDispatcher.fire(...)``, which POSTs a JSON payload to the configured
URL. Fire-and-forget: failures are logged, never raised into the audio path.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

log = logging.getLogger("voiceguard.webhook")


@dataclass
class AlertPayload:
    event: str = "voiceguard.high_risk"
    session_id: str = ""
    score: float = 0.0
    level: str = "HIGH"
    raw_prob: float = 0.0
    consecutive_high: int = 0
    n_chunks: int = 0
    ts: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class WebhookDispatcher:
    def __init__(self, url: str = "", enabled: bool = False, timeout_seconds: float = 5.0) -> None:
        self.url = url
        self.enabled = enabled and bool(url)
        self.timeout = timeout_seconds

    @classmethod
    def from_config(cls, webhook_cfg) -> "WebhookDispatcher":
        return cls(
            url=webhook_cfg.url,
            enabled=webhook_cfg.enabled,
            timeout_seconds=webhook_cfg.timeout_seconds,
        )

    async def fire(self, payload: AlertPayload) -> bool:
        if not self.enabled:
            log.info("webhook disabled; would POST %s", asdict(payload))
            return False
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.url, json=asdict(payload))
            resp.raise_for_status()
            log.info("webhook delivered (%s) for session=%s", resp.status_code, payload.session_id)
            return True
        except Exception as exc:  # noqa: BLE001 — must never break the audio path
            log.warning("webhook delivery failed: %s", exc)
            return False
