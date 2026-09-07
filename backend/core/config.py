"""Typed configuration loading for VoiceGuard.

Reads ``config/config.yaml`` into nested dataclasses. Any leaf can be overridden with an
environment variable ``VG_<SECTION>__<KEY>`` (double underscore = nesting), e.g.
``VG_RISK__HIGH_THRESHOLD=0.8`` or ``VG_FEATURE_EXTRACTOR__BACKEND=indicwav2vec``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_seconds: float = 1.0
    hop_seconds: float = 0.5
    pcm_format: str = "int16"

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_seconds)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_seconds)


@dataclass
class FeatureExtractorConfig:
    backend: str = "dummy"
    model_id: str = "ai4bharat/indicwav2vec-hindi"
    layer: int = -1
    frozen: bool = True
    feat_dim: int = 1024


@dataclass
class ClassifierConfig:
    embed_dim: int = 256
    num_classes: int = 2
    checkpoint: str = "backend/models/aasist_indicw2v.pt"
    device: str = "cpu"

    @property
    def checkpoint_path(self) -> Path:
        p = Path(self.checkpoint)
        return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class RiskConfig:
    window: int = 10
    weighting: str = "linear"
    low_threshold: float = 0.40
    medium_threshold: float = 0.60
    high_threshold: float = 0.75
    high_consecutive: int = 3
    alert_cooldown_seconds: float = 30.0


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: str = ""
    timeout_seconds: float = 5.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] | None = None

    def __post_init__(self) -> None:
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class Config:
    audio: AudioConfig
    feature_extractor: FeatureExtractorConfig
    classifier: ClassifierConfig
    risk: RiskConfig
    webhook: WebhookConfig
    server: ServerConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        return cls(
            audio=AudioConfig(**raw.get("audio", {})),
            feature_extractor=FeatureExtractorConfig(**raw.get("feature_extractor", {})),
            classifier=ClassifierConfig(**raw.get("classifier", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            webhook=WebhookConfig(**raw.get("webhook", {})),
            server=ServerConfig(**raw.get("server", {})),
        )


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply_env_overrides(cfg: Config, prefix: str = "VG_") -> None:
    for key, value in os.environ.items():
        if not key.startswith(prefix) or "__" not in key:
            continue
        section_name, _, leaf = key[len(prefix):].lower().partition("__")
        section = getattr(cfg, section_name, None)
        if section is None or not is_dataclass(section):
            continue
        if leaf in {f.name for f in fields(section)}:
            setattr(section, leaf, _coerce(value, getattr(section, leaf)))


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = Config.from_dict(raw or {})
    _apply_env_overrides(cfg)
    return cfg


_CACHED: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Process-wide singleton config."""
    global _CACHED
    if _CACHED is None or reload:
        _CACHED = load_config()
    return _CACHED
