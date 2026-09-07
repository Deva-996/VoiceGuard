"""Frame-level feature extractors that feed the AASIST backend.

Contract: ``extract(waveform_16k_mono) -> Tensor[T, feat_dim]`` (float32, no batch dim).
``feat_dim`` must match the AASIST backend's input dim (1024 for wav2vec2-large).

Implementations:
  * ``DummyFeatureExtractor``  — deterministic, dependency-light. Used in tests and offline
    smoke checks so the pipeline is exercisable without downloading SSL weights.
  * ``Wav2Vec2Extractor`` — real SSL frontend: any HuggingFace wav2vec2 checkpoint loaded
    via 🤗 transformers (``AutoModel`` + ``AutoFeatureExtractor``). Default on Day 3.
      - ``facebook/wav2vec2-xls-r-300m`` (public, multilingual incl. Indian languages) — the
        Day 3 default, since ``ai4bharat/indicwav2vec-hindi`` is gated on HF.
      - ``ai4bharat/indicwav2vec-hindi`` — the planned frontend; set ``backend: indicwav2vec``
        in config once an HF token with access is available (``huggingface-cli login``).
  * ``IndicWav2VecExtractor`` — thin alias of ``Wav2Vec2Extractor`` kept for import stability.
"""

from __future__ import annotations

import abc

import torch

FRAME_SHIFT_SAMPLES = 320  # 20 ms @ 16 kHz — matches wav2vec2 conv stride

# convenience: backend name -> default HF model id (used only when config omits model_id)
DEFAULT_MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base",          # streaming default: real-time on CPU
    "xlsr": "facebook/wav2vec2-xls-r-300m",        # multilingual, ~1 s/chunk on CPU
    "xls-r": "facebook/wav2vec2-xls-r-300m",
    "indicwav2vec": "ai4bharat/indicwav2vec-hindi",  # gated on HF
    "indic_wav2vec": "ai4bharat/indicwav2vec-hindi",
    "indicw2v": "ai4bharat/indicwav2vec-hindi",
}


class BaseFeatureExtractor(abc.ABC):
    feat_dim: int

    @abc.abstractmethod
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: 1-D float32 tensor at 16 kHz -> (T, feat_dim) float32."""

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.extract(waveform)


def _as_mono_1d(waveform: torch.Tensor) -> torch.Tensor:
    wav = torch.as_tensor(waveform, dtype=torch.float32)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    return wav.reshape(-1)


class DummyFeatureExtractor(BaseFeatureExtractor):
    """Deterministic stand-in for an SSL frontend.

    Frames the waveform every 20 ms and maps each raw frame through a *fixed* random
    projection to ``feat_dim``. Not meaningful acoustically — it only lets us validate
    tensor shapes and the AASIST forward + scoring path end to end.
    """

    def __init__(self, feat_dim: int = 1024, seed: int = 0) -> None:
        self.feat_dim = feat_dim
        gen = torch.Generator().manual_seed(seed)
        self._proj = torch.randn(FRAME_SHIFT_SAMPLES, feat_dim, generator=gen) / (feat_dim ** 0.5)

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = _as_mono_1d(waveform)
        n_frames = max(1, wav.numel() // FRAME_SHIFT_SAMPLES)
        usable = n_frames * FRAME_SHIFT_SAMPLES
        if wav.numel() < usable:
            wav = torch.nn.functional.pad(wav, (0, usable - wav.numel()))
        frames = wav[:usable].reshape(n_frames, FRAME_SHIFT_SAMPLES)
        feats = frames @ self._proj
        return torch.tanh(feats).contiguous()


class Wav2Vec2Extractor(BaseFeatureExtractor):
    """SSL wav2vec2 frontend backed by a HuggingFace checkpoint.

    Works with any wav2vec2 model (``facebook/wav2vec2-xls-r-300m``,
    ``ai4bharat/indicwav2vec-hindi``, ...). Frame features are taken from a chosen
    transformer hidden layer (``layer=-1`` = last). Frozen by default.
    """

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xls-r-300m",
        layer: int = -1,
        frozen: bool = True,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModel  # lazy: heavy, optional dep

        self.model_id = model_id
        self.layer = layer
        self.frozen = frozen
        self.device = torch.device(device)

        self._fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.sampling_rate = int(getattr(self._fe, "sampling_rate", 16000))

        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.feat_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = _as_mono_1d(waveform)
        inputs = self._fe(
            wav.cpu().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        out = self.model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[self.layer]  # (1, T, H)
        return hidden.squeeze(0).float().cpu().contiguous()


# Backward-compatible name — behaviour is identical to Wav2Vec2Extractor.
class IndicWav2VecExtractor(Wav2Vec2Extractor):
    def __init__(
        self,
        model_id: str = "ai4bharat/indicwav2vec-hindi",
        layer: int = -1,
        frozen: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(model_id=model_id, layer=layer, frozen=frozen, device=device)


def build_feature_extractor(cfg) -> BaseFeatureExtractor:
    """Instantiate from a ``FeatureExtractorConfig``."""
    backend = getattr(cfg, "backend", "dummy").lower()
    if backend == "dummy":
        return DummyFeatureExtractor(feat_dim=cfg.feat_dim)
    if backend in DEFAULT_MODEL_IDS or backend in {"hf", "wav2vec"}:
        model_id = getattr(cfg, "model_id", "") or DEFAULT_MODEL_IDS.get(backend, "")
        if not model_id:
            raise ValueError(f"feature_extractor.model_id required for backend {backend!r}")
        return Wav2Vec2Extractor(
            model_id=model_id,
            layer=getattr(cfg, "layer", -1),
            frozen=getattr(cfg, "frozen", True),
        )
    raise ValueError(f"unknown feature_extractor backend: {backend!r}")
