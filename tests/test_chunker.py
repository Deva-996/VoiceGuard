import numpy as np

from backend.audio import AudioChunker, decode_pcm, resample_to_16k


def test_decode_int16_roundtrip():
    src = (np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32) * 32767).astype("<i2")
    out = decode_pcm(src.tobytes(), "int16")
    assert np.allclose(out, [0.0, 0.5, -0.5, 1.0, -1.0], atol=1e-3)


def test_resample_length():
    wav = np.zeros(48000, dtype=np.float32)
    out = resample_to_16k(wav, 48000)
    assert out.shape[0] == 16000


def test_chunker_yields_fixed_windows_with_hop():
    ch = AudioChunker(sample_rate=16000, chunk_seconds=1.0, hop_seconds=0.5, pcm_format="float32")
    pcm = np.ones(16000 * 3, dtype=np.float32).tobytes()  # 3 s
    windows = list(ch.push(pcm))
    # 3 s, 1 s window, 0.5 s hop -> windows starting at 0.0,0.5,1.0,1.5,2.0 s
    assert len(windows) == 5
    assert all(w.shape[0] == 16000 for w in windows)


def test_chunker_accumulates_across_pushes():
    ch = AudioChunker(sample_rate=16000, chunk_seconds=1.0, hop_seconds=1.0, pcm_format="float32")
    half = np.ones(8000, dtype=np.float32).tobytes()
    assert list(ch.push(half)) == []
    windows = list(ch.push(half))
    assert len(windows) == 1 and windows[0].shape[0] == 16000


def test_flush_pads_remainder():
    ch = AudioChunker(sample_rate=16000, chunk_seconds=1.0, hop_seconds=1.0, pcm_format="float32")
    list(ch.push(np.ones(4000, dtype=np.float32).tobytes()))
    tail = ch.flush()
    assert tail is not None and tail.shape[0] == 16000
    assert tail[:4000].sum() == 4000 and tail[4000:].sum() == 0
