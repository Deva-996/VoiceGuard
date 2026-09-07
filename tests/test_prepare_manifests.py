from scripts.prepare_manifests import Row, split_indic


def _rows(n, lang, label):
    return [Row(f"{lang}_{label}_{i}", f"/x/{i}.wav", label, lang, "human", "t") for i in range(n)]


def test_split_indic_ratios_and_determinism():
    genuine = _rows(100, "hi", "bonafide") + _rows(50, "ta", "bonafide")
    fake = _rows(100, "hi", "spoof") + _rows(50, "ta", "spoof")
    a = split_indic(genuine, fake)
    b = split_indic(genuine, fake)

    assert [r.utt_id for r in a["train"]] == [r.utt_id for r in b["train"]]  # deterministic
    total = len(a["train"]) + len(a["dev"]) + len(a["eval"])
    assert total == 300
    assert len(a["train"]) / total > 0.75
    for split in ("train", "dev", "eval"):
        assert {r.label for r in a[split]} == {"bonafide", "spoof"}


def test_split_indic_no_leakage_between_splits():
    genuine = _rows(40, "mr", "bonafide")
    fake = _rows(40, "mr", "spoof")
    parts = split_indic(genuine, fake)
    ids = [r.utt_id for s in ("train", "dev", "eval") for r in parts[s]]
    assert len(ids) == len(set(ids))
