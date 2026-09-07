"""Build unified train / dev / eval manifests across all datasets.

    python scripts/prepare_manifests.py                 # all datasets present in data/raw
    python scripts/prepare_manifests.py --only asvspoof2019,indic

Output: data/manifests/{train,dev,eval}.tsv with columns

    utt_id   path   label   language   source   dataset

  label    : bonafide | spoof
  source   : human | tts:<system> | vc:<system>
  dataset  : asvspoof2019_LA | asvspoof2021_LA | in_the_wild | indictts | fleurs | indic_fake

Curriculum (CLAUDE.md §5):
  train : ASVspoof2019 LA train  + Indic genuine/fake train split
  dev   : ASVspoof2019 LA dev    + Indic dev split
  eval  : ASVspoof2021 LA eval, In-the-Wild, Indic eval split   (each a separate slice)

Audio that lives inside parquet/zip (ASVspoof2019, ASVspoof2021, In-the-Wild) is materialised
once under data/processed/<dataset>/ ; genuine Indic speech is already wav on disk from
scripts/download_datasets.py and fakes from scripts/generate_indian_fakes.py.
"""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
from collections import Counter
from dataclasses import astuple, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
GENERATED = ROOT / "data" / "generated"
MANIFESTS = ROOT / "data" / "manifests"

COLUMNS = ["utt_id", "path", "label", "language", "source", "dataset"]

# language of each IndicTTS / FLEURS directory
INDIC_LANG_CODE = {
    "hindi": "hi", "tamil": "ta", "telugu": "te", "bengali": "bn",
    "marathi": "mr", "gujarati": "gu", "kannada": "kn", "malayalam": "ml",
    "hi_in": "hi", "ta_in": "ta", "te_in": "te", "bn_in": "bn", "mr_in": "mr", "gu_in": "gu",
}


@dataclass
class Row:
    utt_id: str
    path: str
    label: str
    language: str
    source: str
    dataset: str


# ---------------------------------------------------------------- ASVspoof 2019 LA (parquet)
def parse_asvspoof2019(split: str) -> list[Row]:
    import pyarrow.parquet as pq
    import soundfile as sf

    fname = {"train": "train", "dev": "validation", "eval": "test"}[split]
    files = list((RAW / "asvspoof2019_LA" / "data").glob(f"{fname}-*.parquet"))
    if not files:
        return []
    out_dir = PROC / "asvspoof2019_LA" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for pqfile in files:
        tbl = pq.read_table(pqfile)
        cols = {c: tbl[c].to_pylist() for c in tbl.column_names}
        for i in range(tbl.num_rows):
            uid = cols["audio_file_name"][i]
            key = cols["key"][i]
            sysid = cols["system_id"][i]
            dst = out_dir / f"{uid}.flac"
            if not dst.exists():
                audio = cols["audio"][i]
                raw = audio["bytes"] if audio.get("bytes") else Path(audio["path"]).read_bytes()
                wav, sr = sf.read(io.BytesIO(raw))
                sf.write(dst, wav, sr)
            label = "bonafide" if key == 0 else "spoof"
            source = "human" if key == 0 else f"tts_vc:{sysid}"
            rows.append(Row(f"a19_{uid}", str(dst), label, "en", source, "asvspoof2019_LA"))
    return rows


# ---------------------------------------------------------------- ASVspoof 2021 LA eval (tar)
def parse_asvspoof2021_eval() -> list[Row]:
    import soundfile as sf

    base = RAW / "asvspoof2021_LA_eval"
    flac_root = next(base.rglob("ASVspoof2021_LA_eval"), None) if base.exists() else None
    key_file = next(base.rglob("*trial_metadata.txt"), None) or next(base.rglob("*keys*/*LA*.txt"), None)
    if flac_root is None or key_file is None:
        # audio may still be a tarball
        tars = list(base.glob("*.tar.gz")) if base.exists() else []
        if tars and flac_root is None:
            for t in tars:
                with tarfile.open(t) as tf:
                    tf.extractall(base, filter="data")
            return parse_asvspoof2021_eval()
        return []

    out_dir = PROC / "asvspoof2021_LA_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, tuple[str, str]] = {}
    for line in Path(key_file).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        # ASVspoof2021 metadata: SPK  UTT  ...  SRC  KEY(bonafide|spoof) ...
        utt = p[1]
        lab = "bonafide" if "bonafide" in p else ("spoof" if "spoof" in p else None)
        if lab:
            labels[utt] = (lab, p[4] if len(p) > 4 else "-")
    rows: list[Row] = []
    for flac in flac_root.rglob("*.flac"):
        utt = flac.stem
        if utt not in labels:
            continue
        lab, src = labels[utt]
        dst = out_dir / f"{utt}.flac"
        if not dst.exists():
            wav, sr = sf.read(flac)
            sf.write(dst, wav, sr)
        source = "human" if lab == "bonafide" else f"tts_vc:{src}"
        rows.append(Row(f"a21_{utt}", str(dst), lab, "en", source, "asvspoof2021_LA"))
    return rows


# ---------------------------------------------------------------- In-the-Wild (zip)
def parse_in_the_wild() -> list[Row]:
    base = RAW / "in_the_wild" / "release_in_the_wild"
    meta = base / "meta.csv"
    if not meta.exists():
        return []
    rows: list[Row] = []
    with meta.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            f = r.get("file") or r.get("filename")
            lab_raw = (r.get("label") or "").strip().lower()
            label = "bonafide" if lab_raw in {"bona-fide", "bonafide", "real"} else "spoof"
            rows.append(
                Row(
                    utt_id=f"itw_{Path(f).stem}",
                    path=str(base / f),
                    label=label,
                    language="en",
                    source="human" if label == "bonafide" else "tts_vc:unknown",
                    dataset="in_the_wild",
                )
            )
    return rows


# ---------------------------------------------------------------- Indic genuine + fake
def _indic_genuine(root: Path, dataset: str) -> list[Row]:
    rows: list[Row] = []
    if not root.exists():
        return rows
    for lang_dir in sorted(root.iterdir()):
        tsv = lang_dir / "transcripts.tsv"
        if not tsv.exists():
            continue
        lang = INDIC_LANG_CODE.get(lang_dir.name, lang_dir.name[:2])
        with tsv.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                rows.append(
                    Row(r["utt_id"], r["path"], "bonafide", lang, "human", dataset)
                )
    return rows


def _indic_fake() -> list[Row]:
    root = GENERATED / "indic_fake"
    rows: list[Row] = []
    if not root.exists():
        return rows
    for tsv in root.rglob("manifest.tsv"):
        with tsv.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                rows.append(
                    Row(r["utt_id"], r["path"], "spoof", r["language"],
                        f"tts:{r['engine']}", "indic_fake")
                )
    return rows


def split_indic(genuine: list[Row], fake: list[Row]) -> dict[str, list[Row]]:
    """Deterministic 80/10/10 per language, keeping genuine+fake balanced within a split."""
    import random

    out = {"train": [], "dev": [], "eval": []}
    for pool in (genuine, fake):
        by_lang: dict[str, list[Row]] = {}
        for r in pool:
            by_lang.setdefault(r.language, []).append(r)
        for lang, items in by_lang.items():
            rng = random.Random(f"vg-{lang}")
            rng.shuffle(items)
            n = len(items)
            a, b = int(0.8 * n), int(0.9 * n)
            out["train"] += items[:a]
            out["dev"] += items[a:b]
            out["eval"] += items[b:]
    return out


# ---------------------------------------------------------------- write
def write_manifest(name: str, rows: list[Row]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS / f"{name}.tsv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(astuple(r))
    n_spoof = sum(r.label == "spoof" for r in rows)
    langs = Counter(r.language for r in rows)
    dsets = Counter(r.dataset for r in rows)
    print(f"{out.name}: {len(rows)} rows  ({n_spoof} spoof / {len(rows)-n_spoof} bona)")
    print(f"   datasets: {dict(dsets)}")
    print(f"   languages: {dict(langs)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="subset: asvspoof2019,asvspoof2021,in_the_wild,indic")
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()} or {
        "asvspoof2019", "asvspoof2021", "in_the_wild", "indic"
    }

    train: list[Row] = []
    dev: list[Row] = []
    eval_: list[Row] = []

    if "asvspoof2019" in want:
        print("ASVspoof2019 LA ...")
        train += parse_asvspoof2019("train")
        dev += parse_asvspoof2019("dev")
        eval_ += parse_asvspoof2019("eval")
    if "asvspoof2021" in want:
        print("ASVspoof2021 LA eval ...")
        eval_ += parse_asvspoof2021_eval()
    if "in_the_wild" in want:
        print("In-the-Wild ...")
        eval_ += parse_in_the_wild()
    if "indic" in want:
        print("Indic (IndicTTS + FLEURS genuine, generated fakes) ...")
        genuine = _indic_genuine(RAW / "indictts", "indictts") + _indic_genuine(RAW / "fleurs", "fleurs")
        fake = _indic_fake()
        parts = split_indic(genuine, fake)
        train += parts["train"]
        dev += parts["dev"]
        eval_ += parts["eval"]

    write_manifest("train", train)
    write_manifest("dev", dev)
    write_manifest("eval", eval_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
