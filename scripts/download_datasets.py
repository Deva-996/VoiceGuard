"""Fetch training + eval datasets into data/raw/.

    python scripts/download_datasets.py --list             # show sources + what's present
    python scripts/download_datasets.py --only openslr,fleurs
    python scripts/download_datasets.py --all              # everything (~30 GB)

All sources here are open / direct download or non-gated HF mirrors — no registration, no
HF token. `data/` is a junction to a roomy volume on this machine (see CLAUDE.md).

Target layout under data/raw/:
  asvspoof2019_LA/     HF parquet mirror (Bisher/ASVspoof_2019_LA): train/validation/test
  asvspoof2021_LA_eval/  ASVspoof2021_LA_eval.* + keys           (Zenodo)
  in_the_wild/         release_in_the_wild/  meta.csv            (deepfake-total.com)
  indictts/<lang>/     *.wav + transcripts.tsv                   (SPRINGLab IndicTTS via HF)
  fleurs/<lang>/       {split}/*.wav + transcripts.tsv           (google/fleurs via HF)

Genuine Indian-language speech: SPRINGLab/IndicTTS_* on HF is the IIT-Madras IndicTTS corpus
itself, non-gated — no email request needed. We stream a capped number of utts per language
(INDICTTS_N) since the full corpora are ~6-8 GB each. FLEURS adds read-speech speaker variety
and a separate genuine-domain eval slice. See scripts/generate_indian_fakes.py for fakes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# SPRINGLab IndicTTS on HF (IIT-Madras corpus, non-gated). Repo names are inconsistent
# (dash vs underscore) so map them explicitly. Fake generation targets hi/ta/te/bn/mr/gu.
INDICTTS_REPOS = {
    "hindi": "SPRINGLab/IndicTTS-Hindi",
    "tamil": "SPRINGLab/IndicTTS_Tamil",
    "telugu": "SPRINGLab/IndicTTS_Telugu",
    "bengali": "SPRINGLab/IndicTTS_Bengali",
    "marathi": "SPRINGLab/IndicTTS_Marathi",
    "gujarati": "SPRINGLab/IndicTTS_Gujarati",
}
INDICTTS_N = 1600  # utts streamed per language (full corpora are ~6-8 GB each)

# FLEURS: read Wikipedia sentences, transcribed, non-gated — speaker variety + eval slice.
FLEURS_LANGS = ["hi_in", "bn_in", "ta_in", "te_in", "mr_in", "gu_in"]

HF_ASVSPOOF2019 = "Bisher/ASVspoof_2019_LA"  # parquet: train / validation / test + labels

DIRECT = {
    "in_the_wild": (
        "https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download",
        "in_the_wild/release_in_the_wild.zip",
    ),
    "asvspoof2021_LA_eval": (
        "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz",
        "asvspoof2021_LA_eval/ASVspoof2021_LA_eval.tar.gz",
    ),
    "asvspoof2021_LA_keys": (
        "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz",
        "asvspoof2021_LA_eval/LA-keys-full.tar.gz",
    ),
}

SOURCES = ["asvspoof2019", "asvspoof2021", "in_the_wild", "indictts", "fleurs"]


# --------------------------------------------------------------------------- helpers
def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {dest.relative_to(RAW)}")
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(dest), url],
        check=True,
    )


def _extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    print(f"  extract {archive.name} -> {into.relative_to(RAW)}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(into, filter="data")


# --------------------------------------------------------------------------- fetchers
def get_asvspoof2019() -> None:
    from huggingface_hub import snapshot_download

    out = RAW / "asvspoof2019_LA"
    print("ASVspoof 2019 LA  (HF parquet mirror, ~7.5 GB)")
    snapshot_download(
        repo_id=HF_ASVSPOOF2019,
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=["*.parquet", "*.md", "*.json", "*.txt"],
    )


def get_asvspoof2021() -> None:
    print("ASVspoof 2021 LA eval + keys  (Zenodo, ~7.8 GB)")
    for key in ("asvspoof2021_LA_eval", "asvspoof2021_LA_keys"):
        url, rel = DIRECT[key]
        dest = RAW / rel
        try:
            _curl(url, dest)
            _extract(dest, dest.parent)
        except subprocess.CalledProcessError:
            print(f"  !! failed: {url}  (keys URL sometimes moves — grab manually if so)")


def get_in_the_wild() -> None:
    print("In-the-Wild  (deepfake-total.com, ~8.2 GB)")
    url, rel = DIRECT["in_the_wild"]
    dest = RAW / rel
    _curl(url, dest)
    _extract(dest, dest.parent)


def _stream_hf_audio(repo: str, config: str, split: str, n: int, out: Path, lang: str) -> int:
    """Stream up to n examples of an HF audio dataset to wav + append to transcripts.tsv."""
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / "transcripts.tsv"
    new = not tsv.exists()
    written = 0
    with tsv.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(["utt_id", "path", "split", "gender", "transcript"]) + "\n")
        for i, ex in enumerate(ds):
            if i >= n:
                break
            uid = f"{lang}_{split}_{i:06d}"
            a = ex["audio"]
            raw = a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes()
            wav, sr = sf.read(io.BytesIO(raw))
            wpath = out / f"{uid}.wav"
            sf.write(wpath, wav, sr)
            text = (ex.get("text") or ex.get("transcription") or "").replace("\t", " ").replace("\n", " ")
            fh.write("\t".join([uid, str(wpath), split, str(ex.get("gender", "")), text]) + "\n")
            written += 1
    return written


def get_indictts() -> None:
    print(f"IndicTTS (SPRINGLab, HF)  — {INDICTTS_N} utts/lang, {len(INDICTTS_REPOS)} langs")
    for lang, repo in INDICTTS_REPOS.items():
        out = RAW / "indictts" / lang
        tsv = out / "transcripts.tsv"
        if tsv.exists() and len(tsv.read_text(encoding="utf-8").splitlines()) > 100:
            print(f"  have {lang} ({len(tsv.read_text(encoding='utf-8').splitlines())-1} utts)")
            continue
        print(f"  {lang}  <- {repo}")
        try:
            n = _stream_hf_audio(repo, "default", "train", INDICTTS_N, out, lang)
            print(f"    {n} utts")
        except Exception as e:  # noqa: BLE001
            print(f"    !! {lang}: {repr(e)[:150]}")


FLEURS_SPLIT_N = {"train": 600, "validation": 200, "test": 400}


def get_fleurs() -> None:
    print("FLEURS Indian languages  (HF google/fleurs, streamed)")
    for lang in FLEURS_LANGS:
        out = RAW / "fleurs" / lang
        tsv = out / "transcripts.tsv"
        if tsv.exists() and len(tsv.read_text(encoding="utf-8").splitlines()) > 100:
            print(f"  have {lang}")
            continue
        if out.exists():  # incomplete -> restart this language
            import shutil

            shutil.rmtree(out)
        print(f"  {lang}")
        total = 0
        for split, n in FLEURS_SPLIT_N.items():
            try:
                total += _stream_hf_audio("google/fleurs", lang, split, n, out, lang)
            except Exception as e:  # noqa: BLE001
                print(f"    !! {split}: {repr(e)[:120]}")
        print(f"    {total} utts")


FETCHERS = {
    "asvspoof2019": get_asvspoof2019,
    "asvspoof2021": get_asvspoof2021,
    "in_the_wild": get_in_the_wild,
    "indictts": get_indictts,
    "fleurs": get_fleurs,
}


# --------------------------------------------------------------------------- status
def show_status() -> None:
    print(f"data root: {RAW.resolve()}\n")
    checks = {
        "asvspoof2019_LA": RAW / "asvspoof2019_LA" / "data",
        "asvspoof2021_LA_eval": RAW / "asvspoof2021_LA_eval",
        "in_the_wild": RAW / "in_the_wild" / "release_in_the_wild",
        "indictts": RAW / "indictts",
        "fleurs": RAW / "fleurs",
    }
    for name, path in checks.items():
        n = sum(1 for _ in path.rglob("*")) if path.exists() else 0
        print(f"  [{'OK' if n else '--'}] {name:24s} {n:>7d} entries  ({path})")
    print("\nsources:", ", ".join(SOURCES))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show status and exit")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(SOURCES))
    ap.add_argument("--all", action="store_true", help="fetch everything")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.list or (not args.only and not args.all):
        show_status()
        return 0

    todo = SOURCES if args.all else [s.strip() for s in args.only.split(",") if s.strip()]
    bad = [s for s in todo if s not in FETCHERS]
    if bad:
        print(f"unknown source(s): {bad}\nvalid: {SOURCES}")
        return 2

    for s in todo:
        print(f"\n=== {s} ===")
        try:
            FETCHERS[s]()
        except Exception as e:  # noqa: BLE001
            print(f"!! {s} failed: {repr(e)[:200]}")
    print("\ndone.")
    show_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
