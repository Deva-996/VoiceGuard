"""Generate synthetic (spoof) Indian-language speech, content-matched to the genuine set.

    python scripts/generate_indian_fakes.py                 # all langs, default count
    python scripts/generate_indian_fakes.py --langs hi,ta --n 800

For each target language we read transcripts from the genuine IndicTTS set
(data/raw/indictts/<lang>/transcripts.tsv) and re-synthesise the *same sentences* with an
open local TTS, so every fake clip has a genuine counterpart with identical content — the
model then can't cheat on lexical content, only on synthesis artefacts.

Engine: Meta MMS-TTS (facebook/mms-tts-<iso3>) — 16 kHz native, per-language, non-gated, CPU.
Output:
  data/generated/indic_fake/<lang>/mms/<utt_id>.wav
  data/generated/indic_fake/<lang>/manifest.tsv   (utt_id  path  language  engine  text)

The plan (CLAUDE.md §5) also lists Google Cloud TTS / Bhashini; add them as extra engines
here when credentials are available — the manifest's ``engine`` column keeps them separate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "generated" / "indic_fake"

MMS_MODEL = {
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "bn": "facebook/mms-tts-ben",
    "mr": "facebook/mms-tts-mar",
    "gu": "facebook/mms-tts-guj",
}
DIR_TO_CODE = {"hindi": "hi", "tamil": "ta", "telugu": "te", "bengali": "bn",
               "marathi": "mr", "gujarati": "gu"}


def _load_transcripts(lang_code: str, n: int) -> list[tuple[str, str]]:
    """(utt_id, text) pairs from the genuine IndicTTS set for this language."""
    lang_dir = next((d for d, c in DIR_TO_CODE.items() if c == lang_code), None)
    tsv = RAW / "indictts" / lang_dir / "transcripts.tsv" if lang_dir else None
    if not tsv or not tsv.exists():
        return []
    out = []
    with tsv.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            text = (r.get("transcript") or r.get("text") or "").strip()
            if len(text) >= 3:
                out.append((r["utt_id"], text))
            if len(out) >= n:
                break
    return out


def synth_language(lang_code: str, n: int, device: str = "cpu") -> int:
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    model_id = MMS_MODEL[lang_code]
    pairs = _load_transcripts(lang_code, n)
    if not pairs:
        print(f"  {lang_code}: no genuine transcripts found — run download_datasets.py --only indictts")
        return 0

    print(f"  {lang_code}: {len(pairs)} sentences  <- {model_id}")
    model = VitsModel.from_pretrained(model_id).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    sr = model.config.sampling_rate

    wav_dir = OUT / lang_code / "mms"
    wav_dir.mkdir(parents=True, exist_ok=True)
    man = OUT / lang_code / "manifest.tsv"
    written = 0
    with man.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["utt_id", "path", "language", "engine", "text"])
        for src_uid, text in pairs:
            uid = f"fake_mms_{src_uid}"
            dst = wav_dir / f"{uid}.wav"
            if not dst.exists():
                inp = tok(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    wave = model(**inp).waveform.squeeze().cpu().numpy()
                if wave.size < sr * 0.2:  # skip degenerate outputs
                    continue
                sf.write(dst, wave, sr)
            w.writerow([uid, str(dst), lang_code, "mms", text])
            written += 1
            if written % 200 == 0:
                print(f"    {written}/{len(pairs)}")
    print(f"  {lang_code}: wrote {written}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default=",".join(MMS_MODEL), help="comma-separated: " + ",".join(MMS_MODEL))
    ap.add_argument("--n", type=int, default=1200, help="sentences per language")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    bad = [x for x in langs if x not in MMS_MODEL]
    if bad:
        print(f"unknown lang(s): {bad}  (have {list(MMS_MODEL)})")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for lc in langs:
        total += synth_language(lc, args.n, args.device)
    print(f"\ntotal fake clips: {total}  ->  {OUT}")
    return 0


if __name__ == "__main__":
    import os
    import sys

    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)  # HF download threads can crash interpreter teardown; we're done, bail hard
