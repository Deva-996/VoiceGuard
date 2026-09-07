# Training the VoiceGuard detector

The live pipeline auto-loads `backend/models/aasist_indicw2v.pt` if present
(`classifier.checkpoint` in `config/config.yaml`). Until one exists, `fake_prob` is noise.

## Pipeline

```
download_datasets.py ─▶ data/raw/{asvspoof2019_LA, indictts/<lang>, fleurs/<lang>, ...}
generate_indian_fakes.py ─▶ data/generated/indic_fake/<lang>/  (MMS-TTS, content-matched)
prepare_manifests.py ─▶ data/manifests/{train,dev,eval}.tsv
training.features   ─▶ data/processed/feat_cache/<model>_L<layer>/<utt>.npy   (SSL, once)
training.train      ─▶ backend/models/aasist_indicw2v.pt   (frozen frontend + AASIST head)
training.evaluate   ─▶ EER / min-tDCF, pooled and per dataset/language
```

`training/pipeline_run.py` runs all of it: `python -m training.pipeline_run`.

## Where to run it

**GPU (Colab / Kaggle / any CUDA box) — recommended.** `notebooks/train_colab.ipynb`:
set your repo URL, pick a frontend, Run all. ~20–40 min feature extraction + ~1–2 h training
on a T4. Download `aasist_indicw2v.pt`, drop it in `backend/models/`.

**CPU (local).** Works but slow — the frozen-frontend baseline only (stage-2 unfreeze needs a
GPU). Feature-caching ASVspoof2019 (121k clips) with `wav2vec2-base` ≈ overnight; training the
head ≈ overnight. Use `--limit` for a smoke run first:
```
python -m training.pipeline_run --limit 400 --epochs 3
```

## Config

`training/config_train.yaml` — `frontend.model_id` (keep in sync with
`config/config.yaml` `feature_extractor` for train/serve parity), `loss.name`
(`weighted_ce` | `oc_softmax`), `epochs`, `batch_size`, `max_frames`.

- `wav2vec2-base` — real-time serving on CPU, English-pretrained.
- `wav2vec2-xls-r-300m` — multilingual, stronger; serving needs `hop_seconds: 1.0`.
- `ai4bharat/indicwav2vec-hindi` — best for Indian languages; **gated** (`huggingface-cli login`).

Run all three as an A/B once data is in place — `evaluate.py --by language` shows which wins
where.

## Expected results

Frozen `wav2vec2-base` + AASIST: ~8–12 % EER on ASVspoof2019 LA eval (a real detector, not
SOTA). Frozen XLS-R: better. Unfrozen frontend (stage-2, GPU): sub-1 % on ASVspoof, though
In-the-Wild / cross-lingual will be higher — that domain gap is the point of those eval sets.

## Datasets

All non-gated, fetched by `scripts/download_datasets.py` (see CLAUDE.md §5). Big eval sets
(ASVspoof 2021 LA, In-the-Wild) are optional for a first training run — `prepare_manifests.py`
just omits any dataset whose raw files aren't present.
