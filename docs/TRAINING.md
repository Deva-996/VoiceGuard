# Training the VoiceGuard detector

The live pipeline auto-loads `backend/models/aasist_indicw2v.pt` if present
(`classifier.checkpoint` in `config/config.yaml`). Until one exists, `fake_prob` is noise.
If the frontend was fine-tuned, its weights ride inside the same file and
`DetectionPipeline.from_config` applies them over `config.yaml`.

## Pipeline

```
download_datasets.py      ─▶ data/raw/{asvspoof2019_LA, indictts/<lang>, fleurs/<lang>, ...}
generate_indian_fakes.py  ─▶ data/generated/indic_fake/<lang>/   (MMS-TTS, content-matched)
prepare_manifests.py      ─▶ data/manifests/{train,dev,eval}.tsv
training.train            ─▶ backend/models/aasist_indicw2v.pt
training.evaluate         ─▶ EER / min-tDCF, pooled and per dataset/language
```

`training/pipeline_run.py` runs all of it: `python -m training.pipeline_run`.

## Two training modes (`training/train.py` picks automatically; `--mode` to force)

**`e2e` — the quality path (needs a GPU).** Waveform → optional **RawBoost** augmentation →
**trainable** wav2vec2 frontend → AASIST. Frontend frozen for `stage2_unfreeze_epoch` epochs
(head warm-up), then unfrozen at `lr * frontend_lr_mult`. Class-balanced sampler for the
ASVspoof spoof/bonafide skew. This is where sub-1% EER comes from.

**`frozen` — the CPU fallback.** SSL features cached to disk once (`training/features.py`),
AASIST head trains off the cache. No waveform augmentation, frontend never updated. Train set
capped at `frozen_max_per_class` so the cache stays small. ~5–10% EER.

## Where to run

**Kaggle P100 — recommended.** `notebooks/train_kaggle.ipynb`: set `REPO_URL` (+ `HF_TOKEN`
for IndicWav2Vec), **Save & Run All (Commit)** → runs in the background across the 12 h limit,
checkpoint lands in `/kaggle/working`. 30 GPU-hrs/week free.

**Colab.** `notebooks/train_colab.ipynb` — same flow, less reliable for long runs.

**Cloud rental** (RunPod / Lambda / vast.ai) — `git clone`, `pip install -r requirements.txt`,
`python -m training.pipeline_run`. ~$0.30–0.50/hr, one clean sweep.

**Local CPU** — `frozen` mode only, and slow. Smoke first:
`python -m training.pipeline_run --limit 400 --epochs 3`.

## Config — `training/config_train.yaml`

| key | note |
|---|---|
| `frontend.model_id` | `wav2vec2-xls-r-300m` (default, multilingual) · `wav2vec2-base` (serving-fast) · `ai4bharat/indicwav2vec-hindi` (best for Indian langs, gated → `HF_TOKEN`) |
| `frontend.stage2_unfreeze_epoch` | epoch to unfreeze the frontend; `null` = stay frozen |
| `loss.name` | `oc_softmax` (default, generalises to unseen attacks) · `weighted_ce` |
| `augment.rawboost` / `rawboost_mode` | codec/channel/noise aug; mode 5 covers telephone |
| `batch_size` · `grad_accum` | effective batch = product; e2e wants small batch × accum |
| `frozen_max_per_class` | frozen mode only — caps ASVspoof train per class |

## The frontend A/B

Fine-tune `wav2vec2-xls-r-300m` **and** `ai4bharat/indicwav2vec-hindi`, then
`python -m training.evaluate --checkpoint <ckpt> --manifest data/manifests/eval.tsv --by language`
— keep whichever wins on the Indian-language slices (IndicWav2Vec is the plan's premise;
verify it beats XLS-R once both are fine-tuned).

## Expected

- frozen `wav2vec2-base` + AASIST: ~8–12 % EER (ASVspoof2019 LA eval)
- fine-tuned XLS-R / IndicWav2Vec + AASIST + RawBoost: ~0.3–1 % on ASVspoof, higher on
  In-the-Wild / cross-lingual — that domain gap is the point of those held-out sets.

## Datasets

All non-gated, `scripts/download_datasets.py` (CLAUDE.md §5). Big eval sets (ASVspoof 2021 LA,
In-the-Wild) are optional for a first run — `prepare_manifests.py` omits any dataset whose raw
files aren't present.
