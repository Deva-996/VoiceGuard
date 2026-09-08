# VoiceGuard — evaluation results

**Checkpoint:** `backend/models/aasist_indicw2v.pt` — XLS-R-300m frontend (fine-tuned) + AASIST
backend + OC-Softmax, **epoch 10 of 16** (Kaggle run timed out; resumable). Training dev-EER
0.33% (on the MMS-TTS / IndicTTS dev split — not representative of unseen attacks).

**Scoring:** OC-Softmax centre distance → P(spoof). The 2-logit head is untrained under
OC-Softmax and must not be used (see `oc-softmax` note).

**Protocol:** `scripts/eval_local.py` — balanced 400/domain, 4 s segments, CPU. A representative
subset, not the full eval sets. Regenerate: `python -m training.evaluate --checkpoint … --manifest
data/manifests/eval.tsv --by dataset,language`.

## Detection performance

| slice | EER | min-tDCF | spoof n | bona n | notes |
|---|---|---|---|---|---|
| **ALL (pooled)** | **14.09%** | 0.769 | 596 | 823 | |
| ASVspoof-2019 LA | 15.75% | 0.755 | 200 | 200 | eval attacks A07–A19, unseen at train time |
| ASVspoof-2021 LA | 10.87% | 0.558 | 120 | 110 | codec/telephone; ~50% of local FLACs corrupt, scored on the rest |
| In-the-Wild | 12.50% | 0.645 | 200 | 200 | fully unseen domain (celebrity deepfakes) |
| FLEURS (genuine only) | — | — | 0 | 113 | mean P(spoof) 0.16, 86% below 0.5 — mild false-positive rate on unseen natural speech |
| IndicTTS (genuine only) | — | — | 0 | 200 | mean 0.027, 100% correct |
| Indic fakes / MMS-TTS (spoof only) | — | — | 76 | 0 | mean 0.84, 84% caught |

### By language (where both classes present)

| lang | EER | source |
|---|---|---|
| en | 16.31% | ASVspoof 2019/2021 + In-the-Wild |
| hi | 0.00% | MMS-TTS fakes vs FLEURS genuine (n small; MMS-TTS is in-distribution — not a hard test) |

Single-class language slices (bn, gu, mr, ta, te — all genuine): mean P(spoof) 0.03–0.14,
88–100% below threshold. bn is FLEURS (natural) and accounts for most of the residual
false-positives; gu/mr/ta/te are IndicTTS and score ~perfectly.

## Reading

- The detector **works** — clean polarity, strong on the unseen In-the-Wild domain, near-perfect
  on genuine Indian-language corpus speech.
- **~14% pooled EER is mediocre** for anti-spoofing (in-domain SOTA is <1%). Causes: only 10/16
  epochs (5 of 11 planned frontend-finetune epochs); training fakes skew heavily to MMS-TTS
  content-matched pairs, so generalisation to ASVspoof attack families is partial.
- Mild over-trigger on FLEURS-style natural speech (14% of genuine FLEURS clips over 0.5).

## Next

- Resume Kaggle training to 16 epochs (now saves the OC-Softmax centre directly).
- Rebalance the fake mix (more ASVspoof attack diversity vs MMS-TTS).
- IndicWav2Vec A/B.
- Re-extract `ASVspoof2021_LA_eval.tar.gz` — the local copy is ~half corrupt.
