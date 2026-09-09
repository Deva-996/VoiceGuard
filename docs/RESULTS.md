# VoiceGuard — evaluation results

**Checkpoint:** `backend/models/aasist_indicw2v.pt` — XLS-R-300m frontend (fine-tuned) + AASIST
backend + OC-Softmax, **epoch 10** (frontend unfrozen at epoch 6, so 5 epochs of frontend
fine-tuning). Training dev-EER 0.33% — on the MMS-TTS / IndicTTS dev split, *not* representative
of unseen attacks (see "Resume to epoch 15" below).

**Scoring:** OC-Softmax centre distance → P(spoof). The 2-logit head is untrained under
OC-Softmax and must not be used (see `oc-softmax` note).

**Protocol:** `python -m training.evaluate --checkpoint backend/models/aasist_indicw2v.pt
--manifest data/manifests/eval.tsv --by dataset,language --device cpu --per-domain 400
--crop-seconds 4.0 --batch-size 8` — balanced 400/domain, 4 s segments, CPU. A representative
subset (~1.4k clips), not the full eval sets.

## Detection performance

| slice | EER | min-tDCF | spoof n | bona n | notes |
|---|---|---|---|---|---|
| **ALL (pooled)** | **14.09%** | 0.767 | 596 | 823 | |
| ASVspoof-2019 LA | 16.00% | 0.755 | 200 | 200 | eval attacks A07–A19, unseen at train time |
| ASVspoof-2021 LA | 10.87% | 0.550 | 120 | 110 | codec/telephone; ~50% of local FLACs corrupt, scored on the rest |
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

## Resume to epoch 15 — tried, rejected (overfitting)

A second Kaggle run (2026-09-09) resumed from epoch 10 with optimizer state and trained
epochs 11→15 (frontend fine-tuned throughout). Dev-EER improved (0.33% → **0.27%** at
epoch 15) but held-out generalisation **regressed across the board** — same eval protocol:

| slice | epoch 10 | epoch 15 | Δ |
|---|---|---|---|
| ALL (pooled) | 14.09% | 16.42% | +2.3 |
| ASVspoof-2019 LA | 16.00% | 26.00% | **+10.0** |
| ASVspoof-2021 LA | 10.87% | 12.61% | +1.7 |
| In-the-Wild | 12.50% | 17.00% | +4.5 |
| Indic fakes caught @0.5 | 84% | 68% | **−16** |
| FLEURS false-pos (acc@0.5) | 86% | 91% | −5 (better) |
| pooled min-tDCF | 0.77 | 1.00 | saturated |

Epochs 11–15 overfit the MMS-TTS / IndicTTS / ASVspoof2019-train mix: the frontend memorised
the training attack families at the cost of the unseen ones. Dev-EER was a misleading signal
because the dev set is in-distribution. **Epoch 10 restored as the serving checkpoint.** The
epoch-15 weights are kept at `backend/models/aasist_indicw2v.epoch15.bak.pt` for reference.

## Reading

- The detector **works** — clean polarity, decent on the unseen In-the-Wild domain, near-perfect
  on genuine Indian-language corpus speech.
- **~14% pooled EER is mediocre** for anti-spoofing (in-domain SOTA is <1%). The cause is **not**
  "too few epochs" — more epochs made it worse. It's the **training data**: fakes skew heavily
  to MMS-TTS content-matched pairs, so the model never learns the breadth of ASVspoof / real
  deepfake attack signatures.
- Mild over-trigger on FLEURS-style natural speech (~14% of genuine FLEURS clips over 0.5).

## Next

- **Rebalance the fake mix** — this is the lever. More ASVspoof attack diversity (all of the
  2019-train spoof families, not a capped subset) vs MMS-TTS; add a second TTS engine
  (Google Cloud TTS / Bhashini) so "fake" isn't ~synonymous with "MMS-TTS".
- Consider **fewer frontend-finetune epochs** or a lower frontend LR / stronger RawBoost —
  the frontend overfits fast once unfrozen.
- IndicWav2Vec A/B (may generalise better on Indian-language calls than XLS-R).
- Re-extract `ASVspoof2021_LA_eval.tar.gz` — the local copy is ~half corrupt (11 skips / 400).
