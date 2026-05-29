# Decision-Aware and Explainable Deep Learning for Leukemic Blast Detection

Reference implementation for the paper *"A Decision-Aware and Explainable Deep
Learning System for Leukemic Blast Detection"*. The notebook reproduces every
number reported in the manuscript from real model outputs — nothing is hardcoded.

## What it does

The pipeline (`CANCER_clean.ipynb`) runs end-to-end:

1. **Patient-independent split** — patients are parsed from the C-NMC filename
   token (`UID_48_...` -> patient `48`, `UID_H12_...` -> `H12`) and split per class
   so that both classes appear in every partition and no patient crosses partitions.
2. **Baseline classifier** (ResNet-18) -> generates **confidence-based decision
   pseudo-labels** (top-confidence -> ACCEPT, low-confidence -> REVIEW,
   low-quality images -> DEFER).
3. **Dual-head model** — shared ResNet-18 backbone with a classification head
   (blast vs normal) and a decision head (ACCEPT / REVIEW / DEFER), trained with
   the joint loss `L = a*L_class + b*L_decision`.
4. **Selective-prediction evaluation** — coverage, accepted-only accuracy, AUC,
   blast sensitivity, false-negative rate, review rate, with **patient-level**
   bootstrap confidence intervals.
5. **Calibration** — reliability diagram and Expected Calibration Error (ECE).
6. **Baselines under a matched coverage protocol** — Confidence Thresholding,
   MC-Dropout, Deep Ensemble (Table 3).
7. **Decision-level Grad-CAM** + attention metrics (entropy, peakiness,
   cell-focus ratio, center-of-mass distance) with Mann-Whitney U, Cohen's d,
   and Bonferroni correction.
8. **Faithfulness / perturbation** check.
9. **Figure generation** — Fig. 1 (examples), Fig. 4 (Grad-CAM), Fig. 5
   (reliability diagram), saved at 300 dpi.

## How to run (Google Colab)

1. Open `CANCER_clean.ipynb` in Colab and select a GPU runtime
   (Runtime -> Change runtime type -> GPU).
2. Organize the C-NMC data so an `ImageFolder` sees two class folders:
   `C-NMC_training_data/blast/*.bmp` and `.../normal/*.bmp`. Filenames must keep
   the `UID_<patient>_...` token.
3. Edit `DATA_DIR` and `OUT_DIR` in the **Setup** cell.
4. Run all cells. Outputs are written to `OUT_DIR`.

## Reproducibility

- Fixed seed (`SEED = 42`) across Python, NumPy, and PyTorch; cuDNN deterministic.
- The exact patient-level split is saved to `patient_split.json`.
- Final hyper-parameters are set in the Setup cell
  (LR 5e-5, 60 epochs, patience 15, batch 32, alpha=1.0, beta=0.7,
  ACCEPT top 60% / REVIEW bottom 20% pseudo-labels, target coverage 0.70).

## Outputs

| File | Contents |
|---|---|
| `patient_split.json` | per-split image and patient counts (paper section 2.1) |
| `RESULTS_SUMMARY.json` | headline metrics + ECE + attention tests |
| `table3_comparison.json` | Proposed vs Confidence-Thresholding / MC-Dropout / Deep-Ensemble |
| `attention_metrics.json` | ACCEPT-vs-REVIEW attention statistics |
| `figures/` | Fig. 1, Fig. 4, Fig. 5 (300 dpi) |

## Notes and limitations

- The C-NMC test partition is small at the patient level, so the false-negative
  confidence interval is wide even when the point estimate is low; large-scale
  and external (e.g. ALL-IDB) validation is left as future work.
- In this dataset the two classes come from disjoint patient groups (blast
  patients have numeric IDs; normal patients use `H`-prefixed IDs), a known
  C-NMC characteristic noted in the paper.

## Dataset

C-NMC 2019 (B-ALL blast vs normal leukocytes). Obtain it from the official source
and place it as described above; the dataset is not redistributed here.
