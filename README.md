# MultiQRSNet

Code and patient-level predictions for:

> Yalman H, Cimci M, Sahin YH, Abdinli N, Bora C, Buyukbayrak AH, Ozkan E, Ikitimur B, Yalin K.
> *MultiQRSNet: Explainable Deep Learning for ECG-Based Classification of Outflow Tract Ventricular Premature Contraction Origin.* (submitted)

MultiQRSNet is a one-dimensional residual convolutional network that classifies the origin of outflow tract ventricular premature contractions (RVOT vs LVOT; exploratory three-class with intramural origin) from 2.5-s, 12-lead ECG segments sampled at 1 kHz.

## Reported model

All results in the paper were produced with one configuration:

- **Input:** 12 leads, each centred on its median and scaled by its interquartile range, plus their 12 first-order derivatives (24 channels × 2,500 samples).
- **Maximal deflection index (MDI):** a 12-dimensional vector, concatenated with the 256 global-average-pooled features (268 features in total).
- **Head:** Linear 268→128, ReLU, dropout 0.5, Linear 128→2 (or 128→3 for the three-class task).
- **Training:** label-smoothing cross-entropy (0.1), AdamW (lr 5e-4, weight decay 1e-3), cosine annealing with warm restarts (T₀ = 10), batch size 64, signal-only Mixup (p = 0.5, Beta(0.2, 0.2)) and ECG augmentation.
- **Model selection:** a fixed 50 epochs, using the final-epoch weights; no early stopping or checkpoint selection.

Seeds are fixed (42), but training on GPU and Apple-silicon (MPS) back-ends is not bit-wise deterministic. Two runs of the same configuration on the same folds gave 95.1% and 90.3% patient-level accuracy. **The prediction files in `predictions/` are therefore the reproducible record of the reported results.** Retraining will give similar but not identical numbers.

## Reproducing the reported metrics (no patient data needed)

```bash
pip install numpy scipy scikit-learn
python3 reproduce_metrics.py
```

This recomputes every model metric in the paper from the files below.

| Reported result | Script that produced it | File in `predictions/` |
|---|---|---|
| Internal 5-fold CV, primary run: 95.1% (98/103), AUROC 0.978 (Table 2, row 1); segment-level and k-segment analyses | `scripts/multiqrsnet_cv_raw.py` | `internal_cv_primary_segments.csv` |
| Fold assignment (StratifiedGroupKFold, 5 folds, seed 42) | `scripts/multiqrsnet_cv_raw.py` | `cv_fold_assignment.csv` |
| Segment-level fold accuracies | `scripts/multiqrsnet_cv_raw.py` | `internal_cv_primary_fold_accuracy_segment_level.csv` |
| Input-configuration analysis (Supplementary Table S4), including the independent run of the reported configuration, 90.3% (Table 2, row 2) | `scripts/multiqrsnet_ablation_metrics.py` | `input_configuration_patient_level.csv` |
| Public database (Zheng et al.), locked model: 82.6% (276/334), AUROC 0.900 (Table 2, row 4) | `scripts/multiqrsnet_external.py` | `public_database_zheng_patient_level.csv` |
| Independent clinical cohort, locked model: 87.0% (40/46) (Table 2, row 3) | `scripts/multiqrsnet_external.py` | not available (see note) |
| Three-class CV: 70.5% (91/129) (Table 2, row 5) | `scripts/multiqrsnet_cv_3class.py` | not included (see note) |
| Reader comparison: McNemar p values (Table 3) | — | recomputed from the paired counts in `reproduce_metrics.py` |
| Grad-CAM examples (Supplementary Fig. S2) | `scripts/multiqrsnet_figures.py` | — (separate run without augmentation; illustrative only) |

**Notes**

- **Independent clinical cohort:** `multiqrsnet_external.py` writes its patient-level output to a fixed file name, and the clinical-cohort output was later overwritten by the public-database run. The reported 40/46 comes from the console output of that run. Pass a distinct output name when re-running.
- **Three-class task:** out-of-fold predictions were saved to a cloud drive and are not yet included. [Authors: add `three_class_patient_level.csv` if available.]
- **Bootstrap CIs:** confidence intervals resample patients (5,000 replicates). The original CIs used an unseeded generator, so `reproduce_metrics.py` (seed 42) may differ in the last decimal.
- **Patient identifiers:** institutional patients are pseudonymised as `P001`–`P103`. The key is held by the corresponding author and is not distributed. Public-database records keep their original IDs.

## Data

Institutional ECG data cannot be shared publicly because they contain patient information. They are available from the corresponding author on reasonable request, subject to ethics approval and a data-sharing agreement. The public database is available from Zheng et al., *Sci Data* 2020;7:98 (https://doi.org/10.1038/s41597-020-0440-8).

The training scripts expect a cache `.npz` with `X` (N × 12 × 2500, raw 12-lead ECG at 1 kHz), `y` (1 = RVOT, 0 = LVOT; 2 = intramural for three-class) and `grp` (patient ID).

## Reporting

The study follows TRIPOD+AI and the EHRA artificial intelligence checklist; both completed checklists are in the paper's Supplementary Material.

## Contact

Hakan Yalman, MD — hakan.yalman.96@gmail.com
