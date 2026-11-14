# MultiQRSNet

Code and patient-level predictions for:

> Yalman H, Cimci M, Sahin YH, Abdinli N, Bora C, Buyukbayrak AH, Ozkan E, Ikitimur B, Yalin K.
> *MultiQRSNet: Explainable Deep Learning for ECG-Based Classification of Outflow Tract Premature Ventricular Complex Origin.* J Electrocardiol (under revision, JECG-D-26-00848).

MultiQRSNet is a one-dimensional residual convolutional network that classifies the origin of outflow tract premature ventricular complexes (PVCs; RVOT vs LVOT, and an exploratory three-class task including intramural origin) from 2.5-s, 12-lead ECG segments sampled at 1 kHz.

## Final model (revised manuscript)

- **Input:** 12 leads, each centred on its median and scaled by its interquartile range, plus their 12 first-order derivatives (24 channels × 2,500 samples).
- **Architecture:** convolutional stem, three residual blocks with squeeze-and-excitation, global average pooling (256 features), head Linear 256→128, ReLU, dropout 0.5, Linear 128→2 (128→3 for the three-class task).
- **Training:** label-smoothing cross-entropy (0.1), AdamW (lr 5e-4, weight decay 1e-3), cosine annealing with warm restarts (T₀ = 10), batch size 64, 50 epochs (final-epoch weights), signal-only Mixup (p = 0.5, Beta(0.2, 0.2)) and ECG augmentation (baseline wander, amplitude scaling, additive noise, time shift, short signal dropout).
- **Seeds:** five training seeds (42, 1, 2, 3, 4) on identical patient-level folds; the seed-42 model is the locked primary model.

### Changes from the original submission

The original submission also used (i) a 12-dimensional timing feature, called the "maximal deflection index" (MDI), and (ii) global polarity-inversion augmentation. Both were removed in revision:

- The timing feature was computed in a 200-sample window centred on the segment (samples 1,150–1,350), whereas the index PVC lies about 2.0 s into each segment (median sample 1,990). It therefore did not describe the index PVC, was not the maximum deflection index of Daniels et al., and added no information.
- Polarity inversion (all 24 channels multiplied by −1 with probability 0.2) has no physiological basis for a task in which lead polarity is diagnostic. Removing it did not change performance.

The original-submission prediction files are kept in `predictions/` as the record of the submitted results; the revision files are in `predictions/revision/`.

## Repository contents

```
README.md
reproduce_revision.py         recompute every model metric of the revised manuscript from predictions/revision/
reproduce_metrics.py          recompute the metrics of the original submission from predictions/
make_inference_example.py     forward pass of the reported architecture on a synthetic ECG
requirements.txt
LICENSE
predictions/                  original submission: pseudonymised predictions and fold assignment
predictions/revision/         revised manuscript: five-seed predictions, sensitivity analyses, readers
scripts/                      training / evaluation scripts used for the paper
  revision_experiments.py         revision: binary CV, external evaluation, masking and PVC-window runs
  revision_3class.py              revision: three-class CV (final inputs)
  revision_shap.py                revision: out-of-fold SHAP (Fig. 3, Supplementary Fig. S2)
  diag_3class_cache.py            checks a three-class cache against the binary cache
  multiqrsnet_cv_raw.py           binary 5-fold CV (original submission)
  multiqrsnet_ablation_metrics.py input-configuration analysis
  multiqrsnet_external.py         locked-model external evaluation
  multiqrsnet_cv_3class.py        exploratory three-class CV
  multiqrsnet_figures.py          Grad-CAM examples (Supplementary Fig. S1)
  zheng_format.py                 prepares the public Zheng et al. database
reporting/                    completed TRIPOD+AI and EHRA AI checklists
```

Trained weights are not distributed: training is not bit-wise deterministic, and the prediction files are the record of the reported results.

## Reproducing the revised manuscript (no patient data needed)

```bash
pip install numpy scipy scikit-learn
python3 reproduce_revision.py
```

| Reported result (revised manuscript) | File in `predictions/revision/` |
|---|---|
| Internal CV, locked model 91.3% (94/103), AUROC 0.979; five seeds 90.7 ± 1.9% (Table 2, Supplementary Table S6) | `internal_cv_final_5seeds.csv` |
| Independent clinical cohort, locked model 87.0% (40/46); five seeds 85.7 ± 1.9% | `independent_cohort_final_5seeds.csv` |
| Public database (Zheng et al.), locked model 81.4% (272/334), AUROC 0.895; five seeds 83.6 ± 1.5% | `public_database_final_5seeds.csv` |
| Original configuration, five seeds 92.2 ± 1.9% (Supplementary Table S6) | `internal_cv_original_submission_config_5seeds.csv` |
| Without polarity inversion, five seeds 91.8 ± 1.9% (CV) and 87.4 ± 2.4% (independent cohort) | `internal_cv_no_polarity_inversion_5seeds.csv`, `independent_cohort_no_polarity_inversion_5seeds.csv` |
| Masking outside ±150 ms of the index PVC; Grad-CAM share (Supplementary Table S7) | `internal_cv_masking_3seeds.csv` |
| PVC-only 500-ms window, 91.7 ± 1.3% (CV) and 81.3 ± 1.2% (independent cohort) | `internal_cv_pvc_window500_5seeds.csv`, `independent_cohort_pvc_window500_5seeds.csv` |
| Reader comparison and exact McNemar tests (Table 3, Supplementary Table S3); modality-matched re-reading; sub-site accuracy (Supplementary Table S5) | `readers_and_subsites.csv` |
| SHAP channel ranking (Fig. 3) | `shap_channel_ranking.csv` |

Not reproducible from this repository: the adjustment for age, hypertension and diabetes (requires patient characteristics, available on request) and the exploratory three-class task (`revision_3class.py`; predictions to be added).

## Reproducing the original submission

```bash
pip install numpy scipy scikit-learn
python3 reproduce_metrics.py
```

This recomputes the metrics of the original submission (timing feature and polarity inversion; superseded) from the files below.

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
- **Patient identifiers:** development-cohort patients are pseudonymised as `P001`–`P103` (identical in the original and revision files) and independent-cohort patients as `X01`–`X46`. The keys are held by the corresponding author and are not distributed. Public-database records keep their original IDs.

## Data

Institutional ECG data cannot be shared publicly because they contain patient information. They are available from the corresponding author on reasonable request, subject to ethics approval and a data-sharing agreement. The public database is available from Zheng et al., *Sci Data* 2020;7:98 (https://doi.org/10.1038/s41597-020-0440-8).

The training scripts expect a cache `.npz` (never committed; see `.gitignore`) with `X` (N × 12 × 2500, raw 12-lead ECG at 1 kHz), `y` (1 = RVOT, 0 = LVOT; 2 = intramural for three-class) and `grp` (patient ID).

## Reporting

The study follows TRIPOD+AI and the EHRA artificial intelligence checklist; both completed checklists are in the paper's Supplementary Material.

## Contact

Hakan Yalman, MD — hakan.yalman.96@gmail.com
