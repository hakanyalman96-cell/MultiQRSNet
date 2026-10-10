#!/usr/bin/env python3
"""
Recompute every model metric reported in the manuscript from the saved,
pseudonymised predictions in ./predictions — no patient data or GPU needed.

    python3 reproduce_metrics.py

Requires: numpy, scikit-learn, scipy.
Patient-level prediction = mean RVOT probability over a patient's segments;
RVOT if > 0.50. Bootstrap CIs resample patients (5,000 replicates, seed 42);
because the original CIs were computed with an unseeded generator, the last
decimal of a CI bound may differ slightly from the manuscript.
"""
import csv
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

P = "predictions/"


def boot_ci(correct, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, float)
    b = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def report(name, prob, true):
    prob, true = np.asarray(prob, float), np.asarray(true, int)
    pred = (prob > 0.5).astype(int)
    c = pred == true
    lo, hi = boot_ci(c)
    print(f"\n{name}")
    print(f"  patients {len(true)} (RVOT {true.sum()}, LVOT {(true == 0).sum()})")
    print(f"  accuracy {c.sum()}/{len(c)} = {c.mean()*100:.1f}%  (95% CI {lo:.1f}-{hi:.1f})")
    print(f"  RVOT sensitivity {c[true == 1].sum()}/{(true == 1).sum()} = {c[true == 1].mean()*100:.1f}%")
    print(f"  LVOT sensitivity {c[true == 0].sum()}/{(true == 0).sum()} = {c[true == 0].mean()*100:.1f}%")
    print(f"  AUROC {roc_auc_score(true, prob):.3f}  AUPRC {average_precision_score(true, prob):.3f}  "
          f"Brier {brier_score_loss(true, prob):.3f}")
    return c


# ---------- 1. internal cross-validation, primary run ----------
seg = defaultdict(list); lab = {}
with open(P + "internal_cv_primary_segments.csv") as fh:
    for r in csv.DictReader(fh):
        seg[r["patient"]].append(float(r["prob_RVOT"])); lab[r["patient"]] = int(r["true_label_RVOT1_LVOT0"])
pats = sorted(seg)
pp = np.array([np.mean(seg[p]) for p in pats]); pt = np.array([lab[p] for p in pats])
c_primary = report("Internal 5-fold CV, primary run (Table 2, row 1)", pp, pt)

sp = np.concatenate([seg[p] for p in pats]); st = np.concatenate([[lab[p]] * len(seg[p]) for p in pats])
print(f"  segment level: {len(sp)} segments, accuracy {((sp > 0.5) == st).mean()*100:.1f}%, "
      f"AUROC {roc_auc_score(st, sp):.3f}")
n = np.array([len(seg[p]) for p in pats])
print(f"  segments per patient: median {np.median(n):.0f}, range {n.min()}-{n.max()}")

rng = np.random.default_rng(42)
for k in (1, 3, 10):
    accs = []
    for _ in range(1000):
        m = [np.mean(rng.choice(seg[p], min(k, len(seg[p])), replace=False)) for p in pats]
        accs.append(((np.array(m) > 0.5) == pt).mean())
    print(f"  k = {k:2d} segments per patient: mean accuracy {np.mean(accs)*100:.1f}%")

# ---------- 2. input-configuration analysis (Supplementary Table S4) ----------
rows = list(csv.DictReader(open(P + "input_configuration_patient_level.csv")))
t = np.array([int(r["true_label_RVOT1_LVOT0"]) for r in rows])
for cfg in ["ecg_only", "ecg_deriv", "ecg_mdi", "full", "mdi_only", "deriv_only"]:
    label = cfg + ("  <- independent run of the reported configuration (Table 2, row 2)" if cfg == "full" else "")
    report(f"Input configuration: {label}", [float(r["prob_RVOT_" + cfg]) for r in rows], t)

# paired comparison of the two runs of the reported configuration
full = {r["patient"]: float(r["prob_RVOT_full"]) for r in rows}
c_full = np.array([(full[p] > 0.5) == lab[p] for p in pats])
b, d = int((c_primary & ~c_full).sum()), int((~c_primary & c_full).sum())
print(f"\nPrimary vs independent run: {b} vs {d} discordant, exact McNemar p = {binomtest(min(b, d), b + d, 0.5).pvalue:.3f}")

# ---------- 3. public database (Zheng et al.), locked model ----------
rows = list(csv.DictReader(open(P + "public_database_zheng_patient_level.csv")))
report("Public database, Zheng et al. (Table 2, row 4)",
       [float(r["prob_RVOT"]) for r in rows], [int(r["true_label_RVOT1_LVOT0"]) for r in rows])

# ---------- 4. reader comparison (Table 3) ----------
# Per-patient reader classifications are not public; the paired counts reported in Table 3
# (model-only correct vs reader-only correct) reproduce the exact McNemar p values:
for name, b, d in [("Reader A", 29, 0), ("Reader B", 28, 0)]:
    print(f"{name}: {b} vs {d} discordant, exact McNemar p = {binomtest(min(b, d), b + d, 0.5).pvalue:.1e}")
