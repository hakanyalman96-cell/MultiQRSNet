#!/usr/bin/env python3
"""
Recompute every model metric of the revised manuscript from the pseudonymised
predictions in ./predictions/revision — no patient data or GPU needed.

    python3 reproduce_revision.py

Final configuration: 12 leads + first derivatives (24 channels), no timing
feature (formerly "MDI"), no polarity-inversion augmentation. Five training
seeds (42, 1, 2, 3, 4) on identical patient-level folds; seed 42 is the locked
primary model. Patient-level prediction = mean RVOT probability over the
patient's segments; RVOT if > 0.50.

Requires: numpy, scikit-learn, scipy.
"""
import csv
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

P = "predictions/revision/"
SEEDS = ["42", "1", "2", "3", "4"]


def load_wide(fn):
    rows = list(csv.DictReader(open(P + fn)))
    y = np.array([int(r["true_label_RVOT1_LVOT0"]) for r in rows])
    probs = {s: np.array([float(r[f"prob_RVOT_seed{s}"]) for r in rows]) for s in SEEDS}
    ids = [r.get("patient", r.get("zheng_record_id")) for r in rows]
    return ids, y, probs


def metrics(y, p):
    pred = (p > 0.5).astype(int)
    c = pred == y
    return dict(acc=c.mean() * 100, n=c.sum(), N=len(c), rvot=c[y == 1].mean() * 100,
                lvot=c[y == 0].mean() * 100, auroc=roc_auc_score(y, p))


def boot_ci(correct, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, float)
    b = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n)]
    return np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100


def table(title, fn):
    ids, y, probs = load_wide(fn)
    print(f"\n{title}  ({fn}; RVOT {y.sum()}, LVOT {(y == 0).sum()})")
    rows = []
    for s in SEEDS:
        m = metrics(y, probs[s]); rows.append(m)
        extra = ""
        if s == "42":
            lo, hi = boot_ci((probs[s] > 0.5).astype(int) == y)
            extra = f"  95% CI {lo:.1f}-{hi:.1f}"
        print(f"  seed {s:>2}: acc {m['n']}/{m['N']} = {m['acc']:.1f}%  RVOT {m['rvot']:.1f}%  "
              f"LVOT {m['lvot']:.1f}%  AUROC {m['auroc']:.3f}{extra}")
    for k in ["acc", "rvot", "lvot", "auroc"]:
        v = np.array([r[k] for r in rows])
        fmt = ".3f" if k == "auroc" else ".1f"
        print(f"  {k:5s} mean ± SD {v.mean():{fmt}} ± {v.std(ddof=1):{fmt}}  (range {v.min():{fmt}}-{v.max():{fmt}})")
    return ids, y, probs


def mcnemar(model_correct, reader_correct):
    b = int(np.sum(model_correct & ~reader_correct)); c = int(np.sum(~model_correct & reader_correct))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


print("=" * 78)
print("Supplementary Table S6 / Table 2: binary task, five seeds per configuration")
ids, y, final = table("Final model, internal CV", "internal_cv_final_5seeds.csv")
table("Final model, independent clinical cohort", "independent_cohort_final_5seeds.csv")
table("Final model, public database (Zheng et al.)", "public_database_final_5seeds.csv")
table("Original submission configuration (timing feature + polarity inversion), internal CV",
      "internal_cv_original_submission_config_5seeds.csv")
table("Without polarity inversion (timing feature retained), internal CV",
      "internal_cv_no_polarity_inversion_5seeds.csv")
table("Without polarity inversion, independent clinical cohort",
      "independent_cohort_no_polarity_inversion_5seeds.csv")

print("\n" + "=" * 78)
print("Supplementary Table S7: beats other than the index PVC")
table("PVC-only 500-ms window, internal CV", "internal_cv_pvc_window500_5seeds.csv")
table("PVC-only 500-ms window, independent clinical cohort", "independent_cohort_pvc_window500_5seeds.csv")
rows = list(csv.DictReader(open(P + "internal_cv_masking_3seeds.csv")))
print("\nMasking outside ±150 ms of the index PVC at inference (three separate runs)")
for s in ["42", "1", "2"]:
    r = [x for x in rows if x["seed"] == s]
    yy = np.array([int(x["true_label_RVOT1_LVOT0"]) for x in r])
    pf = np.array([float(x["prob_RVOT_full"]) for x in r]); pm = np.array([float(x["prob_RVOT_masked"]) for x in r])
    g = np.median([float(x["gradcam_share_pvc_window"]) for x in r]) * 100
    print(f"  seed {s:>2}: full {((pf > .5) == yy).mean()*100:.1f}% (AUROC {roc_auc_score(yy, pf):.3f})  "
          f"masked {((pm > .5) == yy).mean()*100:.1f}% (AUROC {roc_auc_score(yy, pm):.3f})  Grad-CAM share {g:.1f}%")

print("\n" + "=" * 78)
print("Table 3 / Supplementary Tables S3 and S5: readers and sub-sites (locked model, seed 42)")
rd = {r["patient"]: r for r in csv.DictReader(open(P + "readers_and_subsites.csv"))}
yy = np.array([int(rd[i]["true_label_RVOT1_LVOT0"]) for i in ids]); assert (yy == y).all()
mc = (final["42"] > 0.5).astype(int) == y
for col, name in [("readerA_mindray", "Reader A (Mindray)"), ("readerB_mindray", "Reader B (Mindray)"),
                  ("readerA_carto", "Reader A (CARTO segment)"), ("readerB_carto", "Reader B (CARTO segment)")]:
    rp = np.array([int(rd[i][col]) for i in ids]); rc = rp == y
    b, c, p = mcnemar(mc, rc)
    print(f"  {name:26s} acc {rc.sum()}/{len(rc)} = {rc.mean()*100:.1f}%  RVOT {rc[y==1].mean()*100:.1f}%  "
          f"LVOT {rc[y==0].mean()*100:.1f}%  | model-only {b}, reader-only {c}, exact McNemar p = {p:.1e}")
for s in SEEDS:
    m = (final[s] > 0.5).astype(int) == y
    ps = [mcnemar(m, np.array([int(rd[i][c]) for i in ids]) == y)[2] for c in ["readerA_mindray", "readerB_mindray"]]
    print(f"  seed {s:>2}: McNemar p (A, B) = {ps[0]:.1e}, {ps[1]:.1e}")
sub = np.array([rd[i]["subsite"] for i in ids])
ra = np.array([int(rd[i]["readerA_mindray"]) for i in ids]) == y
rb = np.array([int(rd[i]["readerB_mindray"]) for i in ids]) == y
print("\n  Sub-site                n   model  reader A  reader B")
for s in ["RVOT_septal", "RVOT_free_wall", "LCC", "RCC", "NCC", "LR_commissure", "AMC", "LV_summit", "subvalvular_LVOT"]:
    k = sub == s
    print(f"  {s:20s} {k.sum():4d}   {mc[k].sum():4d}   {ra[k].sum():6d}   {rb[k].sum():6d}")
