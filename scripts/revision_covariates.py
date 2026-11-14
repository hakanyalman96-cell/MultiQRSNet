#!/usr/bin/env python3
"""
Revision analyses that need no retraining (JECG-D-26-00848):
  B3  single-segment model vs readers (paired McNemar), 1,000 random draws of one segment per patient
  B5  model probability adjusted for age, hypertension and diabetes; accuracy by age group
  B6  model and reader accuracy by ablation sub-site

Inputs
  predictions/internal_cv_primary_segments.csv   (in the repository)
  covariates.csv  -- prepared by the corresponding author from the pseudonymisation key, NO names/IDs:
      patient   P001..P103 (same pseudonyms as the repository)
      age       years
      htn       1/0
      dm        1/0
      subsite   e.g. RVOT-septal, RVOT-free-wall, LCC, RCC, NCC, L-R commissure, AMC, LV summit, LVOT-subvalvular
      reader_A  reader A final call, 1 = RVOT, 0 = LVOT
      reader_B  reader B final call, 1 = RVOT, 0 = LVOT

Run:  python3 scripts/revision_covariates.py predictions/internal_cv_primary_segments.csv covariates.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import statsmodels.api as sm

SEG = sys.argv[1] if len(sys.argv) > 1 else "predictions/internal_cv_primary_segments.csv"
COV = sys.argv[2] if len(sys.argv) > 2 else None
rng = np.random.default_rng(42)


def mcnemar_exact(model_ok, reader_ok):
    b = int(np.sum(model_ok & ~reader_ok)); c = int(np.sum(~model_ok & reader_ok))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


seg = pd.read_csv(SEG)
pat = seg.groupby("patient").agg(true=("true_label_RVOT1_LVOT0", "first"), prob=("prob_RVOT", "mean")).reset_index()
pat["ok"] = (pat.prob > 0.5).astype(int) == pat.true
print(f"All segments: {pat.ok.sum()}/{len(pat)} = {pat.ok.mean()*100:.1f}%")

# ---- B3: one random segment per patient, 1,000 draws
groups = {p: g.prob_RVOT.values for p, g in seg.groupby("patient")}
truth = pat.set_index("patient").true
draws = []
for _ in range(1000):
    pr = pd.Series({p: v[rng.integers(len(v))] for p, v in groups.items()})
    draws.append(((pr > 0.5).astype(int) == truth.loc[pr.index]).values)
draws = np.array(draws)  # (1000, n_patients) correctness
acc1 = draws.mean(1)
print(f"Single segment (k = 1, 1,000 draws): mean {acc1.mean()*100:.1f}%, 95% range {np.percentile(acc1, 2.5)*100:.1f}-{np.percentile(acc1, 97.5)*100:.1f}%")

if COV is None:
    print("\nNo covariates.csv given: B3 reader comparison, B5 and B6 skipped."); sys.exit()

cov = pd.read_csv(COV)
d = pat.merge(cov, on="patient", how="inner")
assert len(d) == len(pat), f"covariates matched {len(d)} of {len(pat)} patients"
order = list(groups.keys())
d = d.set_index("patient").loc[order].reset_index()

# ---- B3 continued: readers vs model (all segments, and k = 1)
for r in ["reader_A", "reader_B"]:
    if r not in d: continue
    rok = (d[r].values == d.true.values)
    b, c, p = mcnemar_exact(d.ok.values, rok)
    ps = np.array([mcnemar_exact(draws[i], rok)[2] for i in range(len(draws))])
    bs = np.array([mcnemar_exact(draws[i], rok)[0] for i in range(len(draws))])
    cs = np.array([mcnemar_exact(draws[i], rok)[1] for i in range(len(draws))])
    print(f"{r}: {rok.sum()}/{len(rok)} correct | vs all-segment model: {b} vs {c}, p = {p:.2g} | "
          f"vs single-segment model: median discordant {np.median(bs):.0f} vs {np.median(cs):.0f}, "
          f"median p = {np.median(ps):.2g}, p < 0.05 in {np.mean(ps < 0.05)*100:.1f}% of draws")

# ---- B5: adjusted association and age strata
eps = 1e-4
d["logit_lvot"] = np.log((1 - d.prob + eps) / (d.prob + eps))   # model evidence for LVOT
d["lvot"] = 1 - d.true
X = sm.add_constant(d[["logit_lvot", "age", "htn", "dm"]].astype(float))
fit = sm.Logit(d.lvot, X).fit(disp=0)
ci = fit.conf_int()
print("\nB5 logistic regression for LVOT origin (per unit of model logit; per year of age):")
for k in ["logit_lvot", "age", "htn", "dm"]:
    print(f"  {k:10s} OR {np.exp(fit.params[k]):.2f} (95% CI {np.exp(ci.loc[k, 0]):.2f}-{np.exp(ci.loc[k, 1]):.2f}), p = {fit.pvalues[k]:.3g}")
Xd = sm.add_constant(d[["age", "htn", "dm"]].astype(float)); fd = sm.Logit(d.lvot, Xd).fit(disp=0)
from sklearn.metrics import roc_auc_score
print(f"  AUROC demographics only: {roc_auc_score(d.lvot, fd.predict(Xd)):.3f} | model only: {roc_auc_score(d.lvot, d.logit_lvot):.3f} | both: {roc_auc_score(d.lvot, fit.predict(X)):.3f}")
for lab, m in [("age < 55", d.age < 55), ("age >= 55", d.age >= 55)]:
    sub = d[m]
    print(f"  {lab}: model {sub.ok.sum()}/{len(sub)} = {sub.ok.mean()*100:.1f}% "
          f"(RVOT {sub[sub.true==1].ok.sum()}/{(sub.true==1).sum()}, LVOT {sub[sub.true==0].ok.sum()}/{(sub.true==0).sum()})")

# ---- B6: by sub-site
print("\nB6 accuracy by sub-site (correct / n):")
rows = []
for s, g in d.groupby("subsite"):
    r = dict(subsite=s, n=len(g), model=f"{g.ok.sum()}/{len(g)}")
    for rd in ["reader_A", "reader_B"]:
        if rd in g: r[rd] = f"{(g[rd] == g.true).sum()}/{len(g)}"
    rows.append(r)
tab = pd.DataFrame(rows).sort_values("n", ascending=False)
print(tab.to_string(index=False))
tab.to_csv("revision_subsite_table.csv", index=False)
print("\nwrote revision_subsite_table.csv")
