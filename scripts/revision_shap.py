#!/usr/bin/env python3
"""
SHAP attribution for the final MultiQRSNet configuration (ECG + derivatives, no MDI, no polarity inversion).
Revision of Fig. 3 (JECG-D-26-00848) and supporting analysis for Reviewer 2, comment 2.8.

Out-of-fold: in each of the 5 folds (same StratifiedGroupKFold, random_state=42) a model is trained on the
training patients and explains up to K segments of each held-out patient (GradientExplainer, background =
32 random training segments). Output explained: probability of RVOT origin.

Outputs (./revision_out/):
  shap_fig3_beeswarm.png       Fig. 3: per-channel time-aggregated signed SHAP, (a) LVOT and (b) RVOT patients
  shap_channel_ranking.csv     mean |SHAP| per channel, overall and by class
  shap_time_profile.png        mean |SHAP| over time (all channels), index-PVC window marked
  shap_summary.txt             ranking + share of |SHAP| inside index PVC +/- 150 ms (chance = 12%)
  shap_patient_values.npz      raw patient-level values for re-plotting

Run (Colab, GPU):  python3 scripts/revision_shap.py /content/drive/MyDrive/multiqrsnet_cache_bin.npz [K=5] [seed=42]
"""
import os, sys, random
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiqrsnet_cv_raw import build_inputs
from revision_experiments import train, set_seed, locate_index_pvc
import revision_experiments as RE

CACHE = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
RE.EPOCHS = int(os.environ.get("EPOCHS", 50))
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CH = LEADS + [f"d({l})" for l in LEADS]
HALF = 150


class ProbRVOT(nn.Module):
    def __init__(s, net): super().__init__(); s.net = net
    def forward(s, x):
        m = torch.zeros(x.shape[0], 0, device=x.device)
        return torch.softmax(s.net(x, m), 1)[:, 1:2]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("revision_out", exist_ok=True)
    C = np.load(CACHE, allow_pickle=True)
    Xr = C["X"][:, :12, :].astype(np.float32); y = C["y"].astype(int); grp = np.asarray(C["grp"])
    X, M = build_inputs(Xr); M = np.ascontiguousarray(M[:, :0])
    cen = locate_index_pvc(Xr)
    rng = np.random.default_rng(SEED)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, grp))

    P_sv, P_amp, P_y, P_id, time_abs, inside_share = [], [], [], [], [], []
    for k, (tr, va) in enumerate(folds, 1):
        set_seed(SEED + k)
        net = train(X, M, y, tr, dev, polarity=False)
        model = ProbRVOT(net).eval()
        bg = torch.from_numpy(X[rng.choice(tr, 32, replace=False)]).to(dev)
        ex = shap.GradientExplainer(model, bg)
        for p in sorted(set(grp[va])):
            idx = np.where(grp == p)[0]; idx = rng.choice(idx, min(K, len(idx)), replace=False)
            sv = ex.shap_values(torch.from_numpy(X[idx]).to(dev))
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            sv = sv.reshape(len(idx), 24, -1)                      # (k, 24, T)
            P_sv.append(sv.sum(2).mean(0))                         # time-aggregated signed, patient mean
            amp = np.array([[np.abs(X[i, c, max(cen[i] - HALF, 0):cen[i] + HALF]).mean() for c in range(24)] for i in idx]).mean(0)
            P_amp.append(amp); P_y.append(y[idx[0]]); P_id.append(p)
            a = np.abs(sv).sum(1)                                  # (k, T)
            time_abs.append(np.stack([np.roll(a[j], 1990 - cen[idx[j]]) for j in range(len(idx))]).mean(0))  # aligned on index PVC
            inside_share.append(np.mean([a[j, max(cen[idx[j]] - HALF, 0):cen[idx[j]] + HALF].sum() / (a[j].sum() + 1e-12) for j in range(len(idx))]))
        print(f"fold {k} explained ({len(set(grp[va]))} patients)", flush=True)

    S = np.array(P_sv); A = np.array(P_amp); Y = np.array(P_y); T = np.array(time_abs); IS = np.array(inside_share)
    np.savez("revision_out/shap_patient_values.npz", shap=S, amp=A, y=Y, patient=np.array(P_id), time_abs=T, inside_share=IS, channels=np.array(CH))

    # ranking
    import csv
    rank = np.argsort(-np.abs(S).mean(0))
    with open("revision_out/shap_channel_ranking.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["channel", "mean_abs_all", "mean_abs_RVOT", "mean_abs_LVOT", "mean_signed_RVOT", "mean_signed_LVOT"])
        for c in rank:
            w.writerow([CH[c], f"{np.abs(S[:, c]).mean():.5f}", f"{np.abs(S[Y == 1, c]).mean():.5f}", f"{np.abs(S[Y == 0, c]).mean():.5f}",
                        f"{S[Y == 1, c].mean():.5f}", f"{S[Y == 0, c].mean():.5f}"])
    with open("revision_out/shap_summary.txt", "w") as f:
        f.write("Top channels by mean |SHAP| (patient level, out-of-fold): " + ", ".join(CH[c] for c in rank[:8]) + "\n")
        f.write(f"Share of |SHAP| within index PVC +/-{HALF} ms: median {np.median(IS)*100:.1f}% (IQR {np.percentile(IS,25)*100:.1f}-{np.percentile(IS,75)*100:.1f}%); chance {2*HALF/X.shape[2]*100:.0f}%\n")
        f.write(f"Patients: {len(Y)} (RVOT {int((Y==1).sum())}, LVOT {int((Y==0).sum())}); segments per patient: up to {K}\n")
    print(open("revision_out/shap_summary.txt").read())

    # Fig. 3 beeswarm, top 12 channels
    top = rank[:12]
    fig = plt.figure(figsize=(13, 6))
    for j, (lab, nm) in enumerate([(0, "(a) LVOT"), (1, "(b) RVOT")]):
        ax = fig.add_subplot(1, 2, j + 1); plt.sca(ax)
        m = Y == lab
        shap.summary_plot(S[m][:, top], A[m][:, top], feature_names=[CH[c] for c in top], show=False, plot_size=None, color_bar=(j == 1))
        ax.set_title(nm, loc="left"); ax.set_xlabel("SHAP value (← LVOT · RVOT →)")
    plt.tight_layout(); plt.savefig("revision_out/shap_fig3_beeswarm.png", dpi=300); plt.close()

    # time profile
    t = np.arange(T.shape[1]) - 1990
    plt.figure(figsize=(10, 3))
    for lab, nm in [(1, "RVOT"), (0, "LVOT")]:
        plt.plot(t, T[Y == lab].mean(0), label=nm, lw=1)
    plt.axvspan(-HALF, HALF, color="r", alpha=.1, label="index PVC ±150 ms")
    plt.xlabel("time relative to index PVC (ms)"); plt.ylabel("mean |SHAP|"); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig("revision_out/shap_time_profile.png", dpi=200)
    print("wrote revision_out/shap_*")


if __name__ == "__main__":
    main()
