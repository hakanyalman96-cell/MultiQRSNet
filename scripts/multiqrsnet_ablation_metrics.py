"""
MultiQRSNet — ablation study + full patient-level metrics (reviewer #7, #9, #10).

Run on your Mac (Apple Silicon MPS or CPU) or any CUDA machine:

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python3 multiqrsnet_ablation_metrics.py

Requires: torch, numpy, scikit-learn, scipy
    pip3 install torch numpy scikit-learn scipy

INPUT  : a cache .npz with  X (N,12,2500) raw 12-lead ECG @1 kHz,
         y (N,) in {0=LVOT, 1=RVOT},  grp (N,) patient IDs.
         Edit CACHE below to point at multiqrsnet_cache_bin.npz.

OUTPUT : ablation_results.npz  (all OOF probs per configuration) AND a printed
         table you can paste into the manuscript.  Every configuration uses the
         SAME patient-level 5-fold splits (seed 42), so differences are paired.

Six configurations (reviewer #7):
    1 ecg_only        12 raw leads
    2 ecg_deriv       12 leads + 12 first-order derivatives  (24 ch)
    3 ecg_mdi         12 leads + 12-dim MDI prior
    4 full            24 ch + MDI prior          (the manuscript model)
    5 mdi_only        MDI prior only (no signal branch)
    6 deriv_only      12 derivative channels only

Metrics (reviewer #9, #10), all PATIENT-LEVEL with cluster (patient) bootstrap
95% CI: accuracy, sensitivity, specificity, PPV, NPV, balanced accuracy,
macro-F1, AUROC, AUPRC, Brier score, calibration intercept & slope, and the
confusion matrix.  Positive class = RVOT (label 1); decision threshold = 0.50.
"""
import os, numpy as np, torch
from torch import nn, optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy import stats

# ================= CONFIG — edit this line =================
CACHE = "multiqrsnet_cache_bin.npz"      # your binary cache (103 patients)
SEED, N_SPLITS, EPOCHS, BATCH = 42, 5, 50, 64
# ==========================================================

DEV = ("cuda" if torch.cuda.is_available()
       else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
             else "cpu"))
print("device:", DEV)

# ---------- preprocessing ----------
def robust_normalize(x):
    med = np.median(x); q1, q3 = np.percentile(x, [25, 75])
    return (x - med) / ((q3 - q1) + 1e-6)

def compute_mdi(lead):
    s = lead - np.median(lead); mid = len(s) // 2
    w = s[mid - 100: mid + 100]
    return np.argmax(np.abs(w)) / len(w)

def build_inputs(X_raw):
    N = X_raw.shape[0]
    leads = np.zeros((N, 12, 2500), np.float32)
    deriv = np.zeros((N, 12, 2500), np.float32)
    MDI   = np.zeros((N, 12), np.float32)
    for i in range(N):
        for c in range(12):
            n = robust_normalize(X_raw[i, c].astype(np.float32))
            leads[i, c] = n
            deriv[i, c] = np.gradient(n)
            MDI[i, c]   = compute_mdi(X_raw[i, c])
    return leads, deriv, MDI

def make_signal(leads, deriv, cfg):
    """Return the signal tensor (N,C,2500) for a configuration, or None."""
    if cfg == "ecg_only":  return leads
    if cfg == "ecg_deriv": return np.concatenate([leads, deriv], axis=1)
    if cfg == "ecg_mdi":   return leads
    if cfg == "full":      return np.concatenate([leads, deriv], axis=1)
    if cfg == "mdi_only":  return None
    if cfg == "deriv_only":return deriv
    raise ValueError(cfg)

def uses_mdi(cfg):  return cfg in ("ecg_mdi", "full", "mdi_only")

# ---------- model (configurable) ----------
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__(); h = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(True), nn.Linear(h, ch), nn.Sigmoid())
    def forward(self, x):
        s = x.mean(2); return x * self.fc(s).unsqueeze(2)

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=2, k=5):
        super().__init__(); p = k // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride, p); self.n1 = nn.InstanceNorm1d(cout, affine=True)
        self.c2 = nn.Conv1d(cout, cout, k, 1, p);     self.n2 = nn.InstanceNorm1d(cout, affine=True)
        self.se = SEBlock(cout); self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride), nn.InstanceNorm1d(cout, affine=True))
    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        o = torch.relu(self.n1(self.c1(x))); o = self.n2(self.c2(o)); o = self.se(o)
        return torch.relu(o + idt)

class Net(nn.Module):
    """Signal branch (optional) + MDI branch (optional), fused before the head."""
    def __init__(self, in_ch, mdi_dim, n_classes=2, use_signal=True, use_mdi=True):
        super().__init__(); self.use_signal, self.use_mdi = use_signal, use_mdi
        feat = 0
        if use_signal:
            self.stem = nn.Sequential(nn.Conv1d(in_ch, 64, 7, 1, 3),
                                      nn.InstanceNorm1d(64, affine=True), nn.ReLU(True), nn.MaxPool1d(2))
            self.rb1 = ResidualBlock(64, 64); self.rb2 = ResidualBlock(64, 128); self.rb3 = ResidualBlock(128, 256)
            feat += 256
        if use_mdi: feat += mdi_dim
        self.head = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(128, n_classes))
    def forward(self, x, mdi):
        parts = []
        if self.use_signal:
            h = self.stem(x); h = self.rb1(h); h = self.rb2(h); h = self.rb3(h); parts.append(h.mean(2))
        if self.use_mdi: parts.append(mdi)
        return self.head(torch.cat(parts, 1))

# ---------- augmentation / mixup ----------
def augment(x):
    import random; L = x.shape[1]
    if random.random() < .5:
        t = np.linspace(0, 2*np.pi*random.uniform(1, 3), L); x = x + (0.05*np.sin(t)).astype(np.float32)
    if random.random() < .5: x = x * random.uniform(.85, 1.15)
    if random.random() < .5: x = x + np.random.normal(0, .02, x.shape).astype(np.float32)
    if random.random() < .5: x = np.roll(x, random.randint(-50, 50), axis=1)
    if random.random() < .2: x = -x
    return x

def mixup(x, a=0.2):
    lam = np.random.beta(a, a); lam = max(lam, 1-lam)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[perm]

class DS(torch.utils.data.Dataset):
    def __init__(self, X, M, y, idx, train, use_signal):
        self.X, self.M, self.y, self.idx, self.train, self.us = X, M, y, idx, train, use_signal
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        x = self.X[i].copy() if self.us else np.zeros((1, 1), np.float32)
        if self.train and self.us: x = augment(x)
        return torch.from_numpy(x), torch.from_numpy(self.M[i]), int(self.y[i])

def train_fold(X, M, y, tr, va, in_ch, mdi_dim, use_signal, use_mdi):
    net = Net(in_ch, mdi_dim, use_signal=use_signal, use_mdi=use_mdi).to(DEV)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    tl = torch.utils.data.DataLoader(DS(X, M, y, tr, True, use_signal), batch_size=BATCH, shuffle=True)
    vl = torch.utils.data.DataLoader(DS(X, M, y, va, False, use_signal), batch_size=256, shuffle=False)
    for _ in range(EPOCHS):
        net.train()
        for xb, mb, yb in tl:
            xb, mb, yb = xb.to(DEV), mb.to(DEV), yb.to(DEV)
            if use_signal and np.random.rand() < 0.5: xb = mixup(xb)
            opt.zero_grad(); crit(net(xb, mb), yb).backward(); opt.step()
        sch.step()
    net.eval(); probs = []
    with torch.no_grad():
        for xb, mb, _ in vl:
            probs.extend(torch.softmax(net(xb.to(DEV), mb.to(DEV)), 1)[:, 1].cpu().numpy())
    return np.array(probs)

# ---------- metrics ----------
def cluster_boot(y, p, fn, nb=5000, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        try: out.append(fn(y[idx], p[idx]))
        except Exception: pass
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)

def calib(y, p):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1-eps) / (1 - np.clip(p, eps, 1-eps)))
    lr = LogisticRegression(fit_intercept=True, C=1e12, solver="lbfgs").fit(lp.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])   # intercept, slope

def full_metrics(y, p):
    y = np.asarray(y).astype(int); pred = (p > 0.5).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    acc = (tp+tn)/len(y); sens = tp/(tp+fn+1e-9); spec = tn/(tn+fp+1e-9)
    ppv = tp/(tp+fp+1e-9); npv = tn/(tn+fn+1e-9); bal = (sens+spec)/2
    f1r = 2*ppv*sens/(ppv+sens+1e-9)
    ppv0 = tn/(tn+fn+1e-9); rec0 = tn/(tn+fp+1e-9); f1l = 2*ppv0*rec0/(ppv0+rec0+1e-9)
    macroF1 = (f1r+f1l)/2
    auroc = roc_auc_score(y, p); auprc = average_precision_score(y, p); brier = brier_score_loss(y, p)
    ci, cs = calib(y, p)
    def _acc(a, b): return (( (b>0.5).astype(int)==a).mean())
    lo, hi = cluster_boot(y, p, _acc)
    al, ah = cluster_boot(y, p, roc_auc_score)
    return dict(n=len(y), tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, acc_ci=(lo, hi),
                sens=sens, spec=spec, ppv=ppv, npv=npv, bal=bal, macroF1=macroF1,
                auroc=auroc, auroc_ci=(al, ah), auprc=auprc, brier=brier,
                calib_intercept=ci, calib_slope=cs)

# ---------- run ----------
def run():
    C = np.load(CACHE, allow_pickle=True)
    X_raw = C["X"][:, :12, :].astype(np.float32); y = C["y"].astype(int); grp = C["grp"]
    print(f"data {X_raw.shape} | patients {len(set(grp))} | RVOT {int((y==1).sum())} LVOT {int((y==0).sum())}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    leads, deriv, MDI = build_inputs(X_raw)
    sgkf = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(leads, y, grp))

    configs = ["ecg_only", "ecg_deriv", "ecg_mdi", "full", "mdi_only", "deriv_only"]
    store = {}
    print("\n" + "="*118)
    print(f"{'config':11s} {'acc (95% CI)':22s} {'AUROC (95% CI)':22s} {'AUPRC':7s} {'bal':7s} {'macroF1':8s} {'Brier':7s} {'calib i/slope':14s}")
    print("-"*118)
    for cfg in configs:
        sig = make_signal(leads, deriv, cfg); use_sig = sig is not None
        in_ch = sig.shape[1] if use_sig else 1
        Xc = sig if use_sig else np.zeros((len(y), 1, 1), np.float32)
        AP, AY, AG = [], [], []
        for tr, va in folds:
            pr = train_fold(Xc, MDI, y, tr, va, in_ch, MDI.shape[1], use_sig, uses_mdi(cfg))
            AP.extend(pr); AY.extend(y[va]); AG.extend(grp[va])
        AP, AY, AG = np.array(AP), np.array(AY), np.array(AG)
        pats = sorted(set(AG))
        pp = np.array([AP[AG==p].mean() for p in pats]); pt = np.array([int(AY[AG==p][0]) for p in pats])
        m = full_metrics(pt, pp); store[cfg] = dict(prob=pp, true=pt, ids=np.array(pats), **{k:v for k,v in m.items() if k!='n'})
        print(f"{cfg:11s} {m['acc']*100:5.1f} ({m['acc_ci'][0]*100:4.1f}-{m['acc_ci'][1]*100:4.1f})   "
              f"{m['auroc']:.3f} ({m['auroc_ci'][0]:.3f}-{m['auroc_ci'][1]:.3f})   "
              f"{m['auprc']:.3f}  {m['bal']*100:5.1f}  {m['macroF1']*100:6.1f}  {m['brier']:.3f}  "
              f"{m['calib_intercept']:+.2f}/{m['calib_slope']:.2f}")
    print("="*118)
    # paired delta of the MDI contribution
    f = store["full"]; e = store["ecg_deriv"]
    print(f"\nMDI contribution (full − ecg_deriv): Δacc = {(f['acc']-e['acc'])*100:+.1f} pp, "
          f"ΔAUROC = {f['auroc']-e['auroc']:+.3f}")
    print("Confusion matrix (full):  TP(RVOT) {tp}  FN {fn}  FP {fp}  TN(LVOT) {tn}".format(**{
        'tp':f['tp'],'fn':f['fn'],'fp':f['fp'],'tn':f['tn']}))
    np.savez("ablation_results.npz", **{f"{c}_{k}": v for c, d in store.items()
                                        for k, v in d.items() if k in ("prob","true","ids")})
    print("\nsaved ablation_results.npz  → upload this back to the chat.")

if __name__ == "__main__":
    run()
