#!/usr/bin/env python3
"""
Three-class MultiQRSNet (RVOT / LVOT / intramural), patient-level 5-fold CV.
Labels: 0=LVOT, 1=RVOT, 2=intramural.
Reports: overall & balanced accuracy, per-class recall/precision, macro-F1,
confusion matrix, cluster (patient) bootstrap 95% CIs. No curation.
Works on CUDA / Apple-Silicon MPS / CPU.

Run:
    python multiqrsnet_cv_3class.py [CACHE_NPZ]
Cache must hold raw 12-lead ECG: 'X' (N,>=12,2500), 'y' (N,) in {0,1,2}, 'grp' (N,).
Needs: torch, numpy, scikit-learn, scipy.
"""
import os, sys, random
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score

CACHE = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/multiqrsnet_cache_3class.npz"
EPOCHS, BATCH, SEED = 50, 64, 42
NAMES = {0: "LVOT", 1: "RVOT", 2: "intramural"}

def rnorm(l):
    m = np.median(l); q1, q3 = np.percentile(l, [25, 75])
    return (l - m) / ((q3 - q1) + 1e-6)

def mdi(l):
    s = l - np.median(l); w = s[len(s)//2 - 100: len(s)//2 + 100]
    return np.argmax(np.abs(w)) / len(w)

def build_inputs(Xr):
    N = len(Xr); X24 = np.zeros((N, 24, 2500), np.float32); M = np.zeros((N, 12), np.float32)
    for i in range(N):
        for c in range(12):
            nz = rnorm(Xr[i, c].astype(np.float32))
            X24[i, c] = nz; X24[i, 12 + c] = np.gradient(nz); M[i, c] = mdi(Xr[i, c])
        if i % 2000 == 0: print(f"  build_inputs {i}/{N}", flush=True)
    return X24, M

def augment(x):
    L = x.shape[1]
    if random.random() < 0.5: x = x + (0.05*np.sin(np.linspace(0, 2*np.pi*random.uniform(1,3), L))).astype(np.float32)
    if random.random() < 0.5: x = x * random.uniform(0.85, 1.15)
    if random.random() < 0.5: x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
    if random.random() < 0.5: x = np.roll(x, random.randint(-50, 50), axis=1)
    if random.random() < 0.2: x = -x
    if random.random() < 0.3:
        s = random.randint(0, L - 100); x[:, s:s + random.randint(20, 100)] = 0
    return x

class SE(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__(); h = max(ch // r, 4)
        s.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(True), nn.Linear(h, ch), nn.Sigmoid())
    def forward(s, x): return x * s.fc(x.mean(2)).unsqueeze(2)

class RB(nn.Module):
    def __init__(s, ci, co, st=2, k=5):
        super().__init__(); p = k // 2
        s.c1 = nn.Conv1d(ci, co, k, st, p); s.n1 = nn.InstanceNorm1d(co, affine=True)
        s.c2 = nn.Conv1d(co, co, k, 1, p);  s.n2 = nn.InstanceNorm1d(co, affine=True); s.se = SE(co)
        s.d = nn.Sequential(nn.Conv1d(ci, co, 1, st), nn.InstanceNorm1d(co, affine=True)) if (st != 1 or ci != co) else None
    def forward(s, x):
        idt = x if s.d is None else s.d(x)
        o = torch.relu(s.n1(s.c1(x))); o = s.n2(s.c2(o)); o = s.se(o)
        return torch.relu(o + idt)

class MultiQRSNet(nn.Module):
    def __init__(s, inch=24, md=12, nc=3):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(inch, 64, 7, 1, 3), nn.InstanceNorm1d(64, affine=True), nn.ReLU(True), nn.MaxPool1d(2))
        s.r1 = RB(64, 64); s.r2 = RB(64, 128); s.r3 = RB(128, 256)
        s.head = nn.Sequential(nn.Linear(256 + md, 128), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(128, nc))
    def forward(s, x, m):
        x = s.stem(x); x = s.r1(x); x = s.r2(x); x = s.r3(x); x = x.mean(2)
        return s.head(torch.cat([x, m], 1))

class DS(torch.utils.data.Dataset):
    def __init__(s, X, M, y, idx, tr=False): s.X, s.M, s.y, s.idx, s.tr = X, M, y, idx, tr
    def __len__(s): return len(s.idx)
    def __getitem__(s, k):
        i = s.idx[k]; x = s.X[i].copy()
        if s.tr: x = augment(x)
        return torch.from_numpy(x), torch.from_numpy(s.M[i]), int(s.y[i])

def mixup(x, a=0.2):
    lam = np.random.beta(a, a); lam = max(lam, 1 - lam)
    p = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[p]

def train_fold(X24, M, y, tr, va, dev):
    net = MultiQRSNet(nc=3).to(dev); crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    tl = torch.utils.data.DataLoader(DS(X24, M, y, tr, True), BATCH, shuffle=True)
    vl = torch.utils.data.DataLoader(DS(X24, M, y, va), 256)
    for _ in range(EPOCHS):
        net.train()
        for xb, mb, yb in tl:
            xb, mb, yb = xb.to(dev), mb.to(dev), yb.to(dev)
            if np.random.rand() < 0.5: xb = mixup(xb)
            opt.zero_grad(); crit(net(xb, mb), yb).backward(); opt.step()
        sch.step()
    net.eval(); PR = []
    with torch.no_grad():
        for xb, mb, _ in vl:
            PR.append(torch.softmax(net(xb.to(dev), mb.to(dev)), 1).cpu().numpy())
    return np.concatenate(PR, 0)          # (N_va, 3) softmax probabilities

def cboot_acc(pred, true, nb=5000):
    correct = (pred == true).astype(float); n = len(correct)
    b = [correct[np.random.randint(0, n, n)].mean() for _ in range(nb)]
    return correct.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print("device:", dev, "| cache:", CACHE, flush=True)
    C = np.load(CACHE, allow_pickle=True)
    Xr = C["X"][:, :12, :].astype(np.float32); y = C["y"].astype(int); groups = C["grp"]
    print(f"loaded: {len(Xr)} recordings, {len(set(groups))} patients | classes:",
          {int(c): int((y == c).sum()) for c in np.unique(y)}, flush=True)

    print("building 24-channel inputs + MDI ...", flush=True)
    X24, M = build_inputs(Xr)

    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    AP, AY, AG = [], [], []
    for k, (tr, va) in enumerate(sgkf.split(X24, y, groups), 1):
        pr = train_fold(X24, M, y, tr, va, dev)
        AP.append(pr); AY.extend(y[va]); AG.extend(groups[va])
        acc = (pr.argmax(1) == y[va]).mean()
        print(f"Fold {k}: patients={len(set(groups[va]))} recording-acc={acc*100:.2f}%", flush=True)
    AP = np.concatenate(AP, 0); AY = np.array(AY); AG = np.array(AG)

    # patient-level: mean probability across a patient's recordings -> argmax
    pats = sorted(set(AG))
    p_prob = np.array([AP[AG == p].mean(0) for p in pats])
    p_pred = p_prob.argmax(1)
    p_true = np.array([int(AY[AG == p][0]) for p in pats])

    acc, lo, hi = cboot_acc(p_pred, p_true)
    print(f"\n===== THREE-CLASS PATIENT-LEVEL =====")
    print(f"Overall accuracy = {(p_pred==p_true).sum()}/{len(pats)} = {acc*100:.2f}% "
          f"(cluster-bootstrap 95% CI {lo*100:.2f}-{hi*100:.2f}%)")
    print(f"Balanced accuracy = {balanced_accuracy_score(p_true, p_pred)*100:.2f}%")
    print(f"Macro-F1 = {f1_score(p_true, p_pred, average='macro')*100:.2f}%")
    for c in [1, 0, 2]:
        m = p_true == c
        rec = (p_pred[m] == c).mean() if m.sum() else float('nan')
        prec = (p_true[p_pred == c] == c).mean() if (p_pred == c).sum() else float('nan')
        print(f"  {NAMES[c]:11s}: recall {int((p_pred[m]==c).sum())}/{int(m.sum())}={rec*100:.1f}%  precision={prec*100:.1f}%")
    print("\nConfusion matrix (rows=true, cols=pred) order [LVOT,RVOT,intramural]:")
    print(confusion_matrix(p_true, p_pred, labels=[0, 1, 2]))
    try:
        np.savez("/content/drive/MyDrive/multiqrsnet_oof_3class.npz", prob=AP, true=AY, grp=AG)
    except Exception:
        pass

if __name__ == "__main__":
    main()
