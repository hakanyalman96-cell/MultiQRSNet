#!/usr/bin/env python3
"""
Binary MultiQRSNet (Methods-matching), patient-level 5-fold CV, no curation.
Labels: 0=LVOT, 1=RVOT. Works on CUDA / Apple-Silicon MPS / CPU.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 multiqrsnet_cv_raw.py /path/to/cache_bin.npz
Cache must hold raw 12-lead ECG: 'X' (N,>=12,2500), 'y' (N,) in {0,1}, 'grp' (N,).
Needs: torch, numpy, scikit-learn, scipy.
"""
import os, sys, random
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats

CACHE = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/multiqrsnet_cache_bin.npz"
EPOCHS, BATCH, SEED = 50, 64, 42

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
    def __init__(s, inch=24, md=12, nc=2):
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
    net = MultiQRSNet().to(dev); crit = nn.CrossEntropyLoss(label_smoothing=0.1)
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
    net.eval(); P = []
    with torch.no_grad():
        for xb, mb, _ in vl:
            P.extend(torch.softmax(net(xb.to(dev), mb.to(dev)), 1)[:, 1].cpu().numpy())
    return np.array(P)

def cboot(c, nb=5000):
    c = np.asarray(c, float); n = len(c)
    b = [c[np.random.randint(0, n, n)].mean() for _ in range(nb)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print("device:", dev, "| cache:", CACHE, "| NO curation (all recordings)", flush=True)
    C = np.load(CACHE, allow_pickle=True)
    Xr = C["X"][:, :12, :].astype(np.float32); y = C["y"].astype(int); groups = C["grp"]
    print(f"loaded: {len(Xr)} recordings, {len(set(groups))} patients | RVOT {int((y==1).sum())} LVOT {int((y==0).sum())}", flush=True)

    print("building 24-channel inputs + MDI ...", flush=True)
    X24, M = build_inputs(Xr)

    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    AP, AY, AG, accs = [], [], [], []
    for k, (tr, va) in enumerate(sgkf.split(X24, y, groups), 1):
        pr = train_fold(X24, M, y, tr, va, dev); yt = y[va]
        a = ((pr > 0.5).astype(int) == yt).mean(); accs.append(a)
        AP.extend(pr); AY.extend(yt); AG.extend(groups[va])
        print(f"Fold {k}: patients={len(set(groups[va]))} recording-acc={a*100:.2f}%", flush=True)
    AP, AY, AG, accs = map(np.array, (AP, AY, AG, accs))

    for outdir in ["/content/drive/MyDrive", os.path.dirname(os.path.abspath(CACHE)), "."]:
        try:
            np.savez(os.path.join(outdir, "multiqrsnet_oof_binary_final.npz"), prob=AP, true=AY, grp=AG, accs=accs)
            print("saved out-of-fold to:", outdir); break
        except Exception:
            continue

    sem = accs.std(ddof=1) / np.sqrt(5); t = stats.t.ppf(0.975, 4)
    print(f"\nRecording-level fold-mean = {accs.mean()*100:.2f}% "
          f"(95% CI {(accs.mean()-t*sem)*100:.2f}-{(accs.mean()+t*sem)*100:.2f}%)")
    pats = sorted(set(AG))
    pp = np.array([AP[AG == p].mean() for p in pats]); pg = np.array([int(AY[AG == p][0]) for p in pats])
    corr = ((pp > 0.5).astype(int) == pg); lo, hi = cboot(corr)
    print(f"PATIENT-LEVEL accuracy = {corr.sum()}/{len(pats)} = {corr.mean()*100:.2f}% "
          f"(cluster-bootstrap 95% CI {lo*100:.2f}-{hi*100:.2f}%)")
    for c, nm in [(1, "RVOT"), (0, "LVOT")]:
        m = pg == c; l, h = cboot(corr[m])
        print(f"  {nm} sensitivity = {corr[m].sum()}/{m.sum()} = {corr[m].mean()*100:.2f}% "
              f"(95% CI {l*100:.2f}-{h*100:.2f}%)")

if __name__ == "__main__":
    main()
