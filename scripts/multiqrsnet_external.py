#!/usr/bin/env python3
"""
External validation: train MultiQRSNet on the FULL internal binary cohort,
then evaluate ONCE on an external cohort (patient-level). Prints misclassified
external patients. CUDA / Apple-Silicon MPS / CPU.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 multiqrsnet_external.py INTERNAL.npz EXTERNAL.npz
"""
import sys, random
import numpy as np
import torch
from torch import nn, optim
from scipy import stats

INT = sys.argv[1] if len(sys.argv) > 1 else "multiqrsnet_cache_bin.npz"
EXT = sys.argv[2] if len(sys.argv) > 2 else "sakura_external.npz"
EPOCHS, BATCH, SEED = 50, 64, 42

def rnorm(l):
    m = np.median(l); q1, q3 = np.percentile(l, [25, 75]); return (l - m) / ((q3 - q1) + 1e-6)
def mdi(l):
    s = l - np.median(l); w = s[len(s)//2 - 100: len(s)//2 + 100]; return np.argmax(np.abs(w)) / len(w)
def build_inputs(Xr):
    N = len(Xr); X24 = np.zeros((N, 24, 2500), np.float32); M = np.zeros((N, 12), np.float32)
    for i in range(N):
        for c in range(12):
            nz = rnorm(Xr[i, c].astype(np.float32)); X24[i, c] = nz; X24[i, 12 + c] = np.gradient(nz); M[i, c] = mdi(Xr[i, c])
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
    def __init__(s, X, M, y, tr=False): s.X, s.M, s.y, s.tr = X, M, y, tr
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        x = s.X[i].copy()
        if s.tr: x = augment(x)
        return torch.from_numpy(x), torch.from_numpy(s.M[i]), int(s.y[i])
def mixup(x, a=0.2):
    lam = np.random.beta(a, a); lam = max(lam, 1 - lam)
    p = torch.randperm(x.size(0), device=x.device); return lam * x + (1 - lam) * x[p]
def cboot(c, nb=5000):
    c = np.asarray(c, float); n = len(c); b = [c[np.random.randint(0, n, n)].mean() for _ in range(nb)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    Ci = np.load(INT, allow_pickle=True); Ce = np.load(EXT, allow_pickle=True)
    Xi = Ci["X"][:, :12, :].astype(np.float32); yi = Ci["y"].astype(int)
    Xe = Ce["X"][:, :12, :].astype(np.float32); ye = Ce["y"].astype(int); ge = Ce["grp"]
    print("device:", dev, "| train(internal):", len(Xi), "| test(external):", len(Xe), "patients:", len(set(ge)), flush=True)
    print("building inputs (train)..."); X24i, Mi = build_inputs(Xi)
    print("building inputs (test)...");  X24e, Me = build_inputs(Xe)

    net = MultiQRSNet().to(dev); crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    tl = torch.utils.data.DataLoader(DS(X24i, Mi, yi, True), BATCH, shuffle=True)
    for ep in range(EPOCHS):
        net.train()
        for xb, mb, yb in tl:
            xb, mb, yb = xb.to(dev), mb.to(dev), yb.to(dev)
            if np.random.rand() < 0.5: xb = mixup(xb)
            opt.zero_grad(); crit(net(xb, mb), yb).backward(); opt.step()
        sch.step()
        if (ep+1) % 10 == 0: print(f"  epoch {ep+1}/{EPOCHS}", flush=True)

    net.eval(); prob = []
    vl = torch.utils.data.DataLoader(DS(X24e, Me, ye, False), 256)
    with torch.no_grad():
        for xb, mb, _ in vl:
            prob.extend(torch.softmax(net(xb.to(dev), mb.to(dev)), 1)[:, 1].cpu().numpy())
    prob = np.array(prob)

    pats = sorted(set(ge))
    pp = np.array([prob[ge == p].mean() for p in pats]); pg = np.array([int(ye[ge == p][0]) for p in pats])
    corr = ((pp > 0.5).astype(int) == pg); lo, hi = cboot(corr)
    print(f"\n===== EXTERNAL VALIDATION (patient-level) =====")
    print(f"Accuracy = {corr.sum()}/{len(pats)} = {corr.mean()*100:.2f}% (cluster-bootstrap 95% CI {lo*100:.2f}-{hi*100:.2f}%)")
    for c, nm in [(1, "RVOT"), (0, "LVOT")]:
        m = pg == c; l, h = cboot(corr[m])
        print(f"  {nm} sensitivity = {corr[m].sum()}/{m.sum()} = {corr[m].mean()*100:.2f}% (95% CI {l*100:.2f}-{h*100:.2f}%)")

    NM = {1: "RVOT", 0: "LVOT"}
    wrong = [(pats[i], pg[i], int(pp[i] > 0.5), float(pp[i])) for i in range(len(pats)) if (pp[i] > 0.5) != bool(pg[i])]
    wrong.sort(key=lambda r: abs(r[3] - 0.5), reverse=True)
    print(f"\n--- Misclassified external patients ({len(wrong)}; most confident-wrong first) ---")
    print(f"{'patient':30s} {'true':6s} {'pred':6s} P(RVOT)")
    for pid, tl, pr, mp in wrong:
        print(f"{pid:30s} {NM[tl]:6s} {NM[pr]:6s} {mp:.2f}")
    np.savez("sakura_patient_results.npz", pid=np.array(pats), true=pg, prob=pp)
    print("\nsaved sakura_patient_results.npz")

if __name__ == "__main__":
    main()
