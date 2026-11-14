#!/usr/bin/env python3
"""
Revision experiments for JECG-D-26-00848 (Reviewer 2, comments 2.7 and 2.8; Reviewer 1, comment 1.4).

Re-uses the architecture, preprocessing and training recipe of multiqrsnet_cv_raw.py unchanged,
except for the switches below. Same folds (StratifiedGroupKFold, 5 folds, random_state=42).

Switches
  --no-polarity      remove the global polarity-inversion augmentation (x = -x, p = 0.2)       [B1]
  --seeds 42,1,2,3,4 train once per seed (fold split stays fixed at 42)                          [B2]
  --window 500       train AND test on a PVC-centred crop of this many samples (sinus beats out) [B4b]
  --mask-half 150    at inference, also score each fold model with everything outside
                     centre +/- mask-half samples set to 0 (= the lead median after
                     normalisation); also computes Grad-CAM share inside that window          [B4a, B4c]
  --no-mdi           drop the 12-d timing feature (it was computed outside the index PVC)
  --external A.npz,B.npz  additionally train on the full internal cohort and test once on each
  --tag NAME         prefix of the output files

The index PVC is NOT at the segment centre: diagnostic plots showed it ~0.5 s before the segment end.
With --centre auto (default) it is located per segment as the activity peak within samples 1850-2150.

Outputs (one row per patient per seed), written to ./revision_out/:
  <tag>_cv_patient.csv        patient, seed, true, prob_full[, prob_masked], gradcam_share_in_window
  <tag>_external_patient.csv  patient, seed, true, prob   (if --external)
  <tag>_summary.txt           accuracy, sensitivities, AUROC per seed, then mean +/- SD and range

Runs reported in the revised manuscript (development cache: 103 patients; external caches: independent
clinical cohort and Zheng et al. public database), one call per seed (42, 1, 2, 3, 4):
  original config  : python3 scripts/revision_experiments.py CACHE --seeds S --tag original_seedS
  no polarity      : python3 scripts/revision_experiments.py CACHE --seeds S --no-polarity --external EXT --tag nopolarity_seedS
  final model      : python3 scripts/revision_experiments.py CACHE --seeds S --no-mdi --no-polarity --external EXT,ZHENG --tag final_seedS
  PVC-only window  : python3 scripts/revision_experiments.py CACHE --seeds S --no-mdi --no-polarity --window 500 --external EXT --tag window500_final_seedS
  masking (42,1,2) : python3 scripts/revision_experiments.py CACHE --seeds S --no-mdi --no-polarity --mask-half 150 --tag mask_final_seedS
On Apple-Silicon Macs prefix with PYTORCH_ENABLE_MPS_FALLBACK=1.
"""
import argparse, os, random, sys
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiqrsnet_cv_raw import build_inputs, MultiQRSNet, mixup  # unchanged reported components

EPOCHS, BATCH, SPLIT_SEED, CENTRE = 50, 64, 42, 1250


def augment(x, polarity):
    """Identical to multiqrsnet_cv_raw.augment, with the polarity flip made optional."""
    L = x.shape[1]
    if random.random() < 0.5: x = x + (0.05*np.sin(np.linspace(0, 2*np.pi*random.uniform(1, 3), L))).astype(np.float32)
    if random.random() < 0.5: x = x * random.uniform(0.85, 1.15)
    if random.random() < 0.5: x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
    if random.random() < 0.5: x = np.roll(x, random.randint(-50, 50), axis=1)
    if polarity and random.random() < 0.2: x = -x
    if random.random() < 0.3:
        s = random.randint(0, L - 100); x[:, s:s + random.randint(20, min(100, L - s))] = 0
    return x


class DS(torch.utils.data.Dataset):
    def __init__(s, X, M, y, idx, tr, polarity): s.X, s.M, s.y, s.idx, s.tr, s.pol = X, M, y, idx, tr, polarity
    def __len__(s): return len(s.idx)
    def __getitem__(s, k):
        i = s.idx[k]; x = s.X[i].copy()
        if s.tr: x = augment(x, s.pol)
        return torch.from_numpy(x), torch.from_numpy(s.M[i]), int(s.y[i])


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def train(X, M, y, idx, dev, polarity):
    net = MultiQRSNet(md=M.shape[1]).to(dev); crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    tl = torch.utils.data.DataLoader(DS(X, M, y, idx, True, polarity), BATCH, shuffle=True)
    for _ in range(EPOCHS):
        net.train()
        for xb, mb, yb in tl:
            xb, mb, yb = xb.to(dev), mb.to(dev), yb.to(dev)
            if np.random.rand() < 0.5: xb = mixup(xb)
            opt.zero_grad(); crit(net(xb, mb), yb).backward(); opt.step()
        sch.step()
    return net.eval()


def predict(net, X, M, idx, dev):
    P = []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            P.extend(torch.softmax(net(torch.from_numpy(X[b]).to(dev), torch.from_numpy(M[b]).to(dev)), 1)[:, 1].cpu().numpy())
    return np.array(P)


def gradcam_share(net, X, M, idx, dev, cen, half):
    """Share of positive Grad-CAM relevance (final residual block, predicted class) within +/- half of each segment's index PVC."""
    feats = {}
    h = net.r3.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    out = []
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        xb = torch.from_numpy(X[b]).to(dev); mb = torch.from_numpy(M[b]).to(dev)
        logits = net(xb, mb); a = feats["a"]; a.retain_grad()
        cls = logits.argmax(1)
        net.zero_grad(); logits.gather(1, cls[:, None]).sum().backward()
        w = a.grad.mean(2, keepdim=True)
        cam = torch.relu((w * a).sum(1)).detach().cpu().numpy()  # (B, T')
        T = X.shape[2]; Tp = cam.shape[1]
        pos = (np.arange(Tp) + 0.5) * T / Tp
        tot = cam.sum(1) + 1e-12
        for j, i in enumerate(b):
            inside = (pos >= cen[i] - half) & (pos < cen[i] + half)
            out.append(cam[j, inside].sum() / tot[j])
    h.remove()
    return np.array(out)


def crop(X, cen, w):
    """Window of w samples around each segment's index PVC (shifted to stay inside the segment)."""
    T = X.shape[2]; out = np.empty((len(X), X.shape[1], w), np.float32)
    for i, c in enumerate(cen):
        lo = int(min(max(c - w // 2, 0), T - w)); out[i] = X[i, :, lo:lo + w]
    return out


def patient_level(prob, true, grp):
    pats = np.array(sorted(set(grp)))
    pp = np.array([prob[grp == p].mean() for p in pats]); pt = np.array([int(true[grp == p][0]) for p in pats])
    return pats, pp, pt


def metrics(pt, pp):
    c = (pp > 0.5).astype(int) == pt
    return dict(acc=c.mean(), n_correct=int(c.sum()), n=len(c),
                sens_rvot=c[pt == 1].mean(), sens_lvot=c[pt == 0].mean(), auroc=roc_auc_score(pt, pp))


def locate_index_pvc(Xr, lo=1850, hi=2150):
    """Per-segment position of the index PVC: peak of the multi-lead activity envelope within [lo, hi).
    The diagnostic plots showed a beat at ~1950-2000 ms in every segment (index PVC ~0.5 s before the end)."""
    c = np.zeros(len(Xr), int)
    for i in range(len(Xr)):
        d = np.abs(np.diff(Xr[i], axis=1)); d = d / (np.median(d, axis=1, keepdims=True) + 1e-9)
        e = np.convolve(d.sum(0), np.ones(40) / 40, "same")
        c[i] = lo + int(np.argmax(e[lo:hi]))
    return c


def check_centring(Xr, n=200):
    """Median position of the largest absolute deflection in lead II across segments; should be near CENTRE."""
    sel = np.random.default_rng(0).choice(len(Xr), min(n, len(Xr)), replace=False)
    pk = [np.argmax(np.abs(Xr[i, 1] - np.median(Xr[i, 1]))) for i in sel]
    print(f"centring check: median peak at sample {int(np.median(pk))} (IQR {np.percentile(pk, 25):.0f}-{np.percentile(pk, 75):.0f}); assumed CENTRE = {CENTRE}")


def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("--seeds", default="42"); ap.add_argument("--no-polarity", action="store_true")
    ap.add_argument("--window", type=int, default=0); ap.add_argument("--mask-half", type=int, default=0)
    ap.add_argument("--centre", default="auto", help="'auto' (per-segment index-PVC search in 1850-2150) or a fixed sample number")
    ap.add_argument("--no-mdi", action="store_true", help="drop the 12-d timing feature (head 256->128)")
    ap.add_argument("--external", default="", help="one or more external caches, comma-separated"); ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="only for smoke tests")
    a = ap.parse_args()
    EPOCHS = a.epochs
    polarity = not a.no_polarity; seeds = [int(s) for s in a.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    os.makedirs("revision_out", exist_ok=True)

    C = np.load(a.cache, allow_pickle=True)
    Xr = C["X"][:, :12, :].astype(np.float32); y = C["y"].astype(int); grp = np.asarray(C["grp"])
    print(f"device {dev} | {len(Xr)} segments, {len(set(grp))} patients | MDI: {not a.no_mdi} | polarity flip: {polarity} | window: {a.window or 'full'} | seeds {seeds}", flush=True)
    check_centring(Xr)
    X, M = build_inputs(Xr)  # MDI computed as in the reported model
    if a.no_mdi: M = np.ascontiguousarray(M[:, :0])
    cen = locate_index_pvc(Xr) if a.centre == "auto" else np.full(len(Xr), int(a.centre))
    print(f"index-PVC position: median {int(np.median(cen))}, IQR {np.percentile(cen, 25):.0f}-{np.percentile(cen, 75):.0f}", flush=True)
    np.save(f"revision_out/{a.tag}_pvc_position.npy", cen)
    if a.window:
        X = crop(X, cen, a.window)
    Xmask = None
    if a.mask_half and not a.window:
        Xmask = X.copy()
        for i, c in enumerate(cen):
            Xmask[i, :, :max(c - a.mask_half, 0)] = 0; Xmask[i, :, c + a.mask_half:] = 0

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SPLIT_SEED).split(X, y, grp))
    rows, summ = [], []
    for seed in seeds:
        set_seed(seed)
        P, Pm, G = np.zeros(len(y)), np.zeros(len(y)), np.full(len(y), np.nan)
        for k, (tr, va) in enumerate(folds, 1):
            net = train(X, M, y, tr, dev, polarity)
            P[va] = predict(net, X, M, va, dev)
            if Xmask is not None:
                Pm[va] = predict(net, Xmask, M, va, dev)
                G[va] = gradcam_share(net, X, M, va, dev, cen, a.mask_half)
            print(f"  seed {seed} fold {k} done", flush=True)
        pats, pp, pt = patient_level(P, y, grp); m = metrics(pt, pp)
        line = f"seed {seed}: acc {m['acc']*100:.1f}% ({m['n_correct']}/{m['n']}), RVOT {m['sens_rvot']*100:.1f}%, LVOT {m['sens_lvot']*100:.1f}%, AUROC {m['auroc']:.3f}"
        rec = dict(seed=seed, **m)
        if Xmask is not None:
            _, ppm, _ = patient_level(Pm, y, grp); mm = metrics(pt, ppm)
            _, gs, _ = patient_level(G, y, grp)
            line += f" | masked: acc {mm['acc']*100:.1f}%, AUROC {mm['auroc']:.3f} | Grad-CAM share in PVC window: median {np.median(gs)*100:.1f}%"
            rec.update(masked_acc=mm["acc"], masked_auroc=mm["auroc"], gradcam_median=np.median(gs))
        print(line, flush=True); summ.append((line, rec))
        for i, p in enumerate(pats):
            r = dict(patient=p, seed=seed, true=pt[i], prob_full=pp[i])
            if Xmask is not None: r.update(prob_masked=ppm[i], gradcam_share_in_window=gs[i])
            rows.append(r)

    import csv
    with open(f"revision_out/{a.tag}_cv_patient.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    with open(f"revision_out/{a.tag}_summary.txt", "w") as f:
        f.write(f"cache={a.cache} polarity_flip={polarity} window={a.window} mask_half={a.mask_half}\n")
        for line, _ in summ: f.write(line + "\n")
        for key in ["acc", "sens_rvot", "sens_lvot", "auroc", "masked_acc", "masked_auroc"]:
            v = np.array([r[key] for _, r in summ if key in r])
            if len(v): f.write(f"{key}: mean {v.mean():.4f} SD {v.std(ddof=1) if len(v) > 1 else 0:.4f} range {v.min():.4f}-{v.max():.4f}\n")

    for extpath in [e for e in a.external.split(",") if e]:
        E = np.load(extpath, allow_pickle=True)
        Xe = E["X"][:, :12, :].astype(np.float32); ye = E["y"].astype(int); ge = np.asarray(E["grp"])
        Xe24, Me = build_inputs(Xe)
        if a.no_mdi: Me = np.ascontiguousarray(Me[:, :0])
        if a.window:
            Xe24 = crop(Xe24, locate_index_pvc(Xe) if a.centre == "auto" else np.full(len(Xe), int(a.centre)), a.window)
        erows = []; name = os.path.splitext(os.path.basename(extpath))[0]
        with open(f"revision_out/{a.tag}_summary.txt", "a") as f:
            for seed in seeds:
                set_seed(seed); net = train(X, M, y, np.arange(len(y)), dev, polarity)
                pe = predict(net, Xe24, Me, np.arange(len(ye)), dev)
                pats, pp, pt = patient_level(pe, ye, ge); m = metrics(pt, pp)
                line = f"EXTERNAL {name} seed {seed}: acc {m['acc']*100:.1f}% ({m['n_correct']}/{m['n']}), RVOT {m['sens_rvot']*100:.1f}%, LVOT {m['sens_lvot']*100:.1f}%, AUROC {m['auroc']:.3f}"
                print(line, flush=True); f.write(line + "\n")
                erows += [dict(patient=p, seed=seed, true=pt[i], prob=pp[i]) for i, p in enumerate(pats)]
        with open(f"revision_out/{a.tag}_external_{name}_patient.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(erows[0].keys())); w.writeheader(); w.writerows(erows)
    print(f"wrote revision_out/{a.tag}_*", flush=True)


if __name__ == "__main__":
    main()
