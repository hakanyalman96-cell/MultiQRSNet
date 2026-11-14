#!/usr/bin/env python3
"""
Diagnose (and optionally repair) the three-class cache.

Usage (Colab):
  python diag_3class_cache.py                       # report only
  python diag_3class_cache.py --master intramural_26.txt   # also list missing intramural names
  python diag_3class_cache.py --fix                 # write multiqrsnet_cache_3class_fixed.npz

Labels: 0=LVOT, 1=RVOT, 2=intramural (same 0/1 mapping as the binary cache).
"""
import sys, unicodedata, collections, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="/content/drive/MyDrive/multiqrsnet_cache_bin.npz")
ap.add_argument("--tri", default="/content/drive/MyDrive/multiqrsnet_cache_3class.npz")
ap.add_argument("--master", default=None, help="text file: one intramural patient name per line (26 names)")
ap.add_argument("--fix", action="store_true")
a = ap.parse_args()

def norm(s):
    s = str(s).lower().replace("ı", "i")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s if ch.isalnum())

def load(p):
    C = np.load(p, allow_pickle=True)
    X, y, g = C["X"], C["y"].astype(int), np.array([str(v) for v in C["grp"]])
    print(f"\n{p}\n  X {X.shape}, recordings {len(y)}, patients {len(set(g))}")
    pc = collections.Counter(int(y[g == p][0]) for p in set(g))
    print("  patients per class:", dict(sorted(pc.items())))
    # label consistency within patient
    bad = [p for p in set(g) if len(set(y[g == p])) > 1]
    if bad: print("  !! patients with mixed labels:", bad)
    # empty / flat / NaN recordings
    Xf = X[:, :12, :].astype(np.float32)
    nan = np.isnan(Xf).any((1, 2)); flat = (Xf.std(2) < 1e-6).all(1)
    if nan.any() or flat.any():
        print(f"  !! NaN recordings {nan.sum()}, flat recordings {flat.sum()} ->",
              sorted(set(g[nan | flat])))
    return C, X, y, g

Cb, Xb, yb, gb = load(a.bin)
Ct, Xt, yt, gt = load(a.tri)

nb = {norm(p): p for p in set(gb)}; nt = {norm(p): p for p in set(gt)}
print("\nSpelling differences (same patient, different name):")
for k in sorted(set(nb) & set(nt)):
    if nb[k] != nt[k]: print(f"  bin '{nb[k]}'  vs  3class '{nt[k]}'")
print("\nRVOT/LVOT patients in binary cache but MISSING from 3-class cache:")
miss = sorted(k for k in nb if k not in nt)
for k in miss:
    p = nb[k]; print(f"  {p}  (label {int(yb[gb == p][0])}, {int((gb == p).sum())} recordings in binary cache)")
extra = sorted(k for k in nt if k not in nb and int(yt[gt == nt[k]][0]) != 2)
if extra: print("RVOT/LVOT patients only in 3-class cache:", [nt[k] for k in extra])

intra = sorted(nt[k] for k in nt if int(yt[gt == nt[k]][0]) == 2)
print(f"\nIntramural patients in 3-class cache: {len(intra)} (expected 26)")
for p in intra: print(f"  {p}: {int((gt == p).sum())} recordings")
if a.master:
    want = [l.strip() for l in open(a.master, encoding="utf-8") if l.strip()]
    have = {norm(p) for p in intra}
    print("\nIntramural patients MISSING from 3-class cache:", [w for w in want if norm(w) not in have])

if a.fix:
    # RVOT/LVOT from the binary cache (exactly the 103 patients of the binary task) + intramural from 3-class cache
    keep = yt == 2
    X = np.concatenate([Xb[:, :12, :], Xt[keep][:, :12, :]]).astype(np.float32)
    y = np.concatenate([yb, yt[keep]]); g = np.concatenate([gb, gt[keep]])
    out = a.tri.replace(".npz", "_fixed.npz")
    np.savez_compressed(out, X=X, y=y, grp=g)
    pc = collections.Counter(int(y[g == p][0]) for p in set(g))
    print(f"\nwrote {out}: {len(set(g))} patients, per class {dict(sorted(pc.items()))}")
    if pc.get(2, 0) < 26:
        print("  NOTE: intramural recordings for the missing patients must still be exported from CARTO and added.")
