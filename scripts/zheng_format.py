#!/usr/bin/env python3
"""
Format the Zheng et al. (2020) 12-lead VA-origin database into zheng.npz
matching MultiQRSNet input: 12 leads, 1000 Hz, 2500 samples, VPC-centred.

Zheng data: one CSV per patient, 12 leads, 2000 Hz; a Diagnostics file maps
each file to the arrhythmia origin. This script:
  1. reads each ECG CSV (12 leads),
  2. resamples 2000 -> 1000 Hz,
  3. detects the PVC beat and centres a 2500-sample (2.5 s) window on it,
  4. reorders leads to the model order,
  5. maps origin -> binary (0=LVOT-type, 1=RVOT),
  6. saves zheng.npz  (X:(N,12,2500), y:(N,), grp:(N,) patient ids).

EDIT the CONFIG block to match your downloaded files, then:
    python3 zheng_format.py
    python3 multiqrsnet_external.py multiqrsnet_cache_bin.npz zheng.npz
Needs: numpy, pandas, scipy, biosppy   (pip3 install biosppy)
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

# ================= CONFIG — edit to your files =================
ECG_DIR   = "ECGDataDenoised"          # folder of per-patient CSVs
DIAG_FILE = "Diagnosis.xlsx"           # Zheng diagnostics
FS_IN, FS_OUT, TARGET_LEN = 2000, 1000, 2500

# Zheng diagnostics: HospitalID matches the CSV file name; LeftRight = RVOT/LVOT.
DIAG_ID_COL     = "HospitalID"
DIAG_ORIGIN_COL = "LeftRight"

# Lead order in the Zheng CSV header (verified from 702840.csv):
ZHENG_LEADS = ["aVF","aVL","aVR","I","II","III","V1","V2","V3","V4","V5","V6"]
# Order the MODEL expects:
MODEL_LEADS = ["V1","V2","V3","V4","V5","V6","I","II","III","aVL","aVR","aVF"]

# Map Zheng 'LeftRight' to binary: 1=RVOT (Right), 0=LVOT (Left)
def to_binary(v):
    v = str(v).strip().upper()
    if v == "RIGHT": return 1
    if v == "LEFT":  return 0
    return None
# ==============================================================

def load_csv(path):
    df = pd.read_csv(path)
    if not set(ZHENG_LEADS).issubset(df.columns):     # no/mismatched header -> assume positional
        df = pd.read_csv(path, header=None); df.columns = ZHENG_LEADS[:df.shape[1]]
    return df[ZHENG_LEADS].to_numpy(dtype=np.float32).T   # (12, N) in ZHENG order

def center_on_pvc(sig12):
    """Return a TARGET_LEN window centred on the PVC beat (widest/most-abnormal)."""
    from biosppy.signals import ecg
    lead = sig12[1]                                   # lead II
    try:
        out = ecg.ecg(signal=lead, sampling_rate=FS_OUT, show=False)
        rpeaks = out["rpeaks"]
    except Exception:
        rpeaks = np.array([])
    if len(rpeaks) >= 2:
        # PVC ~ beat whose QRS amplitude deviates most from the median beat amplitude
        amps = np.array([np.ptp(sig12[:, max(0, r-60):r+60]) for r in rpeaks])
        pvc = rpeaks[int(np.argmax(amps))]
    else:
        pvc = sig12.shape[1] // 2
    a = pvc - TARGET_LEN // 2; b = a + TARGET_LEN
    out = np.zeros((12, TARGET_LEN), np.float32)
    s0, s1 = max(0, a), min(sig12.shape[1], b)
    out[:, s0-a:s1-a] = sig12[:, s0:s1]
    return out

def main():
    diag = pd.read_excel(DIAG_FILE) if DIAG_FILE.lower().endswith(("xls", "xlsx")) else pd.read_csv(DIAG_FILE)
    lab = {str(r[DIAG_ID_COL]).strip(): to_binary(r[DIAG_ORIGIN_COL]) for _, r in diag.iterrows()}
    order_idx = [ZHENG_LEADS.index(l) for l in MODEL_LEADS]

    X, y, g = [], [], []
    files = sorted(glob.glob(os.path.join(ECG_DIR, "*.csv")))
    print(f"{len(files)} CSV, {len(lab)} labelled patients")
    kept = skipped = 0
    for f in files:
        pid = os.path.splitext(os.path.basename(f))[0]
        b = lab.get(pid)
        if b is None: skipped += 1; continue
        sig = load_csv(f)                              # (12,N) 2000 Hz, ZHENG order
        sig = resample_poly(sig, FS_OUT, FS_IN, axis=1).astype(np.float32)  # -> 1000 Hz
        sig = center_on_pvc(sig)                        # (12,2500) ZHENG order
        sig = sig[order_idx]                            # -> MODEL order
        X.append(sig); y.append(b); g.append(pid); kept += 1
    X = np.array(X, np.float32); y = np.array(y); g = np.array(g)
    print(f"kept {kept}, skipped {skipped} | RVOT {int((y==1).sum())} LVOT {int((y==0).sum())}")
    np.savez("zheng.npz", X=X, y=y, grp=g)
    print("saved zheng.npz")

if __name__ == "__main__":
    main()
