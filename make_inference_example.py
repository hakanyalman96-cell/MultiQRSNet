"""
End-to-end inference example for MultiQRSNet on a SYNTHETIC 12-lead ECG.

No patient data are needed. The script generates a synthetic VPC-like 12-lead
signal, applies the exact preprocessing used in the paper (24-channel input:
12 median/IQR-normalised leads + 12 first-order derivatives, plus the
12-dimensional maximal deflection index, MDI), builds the reported
architecture (GAP 256 + MDI 12 = 268 -> 128 -> 2) and runs a forward pass.

    python3 make_inference_example.py

Trained weights are not distributed (training is not bit-wise deterministic;
the reported results are documented by the prediction files in predictions/),
so the printed probabilities come from a randomly initialised model and only
demonstrate the input/output contract. Positive class = RVOT (index 1).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from multiqrsnet_cv_raw import MultiQRSNet, build_inputs  # reported architecture and preprocessing


def synthetic_ecg(seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2.5, 2500)
    sig = np.zeros((12, 2500), np.float32)
    for c in range(12):
        base = 0.05 * np.sin(2 * np.pi * 0.3 * t)            # baseline wander
        for beat in (0.6, 1.25, 1.9):                          # wide QRS complexes
            sig[c] += (0.8 + 0.4 * rng.standard_normal()) * np.exp(-((t - beat) ** 2) / (2 * 0.02 ** 2))
        sig[c] += base + 0.02 * rng.standard_normal(2500)
    return sig


def main():
    torch.manual_seed(42)
    x12 = synthetic_ecg()
    np.save("synthetic_ecg.npy", x12)
    x24, mdi = build_inputs(x12[None])                          # (1,24,2500), (1,12)
    net = MultiQRSNet().eval()
    with torch.no_grad():
        p = torch.softmax(net(torch.from_numpy(x24), torch.from_numpy(mdi)), 1)[0].numpy()
    print(f"input: {x24.shape} signal + {mdi.shape} MDI -> head input 268")
    print(f"P(LVOT) = {p[0]:.3f}   P(RVOT) = {p[1]:.3f}   (random weights: not meaningful)")
    print("saved synthetic_ecg.npy")


if __name__ == "__main__":
    main()
