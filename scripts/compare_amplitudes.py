# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Compare probabilities index by index.

  python compare_amplitudes.py --ref "amplitudes_*.jsonl" --test control.jsonl
  python compare_amplitudes.py --ref-amplitudes amp75.txt --test amplitudes_75t.jsonl

The first form compares two runs of this code (fp64 recomputation, alternative scan order).
The second compares with reference amplitude moduli given one per line (IBM's QuiZX amplitudes
for the 75-T circuit, amp_dump.pkl), whose squares are probabilities.
"""
import argparse, glob, json
import numpy as np


def read_rows(path):
    """Read one results file: JSON lines (fields i, p, two_n_p) or CSV (columns index, probability,
    two_n_p), such as the files of the Zenodo record."""
    if path.endswith(".csv"):
        import csv
        with open(path) as fh:
            return [dict(i=int(r["index"]), p=float(r["probability"]), two_n_p=float(r["two_n_p"]))
                    for r in csv.DictReader(fh)]
    return [json.loads(l) for l in open(path) if l.strip()]

ap = argparse.ArgumentParser()
ap.add_argument("--ref", nargs="*", default=[])
ap.add_argument("--ref-amplitudes", default=None)
ap.add_argument("--test", nargs="+", required=True)
ap.add_argument("--tol", type=float, default=1e-3)
a = ap.parse_args()

R = {}
for p in a.ref:
    for f in sorted(glob.glob(p)):
        for r in read_rows(f):
            R[r["i"]] = r["p"]
if a.ref_amplitudes:
    for i, amp in enumerate(np.loadtxt(a.ref_amplitudes)):
        R[i] = float(amp) ** 2
T = [r for p in a.test for f in sorted(glob.glob(p)) for r in read_rows(f)]
print(f"{len(T)} test probabilities, {len(R)} reference probabilities")
dev = []
for t in T:
    if t["i"] not in R:
        print(f"  i={t['i']} missing from the reference"); continue
    rel = abs(t["p"] - R[t["i"]]) / R[t["i"]]; dev.append(rel)
    print(f"  i={t['i']:4d}  test={t['p']:.6e}  reference={R[t['i']]:.6e}  rel. deviation={rel:.2e}")
if dev:
    print(f"median rel. deviation={np.median(dev):.2e}  95th percentile={np.percentile(dev, 95):.2e}  max={max(dev):.2e}  -> "
          f"{'OK' if max(dev) < a.tol else 'CHECK'}")
