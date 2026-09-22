# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Convert the pickled data of Zenodo record 10.5281/zenodo.22233383 (folder v2/) into plain text.
Requires numpy >= 2 (the pickles were written with numpy 2).

  samples_314t_exp1.txt   1389 bitstrings (experiment 1)
  samples_75t.txt         1197 bitstrings (75-T circuit)
  amplitudes_75t_quizx.txt  1197 amplitude moduli from amp_dump.pkl, in the order of samples_75t.txt
Bitstrings follow the Qiskit convention (qubit 0 is the rightmost bit).
"""
import argparse, pickle
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="v2"); a = ap.parse_args()
load = lambda f: pickle.load(open(f"{a.dir}/{f}", "rb"))
f3, f4, amp = load("fig3_data.pkl"), load("fig4_data.pkl"), load("amp_dump.pkl")
sets = {"samples_314t_exp1.txt": f3["doped_data"][0][314],
        "samples_75t.txt": f4["doped_data"][0][75]}
for name, S in sets.items():
    assert all(len(s) == 64 and set(s) <= {"0", "1"} for s in S), name
    open(name, "w").write("\n".join(S) + "\n")
    print(f"{name}: {len(S)} bitstrings, {len(set(S))} distinct")
a75 = np.array(amp, dtype=float)
assert len(a75) == len(sets["samples_75t.txt"])
np.savetxt("amplitudes_75t_quizx.txt", a75, fmt="%.17g")
z = 2.0 ** 64 * a75 ** 2
print(f"amplitudes_75t_quizx.txt: {len(a75)} values; linear XEB from these = {z.mean() - 1:.4f}")
