# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Counterfactual circuit: same Clifford skeleton and same number of T gates as the input circuit,
with the T gates placed uniformly at random after CZ gates instead of at code-compatible sites.
Qubits of the output file are labelled by ring position.

  python make_uniform_counterfactual.py loop64_314t.qasm loop64_314t_uniform.qasm [--seed 0]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
import numpy as np
from dcs_rankwidth import load_circuit

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ops, ring = load_circuit(a.src); n = len(ring)
nT = sum(op[0] == "T" for op in ops)
clifford = [op for op in ops if op[0] != "T"]
slots = [(i, q) for i, op in enumerate(clifford) if op[0] == "cz" for q in op[1]]
rng = np.random.default_rng(a.seed); insert = {}
for k in rng.choice(len(slots), nT, replace=False):
    insert.setdefault(slots[k][0], []).append(slots[k][1])
name = {"SQRT_X": "sx", "X": "x", "S": "s", "Z": "z", "S_DAG": "sdg"}
lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
for i, op in enumerate(clifford):
    lines.append(f"cz q[{op[1][0]}],q[{op[1][1]}];" if op[0] == "cz" else f"{name[op[3]]} q[{op[1][0]}];")
    for q in insert.get(i, []):
        lines.append(f"t q[{q}];")
open(a.dst, "w").write("\n".join(lines) + "\n")
print(f"{a.dst}: {n} qubits, {nT} T gates placed uniformly (seed {a.seed})")
