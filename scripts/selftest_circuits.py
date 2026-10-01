# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""Random ring circuits for self-tests (no pytest dependency)."""
import numpy as np
from qiskit import QuantumCircuit


def ring_circuit(n, depth, n_rot, seed, angles=(np.pi / 4,)):
    """Brickwork CZ circuit on a ring with random sqrt(X) / S sqrt(X) layers and diagonal rotations,
    preferentially late in the circuit (as in the DCS construction)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n); qc.h(range(n)); sites = []
    for L in range(depth):
        s = L % 2
        for a in range(s, n, 2):
            b = (a + 1) % n
            qc.cz(a, b); sites += [(L, a, len(qc.data)), (L, b, len(qc.data))]
        for q in range(n):
            if rng.random() < .5:
                qc.s(q)
            qc.sx(q)
    Ls = np.array([x[0] for x in sites], float)
    pick = rng.choice(len(sites), n_rot, replace=False, p=np.exp(4 * Ls / depth) / np.exp(4 * Ls / depth).sum())
    marks = {}
    for k in pick:
        marks.setdefault(sites[k][2], []).append((sites[k][1], float(rng.choice(angles))))
    out = QuantumCircuit(n)
    for i, ins in enumerate(qc.data):
        out.append(ins.operation, ins.qubits)
        for q, th in marks.get(i + 1, []):
            out.p(th, q)
    return out
