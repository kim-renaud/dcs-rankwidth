# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Tests against exact state vectors (small instances). Run: python -m pytest tests/ -q
  - the T-injection identity on full state vectors (T, T-dagger, generic angles)
  - cut ranks from the stabiliser formula against von Neumann entropies
  - amplitudes from the NumPy and Numba executors against state vectors
  - probabilities for bitstrings given in the Qiskit convention, end to end
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, stim, pytest
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector, partial_trace, entropy
from dcs_rankwidth import (load_circuit, build_injected, reference_angles, cut_ranks,
                           amplitude_reference, CompiledScan)


def ring_circuit(n, depth, n_rot, seed, angles=(math.pi / 4,)):
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


def _prepare(qc, tmp_path):
    f = tmp_path / "c.qasm"; qasm2.dump(qc, str(f))
    ops, ring = load_circuit(str(f)); n = len(ring)
    tab, N, natural = build_injected(ops, n)
    w = {q: [1.0, np.exp(1j * th)] for q, th in reference_angles(ops, n).items()}
    return ops, ring, n, tab, N, natural, w


@pytest.mark.parametrize("n,depth,nrot", [(4, 6, 5), (5, 6, 8), (6, 6, 10)])
def test_injection_identity(n, depth, nrot):
    qc = ring_circuit(n, depth, nrot, seed=n, angles=(math.pi / 4, -math.pi / 4, 0.7341))
    refs = [ins.operation.params[0] for ins in qc.data if ins.operation.name == "p"]
    inj = QuantumCircuit(n + len(refs)); k = n
    for ins in qc.data:
        qs = [qc.find_bit(x).index for x in ins.qubits]
        if ins.operation.name == "p":
            inj.cx(qs[0], k); k += 1
        else:
            getattr(inj, ins.operation.name)(*qs)
    S = Statevector(inj).data.reshape([2] * (n + len(refs)), order="F")
    for th in reversed(refs):
        S = np.tensordot(S, np.array([1.0, np.exp(1j * float(th))]), axes=([S.ndim - 1], [0]))
    assert np.max(np.abs(S.reshape(-1, order="F") - Statevector(qc).data)) < 1e-12


def test_cut_ranks_match_entropies():
    qc = ring_circuit(10, 6, 0, seed=3)
    sim = stim.TableauSimulator(); sim.set_num_qubits(10)
    for ins in qc.data:
        q = [qc.find_bit(x).index for x in ins.qubits]
        {"h": sim.h, "s": sim.s, "sx": sim.sqrt_x, "cz": sim.cz}[ins.operation.name](*q)
    sv = Statevector(qc)
    perm = list(np.random.default_rng(0).permutation(10))
    r = cut_ranks(sim, 10, perm)
    exact = [round(entropy(partial_trace(sv, [q for q in range(10) if q not in perm[:i]]), base=2))
             for i in range(1, 10)]
    assert list(r) == exact


@pytest.mark.parametrize("n,depth,nrot", [(8, 8, 10), (10, 10, 20), (12, 12, 30)])
def test_amplitudes(n, depth, nrot, tmp_path):
    qc = ring_circuit(n, depth, nrot, seed=n)
    ops, ring, n, tab, N, natural, w = _prepare(qc, tmp_path)
    sv = Statevector(qc).data
    C = CompiledScan(tab, N, n, natural); C.alloc(np.complex128)
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = [int(b) for b in rng.integers(0, 2, n)]              # x[q]: bit of QASM qubit q
        xr = [x[q] for q in ring] + [0] * (N - n)
        exact = abs(sv[sum(b << q for q, b in enumerate(x))])
        assert abs(abs(C.amplitude(xr, w=w)) - exact) < 1e-12
        assert abs(abs(amplitude_reference(tab, N, n, natural, xr, w=w)) - exact) < 1e-12


def test_qiskit_bitstring_end_to_end(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    from compute_amplitudes import to_ring
    qc = ring_circuit(10, 10, 15, seed=7, angles=(math.pi / 4, -math.pi / 4, 0.7341))
    ops, ring, n, tab, N, natural, w = _prepare(qc, tmp_path)
    sv = Statevector(qc).data
    C = CompiledScan(tab, N, n, natural); C.alloc(np.complex128)
    for idx in np.random.default_rng(2).choice(2 ** n, 4, replace=False, p=np.abs(sv) ** 2):
        bitstring = format(int(idx), f"0{n}b")                   # Qiskit: qubit 0 rightmost
        p = abs(C.amplitude(to_ring(bitstring, ring, "qiskit") + [0] * (N - n), w=w)) ** 2
        assert abs(p - abs(sv[idx]) ** 2) / abs(sv[idx]) ** 2 < 1e-10
