# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Exact output probabilities of a doped Clifford circuit for a list of bitstrings.

  run    probabilities p(x) = |<x|U|0>|^2 for the bitstrings of a file (one per line)
  check  replaces every T by S (same graph, same plan, same cut ranks) so that the circuit becomes
         Clifford, and compares the result with the exact probability computed by stim

Examples
  python compute_amplitudes.py run --qasm loop64_314t.qasm --order ../data/scan_order_314t.npy \
         --samples samples_exp1.txt --start 0 --count 100 --out amplitudes_0.jsonl
  python compute_amplitudes.py check --qasm loop64_314t.qasm --order ../data/scan_order_314t.npy

The number of threads follows NUMBA_NUM_THREADS. Output lines are appended, so an interrupted job
can be resumed with --start.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse, json, math, time
import numpy as np, stim
from dcs_rankwidth import load_circuit, build_injected, reference_angles, CompiledScan


def _apply_clifford_version(sim, ops):
    """Apply the circuit to a stim simulator, with every T replaced by S (and T-dagger by S-dagger)."""
    gate = {"SQRT_X": sim.sqrt_x, "X": sim.x, "S": sim.s, "Z": sim.z, "S_DAG": sim.s_dag}
    for op in ops:
        if op[0] == "cz":
            sim.cz(*op[1])
        elif op[0] == "c1":
            gate[op[3]](op[1][0])
        else:
            th = (float(op[3]) if len(op) > 3 else math.pi / 4) % (2 * math.pi)
            if abs(th - math.pi / 4) < 1e-9:
                sim.s(op[1][0])
            elif abs(th - 7 * math.pi / 4) < 1e-9:
                sim.s_dag(op[1][0])
            else:
                raise ValueError(f"the T->S check requires angles of +-pi/4, got {th}")


def stim_probability(ops, n, x_ring):
    """Exact probability of x (ring order) for the circuit with T replaced by S."""
    sim = stim.TableauSimulator(); sim.set_num_qubits(n)
    _apply_clifford_version(sim, ops)
    p = 1.0
    for q in range(n):
        z = sim.peek_z(q)
        if z == 0:
            p *= 0.5
        elif z != (-1 if x_ring[q] else 1):
            return 0.0
        sim.postselect_z(q, desired_value=bool(x_ring[q]))
    return p


def stim_sample(ops, n, rng):
    sim = stim.TableauSimulator(); sim.set_num_qubits(n)
    _apply_clifford_version(sim, ops)
    return [int(b) for b in sim.measure_many(*range(n))]


def to_ring(bitstring, ring, bitorder):
    """Convert a bitstring over QASM qubits to ring order. 'qiskit': qubit 0 is the rightmost bit."""
    b = bitstring.strip()
    bits = [int(c) for c in (b[::-1] if bitorder == "qiskit" else b)]
    return [bits[q] for q in ring]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "check"])
    ap.add_argument("--qasm", required=True)
    ap.add_argument("--order", required=True, help="scan order (.npy), or 'natural'")
    ap.add_argument("--samples"); ap.add_argument("--out", default="amplitudes.jsonl")
    ap.add_argument("--bitorder", default="qiskit", choices=["qiskit", "index"])
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=10 ** 9)
    ap.add_argument("--ncheck", type=int, default=2)
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()

    ops, ring = load_circuit(a.qasm); n = len(ring)
    tab, N, natural = build_injected(ops, n)
    order = natural if a.order == "natural" else [int(q) for q in np.load(a.order)]
    t0 = time.time(); C = CompiledScan(tab, N, n, order)
    mem = 2 * (1 << C.rmax) * (16 if a.fp64 else 8) / 2 ** 30
    print(f"# plan: {time.time() - t0:.1f} s  N={N}  r_max={C.rmax}  work=2^{math.log2(C.work):.2f}  "
          f"memory={mem:.0f} GiB", flush=True)
    C.alloc(np.complex128 if a.fp64 else np.complex64)
    ang = reference_angles(ops, n)
    print(f"# non-Clifford rotations: {len(ang)}", flush=True)

    if a.mode == "check":
        w = {q: [1.0, np.exp(2j * th)] for q, th in ang.items()}     # T -> S doubles the angle
        rng = np.random.default_rng(0)
        for i in range(a.ncheck):
            x = stim_sample(ops, n, rng)
            t = time.time()
            ps = abs(C.amplitude(list(x) + [0] * (N - n), w=w)) ** 2
            pe = stim_probability(ops, n, x)
            print(f"check {i}: scan={ps:.6e}  stim={pe:.6e}  rel. deviation={abs(ps - pe) / pe:.1e}  "
                  f"({time.time() - t:.0f} s)", flush=True)
        return

    w = {q: [1.0, np.exp(1j * th)] for q, th in ang.items()}
    lines = [l for l in open(a.samples) if l.strip()][a.start:a.start + a.count]
    with open(a.out, "a") as fo:
        for k, line in enumerate(lines):
            t = time.time()
            p = abs(C.amplitude(to_ring(line, ring, a.bitorder) + [0] * (N - n), w=w)) ** 2
            fo.write(json.dumps(dict(i=a.start + k, x=line.strip(), p=p, two_n_p=p * 2 ** n,
                                     secs=time.time() - t)) + "\n"); fo.flush()
            print(f"{a.start + k}: 2^n p = {p * 2 ** n:.4f}  ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
