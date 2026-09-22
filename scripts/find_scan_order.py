# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Search for a scan order with a small maximal cut rank.

Starts from the natural order (each reference after its data qubit), tries every starting point
of the ring, then runs a local search that moves one vertex at a time and accepts moves that do
not increase (r_max, width of the plateau), with mild annealing on the second criterion.

  python find_scan_order.py circuit.qasm order.npy [--budget 240] [--seed 0]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse, math, time
import numpy as np, stim
from dcs_rankwidth import load_circuit, build_injected, cut_ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm"); ap.add_argument("out")
    ap.add_argument("--budget", type=float, default=240); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=None, help="optional initial order (.npy)")
    a = ap.parse_args()
    ops, ring = load_circuit(a.qasm); n = len(ring)
    tab, N, natural = build_injected(ops, n)
    print(f"# {a.qasm}: {n} qubits, {N - n} non-Clifford rotations, N={N}")
    sim = stim.TableauSimulator(); sim.set_inverse_tableau(tab.inverse())

    def cost(o):
        r = cut_ranks(sim, N, o); m = int(r.max())
        return m, float(np.sum(2.0 ** (r - m)))

    if a.start:
        o = [int(q) for q in np.load(a.start)]
    else:
        blocks, cur = [], []
        for q in natural:
            if q < n and cur:
                blocks.append(cur); cur = []
            cur.append(q)
        blocks.append(cur)
        o = min((sum(blocks[s:] + blocks[:s], []) for s in range(n)), key=cost)
    c = cost(o); print(f"# initial order: r_max={c[0]}")
    rng = np.random.default_rng(a.seed); t0 = time.time(); T = 0.3
    while time.time() - t0 < a.budget:
        i = int(rng.integers(N)); L = int(rng.integers(1, 80))
        j = int(np.clip(i + rng.integers(-L, L + 1), 0, N - 1))
        o2 = o.copy(); e = o2.pop(i); o2.insert(j, e); c2 = cost(o2)
        if c2[0] < c[0] or (c2[0] == c[0] and (c2[1] <= c[1] or rng.random() < math.exp(-(c2[1] - c[1]) / T))):
            o, c = o2, c2
        T *= 0.9999
    np.save(a.out, np.array(o))
    mem = 2 * (2 ** c[0]) * 8 / 2 ** 30
    print(f"# final order: r_max={c[0]}  plateau sum 2^(r-r_max)={c[1]:.1f}  "
          f"memory ~ {mem:.0f} GiB (complex64, two buffers) -> {a.out}")


if __name__ == "__main__":
    main()
