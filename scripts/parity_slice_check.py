# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Probabilities with parity slicing, and comparison with reference values.

  python scripts/parity_slice_check.py loop64_75t.qasm --order data/scan_order_75t.npy --k 4 \
      --samples samples_75t.txt --count 5 --ref-amplitudes amplitudes_75t_quizx.txt

Chooses k parity slices greedily, prints the maximal cut rank before and after, computes the
probabilities of the first `count` bitstrings summed over the 2^k slices, and compares them with
reference amplitude moduli (one per line) if given.
"""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dcs_rankwidth import load_circuit, build_injected, reference_angles
from dcs_rankwidth.t_injection import graph_form
from dcs_rankwidth.parity_slicing import choose_parity_slices, build_parity_sliced
from compute_amplitudes import to_ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm"); ap.add_argument("--order", default="natural")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--samples", required=True); ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--ref-amplitudes", default=None)
    ap.add_argument("--out", default="parity_slice_check.jsonl")
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()
    ops, ring = load_circuit(a.qasm); n = len(ring)
    tab, N, nat = build_injected(ops, n)
    order = nat if a.order == "natural" else [int(q) for q in np.load(a.order)]
    edges, U = graph_form(tab, N)
    t0 = time.time()
    tr, S = choose_parity_slices(N, edges, order, a.k)
    Z = build_parity_sliced(tab, N, n, order, tr)
    mem = 2 * 2 ** Z.rmax * (16 if a.fp64 else 8) / 2 ** 30
    print(f"# {a.k} slices chosen in {time.time() - t0:.0f} s; plan r_max={Z.rmax}, "
          f"work per slice 2^{math.log2(Z.work):.2f}, memory {mem:.0f} GiB", flush=True)
    Z.alloc(np.complex128 if a.fp64 else np.complex64)
    w = {q: [1.0, np.exp(1j * th)] for q, th in reference_angles(ops, n).items()}
    ref = np.loadtxt(a.ref_amplitudes) ** 2 if a.ref_amplitudes else None
    lines = [l for l in open(a.samples) if l.strip()][:a.count]
    with open(a.out, "w") as fo:
        for i, line in enumerate(lines):
            t = time.time()
            p = abs(Z.amplitude(to_ring(line, ring, "qiskit") + [0] * (N - n), w=w)) ** 2
            rec = dict(i=i, x=line.strip(), p=p, two_n_p=p * 2 ** n, secs=time.time() - t)
            if ref is not None:
                rec["rel_dev"] = abs(p - ref[i]) / ref[i]
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"{i}: 2^n p = {p * 2 ** n:.6f}" + (f"   reference {ref[i] * 2 ** n:.6f}   rel. dev {rec['rel_dev']:.1e}" if ref is not None else "")
                  + f"   ({rec['secs']:.0f} s for {2 ** a.k} slices)", flush=True)


if __name__ == "__main__":
    main()
