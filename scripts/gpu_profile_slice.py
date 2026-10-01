# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Where does the time of one slice go? Runs ONE slice of the parity-sliced contraction on one GPU and
reports the wall time, the time spent in the scan kernels (CUDA events), and the host overhead
(one scalar read per step). Run it twice: the first run includes the compilation of the kernel.

  python scripts/gpu_profile_slice.py --qasm loop64_314t.qasm --order data/scan_order_314t_optimized.npy \
      --k 5 --samples samples_314t_exp1.txt
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dcs_rankwidth import load_circuit, build_injected, reference_angles
from dcs_rankwidth.t_injection import graph_form
from dcs_rankwidth.parity_slicing import choose_parity_slices
from gpu_parity_run import gpu_sliced
from compute_amplitudes import to_ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True); ap.add_argument("--order", default="natural")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--samples", required=True)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    ops, ring = load_circuit(a.qasm); n = len(ring)
    tab, N, nat = build_injected(ops, n); edges, _ = graph_form(tab, N)
    order = nat if a.order == "natural" else [int(q) for q in np.load(a.order)]
    tr, _ = choose_parity_slices(N, edges, order, a.k, verbose=False)
    Z = gpu_sliced(tab, N, n, order, tr); Z.alloc(device=a.device); Z.profile = True
    w = {q: [1.0, np.exp(1j * th)] for q, th in reference_angles(ops, n).items()}
    line = [l for l in open(a.samples) if l.strip()][0]
    xr = to_ring(line, ring, "qiskit") + [0] * (N - n)
    print(f"plan r_max={Z.rmax}, {len(Z.steps)} steps, one slice of {2 ** a.k}", flush=True)
    for rep in range(2):
        t = time.time(); Z.amplitude(xr, w=w, part=0, parts=2 ** a.k); dt = time.time() - t
        ev = Z._events; kern = sum(x[0] for x in ev)
        print(f"\nrun {rep}: slice in {dt:.2f} s   scan kernels {kern:.2f} s   host/other {dt - kern:.2f} s", flush=True)
    print("\nslowest scan steps of the last run (s, step, r_out):")
    for s, j, ro in sorted(ev, reverse=True)[:8]:
        print(f"  {s * 1000:8.1f} ms   step {j:3d}   r_out={ro}")
    print("\nfor reference: 30.4 s per slice with the previous kernel (separate reduction and rescaling passes)")


if __name__ == "__main__":
    main()
