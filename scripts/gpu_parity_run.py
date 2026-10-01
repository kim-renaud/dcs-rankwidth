# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Parity-sliced contraction on GPUs.

  # 1. self-test on the GPU (small ring circuit, against the exact state vector)
  python scripts/gpu_parity_run.py --selftest

  # 2. one process per GPU, each summing a share of the slices
  python scripts/gpu_parity_run.py loop64_314t.qasm --order data/scan_order_314t_optimized.npy \
      --k 5 --samples samples_314t_exp1.txt --count 5 --part 0 --parts 4 --out partial_0.jsonl

  # 3. add the partial amplitudes and compare with reference probabilities
  python scripts/gpu_parity_run.py --combine "partial_*.jsonl" --ref "amps_*.jsonl"
"""
import os, sys, glob, json, math, time, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def gpu_sliced(tab, N, n, order, ops_slices):
    from dcs_rankwidth.parity_slicing import build_parity_sliced
    from dcs_rankwidth.slicing import SlicedScan
    from dcs_rankwidth.gpu_kernel import GPUMixin
    Z = build_parity_sliced(tab, N, n, order, ops_slices)
    Z.__class__ = type("GPUSlicedScan", (GPUMixin, SlicedScan), {})
    return Z


def selftest(device):
    from selftest_circuits import ring_circuit
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    from dcs_rankwidth import load_circuit, build_injected, reference_angles
    from dcs_rankwidth.t_injection import graph_form
    from dcs_rankwidth.parity_slicing import choose_parity_slices
    worst = 0.0
    for nq, depth, nrot, seed in [(12, 12, 30, 2), (16, 14, 50, 5)]:
        qc = ring_circuit(nq, depth, nrot, seed=seed); path = f"/tmp/gpu_selftest_{nq}.qasm"; qasm2.dump(qc, path)
        ops, ring = load_circuit(path); n = len(ring); tab, N, nat = build_injected(ops, n)
        w = {q: [1, np.exp(1j * t)] for q, t in reference_angles(ops, n).items()}
        edges, _ = graph_form(tab, N)
        tr, _ = choose_parity_slices(N, edges, nat, 3, verbose=False)
        Z = gpu_sliced(tab, N, n, nat, tr); Z.alloc(device=device)
        sv = Statevector(qc).data
        for t in range(4):
            x = [int(b) for b in np.random.default_rng(t).integers(0, 2, n)]
            xr = [x[q] for q in ring] + [0] * (N - n)
            ex = abs(sv[sum(b << q for q, b in enumerate(x))]) ** 2
            p = abs(Z.amplitude(xr, w=w)) ** 2
            worst = max(worst, abs(p - ex) / ex)
        print(f"selftest n={nq}: r_max with 3 slices = {Z.rmax}, worst relative deviation so far = {worst:.1e}", flush=True)
    ok = worst < 1e-4
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def run(a):
    from dcs_rankwidth import load_circuit, build_injected, reference_angles
    from dcs_rankwidth.t_injection import graph_form
    from dcs_rankwidth.parity_slicing import choose_parity_slices
    from compute_amplitudes import to_ring
    ops, ring = load_circuit(a.qasm); n = len(ring)
    tab, N, nat = build_injected(ops, n)
    order = nat if a.order == "natural" else [int(q) for q in np.load(a.order)]
    edges, _ = graph_form(tab, N)
    t0 = time.time()
    tr, _ = choose_parity_slices(N, edges, order, a.k, verbose=(a.part == 0))
    Z = gpu_sliced(tab, N, n, order, tr)
    mem = 2 * 2 ** Z.rmax * 8 / 2 ** 30
    print(f"# part {a.part}/{a.parts}: {a.k} slices, plan r_max={Z.rmax}, work per slice 2^{math.log2(Z.work):.2f}, "
          f"GPU memory {mem:.0f} GiB, setup {time.time() - t0:.0f} s", flush=True)
    Z.alloc(device=a.device)
    w = {q: [1.0, np.exp(1j * th)] for q, th in reference_angles(ops, n).items()}
    lines = [l for l in open(a.samples) if l.strip()][a.start:a.start + a.count]
    my_slices = len([i for i in range(2 ** a.k) if i % a.parts == a.part])
    with open(a.out, "w") as fo:
        for k, line in enumerate(lines):
            t = time.time()
            amp = Z.amplitude(to_ring(line, ring, "qiskit") + [0] * (N - n), w=w, part=a.part, parts=a.parts)
            dt = time.time() - t
            fo.write(json.dumps(dict(i=a.start + k, x=line.strip(), re=amp.real, im=amp.imag, secs=dt,
                                     slices=my_slices, n=n)) + "\n"); fo.flush()
            print(f"part {a.part}: bitstring {a.start + k}: {dt:.1f} s for {my_slices} slices "
                  f"({dt / my_slices:.2f} s per slice)", flush=True)


def combine(a):
    acc = collections.defaultdict(complex); meta = {}
    for f in sorted(glob.glob(a.combine)):
        for l in open(f):
            if l.strip():
                r = json.loads(l); acc[r["i"]] += complex(r["re"], r["im"]); meta.setdefault(r["i"], r)
    ref = {}
    for f in sorted(glob.glob(a.ref)) if a.ref else []:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "p" in r:
                    ref.setdefault(r["i"], r["p"])
    for i in sorted(acc):
        p = abs(acc[i]) ** 2; n = meta[i]["n"]
        msg = f"bitstring {i}: 2^n p = {p * 2 ** n:.8f}"
        if i in ref:
            msg += f"   reference {ref[i] * 2 ** n:.8f}   rel. dev {abs(p - ref[i]) / ref[i]:.1e}"
        print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--combine", default=None); ap.add_argument("--ref", default=None)
    ap.add_argument("--order", default="natural"); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--samples"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="partial.jsonl")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.device) else 1)
    if a.combine:
        combine(a); return
    run(a)


if __name__ == "__main__":
    main()
