# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
GPU micro-benchmark on the heaviest step of the parity-sliced plan of the 314-T circuit (one GPU, ~15 min).

Times, on the real tables of that step:
  stream     a pure streaming copy of the table (the bandwidth ceiling of one read + write pass)
  baseline   the production kernel (tables in global memory, runtime byte loop)
  smem       tables in shared memory, unrolled byte loop
  ilp4       smem + four independent outputs per thread
and sweeps the launch configuration of the best variant. All variants are checked against the baseline.

  python scripts/gpu_microbench.py --qasm loop64_314t.qasm --order data/scan_order_314t_optimized.npy --k 5
"""
import os, sys, time, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import cupy as cp
from dcs_rankwidth import load_circuit, build_injected
from dcs_rankwidth.t_injection import graph_form
from dcs_rankwidth.parity_slicing import choose_parity_slices, build_parity_sliced
from dcs_rankwidth.gpu_kernel import _SRC
from dcs_rankwidth import gpu_variants as V


def timed(launch, reps=3):
    launch()                                           # warm-up (and compilation)
    cp.cuda.Device().synchronize()
    best = 1e9
    for _ in range(reps):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record(); launch(); e.record(); e.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(s, e) / 1000.0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True); ap.add_argument("--order", default="natural")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--step", type=int, default=-1)
    a = ap.parse_args()
    ops, ring = load_circuit(a.qasm); n = len(ring)
    tab, N, nat = build_injected(ops, n); edges, _ = graph_form(tab, N)
    order = nat if a.order == "natural" else [int(q) for q in np.load(a.order)]
    tr, _ = choose_parity_slices(N, edges, order, a.k, verbose=False)
    Z = build_parity_sliced(tab, N, n, order, tr)
    w = [(1 << s.r_out) * (1 << len(s.kernel)) for s in Z.steps]
    j = int(np.argmax(w)) if a.step < 0 else a.step
    st = Z.steps[j]
    T = Z.tables[j]; combos = Z.combos[j]; nb = T.shape[0]
    n_in, n_out = 1 << st.r_in, 1 << st.r_out
    nc = len(combos)
    print(f"plan r_max={Z.rmax}; step {j}: r_in={st.r_in} r_out={st.r_out} bytes={nb} combos={nc} free={st.free}")
    gb = n_out * (nc + 1) * 8 / 1e9
    print(f"memory traffic of this step: {gb:.0f} GB   table buffers: 2 x {n_in * 8 / 2**30:.0f} GiB", flush=True)
    E = cp.empty(n_in, dtype=np.complex64); out = cp.empty(n_out, dtype=np.complex64)
    E.view(np.float32)[:] = cp.float32(0.5)
    dT, dC = cp.asarray(T.reshape(-1)), cp.asarray(combos)
    sm = cp.cuda.Device().attributes["MultiProcessorCount"]
    f0, f1 = complex(0.6, 0.2), complex(0.3, -0.8)
    mask = np.uint64((1 << st.r_in) - 1)
    args_base = lambda: (E, out, np.uint64(n_out), dT, np.int32(nb), dC, np.int32(nc), np.int32(st.r_in),
                         np.uint64(st.beta), np.uint64(1), np.int32(1 if st.free else 0),
                         np.float32(f0.real), np.float32(f0.imag), np.float32(f1.real), np.float32(f1.imag), np.int32(0))
    args_var = lambda: (E, out, np.uint64(n_out), dT, dC, np.int32(nc), np.int32(st.r_in),
                        np.uint64(st.beta), np.uint64(1), np.int32(1 if st.free else 0),
                        np.float32(f0.real), np.float32(f0.imag), np.float32(f1.real), np.float32(f1.imag), np.int32(0))
    k_stream = cp.RawKernel(V.STREAM, "stream_copy")
    k_base = cp.RawKernel(_SRC, "scan_step")
    pre = f"#define NB {nb}\n"
    k_smem = cp.RawKernel(pre + V.SMEM, "scan_smem")
    k_ilp = cp.RawKernel(pre + V.ILP4, "scan_ilp4")
    grid, block = (sm * 32,), (256,)

    def report(name, t, bytes_gb):
        print(f"  {name:22s} {t * 1000:9.1f} ms   {bytes_gb / t:8.0f} GB/s   {bytes_gb / t / 3350:6.1%} of the 3.35 TB/s peak", flush=True)

    t = timed(lambda: k_stream(grid, block, (E, out, np.uint64(n_out), mask)))
    report("stream copy (ceiling)", t, n_out * 16 / 1e9)
    t0 = timed(lambda: k_base(grid, block, args_base())); report("baseline", t0, gb)
    ref = out[::1024].copy()
    t1 = timed(lambda: k_smem(grid, block, args_var())); report("smem + unroll", t1, gb)
    print("    max deviation from baseline:", float(cp.max(cp.abs(out[::1024] - ref))))
    t2 = timed(lambda: k_ilp(grid, block, args_var())); report("smem + unroll + ilp4", t2, gb)
    print("    max deviation from baseline:", float(cp.max(cp.abs(out[::1024] - ref))))
    mx = cp.ReductionKernel("complex64 z", "float32 m", "z.real() * z.real() + z.imag() * z.imag()", "max(a, b)", "m = a", "0", "maxabs2")
    t = timed(lambda: mx(out)); report("max reduction (CuPy)", t, n_out * 8 / 1e9)
    t = timed(lambda: out.__imul__(np.float32(1.0000001))); report("rescaling (CuPy, in place)", t, n_out * 16 / 1e9)
    print("\nlaunch configuration of the ilp4 kernel (blocks per SM x threads per block):")
    for bps in (4, 8, 16, 32):
        for bs in (128, 256, 512):
            t = timed(lambda: k_ilp((sm * bps,), (bs,), args_var()), reps=2)
            print(f"  {bps:2d} x {bs:3d}: {t * 1000:8.1f} ms  {gb / t:7.0f} GB/s", flush=True)
    best = min(t0, t1, t2)
    est_slice = best * sum(w) / w[j]
    print(f"\nrough estimate of one slice with the best variant (same speed on every step): {est_slice:.1f} s "
          f"(measured with the production kernel: 30.4 s)")


if __name__ == "__main__":
    main()
