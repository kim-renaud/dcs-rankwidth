# GPU execution with parity slicing

The contraction of the 314-T circuit has a maximal cut rank of 36, which needs 1 TiB of memory. Parity
slicing lowers the rank so that the table fits on GPUs, and a CUDA kernel runs the scan on them.

## Parity slicing

Fixing a vertex of the graph state to a value removes it from every cut (*vertex slicing*). Measuring it in
another basis fixes a parity of the vertex and its neighbours instead: after a local complementation (Y) or
a pivot (X) the vertex is removed in the same way. Local complementation preserves every cut rank, so the
choice of the transformation changes which cuts the removal lowers. Each of the 2^k slices is then a
contraction of the reduced graph, with the plan compiled once, and the amplitude is the sum over the slices.
The transformations are exact: `tests/` compares sliced amplitudes with exact state vectors.

On the 314-T circuit, a greedy choice of the transformations gives

| slices | 0 | 3 | 5 | 8 |
|---|---|---|---|---|
| maximal cut rank | 36 | 34 | 32 | 31 |
| table (complex64, two buffers) | 1 TiB | 256 GiB | 64 GiB | 32 GiB |

Slicing vertices alone is less effective (36 to 34 with 6 slices). Each slice doubles the work, so the
number of slices is a trade between memory and time.

## GPU kernel

`dcs_rankwidth/gpu_kernel.py` runs the scan on one GPU with CuPy. The step kernel is memory-bound, and two
choices matter:

- **Renormalisation is fused into the step kernel.** The table is rescaled by a power of two, exact in floating
  point, chosen from the maximal modulus of the previous step; the maximal modulus of the new table is
  obtained in registers (warp reduction, one `atomicMax` per warp). A separate CuPy reduction pass over the
  32 GiB table took 3.2 s, about 100 times slower than the memory bandwidth, and made a slice take 30 s.
- **Weights and sums are in double precision**, rounded to complex64 at the store. Single-precision weights
  (1/sqrt(2), ...) round in the same direction every time, which gave a bias of about 5e-8 per vertex on the
  probabilities, +1.8e-5 for 378 vertices.

## Running

```bash
pip install -r requirements.txt -r requirements-gpu.txt
python scripts/gpu_parity_run.py --selftest          # small circuits against the exact state vector, on the GPU

# one process per GPU, each summing a share of the 2^k slices
python scripts/gpu_parity_run.py loop64_314t.qasm --order data/scan_order_314t_optimized.npy --k 5 \
    --samples samples_314t_exp1.txt --count 5 --part 0 --parts 4 --device 0 --out gpu_partial_0.jsonl
# ... parts 1, 2, 3 on the other GPUs, then
python scripts/gpu_parity_run.py --combine "gpu_partial_*.jsonl" --ref "amplitudes_exp1_*.jsonl"
```

`slurm/gpu_parity_314t.sh` does all of this on one node with four H100 GPUs. The CUB reduction of some CuPy
builds does not compile with CUDA 12.6 headers; the job sets `CUPY_ACCELERATORS=""`, which `gpu_profile_slice.py`
and `gpu_microbench.py` also need.

## Measured performance (Rorqual, H100 80 GB)

| | |
|---|---|
| Plan | 5 parity slices, maximal cut rank 32, 64 GiB per GPU, 32 slices of 2^38.14 updates |
| One slice, one GPU | 1.31 s (1.30 s in the scan kernels, 0.01 s on the host) |
| One probability, four GPUs (8 slices each) | 10.5 to 10.7 s |
| One probability, one GPU | about 42 s |
| Heaviest step (103 GB of traffic) | 36 ms, 2.85 TB/s, 85% of the peak; a plain copy of the table takes 24 ms |
| Before the fused renormalisation | 30.4 s per slice, 243 s per probability on four GPUs |
| CPU node, 192 cores, same circuit | about 100 s (order in `scan_order_314t_optimized.npy`), about 200 s (published order) |

These timings were measured with single-precision weights; the kernel is memory-bound, so the change to
double precision is not expected to affect them. With single-precision weights, the five first probabilities
of experiment 1 agreed with the CPU results to 1.1e-5 to 1.9e-5, all with the same sign, which is the bias
described above.

## Files

| Path | Content |
|---|---|
| `dcs_rankwidth/slicing.py` | vertex slicing, sliced scan |
| `dcs_rankwidth/parity_slicing.py` | local complementation, pivoting, greedy choice of the slices |
| `dcs_rankwidth/gpu_kernel.py` | CUDA kernels and the GPU mixin |
| `dcs_rankwidth/gpu_variants.py` | kernel variants for the micro-benchmark only |
| `scripts/gpu_parity_run.py` | self-test, run on one GPU, combination of partial amplitudes |
| `scripts/parity_slice_check.py` | the same on CPU, against reference amplitudes |
| `scripts/gpu_profile_slice.py`, `scripts/gpu_microbench.py` | time of one slice, bandwidth of the kernels |
| `scripts/selftest_circuits.py` | random ring circuits for the self-test |
| `slurm/gpu_parity_314t.sh`, `slurm/gpu_profile.sh` | jobs for four and one H100 |
