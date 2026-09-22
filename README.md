# dcs-rankwidth

[![tests](https://github.com/kim-renaud/dcs-rankwidth/actions/workflows/tests.yml/badge.svg)](https://github.com/kim-renaud/dcs-rankwidth/actions/workflows/tests.yml)

Exact classical amplitudes for the doped Clifford sampling (DCS) circuit of
[arXiv:2607.25941v3](https://arxiv.org/abs/2607.25941) (IBM, 64 qubits, depth 73, 314 T gates),
by T-gate injection and linear rank-width contraction.

## Method

1. **T injection.** Each diagonal rotation diag(1, e^{iθ}) on qubit q is replaced by a reference
   qubit in |0⟩, a CNOT from q to the reference, and a contraction of the reference with
   w = (1, e^{iθ}) (T gadget; Gottesman and Chuang 1999, Bravyi and Gosset 2016). The circuit
   becomes Clifford on N = n + t qubits and ⟨x|U|0⟩ = (⟨x| ⊗ w^{⊗t}) |S⟩, with |S⟩ a stabiliser state.
2. **Graph-state scan.** |S⟩ is written as a graph state with local Clifford corrections, which are
   absorbed into the contraction vectors. Vertices are summed in a fixed linear order; the partial
   sum is indexed by the F₂ syndrome of the current cut and has 2^r entries, where r is the cut
   rank. This is the linear case of the rank-width contraction of Kuyanov and Kissinger
   ([arXiv:2603.06764](https://arxiv.org/abs/2603.06764), Prop. 4.1 and Cor. 4.2).
3. **Scan order.** `data/scan_order_314t.npy` has maximal cut rank r_max = 36 on the 314-T circuit:
   2^36 complex64 entries per buffer, about 1.1 TB with two buffers, and 2^41.5 sparse updates per
   amplitude (about 3.5 min on 192 cores).

## Layout

| Path | Content |
|---|---|
| `dcs_rankwidth/circuit_io.py` | QASM input, ring reconstruction, cut ranks over F₂ |
| `dcs_rankwidth/t_injection.py` | T injection, graph-state form (stim) |
| `dcs_rankwidth/scan_plan.py` | scan plan compilation; NumPy reference executor |
| `dcs_rankwidth/scan_kernel.py` | parallel Numba executor |
| `scripts/compute_amplitudes.py` | probabilities for a list of bitstrings; T→S check against stim |
| `scripts/find_scan_order.py` | search for a low-cut-rank scan order |
| `scripts/analyze_xeb.py` | linear and log XEB, maximum-likelihood fidelity, KS test |
| `scripts/compare_amplitudes.py` | index-by-index comparison of probabilities |
| `scripts/make_uniform_counterfactual.py` | same skeleton, T gates placed uniformly |
| `scripts/make_random_bitstrings.py` | uniform bitstrings for the negative control |
| `scripts/extract_zenodo_samples.py` | converts IBM's Zenodo pickles to text |
| `scripts/make_figures.py` | the three figures of the submission |
| `tests/` | tests against exact state vectors |
| `slurm/` | job scripts used on Rorqual (Digital Research Alliance of Canada) |
| `data/` | scan orders |

## Data

IBM's circuits and samples: Zenodo [10.5281/zenodo.22233383](https://doi.org/10.5281/zenodo.22233383)
(folder `v2/`). `loop64_314t.qasm` has md5 `054441020510c19c628284de146f5f34`, identical to the
file attached to Quantum Advantage Tracker issue #245. Convert the samples with

    python scripts/extract_zenodo_samples.py --dir v2      # requires numpy >= 2

Per-sample probabilities computed with this code: Zenodo record
[22882048](https://zenodo.org/records/22882048) (`probabilities_exp1.csv`,
`probabilities_75t_validation.csv`). To recompute the fidelity estimates from it:

    python scripts/analyze_xeb.py probabilities_exp1.csv --expect 1389

## Reproducing a result

    pip install -r requirements.txt
    python -m pytest tests/ -q

    # one probability of the 314-T circuit (needs ~1.1 TB of memory)
    python scripts/compute_amplitudes.py run --qasm loop64_314t.qasm \
        --order data/scan_order_314t.npy --samples samples_314t_exp1.txt --count 1 --out p.jsonl

    # full-scale check: T replaced by S, compared with stim
    python scripts/compute_amplitudes.py check --qasm loop64_314t.qasm --order data/scan_order_314t.npy

    python scripts/analyze_xeb.py "amplitudes_exp1_*.jsonl" --expect 1389

Bitstrings follow the Qiskit convention (qubit 0 is the rightmost bit). Probabilities are exact up
to floating-point rounding (relative deviation below 1e-5 in complex64 at r = 36); the global phase
of the amplitude is not tracked.

## Results (experiment 1, 1389 bitstrings)

| Estimator | Value | 95% CI |
|---|---|---|
| Linear XEB | 0.390 | [0.325, 0.457] |
| log-XEB | 0.399 | [0.336, 0.463] |
| Maximum-likelihood fidelity | 0.395 | [0.334, 0.455] |

## Validation

| Check | Result |
|---|---|
| All 1,197 QuiZX amplitudes published by IBM for the 75-T circuit (`amp_dump.pkl`) | median rel. deviation 2.9e-7, max 5.4e-6, same linear XEB (0.3845); 2.3 h on one node |
| Double-precision recomputation at r = 36 (5 probabilities) | max rel. deviation 6.1e-6 |
| Independently searched scan order (3 probabilities) | max rel. deviation 6.0e-7 |
| Negative control, 50 uniformly random bitstrings | linear XEB -0.06 +- 0.24 |

The 75-T comparison is reproduced by `slurm/validation_75t.sh`, after converting `amp_dump.pkl` with
`scripts/extract_zenodo_samples.py`.

## Software

Python 3.11, numba 0.65.1, numpy 1.26.4, qiskit 2.3.1, stim 1.16.0.

## License

Copyright 2026 Kim Renaud, Calcul Québec.

The code is licensed under the Apache License, Version 2.0; see `LICENSE` and `NOTICE`.
The per-sample probabilities deposited on Zenodo are licensed under CC BY 4.0.
