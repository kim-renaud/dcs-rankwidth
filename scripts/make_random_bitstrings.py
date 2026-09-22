# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""Negative control: uniformly random bitstrings, whose XEB should be zero.
  python make_random_bitstrings.py 50 random.txt [--seed 0] [--nbits 64]"""
import argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("n", type=int); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nbits", type=int, default=64)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
open(a.out, "w").write("\n".join("".join(map(str, rng.integers(0, 2, a.nbits))) for _ in range(a.n)) + "\n")
print(f"{a.out}: {a.n} uniform bitstrings of {a.nbits} bits (seed {a.seed})")
