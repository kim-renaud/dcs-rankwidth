#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
# Where does the time of one slice go, and how expensive is a CuPy rescaling pass? One H100, ~20 minutes.
#SBATCH --account=def-kimren
#SBATCH --job-name=gpu-profile
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda
source "$HOME/venv-dcs/bin/activate"
# CuPy's CUB reduction does not compile with this CUDA toolkit: use the standard reduction kernel.
export CUPY_ACCELERATORS=""
cd "$SLURM_SUBMIT_DIR"
QASM=../zenodo/v2/loop64_314t.qasm
ORDER=data/scan_order_314t_optimized.npy
echo "===== 1. one slice with the fused kernel ====="
python scripts/gpu_profile_slice.py --qasm $QASM --order $ORDER --k 5 --samples ../dcs_rorqual/samples_314_exp1.txt
echo
echo "===== 2. micro-benchmark: scan kernels, then the CuPy reduction and rescaling passes ====="
python scripts/gpu_microbench.py --qasm $QASM --order $ORDER --k 5
