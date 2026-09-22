#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=validation-75t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
module load StdEnv/2023 python/3.11
source $HOME/venv-dcs/bin/activate
export NUMBA_NUM_THREADS=$(nproc) OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
# Cross-check against all 1197 QuiZX amplitudes published by IBM for the 75-T circuit
# (r_max = 33, ~128 GiB, about 2.3 h on 192 cores).
python scripts/compute_amplitudes.py run --qasm loop64_75t.qasm --order data/scan_order_75t.npy \
    --samples samples_75t.txt --out amplitudes_75t.jsonl
python scripts/compare_amplitudes.py --ref-amplitudes amplitudes_75t_quizx.txt --test amplitudes_75t.jsonl
