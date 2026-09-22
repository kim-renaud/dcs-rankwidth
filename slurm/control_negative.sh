#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=control-negative
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1150G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
module load StdEnv/2023 python/3.11
source $HOME/venv-dcs/bin/activate
export NUMBA_NUM_THREADS=$(nproc) OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
# Negative control: uniformly random bitstrings; the linear XEB must be compatible with zero.
python scripts/make_random_bitstrings.py 50 samples_random.txt --seed 20260919
python scripts/compute_amplitudes.py run --qasm loop64_314t.qasm --order data/scan_order_314t.npy \
    --samples samples_random.txt --count 50 --out control_negative.jsonl
python scripts/analyze_xeb.py control_negative.jsonl --label "negative control (expected XEB = 0)"
