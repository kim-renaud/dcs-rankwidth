#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=control-scan-order
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1400G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
module load StdEnv/2023 python/3.11
source $HOME/venv-dcs/bin/activate
export NUMBA_NUM_THREADS=$(nproc) OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
# Plan control: 3 probabilities recomputed with an independently searched scan order.
python scripts/find_scan_order.py loop64_314t.qasm scan_order_alt.npy --budget 600 --seed 777
python scripts/compute_amplitudes.py run --qasm loop64_314t.qasm --order scan_order_alt.npy \
    --samples samples_314t_exp1.txt --start 10 --count 3 --out control_scan_order.jsonl
python scripts/compare_amplitudes.py --ref "amplitudes_exp1_*.jsonl" --test control_scan_order.jsonl
