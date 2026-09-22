#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=dcs-amplitudes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1150G
#SBATCH --time=15:00:00
#SBATCH --array=0-13
#SBATCH --output=%x-%A_%a.out
module load StdEnv/2023 python/3.11
source $HOME/venv-dcs/bin/activate
export NUMBA_NUM_THREADS=$(nproc) OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
# Experiment 1: 1389 bitstrings in 14 tasks of 100. r_max = 36 needs ~1.1 TB (large-memory node).
python scripts/compute_amplitudes.py run --qasm loop64_314t.qasm --order data/scan_order_314t.npy \
    --samples samples_314t_exp1.txt --start $((SLURM_ARRAY_TASK_ID*100)) --count 100 \
    --out amplitudes_exp1_${SLURM_ARRAY_TASK_ID}.jsonl
