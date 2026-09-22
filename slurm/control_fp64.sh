#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=control-fp64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=2600G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
module load StdEnv/2023 python/3.11
source $HOME/venv-dcs/bin/activate
export NUMBA_NUM_THREADS=$(nproc) OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
# Precision control: 5 probabilities recomputed in double precision (~2.2 TB).
python scripts/compute_amplitudes.py run --qasm loop64_314t.qasm --order data/scan_order_314t.npy \
    --samples samples_314t_exp1.txt --start 0 --count 5 --fp64 --out control_fp64.jsonl
python scripts/compare_amplitudes.py --ref "amplitudes_exp1_*.jsonl" --test control_fp64.jsonl
