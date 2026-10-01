#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
# Parity-sliced contraction of the 314-T circuit on one node with 4 H100 GPUs.
#SBATCH --account=def-kimren
#SBATCH --job-name=gpu-parity-314t
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda
source "$HOME/venv-dcs/bin/activate"
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
# The installed CuPy CUB reduction fails to compile with this CUDA toolkit.
# Select CuPy's standard reduction kernel before starting any Python process.
export CUPY_ACCELERATORS=""
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv

QASM=../zenodo/v2/loop64_314t.qasm
ORDER=data/scan_order_314t_optimized.npy
SAMPLES=../dcs_rorqual/samples_314_exp1.txt
REF=../dcs_rorqual/amps_0.jsonl

# Validate the GPU kernel before launching the full calculation.
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_parity_run.py --selftest

# Give each process one GPU and eight of the 32 slices.
rm -f gpu_partial_*.jsonl
pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$g" python scripts/gpu_parity_run.py "$QASM" --order "$ORDER" --k 5 \
      --samples "$SAMPLES" --count 5 --part "$g" --parts 4 --device 0 \
      --out "gpu_partial_$g.jsonl" &
  pids+=("$!")
done

# A plain wait can hide failures from earlier background processes.
failed=0
for i in 0 1 2 3; do
  if ! wait "${pids[$i]}"; then
    echo "GPU process $i failed" >&2
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi

for g in 0 1 2 3; do
  if [[ ! -f "gpu_partial_$g.jsonl" ]] || [[ $(wc -l < "gpu_partial_$g.jsonl") -ne 5 ]]; then
    echo "GPU process $g did not produce all 5 amplitudes" >&2
    exit 1
  fi
done

python scripts/gpu_parity_run.py --combine "gpu_partial_*.jsonl" --ref "$REF"