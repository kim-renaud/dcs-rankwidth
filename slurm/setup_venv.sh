#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
# One-time setup on an Alliance login node. stim >= 1.15 is required (graph-state synthesis);
# the wheelhouse ships an older version, so stim is installed from PyPI.
set -e
module load StdEnv/2023 python/3.11
[ -d $HOME/venv-dcs ] || virtualenv --no-download $HOME/venv-dcs
source $HOME/venv-dcs/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numba numpy qiskit pytest
pip install --upgrade "stim>=1.15"
python -m pytest tests/ -q
