#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-cpu}"
case "$backend" in cpu|cu124) ;; *) echo 'Usage: bash environments/setup.sh [cpu|cu124]' >&2; exit 2;; esac
python_bin="${PYTHON_BIN:-python3.10}"
venv_dir="${OCCLUSION_VENV:-.venv-${backend}}"
"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
if [[ "$(uname -s)" == Darwin && "$backend" == cpu ]]; then
  "$venv_dir/bin/python" -m pip install torch==2.5.1
else
  "$venv_dir/bin/python" -m pip install torch==2.5.1 --index-url "https://download.pytorch.org/whl/${backend}"
fi
"$venv_dir/bin/python" -m pip install -r environments/requirements-base.txt
"$venv_dir/bin/python" -m pip install --no-deps -e .
device=cpu
if [[ "$backend" == cu124 ]]; then device=cuda; fi
"$venv_dir/bin/python" scripts/check_environment.py --device "$device"
