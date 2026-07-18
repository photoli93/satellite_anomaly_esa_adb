#!/usr/bin/env bash
set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "No active virtualenv detected. Ubuntu 24.04 blocks system-wide pip installs (PEP 668)."
    echo "Activate the venv first, e.g.: source ~/sat-anom-jetson-env/bin/activate"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "Installed into: ${VIRTUAL_ENV}"
python3 -c "
import onnxruntime, numpy, sklearn, joblib
print(f'onnxruntime  : {onnxruntime.__version__}')
print(f'numpy        : {numpy.__version__}')
print(f'scikit-learn : {sklearn.__version__}')
print(f'joblib       : {joblib.__version__}')
print(f'providers    : {onnxruntime.get_available_providers()}')
"