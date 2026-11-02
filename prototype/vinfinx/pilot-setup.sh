#!/usr/bin/env bash
set -euo pipefail
umask 077
PILOT_ROOT="${1:?Pass the dedicated absolute pilot directory}"
if [ "$PILOT_ROOT" != /home/ubuntu/vinfinx-pilot ]; then
  echo 'Refusing an unexpected install root.'
  exit 1
fi
export CONDA_PKGS_DIRS="$PILOT_ROOT/cache/conda"
export PIP_CACHE_DIR="$PILOT_ROOT/cache/pip"
if [ ! -x "$PILOT_ROOT/env/bin/python" ]; then
  /home/ubuntu/miniconda3/bin/conda create -y --override-channels -c conda-forge --prefix "$PILOT_ROOT/env" python=3.12 tesseract=5.5.3 pip
fi
"$PILOT_ROOT/env/bin/python" -m pip install \
  lightgbm==4.6.0 scikit-learn==1.9.1 numpy==2.3.5 pandas==2.2.3 \
  scipy==1.18.1 joblib==1.6.0 threadpoolctl==3.7.0 pyarrow==21.0.0 \
  reportlab==4.4.9 pdfplumber==0.11.9 pypdf==6.10.0 pypdfium2==5.14.0 \
  pillow==13.0.0 waitress==3.0.2 pytesseract==0.3.13
"$PILOT_ROOT/env/bin/python" -m pip check
"$PILOT_ROOT/env/bin/tesseract" --version
"$PILOT_ROOT/env/bin/python" -m pip freeze > "$PILOT_ROOT/environment.freeze.txt"
echo 'PILOT_ENVIRONMENT_READY'
