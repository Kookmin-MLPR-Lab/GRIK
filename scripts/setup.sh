#!/usr/bin/env bash
# One-shot environment bootstrap.
#   $ bash scripts/setup.sh
# Creates .venv with paper-tested dependencies and downloads LongBench data.
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if [ ! -d data/LongBench ]; then
    echo "[setup] downloading LongBench v1 data..."
    mkdir -p data/LongBench && cd data/LongBench
    curl -L -o data.zip https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
    unzip -q data.zip && rm data.zip
    cd "$REPO"
fi

echo "[setup] done. Activate with:  source .venv/bin/activate"
