#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/conda_env

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Running with GPU support."
else
    echo "No NVIDIA GPU detected. Running without GPU support."
fi

exec uvicorn remote_srv:app --host 0.0.0.0 --port 8000
