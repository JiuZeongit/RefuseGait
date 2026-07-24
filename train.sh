#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 CONFIG [GPU list] [MASTER_PORT]"
    echo
    echo "Examples:"
    echo "  $0 configs/refusegait_davis346_day.yaml 0,1 29501"
    exit 1
fi

CONFIG="$1"
GPUS="${2:-0,1}"
MASTER_PORT="${3:-29501}"

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

cd "$PROJECT_ROOT"

if [ ! -f "$CONFIG" ]; then
    echo "Error: configuration file not found: $CONFIG"
    exit 1
fi

NPROC="$(
    awk -F',' '{print NF}' <<< "$GPUS"
)"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Project root: $PROJECT_ROOT"
echo "Training config: $CONFIG"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of processes: $NPROC"
echo "Master port: $MASTER_PORT"

python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    opengait/main.py \
    --cfgs "$CONFIG" \
    --phase train
