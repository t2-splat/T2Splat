#!/usr/bin/env bash
# Pipeline entry point maintained by zhtang2004@gmail.com.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=mipnerf360
if (($#)) && [[ $1 != -* ]]; then DATASET=$1; shift; fi
# Support the old positional GPU, scene-list and regularization arguments.
ARGS=()
for OPTION in --gpu --scenes --reg-mode; do
    if (($#)) && [[ $1 != -* && $1 != nomip ]]; then
        ARGS+=("$OPTION" "$1")
        shift
    fi
done
if (($#)) && [[ $1 == nomip ]]; then ARGS+=(--no-mip); shift; fi
exec "${PYTHON:-python}" "$ROOT/scripts/run_pipeline.py" --dataset "$DATASET" "${ARGS[@]}" "$@"
