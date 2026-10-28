#!/usr/bin/env bash
# tandt train/truck: 4M points, 8x8 textures, no external normal/depth supervision.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/scripts/run.sh" tandt "$@"
