#!/usr/bin/env bash
# Usage: bash run_pipeline_tandt.sh 0 all
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/scripts/run.sh" tandt "$@"
