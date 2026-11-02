#!/usr/bin/env bash
# Usage: bash run_pipeline_dtu.sh 0 all
# Preview a scene: bash run_pipeline_dtu.sh 0 scan63 --dry-run
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/scripts/run.sh" dtu "$@"
