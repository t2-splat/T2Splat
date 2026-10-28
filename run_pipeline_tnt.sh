#!/usr/bin/env bash
# TNT: all six geometry scenes, 30k, 5.2M points, safe RDT, Heartbeat 0.04.
# No external priors or texture stage; corrected virtual-scan geometry evaluation.
# tandt train/truck rendering uses the separate run_pipeline_tandt.sh preset.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/scripts/run.sh" tnt "$@"
