#!/usr/bin/env bash
# DTU per-scene best preset (configs/dtu.yaml): 20k geometry, 650k point budget.
# Default: lambda_normals=1e-5, final visibility > 0.6, early prune=0.17.
# scan63: normals=1e-6, final=0.5; scan65: early prune=0.14, final=0.5; scan97: final=0.5.
# Legacy RDT, 5% safe Heartbeat, direct visibility pruning, no external priors.
# Stages: geometry -> clean mesh export -> DTU/GOF evaluation.
# Usage after environment activation: bash run_pipeline_dtu.sh 0 all
# Preview a scene: bash run_pipeline_dtu.sh 0 scan24 --dry-run
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/scripts/run.sh" dtu "$@"
