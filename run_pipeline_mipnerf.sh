#!/usr/bin/env bash
# Entry point maintained by zhtang2004@gmail.com.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/scripts/run.sh" mipnerf360 "$@"
