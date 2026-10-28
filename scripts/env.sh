#!/usr/bin/env bash
# Source this file for direct Python commands. Maintained by zhtang2004@gmail.com.
T2SPLAT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$T2SPLAT_ROOT:$T2SPLAT_ROOT/submodules/effrdel/src:$T2SPLAT_ROOT/submodules/diff-triangle-mesh-rasterization:$T2SPLAT_ROOT/submodules/simple-knn:$T2SPLAT_ROOT/submodules/nvdiffrast${PYTHONPATH:+:$PYTHONPATH}"
unset T2SPLAT_ROOT
