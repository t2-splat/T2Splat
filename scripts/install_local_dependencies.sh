#!/usr/bin/env bash
# Install dependencies for this machine after creating the Conda environment.
set -euo pipefail
source /root/autodl-fs/T2-Splat-main/scripts/activate_local.sh
cd /root/autodl-fs/T2-Splat-main
python -m pip install --no-cache-dir --progress-bar off --no-compile -c constraints-local.txt torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir --progress-bar off --no-compile -r requirements.txt -c constraints-local.txt imageio-ffmpeg --index-url https://pypi.org/simple
python - <<'PYFFMPEG'
import imageio_ffmpeg
import sys
from pathlib import Path
link = Path(sys.prefix) / 'bin/ffmpeg'
if not link.exists():
    link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
PYFFMPEG
bash scripts/setup.sh --with-tnt
# nvdiffrast imports its version from installed package metadata.
python -m pip install --no-build-isolation --no-deps --no-cache-dir --no-compile ./submodules/nvdiffrast
python -m pip check
