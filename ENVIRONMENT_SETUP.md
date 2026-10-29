# Local Environment Setup

Project path: `/root/autodl-fs/T2-Splat-main`
Conda environment: `/root/autodl-fs/conda-envs/t2splat`

## Activation

```bash
cd /root/autodl-fs/T2-Splat-main
source scripts/activate_local.sh
```

The activation script loads the Conda environment, CUDA compiler paths, and local extension paths. It also sets `OMP_NUM_THREADS=8`, replacing the invalid value of `0` in the original environment.
Running `conda activate` alone does not set the project's `PYTHONPATH`; always use the script above.

## Version Selection

- Python 3.11.16.
- PyTorch 2.7.1 and torchvision 0.22.1, matching the versions specified in the project README.
- The local `/usr/local/cuda` installation is CUDA 12.8.93, so this environment uses the cu128 build of PyTorch; the example in the project README uses cu126.
- See the [official PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/#v271) for the cu128 build.
- The local GPU is an NVIDIA GeForce RTX 4090 D, and extensions are compiled for compute capability 8.9.
- `constraints-local.txt` pins the local versions of PyTorch, torchvision, and NumPy.

## Reproducing the Installation

```bash
CONDA_ENVS_PATH=/root/autodl-fs/conda-envs \
CONDA_PKGS_DIRS=/root/autodl-fs/.cache/conda-pkgs \
conda create -n t2splat python=3.11 pip -y
bash scripts/install_local_dependencies.sh
```

The installation script installs the core dependencies, four extensions (diff-triangle-mesh-rasterization, simple-knn, effrdel, and nvdiffrast), and ffmpeg within the environment for video export.
The installation log is saved to `setup-logs/install.log`.

`nvdiffrast` also requires its package metadata to be installed with `pip install --no-build-isolation --no-deps ./submodules/nvdiffrast`. Running only `build_ext --inplace` does not provide the metadata needed for version lookup. The local installation script already includes this step.

Exact version snapshots are recorded in `requirements-lock-local.txt` (Python packages, including the local nvdiffrast path) and `conda-explicit-local.txt` (the base Conda environment). Local CUDA/C++ extensions must still be built from the current source with the CUDA toolchain; the version lists do not replace this build step.

## Setup Scope

This setup established the base environment for training, rendering, evaluation, and mesh export. No dataset training was started as part of the setup.
The environment for preprocessing priors with Metric3D and Depth Anything V2 was added on 2026-09-27; see below.
`fused_ssim` is an optional acceleration component; this environment uses the project's built-in SSIM implementation.
LPIPS may download torchvision pretrained weights on first use.

## Recorded Validation Results

The following validation was completed on an RTX 4090 D on 2026-09-26:

- All four extensions were built: diff-triangle-mesh-rasterization, simple-knn, effrdel, and nvdiffrast 0.4.0.
- `pip check`: `No broken requirements found.`
- Existing project tests: **all 111 passed, with none skipped**.
- Environment checks passed for dependency imports, PyTorch CUDA forward/backward operations, torchvision CUDA NMS, simple-knn distances compared with brute-force calculations, nvdiffrast CUDA rasterization/backward operations, libigl/Open3D geometry operations, effrdel BVH, and gstaichi CUDA intersection queries.
- The `--help` commands for `train.py`, `render.py`, `create_glb.py`, and `scripts/run_pipeline.py` all exited successfully; ffmpeg was executable.

To repeat the validation:

```bash
cd /root/autodl-fs/T2-Splat-main
source scripts/activate_local.sh
python scripts/check_environment.py
python -m unittest discover -s tests -v
python -m pip check
```

Validation logs: `setup-logs/environment-check.log`, `setup-logs/tests.log`, and `setup-logs/pip-check.log`.

The main dependencies validated included NumPy 1.26.4, SciPy 1.17.1, Open3D 0.20.0, gstaichi 4.6.0, libigl 2.6.3, pybind11 3.1.0, and OpenCV 4.11.0.86.


## Normal and Depth Prior Environment (2026-09-27)

Official source code and verified weights are stored in `/autodl-fs/data/.cache/t2splat/prior-models/`.
Metric3D v2 ViT-Small generates camera-space normals, and Depth Anything V2 ViT-Large generates inverse depth.
Prior generation scripts, complete version snapshots, and records of weight sources are stored in `runs/tandt_priors_4m_20260927/`.

Added dependencies included mmcv 1.7.2 (the pure Python version), mmengine 0.10.7, timm 1.0.20, einops 0.8.1, and iopath 0.1.10.
The versions of PyTorch, torchvision, NumPy, and the existing training extensions were unchanged; `pip check` passed.
Building mmcv requires pinning `setuptools==80.9.0` in the isolated build environment because its legacy setup.py depends on pkg_resources. The training environment's setuptools version remains unchanged.
Do not install the full legacy requirements list from the Metric3D repository directly, as this may replace training dependencies.

The normal preprocessing used the official RGB mean/standard-deviation normalization and resizing with padding to 616×1064, with outputs restored to the original image size.
The official Metric3D weights were missing only the DINO mask_token, which is not used during inference; all other weights matched.
Depth is encoded as 16-bit inverse-depth PNGs and aligned with sparse COLMAP points using the original project's `utils/make_depth_scale.py`.
CUDA inference for both normals and depth passed on the first image. Prior coverage and scale check results were saved in the experiment directory.
