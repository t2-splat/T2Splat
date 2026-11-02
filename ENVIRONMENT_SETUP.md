# Local Environment Setup

Project path: `/root/autodl-fs/T2-Splat-main`
Conda environment: `/root/autodl-fs/conda-envs/t2splat`

## Activation

```bash
cd /root/autodl-fs/T2-Splat-main
source scripts/activate_local.sh
```

The activation script loads the Conda environment, CUDA compiler paths, and local extension paths, and sets `OMP_NUM_THREADS=8`.
Running `conda activate` alone does not set the project's `PYTHONPATH`; always use the script above.

## Version Selection

- Python 3.11.16.
- PyTorch 2.7.1 and torchvision 0.22.1.
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

`nvdiffrast` also requires its package metadata to be installed with `pip install --no-build-isolation --no-deps ./submodules/nvdiffrast`. Running only `build_ext --inplace` does not provide the metadata needed for version lookup. The local installation script already includes this step.

Exact version snapshots are recorded in `requirements-lock-local.txt` (Python packages, including the local nvdiffrast path) and `conda-explicit-local.txt` (the base Conda environment). Local CUDA/C++ extensions must still be built from the current source with the CUDA toolchain; the version lists do not replace this build step.

`fused_ssim` is an optional acceleration component; this environment uses the project's built-in SSIM implementation.
LPIPS may download torchvision pretrained weights on first use.

## Validation

```bash
cd /root/autodl-fs/T2-Splat-main
source scripts/activate_local.sh
python scripts/check_environment.py
python -m unittest discover -s tests -v
python -m pip check
```

## Normal and Depth Prior Environment

Official source code and verified weights are stored in `/autodl-fs/data/.cache/t2splat/prior-models/`.
Metric3D v2 ViT-Small generates camera-space normals, and Depth Anything V2 ViT-Large generates inverse depth.

Prior dependencies include mmcv 1.7.2 (the pure Python version), mmengine 0.10.7, timm 1.0.20, einops 0.8.1, and iopath 0.1.10.
Building mmcv requires pinning `setuptools==80.9.0` in the isolated build environment because its legacy setup.py depends on pkg_resources. The training environment's setuptools version remains unchanged.
Do not install the full legacy requirements list from the Metric3D repository directly, as this may replace training dependencies.

The cached Metric3D weights omit only the DINO `mask_token`, which is not used during inference.
Depth is encoded as 16-bit inverse-depth PNGs and aligned with sparse COLMAP points using the original project's `utils/make_depth_scale.py`.
