# 本机环境配置

项目路径：`/root/autodl-fs/T2-Splat-main`
Conda 环境：`/root/autodl-fs/conda-envs/t2splat`

## 激活

```bash
cd /root/autodl-fs/T2-Splat-main
source scripts/activate_local.sh
```

激活脚本会载入 Conda 环境、CUDA 编译器路径与项目本地扩展路径，并设置 `OMP_NUM_THREADS=8`，修复原环境中的无效值 `0`。
仅运行 `conda activate` 不会自动设置项目的 `PYTHONPATH`；建议始终使用上述脚本。

## 版本选择

- Python 3.11.16。
- PyTorch 2.7.1、torchvision 0.22.1，与项目 README 指定版本一致。
- 本机 `/usr/local/cuda` 是 CUDA 12.8.93，因此选择 PyTorch 的 cu128 构建；项目 README 的示例使用 cu126。
- cu128 构建见 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v271)。
- 本机 GPU 为 NVIDIA GeForce RTX 4090 D，扩展编译目标为 8.9。
- `constraints-local.txt` 固定 PyTorch、torchvision 和 NumPy 的本机版本。

## 重现安装

```bash
CONDA_ENVS_PATH=/root/autodl-fs/conda-envs \
CONDA_PKGS_DIRS=/root/autodl-fs/.cache/conda-pkgs \
conda create -n t2splat python=3.11 pip -y
bash scripts/install_local_dependencies.sh
```

安装脚本包含核心依赖、四个扩展（diff-triangle-mesh-rasterization、simple-knn、effrdel、nvdiffrast），以及用于视频导出的环境内 ffmpeg。
安装日志位于 `setup-logs/install.log`。

`nvdiffrast` 还需要通过 `pip install --no-build-isolation --no-deps ./submodules/nvdiffrast` 正式安装包元数据，仅执行 `build_ext --inplace` 无法满足其版本读取要求。本机安装脚本已经包含这一步。

精确版本快照：`requirements-lock-local.txt`（Python 包，包含本地 nvdiffrast 路径）和 `conda-explicit-local.txt`（Conda 基础环境）。本地 CUDA/C++ 扩展仍需使用当前源码和 CUDA 工具链构建；版本清单不能代替扩展构建。

## 配置范围

本次配置训练、渲染、评估与网格导出的基础环境。未启动数据集训练。
2026-09-27 已追加配置 Metric3D 与 Depth Anything V2 先验预处理环境，详见下文。
`fused_ssim` 是可选加速组件，当前使用项目内置的 SSIM 实现。
LPIPS 的 torchvision 预训练权重在首次实际使用时可能下载。

## 验证结果

2026-09-26 在 RTX 4090 D 上完成验证：

- 四个扩展已构建：diff-triangle-mesh-rasterization、simple-knn、effrdel、nvdiffrast 0.4.0。
- `pip check`：`No broken requirements found.`
- 项目原有测试：**111 项全部通过，无跳过**。
- 环境自检通过：依赖导入、PyTorch CUDA 前向/反向、torchvision CUDA NMS、simple-knn 距离与暴力计算对照、nvdiffrast CUDA 光栅化/反向、libigl/Open3D 几何操作、effrdel BVH 和 gstaichi CUDA 求交。
- `train.py`、`render.py`、`create_glb.py`、`scripts/run_pipeline.py` 的 `--help` 均正常退出；ffmpeg 可执行。

重新验证：

```bash
cd /root/autodl-fs/T2-Splat-main
source scripts/activate_local.sh
python scripts/check_environment.py
python -m unittest discover -s tests -v
python -m pip check
```

验证日志：`setup-logs/environment-check.log`、`setup-logs/tests.log`、`setup-logs/pip-check.log`。

已验证的主要依赖包括 NumPy 1.26.4、SciPy 1.17.1、Open3D 0.20.0、gstaichi 4.6.0、libigl 2.6.3、pybind11 3.1.0、OpenCV 4.11.0.86。


## 法线与深度先验环境（2026-09-27）

官方源码及校验过的权重位于 `/autodl-fs/data/.cache/t2splat/prior-models/`。
使用 Metric3D v2 ViT-Small 生成相机空间法线，Depth Anything V2 ViT-Large 生成逆深度。
先验生成脚本、完整版本快照及权重来源记录位于 `runs/tandt_priors_4m_20260927/`。

新增依赖包括 mmcv 1.7.2（纯 Python 版本）、mmengine 0.10.7、timm 1.0.20、einops 0.8.1、iopath 0.1.10。
未改动 PyTorch、torchvision、NumPy 与已有训练扩展版本；`pip check` 通过。
构建 mmcv 时需在隔离构建环境约束 `setuptools==80.9.0`，其旧版 setup.py 依赖 pkg_resources；训练环境的 setuptools 保持不变。
不直接安装 Metric3D 仓库的整份旧版 requirements，以免替换训练依赖。

本次法线预处理使用官方 RGB 均值/标准差归一化与 616×1064 缩放补边，输出恢复到原图大小。
Metric3D 官方权重仅缺失推理不使用的 DINO mask_token，其余权重全部匹配。
深度编码为 16 位逆深度 PNG，使用原项目 `utils/make_depth_scale.py` 与 COLMAP 稀疏点对齐。
首图法线与深度 CUDA 推理均已通过，先验覆盖和尺度检查结果保存于实验目录。
