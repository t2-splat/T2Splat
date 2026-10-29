# OpenGL mesh rendering benchmark

`tools/mesh_fps_benchmark.py` measures uncapped, off-screen GPU rendering throughput for a vertex-colored triangle mesh. It uses an OpenGL framebuffer and `GL_TIME_ELAPSED` timer queries. No training code, CUDA, or model checkpoint is needed to benchmark an already exported PLY.

## Requirements and installation

- Python 3.8 or newer; Python 3.11 matches the main project environment.
- macOS, Windows, or Linux with a working OpenGL 4.1 core context and GPU timer queries.
- A desktop/display session that GLFW can use to create a hidden window. Off-screen rendering here does not imply support for a display-less server.

From the repository root:

```bash
python3 -m pip install -r requirements-fps.txt
```

These dependencies are optional and separate from the CUDA training dependencies. A separate Python environment can be used on the machine where rendering performance is measured.

## Input mesh

Export a baked, vertex-colored PLY using `create_ply.py` as described in the main README, or use a compatible PLY from another method. The loader accepts:

- `format binary_little_endian 1.0`;
- scalar `x`, `y`, and `z` vertex coordinates;
- unsigned 8-bit `red`, `green`, and `blue` vertex colors, with optional `alpha`;
- a `face` element with a `vertex_indices` list of exactly three integer indices per face.

Properties are read by name, so RGB and RGBA layouts, reordered properties, and extra scalar properties are supported. RGB input receives an internal alpha value of 255. The renderer treats all triangles as opaque even when the PLY contains alpha. Empty meshes, invalid indices, non-finite positions, and non-triangular faces are rejected. ASCII and big-endian PLY files are not supported.

## Usage

Use one resolution for a 1920×1080 comparison:

```bash
python3 tools/mesh_fps_benchmark.py /path/to/mesh.ply \
  --resolution 1920x1080 --json /path/to/fps.json
```

Run both default resolutions:

```bash
python3 tools/mesh_fps_benchmark.py /path/to/mesh.ply
```

Specify multiple resolutions and adjust the sampling settings:

```bash
python3 tools/mesh_fps_benchmark.py /path/to/mesh.ply \
  --resolution 1920x1080 --resolution 1280x960 \
  --warmup 12 --samples 90
```

Disable MSAA:

```bash
python3 tools/mesh_fps_benchmark.py /path/to/mesh.ply \
  --resolution 1920x1080 --no-msaa
```

Run `python3 tools/mesh_fps_benchmark.py --help` for all options. The parent directory of a requested JSON output must already exist.

## Measurement protocol

| Setting | Default |
| --- | --- |
| Resolutions | 1920×1080 and 1280×960 |
| Warmup | 12 frames per resolution |
| Timed samples | 90 frames per resolution |
| Anti-aliasing | 4× MSAA; `--no-msaa` disables it |
| Depth test | Enabled, `GL_LEQUAL` |
| Backface culling | Enabled; counterclockwise front faces |
| VSync | Disabled; no presentation or buffer swap is timed |
| Timing | `GL_TIME_ELAPSED` around framebuffer clear, matrix update, and indexed mesh draw |
| Shading | Interpolated vertex RGB with a `1/2.2` power transform; opaque output |

The mesh is centered on its axis-aligned bounding box and uniformly scaled by the largest half-extent. Timed frames follow one deterministic orbit around the normalized mesh, using a 45-degree vertical field of view and a slightly varying camera elevation. This is an automatic benchmark camera path, not the dataset's training or test camera trajectory. Bounds are computed separately for each mesh, so different exported extents can change screen coverage. Use consistent scene crops, mesh exports, resolutions, and renderer settings for comparisons.

The timer measures GPU execution of the submitted rendering commands. File loading, mesh upload, framebuffer setup, CPU-side timing, presentation, and any MSAA resolve or readback are not included. The results are GPU rendering throughput, not end-to-end application FPS or the monitor's displayed frame rate.

This benchmark renders the PLY's baked vertex colors. It does not restore view-dependent SH shading, per-face MipTex sampling, or GLB texture atlases; rendering quality can differ from the full trained appearance model.

## Results and JSON

- **GPU FPS** (`theoretical_average_fps`): `1000 / average_gpu_ms`.
- **1% Low** (`one_percent_low_fps`): `1000 / p99_gpu_ms`, the reciprocal of the 99th-percentile frame time; it is not the average FPS of the slowest 1% of frames.
- **Mean GPU ms**: arithmetic mean of the GPU frame times.
- **P99 ms**: the 99th percentile of GPU frame times.

Zero-duration GPU queries are retained in the statistics and counted in `zero_gpu_time_samples`. They can occur for very small or fully culled draws. A run with no measurable GPU time fails rather than reporting infinite FPS.

The JSON report also records the mesh counts and bounds, GPU vendor/renderer/OpenGL version, benchmark settings, and results for each resolution. It contains the input mesh's absolute path; review that field before sharing the report. Results are written locally and are not uploaded by the tool.

## Loader checks

The PLY compatibility tests do not create an OpenGL context:

```bash
python3 -m unittest discover -s tests -p test_mesh_fps_benchmark.py -v
```
