#!/usr/bin/env python3
"""
Measure uncapped GPU rendering throughput for a binary little-endian PLY mesh.

Install dependencies:
    python3 -m pip install -r requirements-fps.txt

Run the two default resolutions:
    python3 tools/mesh_fps_benchmark.py /path/to/mesh.ply

Run one resolution and save JSON:
    python3 tools/mesh_fps_benchmark.py mesh.ply \
        --resolution 1920x1080 --json results.json

The benchmark renders to an off-screen framebuffer and measures each frame with
GL_TIME_ELAPSED. It therefore does not depend on display refresh rate or VSync.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import traceback
from pathlib import Path

try:
    import glfw
    import numpy as np
    from OpenGL import GL
    from plyfile import PlyData, PlyParseError
except ImportError as exc:
    print(
        "Missing dependencies. From the repository root, run: python3 -m pip install -r requirements-fps.txt",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


DEFAULT_RESOLUTIONS = [(1920, 1080), (1280, 960)]
VERTEX_STRIDE = 16  # float x/y/z + uchar r/g/b/a

VERTEX_SHADER = """
#version 410 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

FRAGMENT_SHADER = """
#version 410 core
in vec4 v_color;
layout(location = 0) out vec4 out_color;
void main() {
    vec3 color = pow(max(v_color.rgb, vec3(0.0)), vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure uncapped PLY rendering throughput with GPU timer queries."
    )
    parser.add_argument("mesh", type=Path, help="binary little-endian PLY mesh with RGB or RGBA vertex colors")
    parser.add_argument(
        "--resolution",
        action="append",
        type=parse_resolution,
        metavar="WIDTHxHEIGHT",
        help="repeatable; defaults to 1920x1080 and 1280x960",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=90,
        help="GPU-timed frames per resolution (default: 90)",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_int,
        default=12,
        help="warmup frames per resolution (default: 12)",
    )
    parser.add_argument(
        "--no-msaa",
        action="store_true",
        help="disable 4x MSAA (enabled by default)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="write machine-readable results to a JSON file",
    )
    return parser.parse_args()


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution: {value!r}; expected WIDTHxHEIGHT"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive integers")
    return width, height


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative")
    return number


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read triangle PLYs by their declared properties, then pack XYZ/RGBA for GL."""
    if not path.is_file():
        raise FileNotFoundError(f"PLY file not found: {path}")

    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("Incomplete PLY header: end_header not found")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("PLY header must be ASCII text") from exc
            header_lines.append(line)
            if line == "end_header":
                break

        header_bytes = handle.tell()
        if header_lines[0] != "ply":
            raise ValueError("Missing PLY file signature")
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("Only format binary_little_endian 1.0 is supported")
        handle.seek(0)
        try:
            # Fixed list lengths enable memory mapping for large triangle meshes
            # and reject non-triangular faces instead of reading misaligned data.
            ply = PlyData.read(
                handle, mmap="r", known_list_len={"face": {"vertex_indices": 3}}
            )
        except PlyParseError as exc:
            raise ValueError(f"Invalid triangle PLY: {exc}") from exc

    if "vertex" not in ply or "face" not in ply:
        raise ValueError("PLY must contain vertex and face elements")
    source_vertices = ply["vertex"].data
    source_faces = ply["face"].data
    vertex_count, face_count = len(source_vertices), len(source_faces)
    if vertex_count == 0 or face_count == 0:
        raise ValueError("PLY mesh must contain vertices and triangle faces")
    names = source_vertices.dtype.names or ()
    required = ("x", "y", "z", "red", "green", "blue")
    if any(name not in names for name in required):
        raise ValueError("PLY vertices must contain x, y, z and red, green, blue")
    for name in ("x", "y", "z"):
        if source_vertices.dtype[name].kind not in "fiu":
            raise ValueError(f"Vertex property {name} must be a numeric scalar")
    channels = ("red", "green", "blue") + (("alpha",) if "alpha" in names else ())
    for name in channels:
        if source_vertices.dtype[name] != np.dtype("u1"):
            raise ValueError(f"Vertex color {name} must be an unsigned 8-bit scalar")

    vertices = np.empty(
        vertex_count,
        dtype=[("position", "<f4", (3,)), ("color", "u1", (4,))],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices["position"][:, axis] = source_vertices[name]
    vertices["color"][:] = 255
    for channel, name in enumerate(channels):
        vertices["color"][:, channel] = source_vertices[name]
    positions = vertices["position"]
    if not np.isfinite(positions).all():
        raise ValueError("PLY vertex positions must be finite float32 values")
    if "vertex_indices" not in (source_faces.dtype.names or ()):
        raise ValueError("PLY faces must contain a vertex_indices list")
    face_indices = source_faces["vertex_indices"]
    if face_indices.shape != (face_count, 3) or face_indices.dtype.kind not in "iu":
        raise ValueError("PLY faces must have exactly three integer vertex indices")
    if face_indices.min() < 0 or face_indices.max() >= vertex_count:
        raise ValueError("PLY face vertex indices are outside the vertex array")
    if face_indices.size > np.iinfo(np.int32).max:
        raise ValueError("Mesh exceeds the index-count limit of one OpenGL draw call")
    bounds_min = positions.min(axis=0).astype(np.float64)
    bounds_max = positions.max(axis=0).astype(np.float64)
    indices = np.ascontiguousarray(face_indices, dtype=np.uint32).reshape(-1)

    metadata = {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "header_bytes": header_bytes,
        "vertex_count": int(vertex_count),
        "triangle_count": int(face_count),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
    }
    return vertices, indices, metadata


def init_glfw() -> object:
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform == "darwin":
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(64, 64, "PLY GPU Benchmark", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 4.1 context")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    return window


def compile_shader(shader_type: int, source: str) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        message = GL.glGetShaderInfoLog(shader).decode("utf-8", "replace")
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation failed: \n{message}")
    return shader


def create_program() -> int:
    vertex = compile_shader(GL.GL_VERTEX_SHADER, VERTEX_SHADER)
    fragment = compile_shader(GL.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex)
    GL.glAttachShader(program, fragment)
    GL.glLinkProgram(program)
    GL.glDeleteShader(vertex)
    GL.glDeleteShader(fragment)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        message = GL.glGetProgramInfoLog(program).decode("utf-8", "replace")
        GL.glDeleteProgram(program)
        raise RuntimeError(f"Shader linking failed: \n{message}")
    return program


def delete_gl_objects(delete_function, *object_ids: int) -> None:
    """Delete GL object names with an explicit GLuint array for PyOpenGL 3.1.x."""
    valid_ids = [int(value) for value in object_ids if int(value) != 0]
    if not valid_ids:
        return
    values = (ctypes.c_uint * len(valid_ids))(*valid_ids)
    delete_function(len(valid_ids), values)


def upload_mesh(vertices: np.ndarray, indices: np.ndarray) -> tuple[int, int, int]:
    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)
    ebo = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(
        GL.GL_ARRAY_BUFFER,
        vertices.nbytes,
        vertices,
        GL.GL_STATIC_DRAW,
    )
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(
        0, 3, GL.GL_FLOAT, GL.GL_FALSE, VERTEX_STRIDE, ctypes.c_void_p(0)
    )
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(
        1,
        4,
        GL.GL_UNSIGNED_BYTE,
        GL.GL_TRUE,
        VERTEX_STRIDE,
        ctypes.c_void_p(12),
    )

    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
    GL.glBufferData(
        GL.GL_ELEMENT_ARRAY_BUFFER,
        indices.nbytes,
        indices,
        GL.GL_STATIC_DRAW,
    )
    GL.glBindVertexArray(0)
    return vao, vbo, ebo


def create_framebuffer(width: int, height: int, samples: int) -> tuple[int, int, int]:
    framebuffer = GL.glGenFramebuffers(1)
    color = GL.glGenRenderbuffers(1)
    depth = GL.glGenRenderbuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)

    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, color)
    if samples > 1:
        GL.glRenderbufferStorageMultisample(
            GL.GL_RENDERBUFFER, samples, GL.GL_RGBA8, width, height
        )
    else:
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, width, height)
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_RENDERBUFFER,
        color,
    )

    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth)
    if samples > 1:
        GL.glRenderbufferStorageMultisample(
            GL.GL_RENDERBUFFER, samples, GL.GL_DEPTH_COMPONENT24, width, height
        )
    else:
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height
        )
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_RENDERBUFFER,
        depth,
    )

    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Incomplete off-screen framebuffer: 0x{status:x}")
    return framebuffer, color, depth


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / (length if length else 1.0)


def perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    factor = 1.0 / math.tan(fovy / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = factor / aspect
    matrix[1, 1] = factor
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = 2.0 * far * near / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = normalize(eye - target)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = x_axis
    matrix[1, :3] = y_axis
    matrix[2, :3] = z_axis
    matrix[0, 3] = -np.dot(x_axis, eye)
    matrix[1, 3] = -np.dot(y_axis, eye)
    matrix[2, 3] = -np.dot(z_axis, eye)
    return matrix


def make_mvp(
    center: np.ndarray,
    radius: float,
    angle: float,
    aspect: float,
) -> np.ndarray:
    eye = np.array(
        [
            math.sin(angle) * 3.15,
            1.25 + math.sin(angle * 0.7) * 0.28,
            math.cos(angle) * 3.15,
        ],
        dtype=np.float64,
    )
    projection = perspective(math.pi / 4.0, aspect, 0.05, 20.0)
    view = look_at(
        eye,
        np.zeros(3, dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    model = np.eye(4, dtype=np.float64)
    model[0, 0] = model[1, 1] = model[2, 2] = 1.0 / radius
    model[:3, 3] = -center / radius
    return projection @ view @ model


def benchmark_resolution(
    width: int,
    height: int,
    samples: int,
    warmup: int,
    msaa_samples: int,
    program: int,
    vao: int,
    index_count: int,
    center: np.ndarray,
    radius: float,
) -> dict:
    framebuffer, color, depth = create_framebuffer(
        width, height, msaa_samples
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
    GL.glViewport(0, 0, width, height)
    GL.glUseProgram(program)
    GL.glBindVertexArray(vao)
    location = GL.glGetUniformLocation(program, "u_mvp")

    def draw(angle: float) -> None:
        mvp = make_mvp(center, radius, angle, width / height)
        column_major = np.ascontiguousarray(mvp.T, dtype=np.float32)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, column_major)
        GL.glDrawElements(
            GL.GL_TRIANGLES, index_count, GL.GL_UNSIGNED_INT, None
        )

    for frame in range(warmup):
        draw(frame * 0.05)
    GL.glFinish()

    elapsed_ns = np.empty(samples, dtype=np.float64)
    for frame in range(samples):
        angle = frame * (2.0 * math.pi / samples)
        query = int(np.asarray(GL.glGenQueries(1)).reshape(-1)[0])
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
        draw(angle)
        GL.glEndQuery(GL.GL_TIME_ELAPSED)
        query_result = ctypes.c_uint64()
        GL.glGetQueryObjectui64v(
            int(query),
            GL.GL_QUERY_RESULT,
            ctypes.byref(query_result),
        )
        elapsed_ns[frame] = float(query_result.value)
        delete_gl_objects(GL.glDeleteQueries, query)
        if frame % 10 == 0:
            glfw.poll_events()

    if not np.isfinite(elapsed_ns).all() or elapsed_ns.sum() <= 0:
        raise RuntimeError("GPU timer queries returned no measurable elapsed time")

    mean_ms = float(elapsed_ns.mean() / 1e6)
    median_ms = float(np.percentile(elapsed_ns, 50) / 1e6)
    p99_ms = float(np.percentile(elapsed_ns, 99) / 1e6)
    result = {
        "width": width,
        "height": height,
        "samples": samples,
        "msaa_samples": msaa_samples,
        "average_gpu_ms": mean_ms,
        "median_gpu_ms": median_ms,
        "p99_gpu_ms": p99_ms,
        "zero_gpu_time_samples": int(np.count_nonzero(elapsed_ns == 0)),
        "theoretical_average_fps": 1000.0 / mean_ms,
        "one_percent_low_fps": 1000.0 / p99_ms,
    }

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    delete_gl_objects(GL.glDeleteRenderbuffers, color, depth)
    delete_gl_objects(GL.glDeleteFramebuffers, framebuffer)
    return result


def gl_string(name: int) -> str:
    value = GL.glGetString(name)
    return value.decode("utf-8", "replace") if value else "unknown"


def print_results(metadata: dict, gpu: dict, results: list[dict]) -> None:
    print()
    print(f"Mesh: {metadata['file']}")
    print(
        f"Size: {metadata['vertex_count']:,} vertices / "
        f"{metadata['triangle_count']:,} triangles"
    )
    print(f"GPU: {gpu['renderer']}")
    print(
        f"OpenGL: {gpu['version']} / MSAA: "
        f"{results[0]['msaa_samples']}x"
    )
    print()
    print(
        f"{'Resolution':>12}  {'GPU FPS':>10}  {'1% Low':>10}  "
        f"{'Mean GPU ms':>12}  {'P99 ms':>10}"
    )
    for item in results:
        resolution = f"{item['width']}x{item['height']}"
        print(
            f"{resolution:>12}  "
            f"{item['theoretical_average_fps']:>10.1f}  "
            f"{item['one_percent_low_fps']:>10.1f}  "
            f"{item['average_gpu_ms']:>12.2f}  "
            f"{item['p99_gpu_ms']:>10.2f}"
        )
    print()
    print("Note: Off-screen GPU throughput, independent of display refresh rate and VSync.")


def main() -> int:
    args = parse_args()
    resolutions = args.resolution or DEFAULT_RESOLUTIONS
    msaa_samples = 0 if args.no_msaa else 4

    print(f"Reading: {args.mesh}")
    vertices, indices, metadata = read_ply(args.mesh)
    positions = vertices["position"]
    bounds_min = positions.min(axis=0).astype(np.float64)
    bounds_max = positions.max(axis=0).astype(np.float64)
    center = (bounds_min + bounds_max) * 0.5
    radius = float(np.max(bounds_max - bounds_min) * 0.5) or 1.0
    del positions

    window = init_glfw()
    program = vao = vbo = ebo = 0
    try:
        gpu = {
            "vendor": gl_string(GL.GL_VENDOR),
            "renderer": gl_string(GL.GL_RENDERER),
            "version": gl_string(GL.GL_VERSION),
        }
        program = create_program()
        vao, vbo, ebo = upload_mesh(vertices, indices)
        index_count = int(indices.size)
        del vertices, indices

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)
        if msaa_samples > 1:
            GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glClearColor(0.025, 0.055, 0.08, 1.0)

        results = []
        for width, height in resolutions:
            print(f"Benchmarking {width}x{height}...")
            results.append(
                benchmark_resolution(
                    width=width,
                    height=height,
                    samples=args.samples,
                    warmup=args.warmup,
                    msaa_samples=msaa_samples,
                    program=program,
                    vao=vao,
                    index_count=index_count,
                    center=center,
                    radius=radius,
                )
            )

        payload = {
            "mesh": metadata,
            "gpu": gpu,
            "settings": {
                "vsync": False,
                "offscreen": True,
                "backface_culling": True,
                "msaa_samples": msaa_samples,
                "warmup_frames": args.warmup,
                "timed_frames_per_resolution": args.samples,
                "timer": "GL_TIME_ELAPSED",
                "camera_path": "normalized_mesh_orbit",
            },
            "results": results,
        }
        print_results(metadata, gpu, results)

        if args.json:
            args.json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"JSON saved: {args.json.resolve()}")
        return 0
    finally:
        delete_gl_objects(GL.glDeleteBuffers, ebo, vbo)
        delete_gl_objects(GL.glDeleteVertexArrays, vao)
        if program:
            GL.glDeleteProgram(program)
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nBenchmark cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
