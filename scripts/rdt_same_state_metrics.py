"""Save legacy RDT metrics, optionally adding a same-input surface comparison."""


def compare_and_save(vertices, before_faces, output_dir,
                     max_distance_ratio=0.5, min_normal_cos=0.15,
                     min_quality=0.02, include_surface=True):
    """Run legacy RDT once, optionally selecting a surface from its candidates.

    CPU arrays and completed stages are saved immediately. The input and legacy
    result receive no cleaning, vertex merging, component pruning, or orientation.
    With include_surface=False, the surface module is never imported or called.
    Returns a JSON-compatible summary; stage failures are recorded and re-raised.
    """
    import csv
    from datetime import datetime, timezone
    import inspect
    import json
    from pathlib import Path
    import sys
    import time

    import numpy as np
    from utils.mesh_topology import mesh_topology_metrics

    started = time.perf_counter()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    vertices, before_faces = np.asarray(vertices), np.asarray(before_faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if before_faces.ndim != 2 or before_faces.shape[1] != 3:
        raise ValueError("before_faces must have shape (F, 3)")
    if before_faces.dtype.kind not in "iu":
        raise ValueError("before_faces must have an integer dtype")

    def native(value):
        if isinstance(value, dict):
            return {str(key): native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [native(item) for item in value]
        if isinstance(value, np.ndarray):
            return native(value.tolist())
        if isinstance(value, np.generic):
            return native(value.item())
        if isinstance(value, Path):
            return str(value)
        return value

    def save_json(name, value):
        path = destination / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(native(value), ensure_ascii=False,
                                       indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    summary = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(destination),
        "protocol": {
            "same_vertices_and_before_faces": True,
            "planned_legacy_rdt_calls": 1,
            "include_surface": bool(include_surface),
            "input_or_legacy_cleaning": False,
            "legacy_orient": False,
            "face_components_definition": "connected components through shared edges",
            "triangle_quality_definition": "q = 4 * sqrt(3) * area / (a^2 + b^2 + c^2)",
            "triangle_quality_range": "1 for equilateral, 0 for collapsed triangles",
            "metrics_definition": (
                "Existing mesh_topology_metrics; boundaries are allowed. Invalid-index "
                "and repeated-index faces are counted and excluded from edge/fan graphs. "
                "This does not test global self-intersections."
            ),
        },
        "provenance": {
            "comparison_script": str(Path(__file__).resolve()),
            "metrics_source": str(Path(inspect.getfile(mesh_topology_metrics)).resolve()),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "vertex_dtype": str(vertices.dtype),
            "before_face_dtype": str(before_faces.dtype),
        },
        "arrays": {"vertices": "vertices.npy", "before_faces": "before_faces.npy"},
        "timings_seconds": {},
        "stages": {},
        "legacy_rdt_calls": 0,
    }
    stage_names = ("before", "legacy")
    headings = ("Metric", "Before RDT", "After legacy RDT")
    if include_surface:
        summary["protocol"]["surface_input"] = "exact legacy_faces.npy candidates"
        summary["protocol"]["surface_parameters"] = {
            "max_distance_ratio": float(max_distance_ratio),
            "min_normal_cos": float(min_normal_cos),
            "min_quality": float(min_quality),
        }
        stage_names += ("surface",)
        headings += ("After Safe RDT",)
    rows = (
        ("Triangle count", "faces"),
        ("Face component count", "face_components"),
        ("Faces in the largest component", "largest_component_faces"),
        ("Non-manifold edges / vertices", None),
        ("Median triangle quality", "triangle_quality_median"),
    )

    def value_for(stage, field, pretty=False):
        metrics = summary["stages"].get(stage, {}).get("metrics")
        if metrics is None:
            return "Pending" if pretty else ""
        if field is None:
            edge, vertex = metrics["nonmanifold_edges"], metrics["nonmanifold_vertices"]
            return f"{edge:,} / {vertex:,}" if pretty else f"{edge} / {vertex}"
        value = native(metrics[field])
        if value is None:
            return "Undefined" if pretty else ""
        if pretty and isinstance(value, int):
            return f"{value:,}"
        # Preserve the measured float's full representation, without rounding.
        return str(value)

    def save_reports():
        summary["timings_seconds"]["total_elapsed"] = time.perf_counter() - started
        save_json("summary.json", summary)
        with (destination / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headings)
            for label, field in rows:
                writer.writerow([label] + [value_for(stage, field) for stage in stage_names])
        lines = [
            "# RDT comparison from the same training state" if include_surface
            else "# Before and after legacy RDT", "",
            f"Status: {summary['status']}. Started (UTC): {summary['started_utc']}.", "",
            "| " + " | ".join(headings) + " |",
            "| --- |" + " ---: |" * len(stage_names),
        ]
        for label, field in rows:
            lines.append("| " + " | ".join(
                [label] + [value_for(stage, field, pretty=True) for stage in stage_names]) + " |")
        protocol_text = "Protocol: Legacy RDT runs exactly once. "
        if include_surface:
            protocol_text += "Safe RDT calls `select_rdt_surface` on the same returned candidate faces. "
        protocol_text += ("All stages share the original vertex coordinates. Neither the input nor "
                          "the legacy result undergoes cleaning, vertex merging, or component pruning. "
                          "Legacy RDT uses `orient=False`.")
        lines.extend([
            "", protocol_text, "",
            "Face components are connected through shared edges. Triangle quality is "
            "`q = 4 * sqrt(3) * A / (a^2 + b^2 + c^2)`, where A is the area and a, b, c "
            "are edge lengths: 1 for equilateral triangles and 0 for degenerate triangles. "
            "Table values retain the measured precision.", "",
            "Boundaries are allowed. Non-manifold edge and vertex counts use the project's existing "
            "`mesh_topology_metrics`. These metrics do not test global self-intersections. "
            "See [summary.json](summary.json) for the full metrics, input data types, source paths, "
            "runtime environment, and timings.", "",
        ])
        if include_surface:
            lines.extend([
                f"Safe RDT selection parameters: `max_distance_ratio={max_distance_ratio}`, "
                f"`min_normal_cos={min_normal_cos}`, `min_quality={min_quality}`.", "",
            ])
        lines.extend([
            "Raw arrays: " + ", ".join(
                f"[{path}]({path})" for path in summary["arrays"].values()) + ".", "",
            "## Timings (seconds)", "", "| Stage | Seconds |", "| --- | ---: |",
        ])
        for name, elapsed in summary["timings_seconds"].items():
            lines.append(f"| {name} | {elapsed} |")
        if "error" in summary:
            lines.extend(["", "Error: `" + summary["error"].replace("`", "'") + "`."])
        (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def measure(stage, faces):
        print(f"[RDT comparison] Measuring {stage}: V={len(vertices):,}, F={len(faces):,}", flush=True)
        begin = time.perf_counter()
        metrics = native(mesh_topology_metrics(vertices, faces))
        elapsed = time.perf_counter() - begin
        summary["timings_seconds"][stage + "_metrics"] = elapsed
        summary["stages"][stage] = {"metrics": metrics, "metrics_seconds": elapsed}
        save_json(stage + "_metrics.json", metrics)
        save_reports()
        print(
            f"[RDT comparison] {stage}: F={metrics['faces']:,}, "
            f"components={metrics['face_components']:,}, "
            f"largest={metrics['largest_component_faces']:,}, "
            f"nonmanifold E/V={metrics['nonmanifold_edges']:,}/{metrics['nonmanifold_vertices']:,}, "
            f"quality median={metrics['triangle_quality_median']}", flush=True)

    active_stage = "save_input"
    try:
        print(f"[RDT comparison] Saving unchanged input to {destination}", flush=True)
        begin = time.perf_counter()
        np.save(destination / "vertices.npy", vertices, allow_pickle=False)
        np.save(destination / "before_faces.npy", before_faces, allow_pickle=False)
        summary["timings_seconds"]["save_input"] = time.perf_counter() - begin
        save_reports()
        active_stage = "before_metrics"
        measure("before", before_faces)

        active_stage = "legacy_rdt"
        import rdel
        summary["provenance"]["legacy_rdt_source"] = str(Path(rdel.__file__).resolve())
        print("[RDT comparison] Running legacy RDT once (orient=False, face_selector=None)", flush=True)
        begin = time.perf_counter()
        summary["legacy_rdt_calls"] += 1
        legacy_faces = rdel.run(vertices, before_faces, verbose=True, orient=False,
                                return_debug=False, face_selector=None)
        summary["timings_seconds"]["legacy_rdt"] = time.perf_counter() - begin
        legacy_faces = np.asarray(legacy_faces)
        np.save(destination / "legacy_faces.npy", legacy_faces, allow_pickle=False)
        summary["arrays"]["legacy_faces"] = "legacy_faces.npy"
        summary["provenance"]["legacy_face_dtype"] = str(legacy_faces.dtype)
        active_stage = "legacy_metrics"
        measure("legacy", legacy_faces)

        if include_surface:
            active_stage = "surface_selection"
            from utils.rdt_surface import select_rdt_surface
            summary["provenance"]["surface_selector_source"] = str(
                Path(inspect.getfile(select_rdt_surface)).resolve())
            print("[RDT comparison] Selecting surface from the exact saved legacy candidates", flush=True)
            begin = time.perf_counter()
            surface_faces, stats = select_rdt_surface(
                vertices, before_faces, legacy_faces, max_distance_ratio=max_distance_ratio,
                min_normal_cos=min_normal_cos, min_quality=min_quality)
            summary["timings_seconds"]["surface_selection"] = time.perf_counter() - begin
            surface_faces = np.asarray(surface_faces)
            np.save(destination / "surface_faces.npy", surface_faces, allow_pickle=False)
            summary["arrays"]["surface_faces"] = "surface_faces.npy"
            summary["provenance"]["surface_face_dtype"] = str(surface_faces.dtype)
            summary["surface_selector_stats"] = native(stats)
            save_json("surface_selection.json", stats)
            active_stage = "surface_metrics"
            measure("surface", surface_faces)

        summary["status"] = "complete"
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save_reports()
        print(f"[RDT comparison] Complete: {destination / 'comparison.csv'}", flush=True)
        return native(summary)
    except Exception as error:
        summary["status"] = "failed"
        summary["failed_stage"] = active_stage
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["failed_utc"] = datetime.now(timezone.utc).isoformat()
        save_reports()
        print(f"[RDT comparison] Failed at {active_stage}: {error}", flush=True)
        raise

