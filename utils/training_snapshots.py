"""Read-only geometry snapshots for locating training-stage regressions.

These contain model parameters and raw geometry, not optimizer/resume state.
No cleaning, rendering, topology mutation, or random sampling occurs here.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def save_training_snapshot(scene, iteration, stage, checkpoint_path=None):
    stages = {"before_rdt", "after_rdt", "iteration_end", "training_complete"}
    if stage not in stages:
        raise ValueError(f"Unknown snapshot stage: {stage}")
    if checkpoint_path is None:
        key = iteration if stage in {"iteration_end", "training_complete"} else f"{iteration}_{stage}"
        checkpoint_path = scene.save(key)
    checkpoint_path = Path(checkpoint_path)
    model = scene.triangles
    xyz = model.vertices.detach().cpu().numpy()
    faces = model._triangle_indices.detach().cpu().numpy()
    vertices_ply = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    for axis, name in enumerate(("x", "y", "z")):
        vertices_ply[name] = xyz[:, axis]
    faces_ply = np.empty(len(faces), dtype=[("vertex_indices", "<i4", (3,))])
    faces_ply["vertex_indices"] = faces
    mesh_path = checkpoint_path.parent / "mesh_raw.ply"
    PlyData([PlyElement.describe(vertices_ply, "vertex"),
             PlyElement.describe(faces_ply, "face")], text=False).write(str(mesh_path))
    record = {
        "iteration": int(iteration), "stage": stage,
        "position": "before_iteration_update" if stage in {"before_rdt", "after_rdt"} else "after_iteration",
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path), "raw_mesh": str(mesh_path),
        "vertices": len(xyz), "faces": len(faces),
        "contains_optimizer_state": False,
    }
    (checkpoint_path.parent / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
    with (Path(scene.model_path) / "training_snapshots.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[Snapshot] {stage}@{iteration}: V={len(xyz)} F={len(faces)} -> {mesh_path}", flush=True)
    return record
