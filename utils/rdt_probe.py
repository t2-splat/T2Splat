"""Opt-in diagnostics at the live training RDT event; never alters parameters."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def evaluate_views(scene, model, pipe, background, render, tag, iteration, faces=None):
    root = Path(scene.model_path) / "rdt_probe"
    root.mkdir(parents=True, exist_ok=True)
    original = model._triangle_indices
    try:
        if faces is not None:
            model._triangle_indices = torch.as_tensor(faces, dtype=torch.int32, device=model.vertices.device).contiguous()
        values = []
        cameras = list(scene.getTestCameras())
        for i, camera in enumerate(cameras):
            prediction = render(camera, model, pipe, background)["render"].clamp(0, 1)
            target = camera.original_image.to(prediction.device)
            if getattr(camera, "gt_alpha_mask", None) is not None:
                target = target + background[:, None, None] * (1 - camera.gt_alpha_mask.to(prediction.device))
            mse = (prediction.double() - target.double()).square().mean().item()
            values.append(dict(camera=camera.image_name, psnr=float(-10 * np.log10(max(mse, 1e-30)))))
            if i in (0, len(cameras)//3, 2*len(cameras)//3):
                for name, im in ((tag, prediction), ("gt", target)):
                    array = (im.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    Image.fromarray(array).save(root / f"{name}_{i:02d}.png")
        report = dict(tag=tag, iteration=iteration, views=len(values),
                      psnr=float(np.mean([x['psnr'] for x in values])) if values else None,
                      vertices=len(model.vertices), faces=len(model._triangle_indices),
                      opacity_floor=float(model.opacity_floor), scaling=float(model.scaling), per_view=values)
        (root / f"{tag}.json").write_text(json.dumps(report, indent=2))
        print(f"[RDT probe] {tag}: views={len(values)}, PSNR={report['psnr']}", flush=True)
        return report
    finally:
        model._triangle_indices = original


def surface_coverage(vertices, source, selected, sample_count=100000, seed=71):
    """Area-weighted source samples -> selected mesh; no ground-truth geometry."""
    import open3d as o3d
    from utils.rdt_surface import _geometry
    vertices = np.asarray(vertices, np.float32)
    areas, scales = [], []
    selected_area = 0.
    for begin in range(0, len(source), 200000):
        _, _, length, _, a = _geometry(vertices, source[begin:begin+200000])
        areas.append(a * .5); scales.append(length)
    areas, scales = np.concatenate(areas), np.concatenate(scales)
    for begin in range(0, len(selected), 200000):
        selected_area += .5 * _geometry(vertices, selected[begin:begin+200000])[-1].sum()
    if areas.sum() <= 0 or not len(selected):
        raise ValueError("coverage requires nonempty surfaces")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(source), sample_count, p=areas / areas.sum())
    uv = rng.random((sample_count, 2)); uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    p = vertices[source[ids]]
    samples = (p[:,0] + uv[:,0,None]*(p[:,1]-p[:,0]) + uv[:,1,None]*(p[:,2]-p[:,0])).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices)), o3d.core.Tensor(np.ascontiguousarray(selected, np.uint32)))
    distances = scene.compute_distance(o3d.core.Tensor(samples), nthreads=8).numpy()
    ratio = distances / np.maximum(scales[ids], 1e-20)
    return dict(samples=sample_count, seed=seed, input_area=float(areas.sum()), selected_area=float(selected_area),
                selected_to_input_area=float(selected_area / areas.sum()),
                distance_median=float(np.median(distances)), distance_p95=float(np.quantile(distances, .95)),
                normalized_distance_median=float(np.median(ratio)), normalized_distance_p95=float(np.quantile(ratio, .95)),
                coverage_within_01_local_edge=float((ratio <= .1).mean()),
                coverage_within_025_local_edge=float((ratio <= .25).mean()))


def save_rdt_event(scene, model, debug, iteration):
    root = Path(scene.model_path) / "rdt_probe"
    root.mkdir(parents=True, exist_ok=True)
    vertices = model.vertices.detach().cpu().numpy()
    for tag, faces in (("before", debug["input_faces"]), ("legacy", debug["selected_faces"]), ("surface", debug["output_faces"])):
        np.save(root / f"{tag}_faces.npy", np.asarray(faces, np.int32))
    np.save(root / "vertices.npy", vertices)
    (root / "selection.json").write_text(json.dumps(debug.get("selector_stats", {}), indent=2))
    coverage = surface_coverage(vertices, debug["input_faces"], debug["output_faces"])
    (root / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(f"[RDT probe] coverage: {coverage}", flush=True)
