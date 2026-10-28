# Project adaptations maintained by zhtang2004@gmail.com.
"""Schedule and visibility measurement for gradual training-time simplification."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class VisibilityPruningSchedule:
    mode: str = "legacy"
    start: int = 21000
    end: int = 27000
    interval: int = 1000
    threshold_start: float = 0.05
    threshold_end: float = 0.5
    max_face_ratio: float = 0.02

    def __post_init__(self):
        if self.mode not in ("legacy", "progressive_collapse", "none"):
            raise ValueError(f"Unknown visibility_pruning: {self.mode}")
        if not self.enabled:
            return
        if self.start <= 0 or self.end <= self.start or self.interval <= 0:
            raise ValueError("Visibility pruning needs 0 < start < end and interval > 0")
        if (self.end - self.start) % self.interval:
            raise ValueError("Visibility end must lie on the start + k * interval schedule")
        if not (0 <= self.threshold_start <= self.threshold_end <= 1):
            raise ValueError("Visibility thresholds must satisfy 0 <= start <= end <= 1")
        if not math.isfinite(self.max_face_ratio) or not 0 < self.max_face_ratio < 1:
            raise ValueError("Visibility max_face_ratio must lie in (0, 1)")

    @property
    def enabled(self):
        return self.mode == "progressive_collapse"

    @classmethod
    def from_options(cls, opt):
        return cls(
            mode=opt.visibility_pruning,
            start=opt.visibility_prune_start_iter,
            end=opt.visibility_prune_end_iter,
            interval=opt.visibility_prune_interval,
            threshold_start=opt.visibility_prune_threshold_start,
            threshold_end=opt.visibility_prune_threshold_end,
            max_face_ratio=opt.visibility_prune_max_face_ratio,
        )

    def validate_training(self, opt, rdt_iteration, texture_start=None):
        if not self.enabled:
            return
        if opt.rdt_backend != "surface":
            raise ValueError("Progressive visibility collapse requires --rdt_backend surface")
        if self.start <= rdt_iteration + 1:
            raise ValueError("Visibility collapse must start after the RDT event")
        if self.end >= opt.iterations:
            raise ValueError("Leave optimization iterations after the last visibility collapse")
        if opt.enable_heartbeat:
            hb_end = opt.heartbeat_end_iter or opt.iterations
            if not opt.heartbeat_safety or hb_end >= self.start:
                raise ValueError("Enable DTE safety and finish DTE before visibility collapse")
        if texture_start is not None and self.end >= texture_start:
            raise ValueError("Finish visibility collapse before initializing face textures")

    def threshold_at(self, iteration):
        if (not self.enabled or not self.start <= iteration <= self.end
                or (iteration - self.start) % self.interval):
            return None
        fraction = (iteration - self.start) / (self.end - self.start)
        return self.threshold_start + fraction * (self.threshold_end - self.threshold_start)


@torch.no_grad()
def collect_face_visibility(model, cameras, render_fn, pipe, background):
    """Fresh maximum blending over ALL supplied training cameras, on fixed geometry.

    No stale DTE statistics or random training-camera window is reused. An empty
    or invalid sweep must never turn into an all-zero deletion mask.
    """
    count = len(model._triangle_indices)
    scores = torch.zeros(count, dtype=torch.float32, device=model.vertices.device)
    n_views = 0
    for camera in cameras:
        result = render_fn(camera, model, pipe, background)
        current = result["max_blending"].detach().reshape(-1)
        if current.numel() != count or not torch.isfinite(current).all():
            raise RuntimeError("Invalid per-face visibility returned by renderer")
        torch.maximum(scores, current.to(scores), out=scores)
        n_views += 1
        del result, current
    if not n_views:
        raise ValueError("Visibility measurement requires training cameras")
    return scores


def require_surface_topology(metrics):
    """Fail at the event boundary if the manifold input assumption is violated."""
    fields = ("nonmanifold_edges", "nonmanifold_vertices", "duplicate_faces",
              "winding_conflict_edges", "invalid_index_faces", "repeated_index_faces",
              "zero_area_faces", "nonfinite_faces", "nonfinite_vertices")
    bad = {key: metrics[key] for key in fields if metrics[key]}
    if bad:
        raise RuntimeError(f"Visibility collapse requires valid surface topology: {bad}")


@torch.no_grad()
def finalize_visibility_pruning(model, cameras, render_fn, pipe, background, mode,
                                *, final_visibility_threshold=0.5):
    """Legacy prunes faces; progressive/none only compact unused vertex rows.

    Legacy keeps only faces whose fresh visibility is strictly above the threshold.
    In progressive and none modes no visibility sweep, face deletion or edge
    collapse can occur here. Used positions and surviving faces remain identical.
    """
    if mode not in ("legacy", "progressive_collapse", "none"):
        raise ValueError(f"Unknown visibility_pruning: {mode}")
    if (not math.isfinite(final_visibility_threshold)
            or not 0 <= final_visibility_threshold <= 1):
        raise ValueError("final_visibility_threshold must be finite and lie in [0, 1]")
    before = len(model._triangle_indices)
    if mode == "legacy":
        model.importance_score = collect_face_visibility(model, cameras, render_fn, pipe, background)
        model.prune_triangles(model.importance_score > final_visibility_threshold)
    used = torch.zeros(len(model.vertices), dtype=torch.bool, device=model.vertices.device)
    used[model._triangle_indices.long().flatten()] = True
    if not bool(used.all()):
        model._prune_vertices(used)
    return {"mode": mode, "faces_before": before, "faces_after": len(model._triangle_indices)}
