"""Exact ordered radius suppression without retaining every point's neighbors."""
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors


def greedy_radius_mask(points, radius, workers=8, batch_size=8192, progress=None):
    """Match the original DTU greedy mask, preserving input order and radius.

    A point suppressed by an earlier retained point never becomes eligible again.
    Such queries can be skipped; candidates suppressed within the current batch
    must still be checked immediately before applying their neighborhood.
    """
    if batch_size < 1 or workers < 1:
        raise ValueError('batch_size and workers must be positive')
    count = len(points)
    mask = np.ones(count, dtype=np.bool_)
    if count == 0:
        return mask
    engine = NearestNeighbors(n_neighbors=1, radius=radius, algorithm='kd_tree', n_jobs=workers)
    engine.fit(points)
    last_report = time.monotonic()
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        candidates = np.flatnonzero(mask[start:end]) + start
        if len(candidates):
            neighbors = engine.radius_neighbors(points[candidates], radius=radius, return_distance=False)
            for current, nearby in zip(candidates, neighbors):
                if mask[current]:
                    mask[nearby] = False
                    mask[current] = True
            del neighbors
        if progress is not None and (end == count or time.monotonic() - last_report >= 10):
            progress(end, count)
            last_report = time.monotonic()
    return mask
