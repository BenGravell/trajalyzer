"""Streaming exemplar discovery and nearest-exemplar assignment.

Requires scikit-learn in addition to NumPy. Call ``fit_exemplars`` with a
repeatable batch factory; it never holds the full training matrix in memory.
Supply batches from *training tracks/scenes only* to avoid overlap leakage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .kinematics import infer_kinematics


@dataclass
class ExemplarModel:
    trajectories: np.ndarray  # (K,T,2) real observed trajectories
    features: np.ndarray  # (K,F) same features as test-time assignment
    cluster_sizes: np.ndarray  # (K,) training membership counts
    sample_rate: float
    min_speed: float


def trajectory_features(
    positions: np.ndarray, sample_rate: float = 10.0, min_speed: float = 1.0
) -> np.ndarray:
    """Compact physical-unit features: position, speed, and inferred actions.

    Time is downsampled by two for clustering only. Fixed physical scales keep
    the map stable across batches and test-time data. All samples must remain
    above ``min_speed``; invalid curvature is never silently imputed.
    """
    batch = np.asarray(positions, dtype=np.float32)
    states, actions, valid = infer_kinematics(batch, sample_rate=sample_rate, min_speed=min_speed)
    if not valid.all():
        raise ValueError("Near-zero-speed samples cannot be clustered; filter them first")
    # Clip extreme derivative estimates so a noisy point cannot dominate a
    # geometric distance. These thresholds and feature scales are explicit.
    actions = np.clip(actions, [-5.0, -1.0], [5.0, 1.0])
    features = np.concatenate(
        (
            states[:, ::2, 0] / 20,
            states[:, ::2, 1] / 10,
            states[:, ::2, 3] / 8,
            actions[:, ::2, 0] / 3,
            actions[:, ::2, 1] / 0.2,
        ),
        axis=1,
    )
    return features.astype(np.float32)


def fit_exemplars(
    batch_factory: Callable[[], Iterable[np.ndarray]],
    *,
    n_exemplars: int = 64,
    sample_rate: float = 10.0,
    min_speed: float = 1.0,
    passes: int = 3,
    batch_size: int = 2048,
    seed: int = 42,
) -> ExemplarModel:
    """Fit MiniBatchKMeans and select a real trajectory nearest each center.

    ``batch_factory`` must yield the same training-only batches on each pass.
    Memory scales with batch size, K, and trajectory length, not dataset size.
    A final pass chooses the closest assigned member per cluster. Repeated
    passes improve clustering; no order-sensitive random sampling is needed.
    """
    if n_exemplars < 2 or passes < 1 or batch_size < n_exemplars:
        raise ValueError("Require K >= 2, passes >= 1, and batch_size >= K")
    model = MiniBatchKMeans(
        n_clusters=n_exemplars, batch_size=batch_size, random_state=seed, n_init=3
    )
    for _ in range(passes):
        pending = []
        for batch in batch_factory():
            features = trajectory_features(batch, sample_rate, min_speed)
            if not hasattr(model, "cluster_centers_") and len(features) < n_exemplars:
                pending.append(features)
                if sum(len(x) for x in pending) < n_exemplars:
                    continue
                features = np.concatenate(pending)
                pending.clear()
            model.partial_fit(features)
        if pending:
            raise ValueError("Training set has fewer samples than K")
    best = np.full(n_exemplars, np.inf)
    selected = [None] * n_exemplars
    selected_features = [None] * n_exemplars
    sizes = np.zeros(n_exemplars, dtype=np.int64)
    for batch in batch_factory():
        features = trajectory_features(batch, sample_rate, min_speed)
        ids, distances = pairwise_distances_argmin_min(features, model.cluster_centers_)
        sizes += np.bincount(ids, minlength=n_exemplars)
        for cluster in np.unique(ids):
            members = np.flatnonzero(ids == cluster)
            nearest = members[np.argmin(distances[members])]
            if distances[nearest] < best[cluster]:
                best[cluster] = distances[nearest]
                selected[cluster] = np.asarray(batch[nearest], dtype=np.float32).copy()
                selected_features[cluster] = features[nearest].copy()
    if any(item is None for item in selected):
        raise ValueError("Empty cluster; reduce n_exemplars or increase training data")
    return ExemplarModel(
        np.stack(selected), np.stack(selected_features), sizes, sample_rate, min_speed
    )


def assign_exemplars(model: ExemplarModel, positions: np.ndarray, batch_size: int = 2048):
    """Assign new trajectories to nearest *observed exemplar*, in batches.

    Returns (integer exemplar ID, Euclidean feature distance). Unlike
    ``MiniBatchKMeans.predict``, this compares against real chosen trajectories.
    """
    ids, distances = [], []
    for start in range(0, len(positions), batch_size):
        features = trajectory_features(
            positions[start : start + batch_size], model.sample_rate, model.min_speed
        )
        assigned, distance = pairwise_distances_argmin_min(features, model.features)
        ids.append(assigned.astype(np.int32))
        distances.append(distance.astype(np.float32))
    return np.concatenate(ids), np.concatenate(distances)
