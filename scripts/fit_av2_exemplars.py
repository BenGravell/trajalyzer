"""Fit observed exemplars over every eligible row of a native AV2 feature store.

The source features remain on disk as a memmap. MiniBatchKMeans makes bounded
passes, then one complete pass selects the nearest actual trajectory to each
center. This is a full-store fit, even when the store contains millions of rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min


def fit_store(store: Path, output: Path, *, clusters: int, batch_size: int, passes: int, seed: int):
    info = json.loads((store / "store.json").read_text())
    count, frames, dim = info["count"], info["frames"], info["feature_dim"]
    if count < clusters or clusters < 2 or batch_size < clusters or passes < 1:
        raise ValueError("Require at least K windows, K >= 2, batch size >= K, and passes >= 1")
    if (store / "features.f32").stat().st_size != count * dim * 4:
        raise ValueError("Feature store length mismatch")
    if (store / "positions.f32").stat().st_size != count * frames * 2 * 4:
        raise ValueError("Position store length mismatch")
    features = np.memmap(store / "features.f32", dtype="<f4", mode="r", shape=(count, dim))
    positions = np.memmap(store / "positions.f32", dtype="<f4", mode="r", shape=(count, frames, 2))
    model = MiniBatchKMeans(n_clusters=clusters, batch_size=batch_size, random_state=seed, n_init=3)
    for pass_id in range(passes):
        for start in range(0, count, batch_size):
            model.partial_fit(features[start : start + batch_size])
        print(f"Feature pass {pass_id + 1}/{passes} complete", flush=True)
    best = np.full(clusters, np.inf)
    picked = np.full(clusters, -1, dtype=np.int64)
    sizes = np.zeros(clusters, dtype=np.int64)
    for start in range(0, count, batch_size):
        ids, distances = pairwise_distances_argmin_min(
            features[start : start + batch_size], model.cluster_centers_
        )
        sizes += np.bincount(ids, minlength=clusters)
        for cluster in np.unique(ids):
            members = np.flatnonzero(ids == cluster)
            nearest = members[np.argmin(distances[members])]
            if distances[nearest] < best[cluster]:
                best[cluster] = distances[nearest]
                picked[cluster] = start + nearest
    if np.any(picked < 0):
        raise ValueError("Empty cluster; reduce K or increase the corpus")
    observed = np.asarray(positions[picked], dtype=np.float32)
    selected_features = np.asarray(features[picked], dtype=np.float32)
    scaled = np.rint(observed / 0.05)
    if np.max(np.abs(scaled)) > 32767:
        raise ValueError("Observed exemplar exceeds int16 display range")
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(
        json.dumps({"features": selected_features.tolist()}, separators=(",", ":")) + "\n"
    )
    (output / "exemplars.bin").write_bytes(scaled.astype("<i2").tobytes())
    (output / "model-meta.json").write_text(
        json.dumps(
            {
                "dataset": "Argoverse 2 Motion Forecasting",
                "training_windows": count,
                "count": clusters,
                "frames": frames,
                "feature_dim": dim,
                "selection": "nearest observed trajectory to each MiniBatchKMeans center",
                "cluster_sizes": sizes.tolist(),
                "position_scale_m": 0.05,
                "seed": seed,
                "passes": passes,
            },
            indent=2,
        )
        + "\n"
    )
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    picked = fit_store(
        args.store,
        args.output,
        clusters=args.clusters,
        batch_size=args.batch_size,
        passes=args.passes,
        seed=args.seed,
    )
    print(f"Selected {len(picked)} observed exemplars from {args.store}")


if __name__ == "__main__":
    main()
