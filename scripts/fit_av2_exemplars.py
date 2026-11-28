"""Fit observed exemplars over every eligible row of a native AV2 feature store.

The source features remain on disk as a memmap. MiniBatchKMeans, FAISS k-means,
or FAISS e-means fits the centers, then one complete pass selects the nearest
actual trajectory to each center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import assert_all_finite


def _estimate_cluster_count(
    features: np.ndarray,
    *,
    sample_size: int | None,
    min_cluster_size: int | None,
    min_samples: int,
    seed: int,
):
    sample_size = len(features) if sample_size is None else min(sample_size, len(features))
    uses_full_store = sample_size == len(features)
    if min_cluster_size is None:
        min_cluster_size = max(5, (sample_size + 199) // 200)
    if not 2 <= min_cluster_size <= sample_size or not 1 <= min_samples <= sample_size:
        raise ValueError(
            "HDBSCAN requires 2 <= minimum cluster size <= sample size and "
            "1 <= minimum samples <= sample size"
        )
    sample = features
    if not uses_full_store:
        rows = np.random.default_rng(seed).choice(len(features), size=sample_size, replace=False)
        rows.sort()
        sample = features[rows]
    if not np.isfinite(sample).all():
        raise ValueError("HDBSCAN input contains non-finite trajectory features")
    print(f"Running HDBSCAN on {sample_size:,} of {len(features):,} windows", flush=True)
    start = perf_counter()
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method="eom"
    ).fit_predict(sample)
    print(f"HDBSCAN fit time: {perf_counter() - start:.3f}s", flush=True)
    sizes = np.bincount(labels[labels >= 0])
    noise = sample_size - int(sizes.sum())
    count = len(sizes)
    if count < 2:
        raise ValueError(
            f"HDBSCAN found {count} clusters in {sample_size} windows "
            f"({noise} noise). Adjust the density settings or pass a fixed --clusters count."
        )
    return count, {
        "method": "hdbscan",
        "cluster_selection_method": "eom",
        "uses_full_store": uses_full_store,
        "sample_windows": sample_size,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "sample_cluster_sizes": sizes.tolist(),
        "noise_windows": noise,
        "noise_fraction": noise / sample_size,
    }


def fit_store(
    store: Path,
    output: Path,
    *,
    clusters: int | Literal["auto"],
    batch_size: int,
    seed: int,
    passes: int | None = None,
    density_sample_size: int | None = None,
    min_cluster_size: int | None = None,
    min_samples: int = 10,
    backend: Literal["minibatch", "faiss-kmeans", "emeans"] = "faiss-kmeans",
):
    if backend in ("faiss-kmeans", "emeans"):
        try:
            import faiss
        except ImportError as error:
            raise ValueError(
                "FAISS backends require faiss-cpu. Install the cluster extra."
            ) from error
    elif backend != "minibatch":
        raise ValueError(f"Unknown exemplar fitting backend: {backend}")
    if backend == "emeans":
        try:
            from faiss.contrib.e_means import NumpyEmeans
        except ImportError as error:
            raise ValueError("The emeans backend requires faiss-cpu>=1.16.0") from error

    info = json.loads((store / "store.json").read_text())
    count, frames, dim = info["count"], info["frames"], info["feature_dim"]
    if passes is None:
        passes = 10 if backend == "faiss-kmeans" else 2
    max_clusters = min(count, batch_size) if backend == "minibatch" else count
    if batch_size < 1 or passes < 1:
        raise ValueError("Require batch size >= 1 and passes >= 1")
    if max_clusters < 2 or (clusters != "auto" and not 2 <= clusters <= max_clusters):
        raise ValueError(f"Require 2 <= K <= {max_clusters} for this store and backend")
    if (store / "features.f32").stat().st_size != count * dim * 4:
        raise ValueError("Feature store length mismatch")
    if (store / "positions.f32").stat().st_size != count * frames * 2 * 4:
        raise ValueError("Position store length mismatch")
    features = np.memmap(store / "features.f32", dtype="<f4", mode="r", shape=(count, dim))
    positions = np.memmap(store / "positions.f32", dtype="<f4", mode="r", shape=(count, frames, 2))
    count_selection = {}
    if clusters == "auto":
        clusters, diagnostics = _estimate_cluster_count(
            features,
            sample_size=density_sample_size,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            seed=seed,
        )
        if clusters > max_clusters:
            raise ValueError(
                f"HDBSCAN found {clusters} clusters. Increase --batch-size to at least {clusters}."
            )
        count_selection["count_selection"] = diagnostics
        print(
            f"HDBSCAN estimated {clusters} exemplars from {diagnostics['sample_windows']} windows "
            f"({diagnostics['noise_fraction']:.1%} noise)",
            flush=True,
        )
    fit_start = perf_counter()
    index = None
    if backend in ("faiss-kmeans", "emeans"):
        assert_all_finite(features)
        if backend == "faiss-kmeans":
            # FAISS otherwise caps training at 256 vectors per centroid by default.
            model = faiss.Kmeans(
                dim,
                clusters,
                niter=passes,
                seed=seed,
                max_points_per_centroid=(count + clusters - 1) // clusters,
            )
            model.train(features)
            centers = model.centroids
            center_method = "FAISS k-means"
        else:
            model = NumpyEmeans(dim, clusters, nepochs=passes, seed=seed, capture_assignments=False)
            centers, _ = model.train(features)
            center_method = "FAISS e-means"
        assert_all_finite(centers)
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
    else:
        model = MiniBatchKMeans(
            n_clusters=clusters, batch_size=batch_size, random_state=seed, n_init=3
        )
        for pass_id in range(passes):
            for start in range(0, count, batch_size):
                model.partial_fit(features[start : start + batch_size])
            print(f"Feature pass {pass_id + 1}/{passes} complete", flush=True)
        centers = model.cluster_centers_
        center_method = "MiniBatchKMeans"
    print(f"{center_method} fit time: {perf_counter() - fit_start:.3f}s", flush=True)
    selection_start = perf_counter()
    best = np.full(clusters, np.inf)
    picked = np.full(clusters, -1, dtype=np.int64)
    sizes = np.zeros(clusters, dtype=np.int64)
    for start in range(0, count, batch_size):
        batch = features[start : start + batch_size]
        if index is None:
            ids, distances = pairwise_distances_argmin_min(batch, centers)
        else:
            # Squared L2 preserves the ordering used to select an observed member.
            squared, nearest_ids = index.search(batch, 1)
            ids, distances = nearest_ids[:, 0], squared[:, 0]
        sizes += np.bincount(ids, minlength=clusters)
        for cluster in np.unique(ids):
            members = np.flatnonzero(ids == cluster)
            nearest = members[np.argmin(distances[members])]
            if distances[nearest] < best[cluster]:
                best[cluster] = distances[nearest]
                picked[cluster] = start + nearest
    if np.any(picked < 0):
        raise ValueError("Empty cluster; reduce K or increase the corpus")
    print(f"Observed exemplar selection time: {perf_counter() - selection_start:.3f}s", flush=True)
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
                "selection": f"nearest observed trajectory to each {center_method} center",
                "backend": backend,
                "cluster_sizes": sizes.tolist(),
                "position_scale_m": 0.05,
                "seed": seed,
                "passes": passes,
                **count_selection,
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
    parser.add_argument("--clusters", default="128", help="Fixed count (default: 128) or 'auto'")
    parser.add_argument(
        "--backend",
        choices=("minibatch", "faiss-kmeans", "emeans"),
        default="faiss-kmeans",
        help="Exemplar fitter (default: faiss-kmeans)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Rows per selection batch and MiniBatchKMeans training batch (default: 4096)",
    )
    parser.add_argument(
        "--passes", type=int, help="Training passes (default: 10 for faiss-kmeans, 2 otherwise)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--density-sample-size",
        type=int,
        help="Limit HDBSCAN to this many sampled windows (default: use all windows)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        help="HDBSCAN minimum cluster size (default: 0.5%% of its input, rounded up, at least 5)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=10, help="HDBSCAN neighborhood size (default: 10)"
    )
    args = parser.parse_args()
    try:
        clusters = "auto" if args.clusters == "auto" else int(args.clusters)
    except ValueError:
        parser.error("--clusters must be an integer or 'auto'")
    try:
        picked = fit_store(
            args.store,
            args.output,
            clusters=clusters,
            batch_size=args.batch_size,
            passes=args.passes,
            seed=args.seed,
            density_sample_size=args.density_sample_size,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            backend=args.backend,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"Selected {len(picked)} observed exemplars from {args.store}")


if __name__ == "__main__":
    main()
