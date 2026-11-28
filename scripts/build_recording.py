"""Regenerate the compact web recording from the official INTERACTION archive."""

import argparse
import colorsys
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from trajectory_manifold.clustering import assign_exemplars, fit_exemplars
from trajectory_manifold.kinematics import infer_kinematics
from trajectory_manifold.loader import load_trajectories
from trajectory_manifold.viewer import _log_recording

MIN_SPEED_MPS = 1.0


def cohort_masks(sample):
    lateral = sample[:, -1, 1]
    return {
        "all": np.ones(len(sample), dtype=bool),
        "center": np.abs(lateral) <= 3,
        "positive": lateral > 3,
        "negative": lateral < -3,
    }


def build_focused_recordings(output_dir, sample, cluster_ids, palette):
    """Pack complete RRD streams for the subsets the web viewer can display."""
    index = {}
    spans = {}
    with gzip.GzipFile(output_dir / "focused-recordings.bin", mode="wb", mtime=0) as output:

        def write(key, rows):
            if rows not in spans:
                # Rerun instance IDs follow the order of the browser's filtered indices.
                indices = list(rows)
                with rr.RecordingStream("ego_trajectory_focus") as recording:
                    stream = rr.binary_stream(recording=recording)
                    _log_recording(
                        rr,
                        rrb,
                        sample[indices],
                        sample_rate=10,
                        live=False,
                        cluster_ids=cluster_ids[indices],
                        palette=palette,
                        send_blueprint=False,
                    )
                    data = stream.read()
                spans[rows] = [output.tell(), len(data)]
                output.write(data)
            index[key] = spans[rows]

        for i in range(len(sample)):
            write(f"path-{i}", (i,))
        for cohort, mask in cohort_masks(sample).items():
            for cluster in np.unique(cluster_ids[mask]):
                rows = tuple(np.flatnonzero(mask & (cluster_ids == cluster)))
                write(f"{cohort}-exemplar-{cluster}", rows)

    (output_dir / "focused-recordings.json").write_text(
        json.dumps(index, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cohort", choices=("all", "center", "positive", "negative"))
    mode.add_argument("--focused-only", action="store_true", help="reuse the published web sample")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data"
    if args.focused_only:
        sample = np.fromfile(output_dir / "trajectories.bin", dtype="<i2").reshape(-1, 50, 2) / 100
        cluster_ids = np.fromfile(output_dir / "clusters.bin", dtype="u1")
        info = json.loads((output_dir / "clusters.json").read_text(encoding="utf-8"))
        palette = [list(bytes.fromhex(color.removeprefix("#"))) for color in info["palette"]]
        build_focused_recordings(output_dir, sample, cluster_ids, palette)
        return

    trajectories, metadata = load_trajectories(
        root=root / "data",
        num_trajectories=5000,
        seed=42,
    )
    # Require sustained motion over the whole window before drawing or fitting
    # actions. This avoids unidentifiable heading/curvature at near-zero speed.
    states, _, _ = infer_kinematics(trajectories, sample_rate=10, min_speed=MIN_SPEED_MPS)
    eligible = np.flatnonzero(np.min(states[..., 3], axis=1) >= MIN_SPEED_MPS)
    if len(eligible) < 500:
        raise ValueError(f"Only {len(eligible)} continuously moving windows")
    # Keep a reproducible 500-window subset for the WebAssembly viewer.
    indices = np.random.default_rng(42).choice(eligible, 500, replace=False)
    sample = trajectories[indices]
    # Hold out entire tracks for test-time exemplar assignment. Windows from
    # a track never appear on both sides of the split.
    heldout = np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"{item['source_file']}:{item['track_id']}".encode()).digest()[:4],
                "little",
            )
            % 5
            == 0
            for item in metadata
        ],
        dtype=bool,
    )
    training = eligible[~heldout[eligible]]

    def training_batches():
        for start in range(0, len(training), 2048):
            yield trajectories[training[start : start + 2048]]

    model = fit_exemplars(training_batches, n_exemplars=64, min_speed=MIN_SPEED_MPS, passes=3)
    cluster_ids, distances = assign_exemplars(model, sample)
    palette = [
        tuple(
            round(255 * value) for value in colorsys.hls_to_rgb((i * 0.61803398875) % 1, 0.64, 0.7)
        )
        for i in range(len(model.trajectories))
    ]
    # Use the same cohort boundary as the browser, which reads centimetre positions.
    scaled = np.rint(sample * 100)
    if np.max(np.abs(scaled)) > 32767:
        raise ValueError("Trajectory exceeds Int16 centimetre range")
    masks = cohort_masks(scaled / 100)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, actions, valid = infer_kinematics(sample, sample_rate=10, min_speed=MIN_SPEED_MPS)
    (output_dir / "actions.bin").write_bytes(actions.astype("<f4").tobytes())
    (output_dir / "clusters.bin").write_bytes(cluster_ids.astype("u1").tobytes())
    (output_dir / "distances.bin").write_bytes(distances.astype("<f4").tobytes())
    (output_dir / "heldout.bin").write_bytes(heldout[indices].astype("u1").tobytes())
    (output_dir / "exemplars.bin").write_bytes(
        np.rint(model.trajectories * 100).astype("<i2").tobytes()
    )
    (output_dir / "clusters.json").write_text(
        json.dumps(
            {
                "count": len(model.trajectories),
                "training_windows": len(training),
                "heldout_display_windows": int(heldout[indices].sum()),
                "palette": ["#{:02x}{:02x}{:02x}".format(*rgb) for rgb in palette],
                "cluster_sizes": model.cluster_sizes.tolist(),
                "feature": "positions, speed, acceleration, curvature at 5 Hz; fixed physical scales",
                "assignment": "nearest observed training exemplar in feature space",
                "split": "track ID hash, 20% held out",
            },
            indent=2,
        )
        + "\n"
    )
    cache_dir = root / "data" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / "exemplar_model.npz",
        trajectories=model.trajectories,
        features=model.features,
        cluster_sizes=model.cluster_sizes,
        sample_rate=model.sample_rate,
        min_speed=model.min_speed,
    )
    (output_dir / "cohorts.json").write_text(
        json.dumps(
            {
                "source_windows": len(trajectories),
                "eligible_windows": len(eligible),
                "sample_windows": len(sample),
                "counts": {name: int(mask.sum()) for name, mask in masks.items()},
                "duration_s": 5,
                "sample_rate_hz": 10,
                "dataset": "INTERACTION TC_BGR_Intersection_VA",
                "action_inference": {
                    "window": 7,
                    "degree": 3,
                    "min_speed_mps": MIN_SPEED_MPS,
                    "valid_fraction": float(valid.mean()),
                },
            },
            indent=2,
        )
        + "\n"
    )
    # Int16 centimetres: 100 kB for 500 × 50 × 2, for the canvas fallback.
    (output_dir / "trajectories.bin").write_bytes(scaled.astype("<i2").tobytes())
    build_focused_recordings(output_dir, scaled / 100, cluster_ids, palette)
    for cohort in [args.cohort] if args.cohort else masks:
        subset = sample[masks[cohort]]
        destination = output_dir / f"{cohort}.rrd"
        rr.init(f"ego_trajectory_{cohort}")
        rr.save(str(destination))
        _log_recording(
            rr,
            rrb,
            subset,
            sample_rate=10,
            live=False,
            cluster_ids=cluster_ids[masks[cohort]],
            palette=palette,
        )
        rr.get_global_data_recording().flush()
        print(f"Saved {len(subset)} {cohort} trajectories to {destination}")


if __name__ == "__main__":
    main()
