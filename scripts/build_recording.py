"""Regenerate the compact web recording from the official INTERACTION archive."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from trajectory_manifold.loader import load_trajectories
from trajectory_manifold.viewer import _log_recording


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("all", "center", "positive", "negative"),
                        default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trajectories, _ = load_trajectories(
        root=root / "data", num_trajectories=5000, seed=42,
    )
    # Rerun Web is a WebAssembly viewer. Keep the initial recording compact,
    # with a reproducible subset, while retaining the full 5,000 in the loader.
    indices = np.random.default_rng(42).choice(len(trajectories), 500, replace=False)
    sample = trajectories[indices]
    lateral = sample[:, -1, 1]
    masks = {
        "all": np.ones(len(sample), dtype=bool),
        "center": np.abs(lateral) <= 3,
        "positive": lateral > 3,
        "negative": lateral < -3,
    }
    output_dir = root / "web" / "public"
    if args.cohort == "all":
        (output_dir / "cohorts.json").write_text(json.dumps({
            "source_windows": len(trajectories), "sample_windows": len(sample),
            "counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "duration_s": 5, "sample_rate_hz": 10,
            "dataset": "INTERACTION TC_BGR_Intersection_VA",
        }, indent=2) + "\n")
        # Int16 centimetres: 100 kB for 500 × 50 × 2, for the canvas fallback.
        scaled = np.rint(sample * 100)
        if np.max(np.abs(scaled)) > 32767:
            raise ValueError("Trajectory exceeds Int16 centimetre range")
        (output_dir / "trajectories.bin").write_bytes(
            scaled.astype("<i2").tobytes()
        )
    subset = sample[masks[args.cohort]]
    destination = output_dir / f"{args.cohort}.rrd"
    rr.init(f"ego_trajectory_{args.cohort}")
    rr.save(str(destination))
    _log_recording(rr, rrb, subset, sample_rate=10, live=False)
    rr.get_global_data_recording().flush()
    print(f"Saved {len(subset)} {args.cohort} trajectories to {destination}")


if __name__ == "__main__":
    main()
