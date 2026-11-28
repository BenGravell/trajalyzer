"""Build a bounded static corpus from a repeatable stream of trajectory batches.

The checked-in demo uses existing inferred actions and assignments; external
adapters can feed CorpusWriter.add from millions of windows without retention.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_manifold.corpus import CorpusWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/interaction-corpus"))
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "data" / "published"
    paths = np.frombuffer((source / "trajectories.bin").read_bytes(), dtype="<i2")
    paths = paths.reshape(-1, 50, 2).astype(np.float32) / 100
    actions = np.frombuffer((source / "actions.bin").read_bytes(), dtype="<f4").reshape(-1, 50, 2)
    clusters = np.frombuffer((source / "clusters.bin").read_bytes(), dtype="u1")
    distances = np.frombuffer((source / "distances.bin").read_bytes(), dtype="<f4")
    writer = CorpusWriter(args.output, parquet=args.parquet)
    for start in range(0, len(paths), 128):
        end = start + 128
        writer.add(paths[start:end], actions[start:end], clusters[start:end], distances[start:end])
    print(f"Indexed {writer.finish()['count']} windows in {len(writer.shards)} shards")


if __name__ == "__main__":
    main()
