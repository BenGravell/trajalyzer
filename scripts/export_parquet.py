"""Stream static corpus shards into analytical Parquet partitions using Arrow."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows-per-file", type=int, default=8192)
    args = parser.parse_args()
    if args.rows_per_file < 1:
        parser.error("--rows-per-file must be positive")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["version"] not in (1, 2) or manifest["frames"] not in (50, 110):
        raise ValueError("Unsupported corpus format")
    frames = manifest["frames"]
    position_bytes = frames * 4
    action_bytes = frames * 8
    row_bytes = position_bytes + 8
    pending = defaultdict(list)
    parts = defaultdict(int)

    def flush(key):
        records = pending[key]
        if not records:
            return
        ids, distances, positions, actions = (
            np.concatenate([r[i] for r in records]) for i in range(4)
        )

        def fixed(values):
            return pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), frames * 2)

        table = pa.table(
            {
                "trajectory_id": pa.array(ids),
                "distance": pa.array(distances),
                "positions_xy_m": fixed(positions),
                "actions_accel_curvature": fixed(actions),
            }
        )
        cohort, cluster = key
        folder = args.output / f"cohort={cohort}" / f"cluster={cluster}"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, folder / f"part-{parts[key]:05d}.parquet", compression="zstd")
        parts[key] += 1
        records.clear()

    for shard in manifest["shards"]:
        n = shard["count"]
        payload = (args.corpus / shard["path"]).read_bytes()
        detail = (args.corpus / shard["path"].replace(".bin", ".actions.bin")).read_bytes()
        if len(payload) != n * row_bytes or len(detail) != n * action_bytes:
            raise ValueError(f"Invalid shard {shard['path']}")
        data = np.frombuffer(payload, dtype=np.uint8).reshape(n, row_bytes)
        positions = (
            np.frombuffer(data[:, :position_bytes].copy(), dtype="<i2")
            .reshape(n, frames * 2)
            .astype(np.float32)
            * manifest["position_scale_m"]
        )
        ids = np.frombuffer(data[:, position_bytes : position_bytes + 4].copy(), dtype="<u4")
        distances = np.frombuffer(data[:, position_bytes + 4 : row_bytes].copy(), dtype="<f4")
        actions = np.frombuffer(detail, dtype="<f4").reshape(n, frames * 2)
        key = (shard["cohort"], shard["cluster"])
        pending[key].append((ids, distances, positions, actions))
        if sum(len(batch[0]) for batch in pending[key]) >= args.rows_per_file:
            flush(key)
    for key in list(pending):
        flush(key)
    print(f"Wrote {sum(parts.values())} Parquet partitions/files")


if __name__ == "__main__":
    main()
