"""Bounded-memory static corpus writer for GitHub Pages or an object store.

Each independent shard is directly cacheable. The manifest contains only bucket
counts and URLs; the browser requests a small working set, never the corpus.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class CorpusWriter:
    """Write 50-step ego windows grouped by lateral cohort and exemplar ID.

    Positions are centimetre int16; actions remain float32 because curvature
    is sensitive to quantization. Infer actions before quantizing positions.
    An optional Parquet mirror is for research queries, not the web payload.
    """

    def __init__(self, root, *, sample_rate=10, frames=50, shard_size=128, parquet=False):
        if frames != 50 or shard_size < 1 or sample_rate <= 0:
            raise ValueError("Web format requires 50 frames, positive rate and shard size")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.shard_size = shard_size
        self.pending = defaultdict(list)
        self.parts = defaultdict(int)
        self.shards = []
        self.count = 0
        self.parquet = parquet
        if parquet:
            try:
                import pyarrow as pa  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Install trajalyzer[corpus] for Parquet output") from exc
            (self.root / "parquet").mkdir(exist_ok=True)

    def add(self, positions, actions, clusters, distances):
        p = np.asarray(positions)
        a = np.asarray(actions)
        c = np.asarray(clusters)
        d = np.asarray(distances)
        n = len(p)
        if p.shape != (n, 50, 2) or a.shape != p.shape or c.shape != (n,) or d.shape != (n,):
            raise ValueError("Expected positions/actions (N,50,2), clusters/distances (N,)")
        if not (np.isfinite(p).all() and np.isfinite(a).all() and np.isfinite(d).all()):
            raise ValueError("Non-finite corpus data")
        scaled = np.rint(p * 100)
        if np.max(np.abs(scaled), initial=0) > 32767:
            raise ValueError("Position exceeds signed 16-bit centimetres")
        for i in range(n):
            lateral = p[i, -1, 1]
            cohort = "positive" if lateral > 3 else "negative" if lateral < -3 else "center"
            cluster = int(c[i])
            if cluster < 0 or cluster > 65535 or self.count >= 2**32:
                raise ValueError("Cluster or trajectory ID out of range")
            bucket = (cohort, cluster)
            self.pending[bucket].append(
                (scaled[i].astype("<i2"), a[i].astype("<f4"), self.count, d[i])
            )
            self.count += 1
            if len(self.pending[bucket]) == self.shard_size:
                self._flush(bucket)
        if self.parquet and n:
            self._write_parquet(p, a, c, d, self.count - n)

    def _write_parquet(self, p, a, c, d, first_id):
        import pyarrow as pa
        import pyarrow.parquet as pq

        def fixed(values):
            return pa.FixedSizeListArray.from_arrays(
                pa.array(values.ravel(), type=pa.float32()), 100
            )

        table = pa.table(
            {
                "trajectory_id": pa.array(np.arange(first_id, first_id + len(p), dtype=np.uint32)),
                "cluster": pa.array(c.astype(np.uint16)),
                "final_lateral_m": pa.array(p[:, -1, 1].astype(np.float32)),
                "distance": pa.array(d.astype(np.float32)),
                "positions_xy_m": fixed(p.astype(np.float32)),
                "actions_accel_curvature": fixed(a.astype(np.float32)),
            }
        )
        pq.write_table(
            table, self.root / "parquet" / f"part-{first_id:010d}.parquet", compression="zstd"
        )

    def _flush(self, bucket):
        items = self.pending[bucket]
        if not items:
            return
        cohort, cluster = bucket
        part = self.parts[bucket]
        self.parts[bucket] += 1
        stem = f"shards/{cohort}-{cluster:04d}-{part:06d}"
        destination = self.root / "shards"
        destination.mkdir(exist_ok=True)
        with (
            (self.root / f"{stem}.bin").open("wb") as position_file,
            (self.root / f"{stem}.actions.bin").open("wb") as action_file,
        ):
            for position, action, trajectory_id, distance in items:
                position_file.write(position.tobytes())
                position_file.write(np.asarray([trajectory_id], dtype="<u4").tobytes())
                position_file.write(np.asarray([distance], dtype="<f4").tobytes())
                action_file.write(action.tobytes())
        self.shards.append(
            {"cohort": cohort, "cluster": cluster, "count": len(items), "path": f"{stem}.bin"}
        )
        items.clear()

    def finish(self):
        for bucket in list(self.pending):
            self._flush(bucket)
        manifest = {
            "version": 1,
            "count": self.count,
            "frames": 50,
            "sample_rate_hz": self.sample_rate,
            "position_scale_m": 0.01,
            "shards": self.shards,
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        return manifest
