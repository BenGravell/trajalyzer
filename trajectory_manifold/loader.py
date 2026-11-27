"""Small, dependency-light loader for real INTERACTION vehicle trajectories.

The source ZIP is downloaded into memory; only an optional processed NPZ cache
is written to disk. Coordinates are metres and the sampling grid is seconds.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


INTERACTION_TC_URL = (
    "https://huggingface.co/datasets/INTERACTION-dataset/"
    "INTERACTION-dataset/resolve/main/INTERACTION-Dataset-TC-v1_0.zip"
)
# Official archive hash, checked so a changing upstream file cannot silently
# invalidate a previously cached experiment.
INTERACTION_TC_SHA256 = (
    "9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec"
)
_CACHE_VERSION = 1


def _source_tracks(url: str):
    """Yield (CSV path, track ID, rows) from the archive, without extracting it."""
    request = urllib.request.Request(url, headers={"User-Agent": "trajectory-manifold/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if url == INTERACTION_TC_URL:
        digest = hashlib.sha256(data).hexdigest()
        if digest != INTERACTION_TC_SHA256:
            raise ValueError(f"Unexpected INTERACTION archive SHA256: {digest}")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        paths = sorted(
            name for name in archive.namelist()
            if Path(name).name.startswith("vehicle_tracks_")
            and name.endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
        )
        if not paths:
            raise ValueError("No INTERACTION vehicle track CSV files found in archive")
        for path in paths:
            tracks = defaultdict(list)
            with archive.open(path) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                    reader = csv.DictReader(text)
                    required = {"track_id", "timestamp_ms", "agent_type", "x", "y", "vx", "vy", "psi_rad"}
                    if not required.issubset(reader.fieldnames or ()):
                        raise ValueError(f"Missing columns in {path}: {required - set(reader.fieldnames or ())}")
                    for row in reader:
                        try:
                            tracks[row["track_id"]].append((
                                int(row["timestamp_ms"]) / 1000.0,
                                float(row["x"]), float(row["y"]),
                                float(row["vx"]), float(row["vy"]),
                                float(row["psi_rad"]), row["agent_type"],
                            ))
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"Invalid vehicle row in {path}") from exc
            for track_id, rows in tracks.items():
                yield path, track_id, rows


def _windows(path, track_id, rows, *, duration, sample_rate, stride,
             coordinate_frame, max_gap, max_speed):
    rows.sort(key=lambda row: row[0])
    times = np.array([row[0] for row in rows], dtype=np.float64)
    xy = np.array([row[1:3] for row in rows], dtype=np.float64)
    velocities = np.array([row[3:5] for row in rows], dtype=np.float64)
    headings = np.array([row[5] for row in rows], dtype=np.float64)
    types = [row[6] for row in rows]
    count = round(duration * sample_rate)
    grid = np.arange(count, dtype=np.float64) / sample_rate
    if len(times) < 2:
        return

    # Split at missing observations, duplicate/reversed times, or impossible jumps.
    dt = np.diff(times)
    step_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(dt, 1e-12)
    cuts = np.flatnonzero(
        (dt <= 0) | (dt > max_gap) | (step_speed > max_speed)
        | ~np.isfinite(step_speed)
    ) + 1
    for segment in np.split(np.arange(len(times)), cuts):
        if len(segment) < 2:
            continue
        t, p = times[segment], xy[segment]
        if not (np.isfinite(p).all() and np.isfinite(t).all()):
            continue
        last_start = t[-1] - grid[-1]
        if last_start < t[0] - 1e-9:
            continue
        # Starts are spaced in physical time, independent of source frame rate.
        starts = t[0] + np.arange(
            int(np.floor((last_start - t[0] + 1e-9) / stride)) + 1
        ) * stride
        for start in starts:
            target = start + grid
            sampled = np.column_stack((np.interp(target, t, p[:, 0]),
                                       np.interp(target, t, p[:, 1])))
            index = segment[np.searchsorted(t, start, side="left")]
            angle = headings[index]
            if not np.isfinite(angle):
                vx, vy = velocities[index]
                if not (np.isfinite(vx) and np.isfinite(vy)) or np.hypot(vx, vy) < 0.1:
                    continue
                angle = np.arctan2(vy, vx)
            if coordinate_frame == "ego":
                sampled -= sampled[0]
                c, s = np.cos(angle), np.sin(angle)
                sampled = sampled @ np.array([[c, -s], [s, c]])
            yield sampled.astype(np.float32), {
                "source_file": path,
                "track_id": track_id,
                "start_time_s": float(start),
                "agent_type": types[index],
                "initial_heading_rad": float(angle),
                "coordinate_frame": coordinate_frame,
                "position_unit": "m",
                "time_unit": "s",
            }


def load_trajectories(
    dataset="interaction",
    root="./data",
    num_trajectories=5000,
    duration=5.0,
    sample_rate=10,
    coordinate_frame="ego",
    seed=42,
    *,
    stride=1.0,
    cache=True,
    refresh=False,
    source_url=INTERACTION_TC_URL,
    max_gap=0.3,
    max_speed=70.0,
):
    """Return (N, T, 2) float32 positions and a list of N metadata dicts.

    T = duration * sample_rate must be an integer. Windows use the half-open
    grid [0, duration); stride controls overlap in seconds. Up to N windows
    are sampled without replacement with a local, reproducible RNG. A smaller
    dataset returns fewer windows. ``cache=False`` keeps all data in memory.
    When enabled, root/.cache holds processed NPZ batches, never the raw ZIP.
    ``refresh=True`` recomputes and replaces the cached batch.
    """
    if dataset.lower() != "interaction":
        raise ValueError("Only dataset='interaction' is implemented")
    if coordinate_frame not in ("ego", "world"):
        raise ValueError("coordinate_frame must be 'ego' or 'world'")
    if (not isinstance(num_trajectories, int) or num_trajectories < 1
            or not isinstance(sample_rate, int) or sample_rate < 1
            or not isinstance(seed, int)):
        raise ValueError("num_trajectories and sample_rate must be positive integers; seed must be an integer")
    if not all(np.isfinite(v) and v > 0 for v in (duration, stride, max_gap, max_speed)):
        raise ValueError("duration, stride, max_gap, and max_speed must be positive finite values")
    if abs(duration * sample_rate - round(duration * sample_rate)) > 1e-9:
        raise ValueError("duration * sample_rate must be an integer")

    settings = dict(version=_CACHE_VERSION, dataset=dataset.lower(),
                    source_url=source_url, source_sha256=(
                        INTERACTION_TC_SHA256 if source_url == INTERACTION_TC_URL else None),
                    num_trajectories=num_trajectories, duration=duration,
                    sample_rate=sample_rate, coordinate_frame=coordinate_frame,
                    seed=seed, stride=stride, max_gap=max_gap, max_speed=max_speed)
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    cache_path = Path(root) / ".cache" / f"interaction_{key}.npz"
    if cache and cache_path.is_file() and not refresh:
        with np.load(cache_path, allow_pickle=False) as saved:
            return saved["trajectories"], json.loads(str(saved["metadata_json"].item()))

    candidates, metadata = [], []
    for path, track_id, rows in _source_tracks(source_url):
        for trajectory, info in _windows(
            path, track_id, rows, duration=duration, sample_rate=sample_rate,
            stride=stride, coordinate_frame=coordinate_frame,
            max_gap=max_gap, max_speed=max_speed,
        ):
            candidates.append(trajectory)
            metadata.append(info)
    if not candidates:
        raise ValueError("No valid trajectory windows found; try a shorter duration")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=min(num_trajectories, len(candidates)),
                          replace=False)
    trajectories = np.stack([candidates[i] for i in selected])
    metadata = [metadata[i] for i in selected]

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz",
                                             delete=False) as output:
                temporary = Path(output.name)
                np.savez_compressed(output, trajectories=trajectories,
                                    metadata_json=json.dumps(metadata))
            os.replace(temporary, cache_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return trajectories, metadata


def iter_trajectory_batches(*, dataset="interaction", source_url=INTERACTION_TC_URL,
                            batch_size=2048, duration=5.0, sample_rate=10,
                            stride=1.0, coordinate_frame="ego",
                            max_gap=0.3, max_speed=70.0):
    """Yield (positions, metadata) batches without collecting all windows.

    This reopens the small source archive on each iteration, so a repeatable
    factory can make multiple passes for MiniBatchKMeans on large collections.
    The ZIP stays in memory; only a batch of processed windows is retained.
    """
    if dataset.lower() != "interaction" or batch_size < 1:
        raise ValueError("Require dataset='interaction' and positive batch_size")
    if coordinate_frame not in ("ego", "world") or abs(duration * sample_rate - round(duration * sample_rate)) > 1e-9:
        raise ValueError("Invalid coordinate frame or temporal grid")
    positions, metadata = [], []
    for path, track_id, rows in _source_tracks(source_url):
        for trajectory, info in _windows(
            path, track_id, rows, duration=duration, sample_rate=sample_rate,
            stride=stride, coordinate_frame=coordinate_frame,
            max_gap=max_gap, max_speed=max_speed,
        ):
            positions.append(trajectory)
            metadata.append(info)
            if len(positions) == batch_size:
                yield np.stack(positions), metadata
                positions, metadata = [], []
    if positions:
        yield np.stack(positions), metadata


if __name__ == "__main__":
    batch, details = load_trajectories()
    print(f"Loaded {batch.shape[0]} trajectories of shape {batch.shape[1:]} from "
          f"{len({item['source_file'] for item in details})} CSV files")
