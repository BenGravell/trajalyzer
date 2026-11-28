"""Stream public Argoverse 2 scenario Parquet files into the Rust corpus builder.

No raw dataset is retained locally. PyArrow decodes the source format; the
native process handles track segmentation, flatness, assignment, and sharding.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "https://argoverse.s3.amazonaws.com/"
PREFIX = "datasets/av2/motion-forecasting/"
COLUMNS = [
    "track_id",
    "object_type",
    "timestep",
    "position_x",
    "position_y",
    "heading",
    "velocity_x",
    "velocity_y",
    "scenario_id",
    "city",
]


def read_public(url, *, timeout=60):
    """Retry transient S3/network failures during a long corpus extraction."""
    request = urllib.request.Request(url, headers={"User-Agent": "trajalyzer/0.3"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (
                429,
                500,
                502,
                503,
                504,
            ):
                raise
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))


def scenario_keys(split, *, limit=None, start_after=None):
    """List source Parquet objects with S3 continuation; no AWS account needed."""
    prefix = f"{PREFIX}{split}/"
    token = None
    yielded = 0
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        if start_after and not token:
            query["start-after"] = start_after
        tree = ET.fromstring(read_public(BUCKET + "?" + urllib.parse.urlencode(query)))
        for item in tree.findall("{*}Contents"):
            key = item.findtext("{*}Key")
            if key and Path(key).name.startswith("scenario_") and key.endswith(".parquet"):
                yield key
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
        if tree.findtext("{*}IsTruncated") != "true":
            return
        token = tree.findtext("{*}NextContinuationToken")
        if not token:
            raise ValueError("Truncated S3 listing lacks continuation token")


def fetch_scenario(key, tracks):
    payload = read_public(BUCKET + key)
    table = pq.read_table(pa.BufferReader(payload), columns=COLUMNS)
    rows = table.to_pydict()
    scenario_id = rows["scenario_id"][0]
    if not all(value == scenario_id for value in rows["scenario_id"]):
        raise ValueError(f"Mixed scenario IDs in {key}")
    selected = []
    for i, track in enumerate(rows["track_id"]):
        if tracks == "av" and track != "AV":
            continue
        if (
            tracks == "vehicles"
            and track != "AV"
            and rows["object_type"][i]
            not in (
                "vehicle",
                "bus",
            )
        ):
            continue
        selected.append(
            (
                scenario_id,
                track,
                int(rows["timestep"][i]) * 100,
                rows["position_x"][i],
                rows["position_y"][i],
                rows["velocity_x"][i],
                rows["velocity_y"][i],
                rows["heading"][i],
            )
        )
    return key, rows["city"][0], selected


def stream_scenarios(keys, *, tracks, workers, writer):
    """Ordered, bounded prefetch keeps adjacent scenario rows together."""
    keys = iter(keys)
    pending = deque()
    count = 0
    city_counts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers * 2):
            key = next(keys, None)
            if key is None:
                break
            pending.append(pool.submit(fetch_scenario, key, tracks))
        while pending:
            key, city, rows = pending.popleft().result()
            writer.writerows(rows)
            count += 1
            city_counts[city] = city_counts.get(city, 0) + 1
            if count % 1000 == 0:
                print(f"Streamed {count:,} scenarios; latest {key}", file=sys.stderr)
            key = next(keys, None)
            if key is not None:
                pending.append(pool.submit(fetch_scenario, key, tracks))
    return count, city_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--tracks", choices=("av", "vehicles"), default="av")
    parser.add_argument("--limit-scenarios", type=int, default=1000)
    parser.add_argument("--all", action="store_true", help="Process the entire selected split")
    parser.add_argument("--start-after", help="Resume listing after an exact S3 object key")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--save-slice-map", action="store_true", help="Save the train scenario-to-source-slice map"
    )
    parser.add_argument("--model", type=Path, help="Optional prefit AV2 model for direct packing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit_scenarios < 1 or args.workers < 1:
        parser.error("limit-scenarios and workers must be positive")
    if args.split == "test":
        parser.error("Use train or val: test track annotations require separate handling")
    if args.save_slice_map and args.split != "train":
        parser.error("The published source-slice map covers the train split only")
    keys = scenario_keys(
        args.split, limit=None if args.all else args.limit_scenarios, start_after=args.start_after
    )
    command = ["cargo", "run", "--release", "--manifest-path", "native/Cargo.toml", "--"]
    if args.model:
        command.extend(["--stream", str(args.model), str(args.output)])
    else:
        command.extend(["--extract", str(args.output)])
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Native process has no input pipe")
    try:
        with io.TextIOWrapper(process.stdin, encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ["scenario_id", "track_id", "timestamp_ms", "x", "y", "vx", "vy", "psi_rad"]
            )
            count, cities = stream_scenarios(
                keys, tracks=args.tracks, workers=args.workers, writer=writer
            )
        if process.wait() != 0:
            raise RuntimeError("Native corpus builder failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    args.output.mkdir(parents=True, exist_ok=True)
    if args.save_slice_map:
        mapping = BUCKET + PREFIX + "av2_mf_train_scenario_to_slice.csv"
        with ((args.output / "av2_mf_train_scenario_to_slice.csv").open("wb") as destination,):
            destination.write(read_public(mapping, timeout=120))
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "dataset": "Argoverse 2 Motion Forecasting",
                "split": args.split,
                "track_scope": args.tracks,
                "scenarios": count,
                "complete_split": args.all and args.start_after is None,
                "cities": cities,
                "source": BUCKET + PREFIX,
                "license": "CC BY-NC-SA 4.0",
                "attribution": "© 2021 Argo AI, LLC",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Streamed {count:,} scenarios into {args.output}")


if __name__ == "__main__":
    main()
