"""Validate the packed AV2 training ego corpus before publishing it."""

import argparse
import csv
import json
from pathlib import Path
from struct import Struct


def check_corpus(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    model = json.loads((root / "model-meta.json").read_text(encoding="utf-8"))
    count = manifest["count"]
    frames = manifest["frames"]
    if manifest["version"] != 2 or frames != 110 or manifest["sample_rate_hz"] != 10:
        raise ValueError("Expected the 110-frame AV2 v2 shard format")
    if (source["split"], source["track_scope"], source["complete_split"]) != ("train", "av", True):
        raise ValueError("Expected the complete AV2 training split with ego tracks")
    if count <= 0 or model["training_windows"] != count or model["frames"] != frames:
        raise ValueError("The exemplar model does not match the corpus")
    if (root / "exemplars.bin").stat().st_size != model["count"] * frames * 4:
        raise ValueError("Incorrect exemplar file size")

    position_row = Struct(f"<{frames * 4}xIf")
    trajectories = {}
    paths = set()
    for shard in manifest["shards"]:
        path = Path(shard["path"])
        if path.parent != Path("shards") or path.suffix != ".bin" or path in paths:
            raise ValueError(f"Invalid or repeated shard path: {path}")
        paths.add(path)
        positions = (root / path).read_bytes()
        actions = root / path.with_suffix(".actions.bin")
        rows = shard["count"]
        if (
            len(positions) != rows * position_row.size
            or actions.stat().st_size != rows * frames * 8
        ):
            raise ValueError(f"Incorrect position or action file size: {path}")
        for trajectory_id, _ in position_row.iter_unpack(positions):
            if trajectory_id in trajectories:
                raise ValueError(f"Duplicated trajectory ID: {trajectory_id}")
            trajectories[trajectory_id] = (shard["cohort"], shard["cluster"])
    if len(trajectories) != count:
        raise ValueError("The shard trajectory count does not match the manifest")

    with (root / "index.csv").open(encoding="utf-8", newline="") as index:
        for expected_id, row in enumerate(csv.DictReader(index)):
            group = (row["cohort"], int(row["cluster"]))
            if (
                int(row["trajectory_id"]) != expected_id
                or trajectories.pop(expected_id, None) != group
            ):
                raise ValueError(
                    f"The trajectory index does not match the shards at ID {expected_id}"
                )
    if trajectories:
        raise ValueError("The trajectory index is incomplete")
    print(f"Verified {count:,} AV2 ego trajectories across {len(paths):,} shard pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    check_corpus(parser.parse_args().corpus)
