"""Download the verified INTERACTION sample and run the Rust corpus pipeline."""

import argparse
import hashlib
import io
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_manifold.loader import INTERACTION_TC_SHA256, INTERACTION_TC_URL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/.cache/exemplars.json"))
    parser.add_argument("--output", type=Path, default=Path("web/public/corpus"))
    parser.add_argument("--input", type=Path, help="Local CSV or directory; skips sample download")
    args = parser.parse_args()
    if args.input:
        subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "--manifest-path",
                "native/Cargo.toml",
                "--",
                str(args.input),
                str(args.model),
                str(args.output),
            ],
            check=True,
        )
        return
    request = urllib.request.Request(INTERACTION_TC_URL, headers={"User-Agent": "trajalyzer/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != INTERACTION_TC_SHA256:
        raise ValueError("Unexpected INTERACTION archive SHA256")
    with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(io.BytesIO(archive)) as source:
        root = Path(folder)
        for entry in source.infolist():
            name = Path(entry.filename).name
            if name.startswith("vehicle_tracks_") and name.endswith(".csv"):
                (root / name).write_bytes(source.read(entry))
        subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "--manifest-path",
                "native/Cargo.toml",
                "--",
                str(root),
                str(args.model),
                str(args.output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
