"""Export the scikit-learn exemplar features for the native corpus builder."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.model, allow_pickle=False) as model:
        features = model["features"].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"features": features.tolist()}, separators=(",", ":")))
    print(f"Exported {len(features)} exemplars to {args.output}")


if __name__ == "__main__":
    main()
