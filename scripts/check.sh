#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check trajectory_manifold scripts tests
black --check trajectory_manifold scripts tests
python -m unittest discover -s tests
npm --prefix web run check
npm --prefix web run build
