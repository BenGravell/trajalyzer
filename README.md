# Trajalyzer

Interactive exploration of observed vehicle trajectories. The primary indexed corpus uses [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) (AV2) at its native **11 seconds, 10 Hz, 110 positions**. Complete, moving AV and vehicle/bus tracks are translated and rotated into each vehicle's starting frame. Acceleration and curvature are inferred from the planar trajectory using differential flatness and plotted against the synchronized playback cursor. A separate, earlier Rerun diagnostic view still uses a 5-second, 50-frame [INTERACTION](https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset) sample.

The browser requests bounded position shards and only fetches a selected shard's action data when needed. It shows at most 768 paths at once while paging through the full indexed corpus. The representative colored paths are **observed exemplars**, selected nearest to MiniBatchKMeans centers in the feature space. The deployed AV2 sample uses 100 scenarios and 16 preview exemplars. It is labeled as a preview; its clusters do not claim to represent all AV2 driving behavior.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,corpus,viewer]'
npm --prefix web ci
pre-commit install --hook-type pre-commit --hook-type pre-push
npm --prefix web run dev
```

`./scripts/check.sh` runs Ruff, Black, Python tests, ESLint, Prettier, and the Vite build. The Pages workflow additionally runs Rust format, Clippy, tests, and the AV2 preview build before deployment. The committed `web/public` Rerun recordings and diagnostic binaries are from the INTERACTION sample; `python scripts/build_recording.py` regenerates those if needed. `python scripts/build_corpus.py` remains a small 50-frame protocol fixture against those binaries.

## AV2 pipeline

```bash
python scripts/stream_av2.py --split train --tracks vehicles --limit-scenarios 1000 --output data/av2-sample
python scripts/fit_av2_exemplars.py data/av2-sample data/av2-model --clusters 64
cargo run --release --manifest-path native/Cargo.toml -- --pack data/av2-sample data/av2-model/model.json data/av2-corpus
```

The Python adapter fetches public scenario Parquet objects into bounded memory and streams rows to a Rust extractor. The raw dataset is not saved. The resulting feature and position store is disk-backed; scikit-learn fits in bounded batches across **every eligible stored window**, then selects actual trajectories closest to its centers. Rust packs the assigned corpus into version 2 binary shards with 110 positions at 5 cm resolution, and action samples at 10 Hz. `source.json`, `index.csv`, and `model-meta.json` retain provenance and the fit's true training count. The AV2 workflow and source constraints are detailed in [data sources](docs/data_sources.md).

Kinematic state is `(x, y, yaw, speed)` and the inferred action is `(acceleration, curvature)`. From smoothed velocity `(ẋ,ẏ)` and acceleration `(ẍ,ÿ)`, `yaw = atan2(ẏ,ẋ)`, `speed = hypot(ẋ,ẏ)`, `acceleration = (ẋẍ+ẏÿ)/speed`, and `curvature = (ẋÿ−ẏẍ)/speed³`. A local seven-point cubic fit supplies derivatives. Windows below 1 m/s are excluded because heading and curvature become poorly determined. These are estimates, not recorded control commands.

## Publish a full-data corpus

The [AV2 guide](docs/data_sources.md) gives the full training-split extraction and exemplar commands. Publish the packed directory to an object store/CDN and set the repository Actions variable `VITE_CORPUS_BASE_URL` to its public trailing-slash URL. Configure CORS for the Pages origin. GitHub Pages hosts the app and a small preview at <https://bengravell.github.io/trajalyzer/>; millions of trajectories and their shards belong on scalable storage. An analytical Parquet mirror can be generated with `python scripts/export_parquet.py /path/to/corpus /path/to/parquet` and read with Polars or Arrow scanners.

AV2 is CC BY-NC-SA 4.0, attributed `© 2021 Argo AI, LLC`. The separately labeled INTERACTION diagnostic sample is CC BY 4.0. The current pilot does **not** establish that the full AV2 split has a million eligible full-length moving tracks, and the full-data fit has not yet run.
