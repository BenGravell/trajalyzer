# Trajalyzer

Interactive exploration of observed vehicle trajectories from [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) (AV2) at its native **11 seconds, 10 Hz, 110 positions**. By default, complete, moving autonomous vehicle (AV) ego tracks are translated and rotated into their starting frame. Non-ego vehicle and bus tracks remain available with `--tracks vehicles`.

The browser loads one compressed Rerun recording for the current page, with at most 768 paths. Rerun renders the routes, moving vehicle markers, and inferred acceleration and curvature on a shared timeline. The visibility control filters the loaded recording across all three views. The representative paths are **observed exemplars**, selected nearest to MiniBatchKMeans centers in feature space. The checked-in corpus at `data/corpus` contains all 80,593 eligible ego trajectories from the AV2 training split.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,corpus,viewer]'
npm --prefix web ci
pre-commit install --hook-type pre-commit --hook-type pre-push
npm --prefix web run dev
```

`npm --prefix web run dev` and `npm --prefix web run build` first export recordings into `web/public/corpus/recordings` and reuse files whose inputs match. GitHub Pages serves those static assets with no connection to the AV2 service.

`./scripts/check.sh` runs Ruff, Black, Python tests, ESLint, Prettier, and the Vite build. The Pages workflow also runs Rust format, Clippy, and tests and validates the checked-in AV2 corpus before deployment.

## AV2 pipeline

```bash
python scripts/stream_av2.py --split train --limit-scenarios 1000 --output data/av2-sample
python scripts/fit_av2_exemplars.py data/av2-sample data/av2-model --clusters 64
cargo run --release --manifest-path native/Cargo.toml -- --pack data/av2-sample data/av2-model/model.json data/av2-corpus
```

The Python adapter streams public scenario Parquet objects to a Rust extractor. The feature and position store is disk-backed. FAISS k-means fits centers over **every eligible stored window** by default, and the fitter selects actual trajectories nearest those centers. Rust packs the assigned corpus into version 2 binary shards with 110 positions at 5 cm resolution and action samples at 10 Hz. `source.json`, `index.csv`, and `model-meta.json` retain provenance and the fit's training count. The AV2 workflow and source constraints are detailed in [data sources](docs/data_sources.md).

Kinematic state is `(x, y, yaw, speed)` and the inferred action is `(acceleration, curvature)`. From smoothed velocity `(ẋ,ẏ)` and acceleration `(ẍ,ÿ)`, `yaw = atan2(ẏ,ẋ)`, `speed = hypot(ẋ,ẏ)`, `acceleration = (ẋẍ+ẏÿ)/speed`, and `curvature = (ẋÿ−ẏẍ)/speed³`. A local seven-point cubic fit supplies derivatives. Windows below 1 m/s are excluded because heading and curvature become poorly determined. These are estimates, not recorded control commands.

## Publish a full-data corpus

The [AV2 guide](docs/data_sources.md) gives the commands for refreshing the checked-in corpus. The Pages workflow publishes it at <https://bengravell.github.io/trajalyzer/>. The optional `--tracks vehicles` scope retains non-ego vehicle and bus actors for later experiments. For a larger corpus, run `python scripts/export_web_recordings.py --corpus /path/to/corpus --output /path/to/assets/recordings`. Publish the directory containing `recordings/` to an object store or CDN, set `VITE_CORPUS_BASE_URL` to its public trailing-slash URL at build time, and configure CORS for the Pages origin. An analytical Parquet mirror can be generated with `python scripts/export_parquet.py /path/to/corpus /path/to/parquet` and read with Polars or Arrow scanners.

AV2 is CC BY-NC-SA 4.0, attributed `© 2021 Argo AI, LLC`. The full-data run reports its actual eligibility count.
