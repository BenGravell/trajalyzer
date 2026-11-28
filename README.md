# Trajalyzer

Interactive exploration of observed vehicle trajectories. The primary indexed corpus uses [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) (AV2) at its native **11 seconds, 10 Hz, 110 positions**. By default, complete, moving autonomous vehicle (AV) ego tracks are translated and rotated into their starting frame. Non-ego vehicle and bus tracks remain available with `--tracks vehicles`. Acceleration and curvature are inferred from the planar trajectory using differential flatness and plotted against the synchronized playback cursor. A separate, earlier Rerun diagnostic view still uses a 5-second, 50-frame [INTERACTION](https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset) sample.

The browser requests bounded position and action shards for the current page. It shows at most 768 paths at once while paging through the full indexed corpus. Acceleration and curvature plots overlay every visible path, color them by exemplar, and emphasize the selected path. Group, exemplar, and selected-only controls drive both trajectories and plots. The representative colored paths are **observed exemplars**, selected nearest to MiniBatchKMeans centers in the feature space. The checked-in corpus at `web/public/corpus` contains all 80,593 eligible ego trajectories from the AV2 training split. The app records the actual number of eligible windows and source scenarios.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,corpus,viewer]'
npm --prefix web ci
pre-commit install --hook-type pre-commit --hook-type pre-push
npm --prefix web run dev
```

`./scripts/check.sh` runs Ruff, Black, Python tests, ESLint, Prettier, and the Vite build. The Pages workflow also runs Rust format, Clippy, and tests and validates the checked-in AV2 corpus before deployment. Local and Pages builds serve this copy directly, with no requests to AV2. The Rerun recordings and diagnostic binaries in `web/public` come from the INTERACTION sample and can be regenerated with `python scripts/build_recording.py`. `python scripts/build_corpus.py` remains a small 50-frame protocol fixture against those binaries.

## AV2 pipeline

```bash
python scripts/stream_av2.py --split train --limit-scenarios 1000 --output data/av2-sample
python scripts/fit_av2_exemplars.py data/av2-sample data/av2-model --clusters 64
cargo run --release --manifest-path native/Cargo.toml -- --pack data/av2-sample data/av2-model/model.json data/av2-corpus
```

The Python adapter fetches public scenario Parquet objects into bounded memory and streams rows to a Rust extractor. The raw dataset is not saved. The resulting feature and position store is disk-backed; scikit-learn fits in bounded batches across **every eligible stored window**, then selects actual trajectories closest to its centers. Rust packs the assigned corpus into version 2 binary shards with 110 positions at 5 cm resolution, and action samples at 10 Hz. `source.json`, `index.csv`, and `model-meta.json` retain provenance and the fit's true training count. The AV2 workflow and source constraints are detailed in [data sources](docs/data_sources.md).

Kinematic state is `(x, y, yaw, speed)` and the inferred action is `(acceleration, curvature)`. From smoothed velocity `(ẋ,ẏ)` and acceleration `(ẍ,ÿ)`, `yaw = atan2(ẏ,ẋ)`, `speed = hypot(ẋ,ẏ)`, `acceleration = (ẋẍ+ẏÿ)/speed`, and `curvature = (ẋÿ−ẏẍ)/speed³`. A local seven-point cubic fit supplies derivatives. Windows below 1 m/s are excluded because heading and curvature become poorly determined. These are estimates, not recorded control commands.

## Publish a full-data corpus

The [AV2 guide](docs/data_sources.md) gives the commands for refreshing the checked-in corpus. The Pages workflow publishes it at <https://bengravell.github.io/trajalyzer/>. The optional `--tracks vehicles` scope retains non-ego vehicle and bus actors for later experiments. For a larger corpus, publish the packed directory to an object store/CDN and set `VITE_CORPUS_BASE_URL` to its public trailing-slash URL when building the app. Configure CORS for the Pages origin. An analytical Parquet mirror can be generated with `python scripts/export_parquet.py /path/to/corpus /path/to/parquet` and read with Polars or Arrow scanners.

AV2 is CC BY-NC-SA 4.0, attributed `© 2021 Argo AI, LLC`. The separately labeled INTERACTION diagnostic sample is CC BY 4.0. The full-data run reports its actual eligibility count; no million-window threshold is assumed.
