# Argoverse 2 Motion Forecasting source

The primary corpus uses the [AV2 Motion Forecasting dataset](https://www.argoverse.org/av2.html): 250,000 eleven-second scenarios sampled at 10 Hz. The public unsigned S3 source is `s3://argoverse/datasets/av2/motion-forecasting/` in `us-east-1`. Its per-scenario Parquet files make bounded extraction possible without retaining the raw archive. PyArrow decodes each fetched scenario and streams its rows to the Rust pipeline; the derived feature store and packed viewer corpus do occupy storage.

Each complete track produces **one** 110-position window on the half-open grid `0, 0.1, …, 10.9` seconds. The default `av` scope includes only the autonomous ego vehicle. The optional `vehicles` scope also includes other vehicle and bus tracks. A continuous window must maintain at least 1 m/s under the native seven-point local cubic derivative estimate. The AV-only scope has at most one window per scenario; no million-window target is imposed. In an initial, lexicographically selected 35-scenario pilot, 19 AV and 55 other vehicle tracks survived; that sample is not representative of the whole collection.

## Stream, fit, and pack

Install Python with the `corpus` extra (PyArrow and scikit-learn), and Cargo. From the repository root:

```bash
pip install -e '.[corpus]'
python scripts/stream_av2.py --split train --all --save-slice-map --output data/av2-train
python scripts/fit_av2_exemplars.py data/av2-train data/av2-model --clusters 128 --passes 2
cargo run --release --manifest-path native/Cargo.toml -- --pack data/av2-train data/av2-model/model.json data/av2-corpus
```

For a bounded trial, replace `--all` with `--limit-scenarios 1000` and reduce `--clusters` if necessary. Extraction writes `positions.f32`, `features.f32`, `index.csv`, `store.json`, and `source.json`; it never materializes a whole raw AV2 archive. The fit memory maps all eligible windows, makes bounded feature passes, and chooses an **observed** member nearest each center. Packing assigns each window to those observed exemplars and writes independent, cacheable position and action shards. The feature store can be discarded after verification and publication. Each extracted path uses about 880 bytes for positions and 1,100 bytes for features, plus metadata and staging overhead. Plan for S3 transfer, CPU time, and disk space on a worker near `us-east-1` for the full split.

The deployment workflow executes the **complete AV2 training-split ego extraction**, fits 128 observed exemplars, packs all eligible windows, and publishes them on GitHub Pages. The app displays the actual scenario and fit counts from the run. For larger non-ego corpora, publish the packed directory to an object store/CDN and configure `VITE_CORPUS_BASE_URL` at build time; allow the Pages origin with CORS and publish the manifest only after its referenced shards and exemplar files are available. GitHub Pages has a 1 GB published-site limit, so check the packed size before using it for expanded actor scopes.

`index.csv` records scenario, track and start time. `--save-slice-map` retains the maintainers' [scenario-to-source-slice mapping](https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/av2_mf_train_scenario_to_slice.csv). Group source slices for train/test evaluation when possible. That map does not identify complete acquisition logs, so it does not prove log-independent validation. `--start-after` can partition S3 keys into distinct extraction jobs, but the current fitter accepts one store; combine stores with a validated merger or process the full split as one stream before claiming a full-data fit.

AV2 is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Attribute `© 2021 Argo AI, LLC` and preserve the noncommercial, share-alike conditions for derived data. The old INTERACTION diagnostic recording is separately labeled and uses CC BY 4.0.
