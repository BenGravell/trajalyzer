# Argoverse 2 Motion Forecasting source

The primary corpus uses the [AV2 Motion Forecasting dataset](https://www.argoverse.org/av2.html): 250,000 eleven-second scenarios sampled at 10 Hz. The public unsigned S3 source is `s3://argoverse/datasets/av2/motion-forecasting/` in `us-east-1`. Its per-scenario Parquet files make bounded extraction possible without retaining the raw archive. PyArrow decodes each fetched scenario and streams its rows to the Rust pipeline; the derived feature store and packed viewer corpus do occupy storage.

Each complete track produces **one** 110-position window on the half-open grid `0, 0.1, …, 10.9` seconds. The default `av` scope includes only the autonomous ego vehicle. The optional `vehicles` scope also includes other vehicle and bus tracks. A continuous window must maintain at least 1 m/s under the native seven-point local cubic derivative estimate. The AV-only scope has at most one window per scenario; no million-window target is imposed. In an initial, lexicographically selected 35-scenario pilot, 19 AV and 55 other vehicle tracks survived; that sample is not representative of the whole collection.

## Stream, fit, and pack

Install Python with the `corpus` and `cluster` extras (PyArrow, FAISS, and scikit-learn), and Cargo. From the repository root:

```bash
pip install -e '.[corpus,cluster]'
python scripts/stream_av2.py --split train --all --save-slice-map --output data/av2-train
python scripts/fit_av2_exemplars.py data/av2-train data/av2-model
cargo run --release --manifest-path native/Cargo.toml -- --pack data/av2-train data/av2-model/model.json data/av2-corpus
python scripts/check_av2_corpus.py data/av2-corpus
```

For a bounded trial, replace `--all` with `--limit-scenarios 1000` and reduce `--clusters` if necessary. Extraction writes `positions.f32`, `features.f32`, `index.csv`, `store.json`, and `source.json`; it never materializes a whole raw AV2 archive. The fit memory maps all eligible windows, makes bounded feature passes, and chooses an **observed** member nearest each center. Packing assigns each window to those observed exemplars and writes independent, cacheable position and action shards. The feature store can be discarded after verification and publication. Each extracted path uses about 880 bytes for positions and 1,100 bytes for features, plus metadata and staging overhead. Plan for S3 transfer, CPU time, and disk space on a worker near `us-east-1` for the full split.

Use `--clusters auto` to estimate the exemplar count with HDBSCAN on every stored window. HDBSCAN selects persistent density groups with excess of mass and counts the groups it finds. The fitter then uses that count across the full store, including windows marked as noise by HDBSCAN. Density estimation can take considerably longer on large stores. Use `--density-sample-size 8192` for a faster estimate from a seeded sample. Set `--min-cluster-size` (default 0.5% of HDBSCAN input windows, rounded up, with a minimum of 5) or `--min-samples` (default 10) to change the density criteria. The fit records whether HDBSCAN used the full store, how many windows it saw, the group sizes, and the noise fraction in `model-meta.json`. If fewer than two groups are found, the command reports the result and exits. Adjust the density settings or use a fixed count in that case.

The default fitter uses CPU FAISS k-means with 128 exemplars and 10 Lloyd iterations on all stored windows. FAISS's training subsampling is disabled. The final pass uses exact L2 search to choose observed trajectories, with `--batch-size` controlling how many rows are read at once. Use `--clusters` and `--passes` to adjust the fit. The alternative backends are `--backend minibatch` and `--backend emeans`, both with two training passes by default. E-means requires FAISS 1.16 or later, available via `pip install -e '.[emeans]'`. The fitter prints separate times for HDBSCAN, center fitting, and observed-exemplar selection.

The checked-in corpus at `data/corpus` comes from the complete AV2 training-split ego extraction in [Pages run 46116610225](https://github.com/BenGravell/trajalyzer/actions/runs/46116610225). It contains 80,593 eligible windows across 768 position and action shard pairs, with 128 observed exemplars. The Pages build validates those shards, then exports compressed Rerun recordings without contacting AV2. To refresh it, run the full commands above into a fresh output directory, then replace the checked-in corpus directory with the validated output and commit the changes together.

For larger non-ego corpora, export the packed directory with `python scripts/export_web_recordings.py --corpus /path/to/corpus --output /path/to/assets/recordings`. Publish the assets directory to an object store or CDN, configure `VITE_CORPUS_BASE_URL` at build time, and allow the Pages origin with CORS. Upload the recordings and page metadata before publishing `recordings/index.json`. Check the size of the generated assets before using GitHub Pages for expanded actor scopes.

`index.csv` records scenario, track and start time. `--save-slice-map` retains the maintainers' [scenario-to-source-slice mapping](https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/av2_mf_train_scenario_to_slice.csv). Group source slices for train/test evaluation when possible. That map does not identify complete acquisition logs, so it does not prove log-independent validation. `--start-after` can partition S3 keys into distinct extraction jobs, but the current fitter accepts one store; combine stores with a validated merger or process the full split as one stream before claiming a full-data fit.

AV2 is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Attribute `© 2021 Argo AI, LLC` and preserve the noncommercial, share-alike conditions for derived data.
