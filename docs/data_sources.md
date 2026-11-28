# Argoverse 2 Motion Forecasting source

The primary corpus uses the [AV2 Motion Forecasting dataset](https://www.argoverse.org/av2.html): 250,000 eleven-second scenarios sampled at 10 Hz. The public unsigned S3 source is `s3://argoverse/datasets/av2/motion-forecasting/` in `us-east-1`. Its per-scenario Parquet files make bounded extraction possible without retaining the raw archive. PyArrow decodes each fetched scenario and streams its rows to the Rust pipeline; the derived feature store and packed viewer corpus do occupy storage.

Each complete track produces **one** 110-position window on the half-open grid `0, 0.1, …, 10.9` seconds. The `vehicles` scope includes the AV and vehicle/bus tracks. A continuous window must maintain at least 1 m/s under the native seven-point local cubic derivative estimate. The AV-only scope has an absolute upper bound of 250,000 windows; a million eligible 11-second windows requires multiple vehicle tracks per scenario and has **not** been established. In an initial, lexicographically selected 35-scenario pilot, 19 AV and 55 other vehicle tracks survived; that sample is not representative of the whole collection.

## Stream, fit, and pack

Install Python with the `corpus` extra (PyArrow and scikit-learn), and Cargo. From the repository root:

```bash
pip install -e '.[corpus]'
python scripts/stream_av2.py --split train --tracks vehicles --all --save-slice-map --output data/av2-train
python scripts/fit_av2_exemplars.py data/av2-train data/av2-model --clusters 128 --passes 2
cargo run --release --manifest-path native/Cargo.toml -- --pack data/av2-train data/av2-model/model.json data/av2-corpus
```

For a bounded trial, replace `--all` with `--limit-scenarios 1000` and reduce `--clusters` if necessary. Extraction writes `positions.f32`, `features.f32`, `index.csv`, `store.json`, and `source.json`; it never materializes a whole raw AV2 archive. The fit memory maps all eligible windows, makes bounded feature passes, and chooses an **observed** member nearest each center. Packing assigns each window to those observed exemplars and writes independent, cacheable position and action shards. The feature store can be discarded after verification and publication. Each extracted path uses about 880 bytes for positions and 1,100 bytes for features, plus metadata and staging overhead. Plan for S3 transfer, CPU time, and disk space on a worker near `us-east-1` for the full split.

The deployment workflow builds a **100-scenario preview** with 16 preview exemplars. Its displayed training count and scenario count distinguish those from a full-corpus model. To expose full-data exemplars, run the commands above, publish the packed corpus directory to an object store/CDN, and set the GitHub Actions variable `VITE_CORPUS_BASE_URL` to its trailing-slash URL. Allow the Pages origin with CORS and publish the manifest only after its referenced shards and exemplar files are available. Pages itself hosts the app and the small preview; a full corpus belongs on object storage.

`index.csv` records scenario, track and start time. `--save-slice-map` retains the maintainers' [scenario-to-source-slice mapping](https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/av2_mf_train_scenario_to_slice.csv). Group source slices for train/test evaluation when possible. That map does not identify complete acquisition logs, so it does not prove log-independent validation. `--start-after` can partition S3 keys into distinct extraction jobs, but the current fitter accepts one store; combine stores with a validated merger or process the full split as one stream before claiming a full-data fit.

AV2 is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Attribute `© 2021 Argo AI, LLC` and preserve the noncommercial, share-alike conditions for derived data. The old INTERACTION diagnostic recording is separately labeled and uses CC BY 4.0.
