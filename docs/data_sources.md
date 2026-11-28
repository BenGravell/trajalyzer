# Source data for million-window experiments

## Primary source: Argoverse 2 Motion Forecasting

Official source: <https://www.argoverse.org/av2.html>. Public, unsigned S3 bucket:
`s3://argoverse/datasets/av2/motion-forecasting/` in `us-east-1`.
The source has 250,000 eleven-second scenarios, 10 Hz object tracks, and six
cities. Each scenario contains a Parquet table with track ID, object type,
position, velocity, heading, timestep, and city. The official downloadable
archives total 58 GB. Use the per-scenario public S3 objects to start small,
or the official archives for a full production extraction. No account is
required for unsigned bucket access.

The AV track has 110 samples in the scenarios inspected. A complete 11-second
track can yield **at most seven** five-second windows at one-second stride;
near-zero-speed filtering reduces this. In a preliminary, lexicographically
selected 35-scenario pilot, 150/245 candidate AV windows and 692/2892 other
vehicle windows survived the sustained 1 m/s test. These are measurements of
that pilot, not corpus-wide estimates. The AV-only corpus may approach a
million eligible windows; all dynamic vehicle tracks offer a larger pool.
Measure the actual count across a representative sample before projecting
capacity or choosing a hosting budget.

### Stream without a raw local download

```bash
pip install -e '.[corpus]'
python scripts/stream_av2.py --split train --tracks av --limit-scenarios 1000 --output data/av2-1k
python scripts/stream_av2.py --split train --tracks vehicles --all --save-slice-map --output /data/av2-train
```

The adapter lists public S3 objects with continuation tokens, fetches and
decodes a bounded number of Parquet scenarios concurrently using PyArrow, and
streams normalized rows into one long-lived Rust process. Rust holds one
scenario at a time, filters motion, infers actions, assigns exemplars, and
writes the existing independently fetchable shard format. Raw Parquet is not
retained. `--start-after` can begin a separate batch after an S3 object key;
use a distinct output directory for each batch. For large-scale production,
run on compute near the S3 bucket and publish derived shards to object storage;
the GitHub Pages site can point to that CDN via `VITE_CORPUS_BASE_URL`.

The checked-in exemplar features were fit to INTERACTION. AV2 assignments
against them are provisional diagnostics. Fit new exemplars on AV2 training
tracks before presenting the clusters as a representative AV2 palette.
The object schema supports AV-only or all vehicle/bus tracks. Keep trajectory
IDs, scenario IDs, source split, city, track IDs, and window start time for
analysis. Separate overlapping windows from the same source scenario in any
train/test evaluation.

Argoverse 2 data is **CC BY-NC-SA 4.0**. A published derived corpus needs the
license, attribution `© 2021 Argo AI, LLC`, and a noncommercial, share-alike
use consistent with the source terms. The source Python API is MIT licensed.
Do not silently combine an AV2-derived public corpus with the current CC BY
INTERACTION demonstration without labeling each source and its terms.

The Argoverse maintainers have published a training-scenario-to-source-slice
CSV at
<https://argoverse.s3.amazonaws.com/datasets/av2/motion-forecasting/av2_mf_train_scenario_to_slice.csv>.
Save it with `--save-slice-map` for more conservative train grouping. Their
clarification says it does **not** identify complete acquisition logs, and
validation/test source-log independence is not guaranteed:
<https://github.com/argoverse/av2-api/issues/341>.

## Other sources

| Source | Scale and format | Access and role |
| --- | --- | --- |
| Full INTERACTION | 11 locations, CSV archives; the full drone archive is 122 MB | CC BY 4.0 on its [official Hugging Face repository](https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset/tree/main). Expand this first for a lower-friction mixed-maneuver baseline, but count eligible windows before assuming million scale. |
| Waymo Open Motion | 103,354 segments of 20 s and 10.8 million tracked objects; sharded TFRecord/protobuf | Largest rich follow-on, but registration and license acceptance are required and the decoding stack is heavier. [Official overview](https://waymo.com/open/about/). |
| highD | 110,500 highway vehicles; CSV | Manual access request and limited road/maneuver diversity. Useful for highway studies, not the primary million-window source. [Official access page](https://levelxdata.com/highd-dataset/). |

For intrinsic-dimensionality estimates, report the number of source scenarios,
source slices, original tracks, and derived overlapping windows separately.
The million-window target is a computational scale goal, not a claim of a
million independent driving examples.
