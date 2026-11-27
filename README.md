# Trajalyzer

Interactive exploration of real vehicle trajectories from the [INTERACTION
dataset](https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset)
(CC BY 4.0). Vehicle tracks from the traffic-camera intersection subset are
converted to five-second, 10 Hz ego-relative windows. The Rerun web recording
contains a reproducible 500-window sample from a 5,000-window batch, restricted
to windows that sustain at least 1 m/s throughout. An Astryx
React shell provides lateral-displacement cohorts, playback, timeline scrubbing,
dataset context, and a responsive inspector around the embedded Rerun viewer.
Acceleration and curvature plots share the viewer's timeline cursor. The
selected path is tagged with its nearest one of 64 observed training exemplars;
Rerun and the 2D fallback use the same exemplar colors.
On browsers without WebGL2, a lightweight 2D canvas uses the same sample and
controls. This first subset is one intersection and is not representative of
every driving behavior.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,viewer]'
npm --prefix web ci
pre-commit install --hook-type pre-commit --hook-type pre-push
npm --prefix web run dev
```

From the repository root, one command regenerates all four Rerun recordings,
action sequences, cluster assignments, exemplars, and the canvas sample:

```bash
python scripts/build_recording.py
```

The committed `web/public` assets are ready to use. Regenerate them only when
the dataset sampling, inference, or clustering code changes.

The archive is downloaded into memory. Processed windows are cached under
`data/.cache`; the original ZIP is never stored locally. The core loader only
requires NumPy. You can also run the Python Rerun viewer with
`python -m trajectory_manifold.viewer`.

## Fast feedback

Run `./scripts/check.sh` before publishing. It runs Ruff, Black's format check,
the Python unit tests, ESLint, Prettier, and the Vite production build. Commit
hooks run the fast formatters and linters on changed files; the push hook runs
the full gate. GitHub Actions runs that same script before deploying Pages.
Use `black trajectory_manifold scripts tests` and `npm --prefix web run format`
to apply formatting. `ruff check --fix trajectory_manifold scripts tests`
handles safe Python lint fixes.

## Kinematic actions and exemplars

`trajectory_manifold.kinematics.infer_kinematics` returns arrays for
`[x, y, yaw, speed]`, `[acceleration, curvature]`, and a validity mask.
With velocity `(ẋ,ẏ)` and acceleration `(ẍ,ÿ)` estimated by a local cubic
least-squares fit over seven samples, the flatness equations are
`yaw = atan2(ẏ,ẋ)`, `speed = hypot(ẋ,ẏ)`,
`acceleration = (ẋẍ+ẏÿ)/speed`, and
`curvature = (ẋÿ−ẏẍ)/speed³`. Position and time retain metres and seconds.
At low speed, heading and curvature are not identifiable; this web sample
excludes such windows. These are derivative estimates, not measured controls.

`trajectory_manifold.clustering.fit_exemplars(batch_factory, n_exemplars=64)`
uses scikit-learn MiniBatchKMeans in repeatable batches and chooses an actual
training trajectory nearest each center. `assign_exemplars(model, new_batch)`
then compares new trajectories against those observed exemplars. Features
combine downsampled positions, speed, acceleration, and curvature with fixed
physical scales. The builder holds out entire tracks by hash before fitting,
then assigns all displayed paths, including held-out tracks. It saves a reusable
`data/.cache/exemplar_model.npz` with feature vectors and trajectories.
The four display cohorts remain geometric descriptions, not behavior labels.

For larger collections, pass a repeatable batch factory that reads one chunk
of training tracks at a time. The fit stores one feature batch and the
exemplars, not the full trajectory matrix. Its multiple passes require that
the factory can reopen the source. `iter_trajectory_batches` in the loader
streams INTERACTION windows without assembling the full array. Filter low-speed
windows and split by original track or scene before making training batches.

## Publish

The Pages workflow builds `web/` on pushes to `main`. In the GitHub repository,
select **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The expected project URL is `https://bengravell.github.io/trajalyzer/` after a
successful deployment. The generated recordings and viewer package use matching
Rerun version 0.31.1. The Astryx Core, theme, and CLI share version 0.7.17.

INTERACTION Dataset: Zhan et al., *INTERACTION Dataset: An INTERnational,
Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios
with Semantic Maps*, arXiv:1910.03088 (2019).

## Indexed corpus mode

The landing view now reads `web/public/corpus/manifest.json`, requests at most
32 position shards concurrently in a six-request pool, and displays at most
768 windows. It downloads a selected shard's actions on demand. Filters count
all matching windows in the manifest while the display remains bounded; the
position slider refers to **loaded** paths. The Rerun diagnostic view remains
available on demand and is loaded as a separate JavaScript module.

The checked-in corpus is a 500-window protocol demonstration built from the
previous sample:

```bash
python scripts/build_corpus.py
```

For a larger INTERACTION CSV collection, fit exemplars on training tracks with
the existing scikit-learn pipeline, export the model, and run the native builder:

```bash
python scripts/export_exemplars.py data/.cache/exemplar_model.npz data/.cache/exemplars.json
python scripts/build_native_corpus.py --input /path/to/interaction/csv --model data/.cache/exemplars.json --output /path/to/corpus
```

Omit `--input` to fetch and verify the small official TC archive automatically.
The Rust builder groups one source file at a time through 64 temporary disk
partitions, segments tracks on time gaps and impossible jumps, resamples 50
positions at 10 Hz, excludes windows below 1 m/s, infers acceleration and
curvature with the same seven-point cubic local fit, assigns the nearest
observed exemplar, and writes independent, cacheable shards. It also writes
`index.csv` with source track and start time for provenance. The static shard
protocol stores 208 bytes per path (100 int16 centimetre coordinates, uint32
trajectory ID, float32 exemplar distance); actions occupy a separate 400-byte
file per path. The collection is never held in memory as one array. Native
Rust tests, format, and Clippy run in the Pages workflow.

Optionally export a columnar analytical mirror:

```bash
pip install -e '.[corpus]'
python scripts/export_parquet.py /path/to/corpus /path/to/parquet
```

The Parquet files are partitioned by cohort and cluster with fixed-size Arrow
lists. Read them with Polars `scan_parquet` or Arrow Dataset scanners; the web
app uses the lean binary shards rather than parsing Parquet in a mobile browser.
For a corpus too large for GitHub Pages, upload the corpus directory to an
object store/CDN and set `VITE_CORPUS_BASE_URL` (ending in `/`) in the Pages
build environment. Configure CORS for the Pages origin. The Pages repository
then contains the application and small diagnostic sample, while the CDN serves
immutable corpus shards. Changing the manifest should be deployed atomically
with its shards. The currently bundled source has only one intersection and
must not be interpreted as a million independent behaviors.
