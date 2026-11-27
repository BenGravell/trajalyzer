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
