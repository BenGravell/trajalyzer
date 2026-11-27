# Trajalyzer

Interactive exploration of real vehicle trajectories from the [INTERACTION
dataset](https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset)
(CC BY 4.0). Vehicle tracks from the traffic-camera intersection subset are
converted to five-second, 10 Hz ego-relative windows. The Rerun web recording
contains a reproducible 500-window sample from a 5,000-window batch. An Astryx
React shell provides lateral-displacement cohorts, playback, timeline scrubbing,
dataset context, and a responsive inspector around the embedded Rerun viewer.
On browsers without WebGL2, a lightweight 2D canvas uses the same sample and
controls. This first subset is one intersection and is not representative of
every driving behavior.

## Run locally

```bash
cd web
npm ci
npm run dev
```

From the repository root, regenerate the four `web/public/*.rrd` recordings
and the canvas sample from the official 4.5 MB archive:

```bash
python -m pip install numpy rerun-sdk==0.31.1
for cohort in all center positive negative; do
  python scripts/build_recording.py --cohort "$cohort"
done
```

The archive is downloaded into memory. Processed windows are cached under
`data/.cache`; the original ZIP is never stored locally. The core loader only
requires NumPy. You can also run the Python Rerun viewer with
`python -m trajectory_manifold.viewer`.

## Publish

The Pages workflow builds `web/` on pushes to `main`. In the GitHub repository,
select **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The expected project URL is `https://bengravell.github.io/trajalyzer/` after a
successful deployment. The generated recordings and viewer package use matching
Rerun version 0.31.1. The Astryx Core, theme, and CLI share version 0.7.17.

INTERACTION Dataset: Zhan et al., *INTERACTION Dataset: An INTERnational,
Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios
with Semantic Maps*, arXiv:1910.03088 (2019).
