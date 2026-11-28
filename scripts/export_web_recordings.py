"""Export bounded, independently loadable Rerun recordings for the static web explorer."""

import argparse
import colorsys
import gzip
import hashlib
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from uuid import uuid4

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

ROOT = Path(__file__).resolve().parents[1]
MAX_SHARDS = 6
MAX_PATHS = 768


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def read_shard(root, shard, frames, scale):
    positions = root / shard["path"]
    rows = np.fromfile(
        positions,
        dtype=np.dtype([("xy", "<i2", (frames, 2)), ("id", "<u4"), ("distance", "<f4")]),
    )
    if len(rows) != shard["count"]:
        raise ValueError(f"Invalid trajectory shard: {positions}")
    actions = positions.with_suffix(".actions.bin")
    values = np.fromfile(actions, dtype="<f4").reshape(len(rows), frames, 2)
    return [
        (int(row["id"]), shard["cluster"], float(row["distance"]), xy, action)
        for row, xy, action in zip(rows, rows["xy"].astype(np.float32) * scale, values, strict=True)
    ]


def log_motion(recording, entity, paths, color, time, *, highlight=False):
    count, frames, _ = paths.shape
    recording.log(
        entity,
        rr.LineStrips2D(
            list(paths),
            colors=[*color, 230 if highlight else 100],
            radii=rr.Radius.ui_points(2 if highlight else 1),
        ),
        rr.Points2D.from_fields(colors=color, radii=rr.Radius.ui_points(4 if highlight else 2.5)),
        static=True,
    )
    positions = paths.transpose(1, 0, 2).reshape(-1, 2)
    recording.send_columns(
        entity,
        indexes=[time],
        columns=rr.Points2D.columns(positions=positions).partition(lengths=[count] * frames),
    )


def write_recording(destination, rows, exemplars, frames, sample_rate):
    time = rr.TimeColumn("elapsed", duration=np.arange(frames) / sample_rate)
    groups = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)
    metrics = (("acceleration", "Acceleration (m/s²)"), ("curvature", "Curvature (m⁻¹)"))
    plots = [
        rrb.TimeSeriesView(
            name=title,
            origin=f"/{metric}",
            plot_legend=rrb.PlotLegend(visible=False),
            overrides={
                f"/{metric}/{cluster}": rr.SeriesLines.from_fields(visible_series=True, widths=1)
                for cluster in groups
            },
        )
        for metric, title in metrics
    ]
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(
                name="Trajectories (m)",
                origin="/motion",
                contents=["+ /motion/batches/**", "+ /motion/origin"],
            ),
            rrb.Horizontal(*plots),
            row_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="elapsed", play_state="paused", loop_mode="off"),
    )
    with rr.RecordingStream(
        destination.name.removesuffix(".rrd.tmp"), recording_id=str(uuid4())
    ) as recording:
        recording.save(destination, default_blueprint=blueprint)
        recording.log(
            "/motion/origin",
            rr.Points2D([[0, 0]], colors=[220, 220, 220], radii=rr.Radius.ui_points(3)),
            static=True,
        )
        for cluster in sorted({row[1] for row in rows if row[1] < len(exemplars)}):
            xy = exemplars[cluster] * [1, -1]
            recording.log(
                f"/motion/exemplars/{cluster}",
                rr.LineStrips2D([xy], colors=[238, 238, 238], radii=rr.Radius.ui_points(2)),
                static=True,
            )
        for cluster, group in groups.items():
            color = [
                round(value * 255)
                for value in colorsys.hls_to_rgb(cluster * 0.381966 % 1, 0.64, 0.72)
            ]
            ids = [row[0] for row in group]
            paths = np.stack([row[3] for row in group]).astype(np.float32)
            paths[:, :, 1] *= -1
            log_motion(recording, f"/motion/batches/{cluster}", paths, color, time)
            for i, path_id in enumerate(ids):
                log_motion(
                    recording,
                    f"/motion/paths/{cluster}/{path_id}",
                    paths[i : i + 1],
                    color,
                    time,
                    highlight=True,
                )
            actions = np.stack([row[4] for row in group], axis=1)
            for dimension, (metric, _) in enumerate(metrics):
                values = actions[:, :, dimension].reshape(-1)
                if not np.isfinite(values).any():
                    continue
                entity = f"/{metric}/{cluster}"
                recording.log(
                    entity,
                    rr.SeriesLines(colors=[*color, 125], names=[str(path_id) for path_id in ids]),
                    static=True,
                )
                recording.send_columns(
                    entity,
                    indexes=[time],
                    columns=rr.Scalars.columns(scalars=values).partition(
                        lengths=[len(ids)] * frames
                    ),
                )
        recording.flush()


def write_page(directory, key, rows, exemplars, frames, sample_rate, inputs):
    if len(rows) > MAX_PATHS:
        rows = [
            rows[((2 * index + 1) * len(rows)) // (2 * MAX_PATHS)] for index in range(MAX_PATHS)
        ]
    fingerprint = hashlib.sha256(Path(__file__).read_bytes())
    for path in inputs:
        fingerprint.update(path.read_bytes() if path.exists() else b"missing")
    fingerprint.update(rr.__version__.encode())
    revision = fingerprint.hexdigest()[:16]
    recording = directory / f"{key}-{revision}.rrd-gzip"
    if not recording.exists():
        raw = directory / f"{key}.rrd.tmp"
        temporary = directory / f"{key}.gz.tmp"
        try:
            write_recording(raw, rows, exemplars, frames, sample_rate)
            with (
                raw.open("rb") as source,
                temporary.open("wb") as target,
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=target, compresslevel=3, mtime=0
                ) as compressed,
            ):
                shutil.copyfileobj(source, compressed)
            os.replace(temporary, recording)
        finally:
            raw.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
    metadata = {
        "recording": recording.name,
        "rows": [[path_id, cluster, distance] for path_id, cluster, distance, *_ in rows],
    }
    return metadata


def export_corpus_page(output, key, corpus, chosen, exemplars, frames, rate, scale, inputs):
    rows = []
    for shard in chosen:
        path = corpus / shard["path"]
        inputs.extend((path, path.with_suffix(".actions.bin")))
        rows.extend(read_shard(corpus, shard, frames, scale))
    metadata = write_page(output, key, rows, exemplars, frames, rate, inputs)
    page = f"{key}.json"
    write_json(output / page, metadata)
    return page


def export_corpus(corpus, output, workers):
    manifest_file = corpus / "manifest.json"
    model_file = corpus / "model-meta.json"
    exemplar_file = corpus / "exemplars.bin"
    source_file = corpus / "source.json"
    manifest = read_json(manifest_file)
    model = read_json(model_file) if model_file.exists() else {}
    source = read_json(source_file) if source_file.exists() else {}
    frames = manifest["frames"]
    rate = manifest["sample_rate_hz"]
    scale = manifest["position_scale_m"]
    exemplars = (
        np.fromfile(exemplar_file, dtype="<i2").reshape(-1, frames, 2) * model["position_scale_m"]
        if exemplar_file.exists()
        else []
    )
    shards = manifest["shards"]
    clusters = sorted({shard["cluster"] for shard in shards})
    partitions = {"all": shards} | {
        str(cluster): [shard for shard in shards if shard["cluster"] == cluster]
        for cluster in clusters
    }
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for group, eligible in partitions.items():
            pages = (len(eligible) + MAX_SHARDS - 1) // MAX_SHARDS
            pending[group] = [
                pool.submit(
                    export_corpus_page,
                    output,
                    f"av2-{group}-{page}",
                    corpus,
                    eligible[page::pages],
                    exemplars,
                    frames,
                    rate,
                    scale,
                    [model_file, exemplar_file, manifest_file],
                )
                for page in range(pages)
            ]
        groups = {
            group: {
                "count": sum(shard["count"] for shard in partitions[group]),
                "pages": [result.result() for result in results],
            }
            for group, results in pending.items()
        }
    return {
        "dataset": source.get("dataset", "Indexed trajectories"),
        "frames": frames,
        "sampleRate": rate,
        "count": manifest["count"],
        "source": source,
        "trainingWindows": model.get("training_windows"),
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "web/public/corpus/recordings")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = export_corpus(args.corpus, args.output, args.jobs)
    write_json(args.output / "index.json", index)
    used = {
        read_json(args.output / page)["recording"]
        for group in index["groups"].values()
        for page in group["pages"]
    }
    for recording in args.output.glob("*.rrd-gzip"):
        if recording.name not in used:
            recording.unlink()
    print(f"Prepared {len(used)} Rerun recordings in {args.output}")


if __name__ == "__main__":
    main()
