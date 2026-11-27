"""Interactive Rerun viewer for the INTERACTION ego-trajectory sample.

Install:  python -m pip install numpy rerun-sdk
Run:      python -m trajectory_manifold.viewer
Phone:    python -m trajectory_manifold.viewer --host 192.168.1.23

The browser viewer offers pan, zoom, point inspection, and timeline playback.
Use --host with this computer's LAN address to print a link for a phone on
the same network. Both ports (9090 and 9876 by default) must be reachable.
"""

from __future__ import annotations

import argparse
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

import numpy as np

try:
    from .loader import load_trajectories
except ImportError:  # Also allow: python trajectory_manifold/viewer.py
    from loader import load_trajectories


def _colors(trajectories: np.ndarray) -> np.ndarray:
    """Color by final lateral displacement, a visual cue rather than a label."""
    lateral = trajectories[:, -1, 1]
    colors = np.empty((len(trajectories), 3), dtype=np.uint8)
    colors[:] = (54, 211, 192)       # near center
    colors[lateral > 3] = (255, 174, 87)
    colors[lateral < -3] = (166, 144, 255)
    return colors


def _log_recording(rr, rrb, batch: np.ndarray, sample_rate: int, live: bool) -> None:
    count, frames, _ = batch.shape
    colors = _colors(batch)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial2DView(origin="/trajectories", name="Ego trajectories (metres)"),
    ))
    # One batched static entity keeps thousands of paths cheap to draw.
    rr.log(
        "trajectories/paths",
        rr.LineStrips2D(
            strips=list(batch), colors=colors, radii=rr.Radius.ui_points(1.2),
        ),
        static=True,
    )
    rr.log(
        "trajectories/origin", rr.Points2D([[0, 0]],
                                             colors=[[255, 255, 255]],
                                             radii=rr.Radius.ui_points(5)),
        static=True,
    )

    if live:
        # Stream simulated vehicle positions at actual sample cadence.
        start = time.monotonic()
        for frame in range(frames):
            rr.set_time("elapsed", duration=frame / sample_rate)
            rr.log("trajectories/vehicles", rr.Points2D(
                batch[:, frame], colors=colors, radii=rr.Radius.ui_points(3),
            ))
            time.sleep(max(0.0, start + (frame + 1) / sample_rate - time.monotonic()))
    else:
        # Preload all positions in one columnar operation; the Rerun timeline
        # then gives smooth scrubbing and repeated playback without Python work.
        flattened = np.ascontiguousarray(batch.transpose(1, 0, 2)).reshape(-1, 2)
        all_colors = np.tile(colors, (frames, 1))
        rr.send_columns(
            "trajectories/vehicles",
            indexes=[rr.TimeColumn("elapsed", duration=np.arange(frames) / sample_rate)],
            columns=rr.Points2D.columns(
                positions=flattened, colors=all_colors,
            ).partition(lengths=[count] * frames),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-trajectories", type=int, default=5000,
                        help="batch size loaded by the data loader (default: 5000)")
    parser.add_argument("--display", type=int, default=500,
                        help="maximum paths shown; reduce for slower phones (default: 500)")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="stream positions at sample rate; default preloads for playback")
    parser.add_argument("--native", action="store_true",
                        help="open the native desktop viewer instead of the web viewer")
    parser.add_argument("--host", default="127.0.0.1",
                        help="host/IP visible to your browser or phone (for example LAN IP)")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--no-open", action="store_true",
                        help="print the URL without opening a browser tab")
    args = parser.parse_args()
    if args.display < 1 or args.num_trajectories < 1:
        parser.error("--display and --num-trajectories must be positive")

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise SystemExit("Install the viewer with: python -m pip install rerun-sdk") from exc

    batch, metadata = load_trajectories(
        root=args.data_root, num_trajectories=args.num_trajectories,
        duration=args.duration, sample_rate=args.sample_rate, seed=args.seed,
        coordinate_frame="ego", cache=not args.no_cache, refresh=args.refresh,
    )
    # Deterministic display subset. Loading and display sizes are independent.
    if len(batch) > args.display:
        indices = np.random.default_rng(args.seed).choice(
            len(batch), size=args.display, replace=False,
        )
        batch = batch[indices]
        metadata = [metadata[i] for i in indices]
    print(f"Displaying {len(batch)} windows from "
          f"{len({(m['source_file'], m['track_id']) for m in metadata})} tracks")

    rr.init("ego_trajectory_manifold")
    if args.native:
        rr.spawn(port=args.grpc_port)
    else:
        rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False)
        source = f"rerun+http://{args.host}:{args.grpc_port}/proxy"
        url = f"http://{args.host}:{args.web_port}/?url={quote(source, safe='')}"
        print(f"Open the viewer: {url}")
        if not args.no_open:
            webbrowser.open(url)

    _log_recording(rr, rrb, batch, args.sample_rate, args.live)
    print("Use the timeline to scrub or play. Press Ctrl+C to stop the server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
