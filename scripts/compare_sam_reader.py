"""Compare SAM 2's glass tracks with the bright-top reader: when each glass left, and where
it ended, against Claude's end points.

PYTHONPATH=. .venv/bin/python scripts/compare_sam_reader.py --videos task2_triangle_sunlight_take1 ...
"""

import argparse
import json

import numpy as np
import yaml

from scripts.glass_from_video import CONFIG_PATH, VideoReadError, read_video
from src.perception.glass_tracker import GlassVideoConfig
from src.simulation.scene import ROOT

POINTS = ROOT / "data/processed/task2_setups/claude_points.json"
MOVED_PIXELS = 40.0
STAY_SECONDS = 1.0
OUTPUT = ROOT / "docs/results/sam2_vs_reader.json"


def sam_departure(track: list, fps: float) -> float | None:
    """When the glass's outline moved away from its start, or vanished, and stayed so."""
    start = next(((x, y) for _, x, y, _ in track if x is not None), None)
    if start is None:
        return None
    moved = [
        x is None or np.hypot(x - start[0], y - start[1]) > MOVED_PIXELS
        for _, x, y, _ in track
    ]
    frames = [frame for frame, *_ in track]
    need = STAY_SECONDS * fps
    for index, is_moved in enumerate(moved):
        if is_moved and all(moved[index:]) and frames[-1] - frames[index] >= need:
            return frames[index] / fps
    return None


def sam_end(track: list, fps: float) -> tuple[float, float] | None:
    """The outline's top over the last second, where it ended."""
    last = track[-1][0]
    tail = [(x, y) for frame, x, y, _ in track if x is not None and frame >= last - fps]
    if not tail:
        return None
    return float(np.median([p[0] for p in tail])), float(np.median([p[1] for p in tail]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", nargs="+", required=True)
    args = parser.parse_args()
    saved = json.loads(POINTS.read_text())
    config = GlassVideoConfig(**yaml.safe_load(CONFIG_PATH.read_text())["real_video"])
    report = []
    for stem in args.videos:
        sam = json.loads((ROOT / "data/processed" / stem / "sam_tracks.json").read_text())
        fps = sam["fps"]
        ends = saved[f"{stem}@end"]
        row = {"video": stem, "sam_seconds": sam["seconds"], "glasses": []}
        try:
            demo, _ = read_video(
                ROOT / "data/raw" / f"{stem}.mov", config,
                [tuple(p) for p in saved[stem]], [tuple(p) for p in ends],
            )  # fmt: skip
            reader_times = sorted(demo.moved_at)
        except VideoReadError as error:
            reader_times, row["reader_refused"] = [], str(error)
        for glass, track in sam["tracks"].items():
            left = sam_departure(track, fps)
            end = sam_end(track, fps)
            nearest_end = (
                None if end is None
                else round(min(float(np.hypot(end[0] - q[0], end[1] - q[1])) for q in ends), 1)
            )  # fmt: skip
            row["glasses"].append(
                {
                    "glass": int(glass),
                    "sam_left_at": None if left is None else round(left, 1),
                    "sam_end_pixels_from_nearest_claude_end": nearest_end,
                    "outline_lost_frames": sum(1 for _, x, _, _ in track if x is None),
                }
            )
        row["reader_left_at"] = [round(t, 1) for t in reader_times]
        report.append(row)
        print(json.dumps(row))
    OUTPUT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
