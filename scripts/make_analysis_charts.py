"""Draw the analysis charts for docs/ANALYSIS.md from saved results.

1. Where on the table Task A passes and fails, for one arm and for two arms (simulator).
2. Every real video, grouped, as built / robot failed / refused.

PYTHONPATH=. .venv/bin/python scripts/make_analysis_charts.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path("docs/results")
OUTPUT = Path("docs/analysis")
SEEDS = (8500, 8700, 8800, 8900, 9000)
PASS, FAIL, REFUSED = "#1b7f3b", "#b3261e", "#8a6d00"
TABLE_DEPTH_CM, TABLE_WIDTH_CM = 50, 80
NEIGHBOURS_CM = 9.0

# Real videos, grouped. Counts come from docs/results and docs/JOURNEY.md (one run each).
REAL_GROUPS = [
    ("Task A: plain takes 1-8\n(developed on these)", 7, 0, 0),
    ("Task A: hard takes 1-4\n(after one rule per failure)", 4, 0, 0),
    ("Task A: sunlight", 0, 1, 0),
    ("Task A: cluttered takes 1-6", 5, 1, 0),
    ("Task A: plain takes 9-11\n(fair test, after all fixes)", 3, 0, 0),
    ("Task A: fast, fast + high lift,\ncorners (fair test)", 1, 1, 1),
    ("Task A: quick takes 12-13\n(fair test)", 0, 0, 2),
    ("Task B: all 8 videos", 2, 2, 4),
]


def _runs(mode: str) -> list[dict]:
    runs = []
    for seed in SEEDS:
        path = RESULTS / f"triangle_{mode}_whole_seed_{seed}.json"
        if path.exists():
            runs += json.loads(path.read_text())
    return runs


def _pass_rate_map(runs: list[dict], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Share of nearby layouts that passed, judged by where the tower was."""
    towers = np.array([run["tower"] for run in runs])
    passed = np.array([run["success"] for run in runs], float)
    rate = np.full((len(ys), len(xs)), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            near = np.hypot(towers[:, 0] - x, towers[:, 1] - y) <= NEIGHBOURS_CM
            if near.sum() >= 3:
                rate[i, j] = passed[near].mean()
    return rate


def where_it_passes() -> Path:
    xs = np.linspace(0, TABLE_DEPTH_CM, 60)
    ys = np.linspace(-TABLE_WIDTH_CM / 2, TABLE_WIDTH_CM / 2, 90)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for axis, (mode, title) in zip(axes, [("nearest", "One arm"), ("two", "Two arms")]):
        runs = _runs(mode)
        rate = _pass_rate_map(runs, xs, ys)
        image = axis.imshow(
            rate.T, origin="lower", extent=(-40, 40, 0, 50), cmap="RdYlGn", vmin=0, vmax=1,
            aspect="equal", alpha=0.85,
        )
        for run in runs:
            x, y = run["tower"]
            axis.plot(y, x, "o" if run["success"] else "x", ms=6, mew=2,
                      color="black" if run["success"] else "#7a0000")
        built = sum(run["success"] for run in runs)
        axis.set_title(f"{title}: {built} of {len(runs)} random layouts built", fontsize=13)
        axis.set_xlabel("cm to the robot's left (+) or right (-)")
        axis.set_xlim(40, -40)
        axis.axhline(0, color="#333", lw=3)
        axis.text(0, 2, "robot's edge of the table", ha="center", fontsize=9)
    axes[0].set_ylabel("cm in from the robot's edge (where the triangle was built)")
    figure.colorbar(image, ax=axes, shrink=0.8, label="share of nearby layouts built")
    figure.suptitle(
        "Task A in the simulator: where on the table the triangle can be built\n"
        "dot = built, x = failed; colour = share of layouts within 9 cm that were built",
        fontsize=13,
    )
    path = OUTPUT / "task_a_where_it_passes.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path


def real_video_results() -> Path:
    figure, axis = plt.subplots(figsize=(11, 5.4))
    labels = [group[0] for group in REAL_GROUPS]
    built = np.array([group[1] for group in REAL_GROUPS])
    failed = np.array([group[2] for group in REAL_GROUPS])
    refused = np.array([group[3] for group in REAL_GROUPS])
    rows = np.arange(len(labels))[::-1]
    axis.barh(rows, built, color=PASS, label="built")
    axis.barh(rows, failed, left=built, color=FAIL, label="robot failed (mostly reach)")
    axis.barh(rows, refused, left=built + failed, color=REFUSED, label="refused while reading")
    for row, b, f, r in zip(rows, built, failed, refused):
        axis.text(b + f + r + 0.15, row, f"{b} of {b + f + r}", va="center", fontsize=11)
    axis.set_yticks(rows, labels, fontsize=10)
    axis.set_xlabel("videos")
    axis.set_xlim(0, 16)
    axis.legend(loc="upper right", frameon=False)
    axis.set_title(
        "Every real video, one run each (robot simulated, pelvis pinned)\n"
        "Earlier groups were developed on those same videos; groups marked fair test were not",
        fontsize=12,
    )
    axis.spines[["top", "right"]].set_visible(False)
    path = OUTPUT / "real_video_results.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in (where_it_passes(), real_video_results()):
        print(path)


if __name__ == "__main__":
    main()
