"""Train the small outcome model on saved Task A runs, and test it on layouts it never saw.

Every layout was run both ways, with one arm (the nearest) and with two arms, so for each
test layout we know what each choice really gave. The model picks one way per layout, and is
compared with always choosing one way, and with a simple hand rule.

PYTHONPATH=. .venv/bin/python scripts/outcome_test.py
"""

import json

from src.reasoning.outcome_model import MODES, features, train
from src.simulation.scene import ROOT

RESULTS = ROOT / "docs/results"
TRAIN_SEEDS = (8500, 8800, 8900)
TEST_SEEDS = (8700, 9000)
RULE_LIMITS_CM = (6, 8, 10, 12, 14, 16, 18, 20)


def load(seed: int) -> list[dict]:
    """One entry per layout, with the outcome of each way of doing it."""
    runs = {
        mode: json.loads((RESULTS / f"triangle_{mode}_whole_seed_{seed}.json").read_text())
        for mode in MODES
    }
    layouts = []
    for nearest, two in zip(runs["nearest"], runs["two"], strict=True):
        if nearest["starts"] != two["starts"] or nearest["tower"] != two["tower"]:
            raise ValueError(f"Seed {seed}: the two files don't hold the same layouts")
        layouts.append(
            {
                "starts": nearest["starts"],
                "tower": nearest["tower"],
                "nearest": nearest["success"],
                "two": two["success"],
            }
        )
    return layouts


def score(layouts: list[dict], choose) -> int:
    return sum(layout[choose(layout)] for layout in layouts)


def rule(limit: float):
    return lambda layout: "two" if abs(layout["tower"][1]) < limit else "nearest"


def main() -> None:
    training = [layout for seed in TRAIN_SEEDS for layout in load(seed)]
    testing = [layout for seed in TEST_SEEDS for layout in load(seed)]
    rows = [
        (features(layout["starts"], layout["tower"], mode), layout[mode])
        for layout in training
        for mode in MODES
    ]
    model = train(rows)
    best_limit = max(RULE_LIMITS_CM, key=lambda limit: score(training, rule(limit)))
    choices = {
        "always one arm (nearest)": lambda layout: "nearest",
        "always two arms": lambda layout: "two",
        f"hand rule: two arms if the tower is within {best_limit} cm of the middle": rule(
            best_limit
        ),
        "outcome model": lambda layout: model.best_mode(layout["starts"], layout["tower"]),
        "best possible (knowing the answers)": lambda layout: (
            "two" if layout["two"] else "nearest"
        ),
    }
    report = {
        "training_layouts": len(training),
        "test_layouts": len(testing),
        "rule_limit_cm_chosen_on_training": best_limit,
    }
    print(f"Trained on {len(training)} layouts, tested on {len(testing)} new ones")
    for name, choose in choices.items():
        built = score(testing, choose)
        report[name] = built
        print(f"  {name}: {built} of {len(testing)}")
    (RESULTS / "outcome_model_test.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
