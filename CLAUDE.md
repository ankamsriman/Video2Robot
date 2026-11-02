# Video2Robot

A simulated Unitree G1 humanoid watches a video of a person stacking steel glasses at a table, works out the task, and repeats it in MuJoCo. Solo hackathon project.

## Read these first
- `PLAN.md`: phases, test gates and live status. Check it to see what is next.
- `CONVERSATION_LOG.md`: why decisions were made.
- `task.md`: the official challenge text. Never edit it.
- `notes.md`: full from-scratch explanation of the project for the user. Keep it accurate as things change.

## How to work with this user
- The user is new to Python and robotics. Explain results and robotics ideas in plain language.
- Finish each phase by passing its gate, then explain the result and update `PLAN.md`, `CONVERSATION_LOG.md` and the status below.
- Report misses honestly. Anything simplified in simulation must be labelled in results.
- Commit locally after each phase. Do not push unless asked.

## Decisions
- Robot: Unitree G1 from MuJoCo Menagerie, standing at a table, with a pinned pelvis. Results must say so.
- Project location: stays on the iCloud-synced Desktop by user choice, so the code survives losing the laptop. If imports become slow, check for dataless files.
- Tasks: Task 2, a triangle of three steel glasses, and Task 3, a fourth glass flipped on top. Sorting from real video was dropped by the user, who has no blocks.
- Camera: fixed in place, with no marker. The table's corners and measured size (80 x 50 cm) give real positions.
- Compute: Mac for simulation and logic; hackathon or cloud GPU for SAM 2, with results cached to disk.
- Deadline: Wed 4 Nov 2026, 23:59 IST (official).
- Order: A (video to plan to scripted robot), then B (retarget hand path), then C (train a policy), each only if time permits. See `PLAN.md`.
- Vision AI model: **deferred by the user.** Raise it at the start of Phase 4, not before.

## Novelty to protect
1. Measured positions decide what moved and where. A geometric planner works with no AI model, so the hard rule can be met without an API key.
2. The table's corners and measured size turn pixels into real centimetres, with no marker. Tops above the table are corrected for height.
3. Identical steel glasses are told apart by where they rest, which keeps the order.
4. The robot's result is checked against the demo. Sorting also retries; retries didn't help the glasses, so they aren't used there.

The closed-loop check may use only information from the video, never hand-written labels.

## Commands
```bash
scripts/setup.sh                    # create environment, download the G1 model
PYTHONPATH=. .venv/bin/mjpython scripts/watch_live.py   # watch the task live in MuJoCo
PYTHONPATH=. .venv/bin/python scripts/phase1_gate.py --trials 20 --first-seed 500   # run the gate
PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task2_triangle_right_take1.mov   # real video to robot
PYTHONPATH=. .venv/bin/python scripts/area_test.py triangle --trials 30 --seed 8123   # what the robot can do
PYTHONPATH=. .venv/bin/python main.py --video data/raw/video_01.mp4 --step full   # earlier sorting pipeline
PYTHONPATH=. .venv/bin/python scripts/check_video.py --video data/raw/video_01.mp4   # check a test video
PYTHONPATH=. .venv/bin/python scripts/stress_retry.py   # measure what retries add
.venv/bin/pytest                    # run tests
.venv/bin/pytest --cov=src --cov-report=term-missing
```

## Constraints
- M1 Mac with 8 GB RAM and under 10 GB of free disk. Do not download large models locally without checking space first.
- Isaac Sim and Isaac Lab cannot run here.
- All tunable numbers live in `configs/default.yaml`.
- Use frozen dataclasses or frozen Pydantic models, type annotations, and `logging` rather than `print` in `src/`.
- Target 80% test coverage on logic modules, writing tests first. Check simulator visuals and model wrappers through phase gates.

## Current status
Real-video pipeline works on the user's steel glasses, with no marker: `scripts/glass_from_video.py`. Task 2 from `task2_triangle_right_take1` builds the triangle (one run). Task 3 from `task3_flip_right_take1` builds and flips with no glass moved (one run): the layout picks the arm and standing spot, and the robot starts there, since it can't walk. Standing-spot tests: 19 of 21 on the video's layout wobbled, 25 of 30 on random layouts (`scripts/stand_test.py`). Task 2 uses the nearest arm: 24 of 30 against 10 of 30 for the right arm on left-side layouts, no gain on spread layouts. Robot area tests: 28 of 30 for each task, pinned pelvis. Eight later Task 2 takes: 0 succeeded with the reader as it stood; after adding a rule for an unseen top glass, 4 were read and 2 succeeded with the left arm (takes 2 and 5, not a fair test of that rule). Four are refused because a starting glass is missed. Held-out takes haven't been run. Sorting (marker and colours) is earlier work, never run on real footage. The repo is private until the user says otherwise.
