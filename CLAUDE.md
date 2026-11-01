# Video2Robot

A simulated Unitree G1 humanoid watches a video of a person sorting objects at a table, works out the task, and repeats it in MuJoCo. Solo hackathon project.

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
- Task: sort 3-4 objects into 2 containers.
- Camera: phone on a tripod, fixed in place, with a printed ArUco marker in the shot.
- Compute: Mac for simulation and logic; hackathon or cloud GPU for SAM 2, with results cached to disk.
- Deadline: Wed 4 Nov 2026, 23:59 IST (official).
- Order: A (video to plan to scripted robot), then B (retarget hand path), then C (train a policy), each only if time permits. See `PLAN.md`.
- Vision AI model: **deferred by the user.** Raise it at the start of Phase 4, not before.

## Novelty to protect
1. Measured positions decide what moved and where. A geometric planner works with no AI model, so the hard rule can be met without an API key.
2. A marker turns pixels into real centimetres.
3. The robot's result is checked against the demo and retried. Headline metrics: attempts to success and execution success rate.

The closed-loop check may use only information from the video, never hand-written labels.

## Commands
```bash
scripts/setup.sh                    # create environment, download the G1 model
PYTHONPATH=. .venv/bin/mjpython scripts/watch_live.py   # watch the task live in MuJoCo
PYTHONPATH=. .venv/bin/python scripts/phase1_gate.py --trials 20 --first-seed 500   # run the gate
PYTHONPATH=. .venv/bin/python main.py --video data/raw/video_01.mp4 --step full   # video to robot
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
Phases 0 and 1 complete (20 of 20 on new setups, judged on where objects end up, with a pinned pelvis). Code for Phases 2, 3 and 5 exists but is untested on real video. Waiting for the user's videos. Cup triangle is on branch `cup-task`; the flipped fourth cup is on `cup-flip`. Neither is merged.
