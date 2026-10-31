# Video2Robot

A simulated Unitree G1 humanoid watches a video of a person sorting objects at a table, works out the task, and repeats it in MuJoCo. Solo hackathon project.

## Read these first
- `PLAN.md`: phases, test gates and live status. Check it to see what is next.
- `CONVERSATION_LOG.md`: why decisions were made.
- `task.md`: original spec. Never edit it.

## How to work with this user
- The user is new to Python and robotics. Explain results and robotics ideas in plain language.
- Finish each phase by passing its gate, then explain the result and update `PLAN.md`, `CONVERSATION_LOG.md` and the status below.
- Report misses honestly. Anything simplified in simulation must be labelled in results.
- Commit locally after each phase. Do not push unless asked.

## Decisions
- Robot: Unitree G1 from MuJoCo Menagerie, standing at a table. This replaces the 6-DOF arm in `task.md` because the organisers prefer a humanoid.
- Task: sort 3-4 objects into 2 containers.
- Camera: phone on a tripod, fixed in place, with a printed ArUco marker in the shot.
- Compute: Mac for simulation and logic; hackathon or cloud GPU for SAM 2, with results cached to disk.
- Deadline: unconfirmed. Plan for Nov 4, 2026.
- Vision AI model: **deferred by the user.** Raise it at the start of Phase 4, not before.

## Novelty to protect
1. The AI decides what moved; measured positions decide where.
2. A marker turns pixels into real centimetres.
3. The robot's result is checked against the demo and retried. Headline metrics: attempts to success and execution success rate.

The closed-loop check may use only information from the video, never hand-written labels.

## Commands
```bash
scripts/setup.sh                    # create environment, download the G1 model
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
Phase 0 complete. Next: Phase 1.
