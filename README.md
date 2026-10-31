# Video2Robot

Record yourself doing a tabletop task on a phone, and a simulated Unitree G1 humanoid repeats it in MuJoCo.

> **Draft.** Real-video results are still to come. Numbers below come from simulation and a rendered test video.

## How it works

```
phone video
  -> follow each coloured object, and convert pixels to table positions using a printed marker
  -> work out the plan: which objects moved, where to, and in what order
  -> G1 carries out the plan in physics, gripping by friction
  -> measure the result against the video's plan, and retry failed steps
```

Your recording decides the task, the order and the starting positions. Without it, the robot has nothing to do.

## Design choices

| Choice | Why |
|---|---|
| Copy what happens to objects, not the hand | A human hand and a G1 hand differ, so the outcome is what transfers. This idea comes from Robot See Robot Do |
| Positions decide the plan | SeeDo reports spatial errors as its largest failure source. Measuring positions avoids asking a model to judge them |
| Fixed camera and printed marker | Gives real distances from one cheap phone |
| Judge results by where objects end up | The goal is the object's position, not the hand's |
| Retry from measured failures | Gives a number to improve: attempts to success |

## Results

All of these use hand-written plans unless stated, and a **pinned pelvis**.

| Test | Result |
|---|---|
| Sorting 3 blocks, new random setups (shifts of up to 3 cm) | 20 of 20 |
| Rendered video to plan: block position error | 0.1-0.7 cm |
| Rendered video through to robot | Correct plan, task completed |
| Cup triangle (separate `cup-task` branch), new setups | 30 of 30 |
| Cup triangle with 15 g cups | 10 of 10 |
| Real phone video | **Not yet run** |

## What worked and what didn't

**Worked:**
- Gripping by friction alone, with no shortcut attaching objects to the hand.
- Cup stacking, which went better than expected.

**Didn't work, or needed fixing:**
- **First full test: 0 of 10.** The arm snapped to its pose at the start and knocked objects over. `notes.md` section 10 lists every bug found.
- **Unpinned robot:** it balances while idle but falls over reaching far, so the pelvis stays pinned.
- **My first check was wrong:** it reported 16 of 20, but in all 4 failures the object had landed correctly. It was judging the hand. I changed it and retested on new setups.

## Limitations

- Pelvis pinned, so there's no real balance.
- Three-finger Dex3 hand, not the five-finger hands the challenge suggests.
- Simplified objects: simple rigid shapes, and cups heavier than real paper cups.
- The simulated robot knows exact object positions once the scene is built.
- Colour tracking needs distinct, matte colours.
- Reach is small: a patch of roughly 20 x 40 cm, as the arm is short and the pelvis is fixed.

## Running it

```bash
scripts/setup.sh
.venv/bin/pytest

# Watch the robot do the task live
PYTHONPATH=. .venv/bin/mjpython scripts/watch_live.py

# Run on a video, after setting colours in configs/default.yaml
PYTHONPATH=. .venv/bin/python main.py --video data/raw/video_01.mp4 --step full
```

Runs on a laptop CPU. It was developed on an M1 Mac with 8 GB of RAM.

## Data

To be added: recorded video samples in `data/raw/sample/`, and how they were collected. `docs/RECORDING_GUIDE.md` is the collection procedure.

## More detail

`notes.md` explains the whole project from scratch.

## Credits

Built on MuJoCo, MuJoCo Menagerie, mink, MediaPipe and OpenCV. Ideas from SeeDo (VLM See, Robot Do) and Robot See Robot Do. Built with help from Claude Code.
