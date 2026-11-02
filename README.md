# Video2Robot

I film myself stacking steel glasses at a table. A simulated Unitree G1 watches the footage, works out which glass moved where and in what order, and repeats the task in MuJoCo.

![Task 2: my video beside the robot, at the end of the task](docs/videos/task2_final.jpg)

| Task | From my video | Video |
|---|---|---|
| **Task 2:** three glasses into a triangle | Order and positions come from the video, and the robot builds it | [Side by side](docs/videos/task2_video_to_robot.mp4) |
| **Task 3:** a fourth glass flipped and stood on top, opening up | The video gives the order and the flip, but positions had to be **moved to fit the robot** | [Side by side](docs/videos/task3_video_to_robot_fitted.mp4) |

![Task 3: my video beside the robot, with positions moved to fit](docs/videos/task3_final.jpg)

## How my recording drives the robot

```
phone video of steel glasses, no marker
  -> table corners and measured table size: pixels to real centimetres
  -> bright closed glass tops: where each glass rests, and when it leaves
  -> pair each spot left with the next new spot: the moves, in order
  -> dark top on the finished tower: the last glass was flipped
  -> G1 repeats it in physics, gripping by friction
  -> check where the glasses really ended up
```

Without the recording, the robot has no starting positions, no tower position and no order. The robot's grip and flip are programmed, and they aren't learned from my hand. So it copies **what happened and where**, not how my hand moved.

## Design choices

| Choice | Why |
|---|---|
| Copy what happens to objects, not the hand | A human hand and the G1's three-finger hand differ, so the outcome is what transfers. This idea comes from Robot See Robot Do |
| Identical steel glasses | Colour tracking can't tell them apart, so a glass is identified by where it rests. The order survives, with no labels and no large vision model |
| The table as the ruler, no marker | Its four corners and measured size (80 x 50 cm) give real distances and the camera's position, with nothing printed |
| Correct for height | A glass top is 11.5 cm above the table, and a stacked one 23 cm. Without this, positions come out several centimetres wrong |
| Refuse rather than guess | If a video doesn't read as a clear task, the system says so and stops |
| Report every compromise | Positions moved to fit the robot are listed in cm, in each report and on screen |

## Results

Everything runs in MuJoCo with a **pinned pelvis**. The simulated glass matches my measured size (11.5 cm tall, 7.3 cm rim, 5.5 cm closed end), but it is solid and rigid, and its weight is a guess.

### From my real videos

| Video | Reading | Robot |
|---|---|---|
| `task2_triangle_right_take1` | 3 of 3 moves in order | **Built the triangle**, tower 0.2 cm from where the video put it |
| `task3_flip_right_take1`, positions moved to fit | 4 of 4 moves in order, flip recognised | **Built and flipped**, after moving glass starts by 9-33 cm and the tower by 31 cm |
| `task3_flip_right_take1`, exact positions | As above | **Failed:** dropped the fourth glass in the twist |
| `task2_triangle_left_take1`, my first video | **Refused:** 8 spots found, 6 expected | Not run. An earlier version read it, and the robot couldn't reach a glass on its far left |
| `task3_flip_left_take1` | **Refused:** the order didn't make sense | Not run |

**How far to trust this.** The Task 2 success is one run of one video, and I set the reading's settings while looking at these same videos. It hasn't been tested on a take kept apart from tuning.

### Checks on the reading

I have no ground-truth ruler measurements for the glass positions, so these are indirect:

| Check | Result |
|---|---|
| The two lower glasses touch, so their tops should be about one rim width apart (7.3 cm) | 7.9 cm and 8.2 cm |
| The top glass should sit midway between the lower two | 0.2 cm and 0.7 cm off |
| The table's corners, fitted back into the picture | About 1 pixel |
| Hand-picked rough corners moved by up to 20 pixels, 25 tries each | Positions change by 0.11 cm at most |
| Flip check at the tower's top | Correct for both right-side videos: dark opening in Task 3, none in Task 2 |

### What the robot can physically do

These tests use simulator positions and a hand-written plan, so they don't involve video. Each one is a full task from random spots. The areas were narrowed using earlier failures, then retested on fresh random setups.

| Task | Setups | Result |
|---|---|---|
| Triangle, across the table's right side | Glasses 12-34 cm from the edge, centre to 30 cm right. Tower middle 24-36 cm from the edge, 12 cm left to 24 cm right | **28 of 30** |
| Flipped tower | Glasses 13-19 cm from the edge, 14 cm left to 10 cm right. Glass to flip 23-28 cm from the edge, 4 cm left to 10 cm right. Tower middle 20-27 cm from the edge, 26-31 cm right | **28 of 30** |

Run these with `scripts/area_test.py`. Raw results and earlier area tests are in `docs/results/`. Pictures of the areas are in `docs/recording_views/`.

## What worked and what didn't

**Worked:**
- Following identical steel glasses by their bright tops, including recovering the order.
- Using the table in place of a printed marker.
- Gripping by friction alone, with no shortcut attaching the glass to the hand.

**Didn't work, or needed fixing:**
- **Real glasses broke the robot's motion, 0 of 6.** A motion that passed 30 of 30 with smaller cups failed with my taller glass. It needed a higher grip, levelling, measured aiming, and a higher lift, because the hand was knocking over the finished tower on its way back.
- **The robot can't flip anywhere.** Across its chest, its right arm hits its own body. Flipping elsewhere and carrying the glass across made the arm swing and destroy the tower, so I removed that.
- **Hand tracking wasn't useful for timing.** It found the hand in 93-96% of frames, but picked 13 moments for 6 real ones and 17 for 8. The system uses glass rest spots instead.
- **Colour tracking and the marker were built first, then dropped for the real task.** My glasses are steel, and I recorded without a marker.
- **Grip retries didn't help.** Rerunning the failed setups with up to three grip attempts rescued none of them, so I removed the retries. The working fix was keeping glasses 15 cm from the tower.
- **My table estimate exposed a measuring error.** I was first told 80 x 60 cm, and the video said otherwise. The real size was 80 x 50.

## Limitations

- **Pinned pelvis:** no balance, and reach is limited.
- **Three-finger Dex3 hand,** not a five-finger hand.
- **Programmed motion:** no learned policy, and no retargeting of my hand's motion.
- **Task 3 from video needs positions moved,** unless it is recorded inside the flip's working area.
- **Simulator knowledge:** once a run starts, the robot uses exact glass positions from the simulator, not from its own camera.
- **Hand-picked corners:** someone picks four rough table corners once per camera position, in `data/labels/`. A fit to each video's table edges then sharpens them.
- **Narrow conditions:** a fixed camera, a dark table, bright closed tops and fairly bright light. The settings suit my setup and would probably need changing elsewhere.
- **Flip check:** tested only on my two right-side videos.

## Running it

```bash
scripts/setup.sh
.venv/bin/pytest

# A sample video, through to the robot. Output goes to data/processed/
PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/sample/task2_triangle_right_take1.mp4
PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/sample/task3_flip_right_take1.mp4 --fit-to-robot

# Put the person and robot side by side, after running the matching line above
PYTHONPATH=. .venv/bin/python scripts/side_by_side.py --video data/raw/sample/task2_triangle_right_take1.mp4

# Test what the robot can do, with random setups
PYTHONPATH=. .venv/bin/python scripts/area_test.py triangle --trials 30 --seed 8123
PYTHONPATH=. .venv/bin/python scripts/area_test.py flip --trials 30 --seed 9101
```

It runs on a laptop CPU, and was developed on an M1 Mac with 8 GB of RAM. A robot run takes a few minutes.

## Data

Samples are in `data/raw/sample/`, with the collection note in [`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md). The samples are compressed, at full size, to keep the repo small. They read within 0.1 cm of the originals.

## Earlier work in this repo

Before recording the steel glasses, I built a marker-and-colour pipeline for sorting blocks into trays. It's still here and tested on rendered video and in simulation (20 of 20 new setups, with a retry loop), but I never ran it on real footage, because I didn't have the blocks. `notes.md` explains the whole project from scratch.

## Credits

Built on MuJoCo, MuJoCo Menagerie, mink, MediaPipe and OpenCV. Ideas from SeeDo (VLM See, Robot Do) and Robot See Robot Do. Built with help from Claude Code.
