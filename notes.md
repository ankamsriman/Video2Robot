# Video2Robot: complete project notes

These notes explain the whole project from zero. You don't need any Python or robotics background. Read them in order the first time, then use the glossary at the end as a reference.

Throughout, **[Built]** means the code exists and has been tested. **[Planned]** means it doesn't exist yet.

**To follow one video through every step, with real numbers, see [`docs/PIPELINE_EXPLAINED.md`](docs/PIPELINE_EXPLAINED.md).**

## Contents
1. [The project in one story](#1-the-project-in-one-story)
2. [What the challenge asks for](#2-what-the-challenge-asks-for)
3. [Robotics ideas from zero](#3-robotics-ideas-from-zero)
4. [The Unitree G1 robot](#4-the-unitree-g1-robot)
5. [Types of AI models](#5-types-of-ai-models)
6. [What the mentors and papers said](#6-what-the-mentors-and-papers-said)
7. [Our design and why](#7-our-design-and-why)
8. [The code, file by file](#8-the-code-file-by-file)
   - [The vision and planning code](#8b-the-vision-and-planning-code)
   - [Task 2: the cup triangle](#8c-task-2-the-cup-triangle)
   - [Task 3: flipping a fourth cup](#8d-task-3-flipping-a-fourth-cup)
9. [Worked example: putting the apple in the box](#9-worked-example-putting-the-apple-in-the-box)
10. [What went wrong and what it taught us](#10-what-went-wrong-and-what-it-taught-us)
11. [Results and how to read them](#11-results-and-how-to-read-them)
12. [What comes next](#12-what-comes-next)
13. [How the project is run](#13-how-the-project-is-run)

**Part Two: your real videos and real glasses**

14. [Reading your videos: steel glasses, no marker](#14-reading-your-videos-steel-glasses-no-marker)
15. [Task 3: choosing the arm and where to stand](#15-task-3-choosing-the-arm-and-where-to-stand)
16. [Choosing arms for Task 2, and the logistic regression model](#16-choosing-arms-for-task-2-and-the-logistic-regression-model)
17. [The vision language model (Claude) helper](#17-the-vision-language-model-claude-helper)
18. [Copying your hand's carry](#18-copying-your-hands-carry---hand-path)
19. [Live hand control, and matching a recorded video](#19-live-hand-control-and-matching-a-recorded-video)
20. [Step checks: finding where a run fails](#20-step-checks-finding-where-a-run-fails)
21. [All results on your real videos, and how to read them](#21-all-results-on-your-real-videos-and-how-to-read-them)
22. [How the project got here: one idea at a time](#22-how-the-project-got-here-one-idea-at-a-time)
23. [Glossary](#23-glossary)

---

## 1. The project in one story

Imagine you sit at a table and film yourself putting an apple in a box. You hand the video to a robot that has never seen this task. It should be able to:

1. **Watch** the video and notice the moments that matter, such as "the hand slowed down here, so something was probably picked up".
2. **Identify** what's on the table: an apple, a lemon, a box.
3. **Understand** what you did and write it down: "put the apple in the box".
4. **Do it** with its own body, which is nothing like yours.
5. **Check** whether it worked, and try again if it didn't.

Doing this on a real humanoid would be expensive and risky, so we use a **simulator**: a program that acts like a video game with accurate physics. If the robot's fingers don't squeeze hard enough in the simulator, the object really does slip.

**Where we are (4 Nov):** the robot now learns from **your real webcam videos** of steel glasses. For Task 2, the triangle, it builds 8 of your 9 videos with Claude's help, and 3 of 9 without it. For Task 3, the flip, it builds 2 of 9; most failures are glasses too far for a robot that can't walk. The challenge's hard rule is met: the plan and every glass position come from your video.

**To read the project as a story of ideas, from the first failure to today, start with chapter 22.** Chapters 1 to 13 explain the ideas using the first version, which sorted coloured blocks. **Part Two (chapters 14 to 22) covers your real videos**, and chapter 21 has the numbers to quote.

---

## 2. What the challenge asks for

`task.md` now holds the official challenge text.

**The offer:** Awign and Armature AI Labs, with their partners, are hiring robot-learning interns. The strongest submissions can get a 2-month, full-time paid internship in Bangalore, and strong performers may be considered for full-time roles.

**The task:** record yourself doing a simple task on your phone, then use that footage to make a humanoid do it in simulation.

**The one hard rule:** *"the data you personally recorded has to play a real role in your approach. Everything else is open."*

**Preferred, not required:** Unitree G1 and MuJoCo. Other robots are welcome, and retargeting well to a less common body counts in your favour.

**Their example:** film your hand picking up a cup and placing it on a coaster, retarget your hand poses to a five-finger G1 hand, post-train a small VLA such as SmolVLA, then show the G1 doing it in MuJoCo, using fingertip contact to confirm the grasp.

**Suggested directions:**
- Post-train a policy on your recordings.
- Retarget human hand motion to a five-finger hand, using touch signals.
- Bootstrap a policy and improve it with reinforcement learning.
- Take the world-model route, showing video or state prediction.
- Make a standard policy run much faster than the baseline.

**What to submit, by Wed 4 Nov 2026, 23:59 IST:** a **public GitHub repository** containing:
- A README or short deck: how to run it, example videos or GIFs, your design choices, and **what worked and what didn't**.
- Your recorded data, or a representative sample, plus a note on how you collected it.

**How it's judged:**
- **Creativity:** as long as your own data genuinely matters to the result.
- **Results:** how well the robot performs, and/or the quality of world-model predictions.
- **Clarity:** "a simple implementation and an honest, clear presentation. No AI slop please."

### How our current plan compares

| Requirement | Status | Comment |
|---|---|---|
| Phone recording | Waiting for your videos | Fixed tripod. Egocentric "works well" but isn't required |
| Your data plays a real role | **Code ready, not yet met** | Your video decides the task, the order and the object positions; tested only on a rendered video |
| G1 in MuJoCo | **Built** | The preferred choice |
| Five-finger hand | **Gap** | Our model has three fingers per hand (Dex3). They suggest "Inspire, Dex5, or similar" |
| Listed direction | **Different** | Our closed-loop approach isn't on the list, though "everything else is open" |
| Public repo with data | Ready | Videos placed in `data/raw/sample/` will be included |
| Honest reporting | Being done | Failures and limitations are recorded in these notes and in the results |

### The biggest open decision

The earlier version of `task.md` was a step-by-step pipeline with no training, so I built the plan around it. The official text gives **post-training as its main example**, and four of its five directions involve learned models. Training still isn't mandatory, but we should now decide deliberately whether to add it. Section 12 explains where it fits. None of the work so far would be wasted.

---

## 3. Robotics ideas from zero

### Simulator and physics
A **simulator** computes what happens next, over and over in tiny time steps. Ours uses **MuJoCo**, a widely used robotics physics simulator, with a step of **0.002 seconds**. That's 500 steps for every simulated second.

Each step works out gravity, motor forces, collisions and friction, then moves everything slightly. Nothing is pre-animated. When the robot lifts the apple, it's friction between the fingers and the block that holds it up.

### Model and data
MuJoCo separates two things:
- **Model:** what doesn't change, such as body shapes, masses and joint limits. In the code it's `MjModel`. Think of it as the blueprint.
- **Data:** what's true right now, such as joint angles, speeds and contacts. In the code it's `MjData`. Think of it as a snapshot of this moment.

### Coordinates
To describe where something is, we use three numbers in metres. Our convention puts the origin on the floor, under the robot's pelvis:

```
         +z (up)
          |
          |     +x (forward, away from the robot)
          |    /
          |   /
          |  /
+y <------+
(robot's left)
```

So the apple at `x=0.35, y=-0.28` is 35 cm in front of the robot and 28 cm to its right. Negative `y` means right.

### Pose: position plus orientation
Knowing *where* your hand is isn't enough. It could be there with the palm up or the palm down. **Position** plus **orientation** is called a **pose**.

### Joints and degrees of freedom
A **joint** connects two body parts and lets them move relative to each other. Each independent way of moving is one **degree of freedom (DOF)**. Your elbow is about 1 DOF. Your shoulder is about 3.

### Motors and position control
A **motor** (also called an actuator) applies force at a joint. Our G1's motors use **position control**: you give a target angle, and the motor pushes in proportion to the error. It behaves like a spring pulling the joint toward its target:

```
torque = stiffness x (target angle - current angle) - damping x (current speed)
```

Most stiffness values in this model are 500. If the error is 0.02 radians, the motor produces 500 x 0.02 = 10 N·m of torque. **Damping** resists speed, so the joint doesn't overshoot and wobble.

Angles are in **radians**. One radian is about 57 degrees, so 0.20 rad is about 11 degrees.

### Forward and inverse kinematics
Kinematics is the geometry of motion, ignoring forces.
- **Forward kinematics (FK):** "If these are the joint angles, where is the hand?" There's exactly one answer, and it's easy to compute.
- **Inverse kinematics (IK):** "I want the hand *here*. What joint angles get it there?" This is harder. There may be many answers, or none.

**Everyday example:** touch your nose, then move your elbow without moving your fingertip. Many arm poses give the same fingertip position. That's why IK needs extra preferences to pick one, such as "stay close to a natural posture".

We use a library called **mink**. It takes small steps: compare where the hand is with where it should be, nudge the joints in the direction that reduces the error, and repeat.

### Smooth motion
If you command a distant target suddenly, the arm snaps toward it violently. So we split each motion into many small waypoints along a smooth curve called **minimum jerk**, which starts and stops gently, much like a human reach:

| Time through the move | Distance covered |
|---|---|
| 0% | 0% |
| 25% | about 10% |
| 50% | 50% |
| 75% | about 90% |
| 100% | 100% |

### Open loop and closed loop
- **Open loop:** give the instruction and assume it worked. Like walking across a room with your eyes shut.
- **Closed loop:** measure what actually happened and correct it. Like walking with your eyes open.

This idea appears twice in the project. At a small scale, the hand measures its position and corrects itself **[Built]**. At a large scale, the robot compares its whole result with your video and retries **[Planned]**. The large one is the project's main idea.

### Grasping
Holding something depends on **friction**: the harder you squeeze, the more grip you get. A grip is **stable** only when the contact points resist the object's turning and slipping. Hold a book between your finger and thumb at one corner, and it swings like a hinge. That exact problem appeared in this project (section 10).

### Centre of mass and balance
The **centre of mass** is the average position of a body's weight. A standing body stays up only while its centre of mass is above its feet. Lean too far forward without moving your hips back, and you fall. That's what happens to our G1 when it's unpinned and reaches far.

---

## 4. The Unitree G1 robot

The G1 is a real humanoid, about 1.3 m tall. Its simulation model comes from **MuJoCo Menagerie**, a free collection of robot models. It's the challenge's default robot.

| Part | Joints | Used in our task? |
|---|---|---|
| Legs | 12 (6 each) | Held still |
| Waist | 3 (twist, side-tilt, lean) | Twist and lean are used |
| Arms | 14 (7 each) | Right arm moves; left arm hangs still |
| Hands | 14 (7 each) | Right hand grips |
| **Total motors** | **43** | |

**Arm joints, from shoulder to wrist:** shoulder pitch, roll and yaw, elbow, then wrist roll, pitch and yaw.

- **Pitch** is nodding, or tipping forward and back.
- **Roll** is tilting sideways.
- **Yaw** is turning left and right.

**The hand:** three fingers, a thumb, an index and a middle finger. Each finger's joints curl it, and the thumb has three joints. This is the Dex3 hand. The challenge suggests five-finger hands such as Inspire or Dex5, so this is a known gap.

### Measured reach
- Shoulder height is about 1.08 m.
- Shoulder to elbow is about 20 cm, and elbow to wrist about 18 cm, so the arm is short.
- The arm alone reaches only a small patch. Adding the waist makes the reachable area much larger, which is why the code uses it.

### Measured working area
These are full physics runs, one per position, with one object moved and placed in a tray. **70 of 78 passed.** `#` passed, `.` failed. Forward is from the pelvis, and negative sideways is the robot's right.

**Where an object can start** (with the tray in its usual place):

```
forward\\sideways  -42  -35  -28  -21  -14   -7    0   +7  +14 cm
42 cm               .    #                        #    #    #    #
38 cm               #    #    #    #    #    #    #    #    #
34 cm               #    #    #    #    #    #    #    #    #
30 cm               #    #    #    #    #    #    #    #    #
26 cm               #    #    #    #    #    #    #    #    #
```

**Where a tray can be:**

```
forward\\sideways  -35  -28  -21  -14   -7    0   +7  +14 cm
58 cm               .    .    .    .    #    #    #    #
54 cm               .    .    #    #    #    #    #    #
50 cm               .    #    #    #    #    #    #    #
46 cm               #    #    #    #    #    #    #    #
42 cm               #                        #    #    #
```

Blank spots were skipped, because another item was already there. With only one run per spot, this is a guide rather than a guarantee.

**The earlier estimate of 20 x 40 cm was too cautious.** It came from a check that I later found was too strict.

### Why the pelvis is pinned
The model normally has a **free joint** at the pelvis, meaning the whole robot can move and fall. We delete that joint, which fixes the pelvis in the air at its normal standing height. The robot can't lose its balance, even though its feet are on the floor.

We tested the alternative:
- **Standing idle, unpinned:** the G1 balances fine, with about 1 degree of sway.
- **Reaching far, unpinned:** it falls over.

So **all current results use a pinned pelvis**, and any presentation must say so. Real balance would require moving the hips to counterbalance the lean, which is a planned stretch goal.

---

## 5. Types of AI models

These terms are easy to mix up.

| Type | Input | Output | Our project |
|---|---|---|---|
| **VLM** (vision language model), like Gemini or GPT-4o | Images and a question | Text | **Planned**, to write the plan |
| **VLA** (vision-language-action), like SmolVLA, GR00T, pi0 | Camera images and a command | Robot motions | Not used so far |
| **ACT** (Action Chunking with Transformers) | Camera images and joint state | A short burst of future motions | Possible addition |
| **World model** | Current state and an action | A prediction of what happens next | Not used so far |
| **RL policy** (often trained with PPO) | Robot state | Motions, learned through trial and reward | Not used so far |

**An analogy.** You're teaching someone to make tea:
- A **VLM** watches and writes down the recipe.
- A **VLA or ACT** is muscle memory, learned from watching many examples.
- A **world model** is imagining what will happen if you tip the kettle.
- **RL** is learning through trial and error, with a score for each attempt.
- Our **hand-written `pick` and `place`** are like following exact written instructions for moving your arm.

### Pre-trained, training and post-training
- **Pre-trained:** someone else already trained it, and you just use it. Gemini is an example.
- **Training:** teaching a model from scratch on your data, which takes lots of data and compute.
- **Post-training** (also called fine-tuning): starting from a pre-trained model and specialising it for your task with your own data. This is what the challenge's example does.

---

## 6. What the mentors and papers said

### First session (NVIDIA-focused)
It covers NVIDIA's tools: Isaac Sim, Isaac Lab, Newton physics, Cosmos world models and the GR00T foundation model. Its pipeline is human video → reconstruct the scene → retarget motion → generate 100–200 episodes → post-train → evaluate.

Isaac Sim needs an NVIDIA RTX graphics card, so it can't run on your Mac. The mentor also said GPUs aren't a hard requirement.

### Second session (practical advice)
- **Pick one task and execute it perfectly.** Then either make it more efficient (the example was 7 attempts down to 3) or move to something harder.
- **Build a digital twin** and keep the camera fixed, with accurate coordinate frames.
- **"It's all depending on how well you post train the model"**, and annotate your data well.
- ACT did better than SmolVLA for teams at a recent hackathon.
- Try world models, using Dream Labs credits for exploration.
- **Novelty is the first deliverable.**

### The five papers

| Paper | Main idea | Use to us |
|---|---|---|
| **SeeDo** (VLM See, Robot Do) | A VLM turns video into a step-by-step plan, helped by keyframes and object outlines | Basis of our pipeline, prompts and metrics |
| **Robot See Robot Do** | Copy the *object's* motion, not the hand's, because robot and human bodies differ | Our actions focus on where objects end up |
| **EgoHumanoid-V2** | Post-trains a VLA for the G1 on human demos; mean score 51.67% with 200 demos per task | Shows what post-training takes: motion-capture gear and many demos |
| **SuperMimic** | Trains whole-body control through a learned world model, using far less compute | Direction for whole-body control; needs a strong GPU |
| **Whole-Body Planning** | Moves a humanoid through tight spaces | Mostly unrelated to our task |

### SeeDo in more detail
It has three helpers around the AI:

1. **Keyframes:** a video has thousands of frames and the AI can't handle them all. Hands slow down when picking and placing, so SeeDo plots hand speed and keeps the slowest moments ("valleys").
2. **Visual prompts:** it draws an outline and ID on each object, so the AI can tell similar objects apart.
3. **Small questions in order (chain of thought):** which frames show the hand working? What's being picked? What's the reference object? In which of 6 directions was it placed?

**Why it matters:** feeding uniformly sampled frames into the same AI scored near zero, against 60.53% task success for full SeeDo on vegetable sorting. On its hardest task, block stacking, full task success was 21.62%.

**Its own stated limits, which give us our openings:**
- Spatial errors were its largest failure source (64.10% of failures on block stacking with visual prompts).
- It uses only qualitative relations like "left of", with no real distances.
- It scores the plan's accuracy, not whether the robot actually succeeded.

---

## 7. Our design and why

```
video -> keyframes -> tracked objects + table positions (cm) -> plan.json -> G1 in MuJoCo -> check result
                                                                   ^                            |
                                                                   +---- diagnose and retry ----+
```

| Decision | Why |
|---|---|
| Build the simulator first | It was the biggest risk, and it guarantees something to show even if vision fails later |
| One shared plan format | The hand-written plan can later be swapped for an AI-written plan without touching the robot code |
| Fixed camera and printed marker | Lets us convert pixels to centimetres |
| Measured positions decide the plan | Targets SeeDo's biggest failure source, and needs no AI model or API key |
| Retry loop | Gives the mentors' metric, attempts to success |
| Honest labelling | Clarity is a judging criterion, and it's simply correct |

**How your data would play its real role:** your video decides which objects move, where they go, in what order, and where they started. Without your video, the robot wouldn't know the task. That's a real role, but post-training would make it stronger, since your data would then also shape *how* the robot moves.

---

## 8. The code, file by file

```
configs/default.yaml        all tunable numbers
data/labels/sample_plan.json   the hand-written plan
src/schemas/plan.py         the plan format
src/verification/relations.py  geometry: where things should go, and where they are
src/simulation/scene.py     builds the digital twin
src/simulation/ik.py        works out joint angles
src/simulation/robot_env.py runs the robot in physics
src/simulation/primitives.py   pick, place, and running a plan
src/simulation/recorder.py  saves videos
scripts/                    tools you can run
tests/                      automatic checks
```

The files above move the robot. Section 8B covers the files that watch the video, and `main.py`, which joins everything.

### A little Python first
- **Function:** a named piece of work with inputs and an output. `min_jerk(0.5)` returns `0.5`.
- **Class:** bundles data with the functions that use it. `RobotEnv` holds the simulation and knows how to move the robot.
- **Dataclass:** a simple record. `frozen=True` means it can't be changed after creation, which prevents accidental changes elsewhere.
- **Enum:** a fixed list of allowed values.
- **Tuple and dict:** a tuple is a fixed list, like `(0.35, -0.28)`. A dict looks up values by name, like `{"apple": ...}`.
- **Library names:** `numpy` does number crunching, `pydantic` checks data formats, `yaml` reads config files, `pytest` runs tests.

### `configs/default.yaml`: the settings
All tunable numbers live here, so you never have to dig through code to change them.

| Setting | Value | Meaning |
|---|---|---|
| `table.top_height` | 0.72 | Table surface is 72 cm high |
| `apple.size` | 0.06, 0.07, 0.07 | 6 cm front-to-back, 7 cm side-to-side, 7 cm tall |
| `apple.mass` | 0.08 | 80 grams |
| `grip_height` | 0.04 | Grip 4 cm above the table |
| `hover_height` | 0.17 | Travel 17 cm above the table, clearing everything |
| `release_clearance` | 0.015 | Let go 1.5 cm above the target height |
| `hand_speed` | 0.20 | Travel at about 20 cm/s |
| `control_rate_hz` | 25 | Send new motor targets 25 times per second |
| `finger_force_limit` | 1.0 | Finger motors limited to 1 N·m |
| `hand_pitch` | 0.20 | Tip the fingers down by 0.2 rad, about 11 degrees |
| `grip_squeeze` | 0.015 | Aim fingers 1.5 cm inside the object to create squeeze |
| `release_margin` | 0.035 | Open 3.5 cm wider than the object to let go |

Objects and sizes are placeholders until you send real measurements.

### `src/schemas/plan.py`: the plan format
A plan is an ordered list of steps. Each step says "put this object in this relation to that thing":

```json
{ "object": "apple", "reference": "box", "relation": "in" }
```

`Relation` allows SeeDo's six options: `in`, `on_top_of`, `at_the_back_of`, `in_front_of`, `to_the_left`, `to_the_right`.

**Pydantic** rejects bad plans immediately. An unknown relation like `"beside"`, an empty plan, or placing an apple relative to itself all cause an error. This is valuable because a future AI model could return nonsense, and this stops it at the door.

### `src/verification/relations.py`: geometry
This file holds pure maths with no simulator involved, so it's easy to test.

**`Box`** describes an item by its centre and half-sizes. Half-size means half the width, which makes the maths simpler.

**`placement_target(obj, reference, relation, gap)`** answers "where should the object's centre end up?"

Example: `to_the_left` of the bowl. The bowl's half-width is 9 cm, the object's is 3.5 cm, and the gap setting is 3 cm. So the object's centre goes 9 + 3.5 + 3 = 15.5 cm to the left of the bowl's centre.

**`classify_relation(obj, reference, container)`** does the reverse: given where things are now, what's the relation? An object counts as `in` only if it's inside the outline **and** its bottom is below the container's rim, so hovering above doesn't count. Otherwise, it uses whichever direction is larger.

**`free_slot(obj, container, occupants)`** picks a spot inside a tray. A tray has two slots, side by side, never the centre, because an object in the centre would leave no room for a second. If both are taken, it reports that the container is full.

### `src/simulation/scene.py`: the digital twin
`build_model` starts from the G1 file, then:
1. Deletes the saved start pose, because adding objects would make it the wrong size.
2. Removes the pelvis free joint if `pinned=True`.
3. Adds a **grasp site** on each wrist, an invisible point at the middle of the grip. It's 12.5 cm along the hand and 8 cm below the palm side, as measured in calibration.
4. Adds the table.
5. Adds containers, as a floor and four thin walls that never move.
6. Adds objects, as free-moving blocks with friction.
7. Compiles everything into a model.

`positions` lets the test script move items to random spots.

### `src/simulation/ik.py`: working out joint angles
- **`palm_down_rotation`** gives the required hand orientation: pointing forward, palm toward the table, tipped down by `hand_pitch`.
- **`waist_yaw_for_target`** turns the torso, as a person would, so that the right shoulder faces the target. Example: for the apple at (0.35, −0.28), the torso turns about 0.45 rad (26 degrees) to the right.
- **`ArmIK.solve`** gives mink three instructions: reach this pose, stay near a natural posture with a slight forward lean, and don't move anything except the right arm and waist lean. It iterates until the error is tiny, up to 200 rounds, and returns the joint angles.

Twist is calculated directly rather than solved for. The solver is free to adjust 8 joints: 7 in the arm, plus the waist lean.

### `src/simulation/robot_env.py`: physics control
`RobotEnv` owns the simulation. Its important parts are:

**`hand_pose_for_width(width)`** turns a grip width into finger angles, using a calibration table. Between table entries it interpolates, which means blending the two nearest. For a width of 4.5 cm, which lies between 3 cm and 5 cm, it blends them 25% to 75%.

**`advance(duration)`** runs physics. For every 0.002 s step, it sets each motor's target and adds **gravity compensation**:

```python
ctrl = command + gravity_torque / stiffness
```

Without it, gravity would make the arm sag. Motors have to be slightly "wrong" to produce holding torque, so we shift the target by exactly that amount.

**`move_hand(target)`** is the main motion:
1. Work out the distance and duration from the hand speed.
2. For each waypoint along the minimum-jerk curve, solve IK and send the joint targets.
3. Run physics between waypoints. At 25 waypoints per second and 500 physics steps per second, that's 20 physics steps per waypoint.
4. Run `_correct_hand`: measure the real hand position, and nudge the aim to remove any error.
5. **Measure the final error. Return `True` only if the hand is within 2 cm.** This is logged as a warning, and steps are judged on where the object ends up.

**`set_fingers(pose)`** moves fingers smoothly to a pose.

**`_start_palm_down`** begins the robot in the correct hand orientation, so that it never has to snap into it.

### `src/simulation/primitives.py`: the robot's actions
"Primitive" means a basic reusable action.

**`pick(env, config, name)`:**
1. Open the fingers to 10 cm.
2. Move above the object.
3. Lower to grip height, slowly.
4. Close the fingers.
5. Wait for things to settle.
6. Lift.
7. **Verify:** did it rise at least 5 cm, and is it within 6 cm of the hand?

**`place(env, config, step, placed)`:**
1. Work out the target.
2. Measure how the object sits in the hand (`grip_offset`), so the *object*, not the hand, lands on target.
3. Move above the target, then lower.
4. Open enough to release.
5. Wait, lift straight up, and open fully.

**`execute_plan`** runs all steps, stopping at the first failure, then re-measures everything. **`ExecutionResult.success`** is true only if every step worked **and** everything is still in the right place at the end. So knocking the apple out while placing the lemon counts as failure.

### `src/simulation/recorder.py`
It renders frames at 960x720 and 30 frames per second, writing directly to a video file instead of holding them in memory, which matters on an 8 GB Mac.

### `scripts/`

| Script | What it does |
|---|---|
| `setup.sh` | Creates the Python environment and downloads the robot model |
| `reach_map.py` | Prints a map of where the hand can reach on the table |
| `phase1_gate.py` | Runs the test: randomises positions and runs the plan many times |
| `make_marker.py` | Creates `docs/marker.pdf` |
| `workspace_map.py` | Tests full runs across the table, to find the real working area |
| `stress_retry.py` | Makes the task harder, to measure what retries add |
| `run_keyframes.py` | Finds key moments in a video, and saves a speed plot |
| `check_video.py` | Checks a test video, and says what to fix |
| `read_colors.py` | Suggests colour settings from points on a video frame |
| `watch_live.py` | Runs the task live in the MuJoCo window |
| `live_teleop.py` | Your webcam's view of your hand drives the robot live |

### `tests/`
`pytest` runs about 120 tests. Some test functions run several cases.

| File | What it checks | Example |
|---|---|---|
| `test_g1_model.py` | Model loads and renders | Image isn't blank |
| `test_plan_schema.py` | Plan format | Rejects `"beside"` |
| `test_relations.py` | Geometry | Hovering isn't `in` |
| `test_robot_env.py` | Motion and full physics | Hand blocked by the table reports failure |

We usually write the test **first**, watch it fail, then write the code. That proves the test can actually catch the problem.

---

## 8B. The vision and planning code

This code turns a video into a plan. It's only been tested on a rendered video, where every true position is known.

```
main.py                         one command that runs everything
src/keyframe_selector/          finds your hand and the key moments
src/perception/homography.py    pixels to table centimetres, using the marker
src/perception/color_tracker.py follows coloured objects
src/perception/observe_video.py joins the two above into per-frame positions
src/perception/read_colors.py   suggests colour settings from a video frame
src/reasoning/demo_observer.py  works out the plan with no AI
src/reasoning/vlm_planner.py    works out the plan with an AI, once one is chosen
src/verification/closed_loop.py the retry loop
src/evaluation/metrics.py       TSR, FSR and SSR scores
src/pipeline/run_video.py       runs the stages and saves their output
src/pipeline/synthetic_demo.py  renders the test video
```

### `speed_analyzer.py`: finding key moments
Your hand slows down when it picks or places something. So we measure its speed in each frame and look for the dips.

**Example:** if your hand moves from image position 0.40 to 0.41 in one frame at 30 frames per second, its speed is 0.01 x 30 = 0.3 image widths per second.

Raw speed is jittery, so a filter called **Savitzky-Golay** smooths it. It fits short curves through nearby points, which keeps real dips in place better than plain averaging. Then a dip only counts if it:
- is deep enough compared with its surroundings (its **prominence**)
- is at least 0.8 seconds from the previous dip, so one pause doesn't count twice
- isn't in the first or last half-second, when your hand is just arriving or leaving
- happens in a frame where your hand was really visible, so a gap where it left the frame can't fake a dip

### `hand_detector.py`: following your hand
**MediaPipe** finds 21 points on your hand in each frame, such as fingertips and knuckles. We save all 21 in `hand_path.npz`, not just the speed. Option B would need that full path, and saving it now avoids redoing the work.

### `homography.py`: pixels to centimetres
A flat table photographed at an angle looks like a trapezoid. A **homography** is the maths that undoes that distortion. It can be worked out from four points whose real positions we know, and the printed marker gives us exactly that: four corners of a 10 cm square.

**Limit:** it only works for points on the table's surface. The top of a tall object appears further away than it is. So we use the **bottom edge** of each object, where it touches the table. The camera sees the object's near side, so `base_position` then moves half the object's depth further away to find its centre.

**Example:** an apple 6 cm deep whose near edge maps to x = 0.38 has its centre 3 cm further along.

### `color_tracker.py`: following objects
It converts each frame to **HSV**, which separates a colour's hue from its brightness. That means a shadow changes the brightness but mostly leaves the hue alone, which RGB doesn't do.

For each object it keeps pixels in that object's hue range, cleans up specks, and takes the largest blob. If much of the object is hidden, such as behind your hand, it reports nothing for that frame rather than a misleading position.

**Red is a special case:** hue is a circle, and red sits at both ends, so its range wraps around, such as 170 to 8.

This is the simple method. SAM 2 would handle tricky objects better, but needs a GPU.

### `demo_observer.py`: the plan with no AI
This was a design change made while building, and it's useful: once we know where everything is, the plan follows from simple logic.

1. Find each object's position at the start and end of the video.
2. If it moved more than 5 cm, you moved it.
3. If it ended inside a container's outline, the relation is `in`. Otherwise, it's the direction from the nearest item.
4. Order the steps by when each object first left its starting position.

**Example:** the lemon leaves first at 3 s, and ends inside the box. The apple leaves at 6 s, and also ends in the box. So the plan is: lemon in box, then apple in box.

**Why it matters:** your video can drive the robot with **no API key and no GPU**, which protects the hard rule. **Limit:** it only sees each object's final position, so moving an object twice counts once.

### `vlm_planner.py`: the plan with an AI
This follows SeeDo's question sequence:
1. Is the hand working on an object in this frame? Others are discarded.
2. Pair the remaining frames: pick, place, pick, place.
3. For each pair, ask what's picked, what the reference is, and what the relation is.

The answers are checked: an item that isn't in the scene raises an error instead of reaching the robot. It can also use measured relations instead of the AI's, so we can compare them, as SeeDo's biggest weakness is spatial errors.

**Status:** tested only with a fake AI that plays back scripted answers. It needs the model choice you've postponed, and connecting one is a small job. Its accuracy is unknown until then.

### `closed_loop.py`: the retry loop
A closed loop checks what really happened and corrects it, as in section 3.

1. Try the step.
2. Measure where the object ended up. **It compares with the plan from your video, never your hand-written labels**, so it can't cheat.
3. If it failed, adjust and retry:
   - **Failed pick:** grip harder: fingers aim 1 cm tighter, and the force limit rises 50% per retry, up to 2 N·m.
   - **Failed placement:** move 30% slower per retry.
4. At the end, recheck everything, and redo any object that got knocked away.

It also counts attempts, which gives the mentors' metric: attempts to success.

### `metrics.py`: scoring plans
Section 12 explains TSR, FSR and SSR with an example. For SSR, it uses the **longest common subsequence**: the longest run of correct steps kept in order. SeeDo's released code does the same, though its paper prints a different method.

### `synthetic_demo.py`: the test video
Testing with real video alone would make it hard to measure accuracy, since we wouldn't know the true positions. So this renders a table with the marker and coloured blocks that glide to their targets, with every position known.

**Results:** the plan was correct, and blocks were located to within 0.1-0.7 cm, and the tray to 2.2 cm. **Real video will be worse:** it has shadows, blur and a real hand, none of which this test includes.

### `main.py`: one command
It runs these stages, and each saves its output so it can be rerun alone:

| Stage | Output in `data/processed/<video>/` |
|---|---|
| `keyframes` | `hand_path.npz`, `keyframes.json` |
| `perception` | `tracks.npz` |
| `plan` | `observation.json` |
| `sim` | `rollout.mp4`, `result.json` |

**Note:** the no-AI planner doesn't use key moments yet, since object tracking is enough for it. They matter for the AI planner.

---

## 8C. Task 2: the cup triangle

**This lives on a separate git branch, `cup-task`,** so it can't affect sorting. A branch is a separate line of history, and to look at it you'd run `git checkout cup-task`.

**The task:** three cups start apart at random positions. Two end up side by side on the table, and the third rests across their tops.

```
         [red]
    [yellow][green]
```

### How the cups are modelled
- **Solid, rigid and upside-down:** wide at the bottom, narrow at the top. A flat top gives the third cup something to rest on.
- **60 g,** heavier than a real paper cup. It's a simplification to state in your README, though 15 g cups also passed.

### What it took
1. **The first problem was slip.** The cup slowly slid through the fingers, about 2 cm in 3 seconds, because its sides slope. Gripping harder didn't help much.
2. **The fix was a more accurate friction setting.** MuJoCo's default friction model is fast but approximate. The **elliptic cone** with a higher **impratio** setting is more accurate for grasping, and reduced slip to about 1 mm.
3. **The spacing:** cup centres are 8.5 cm apart, so the fingers fit, yet the top cup's 8 cm base still overlaps both.

### What counts as a success
`triangle.py` measures the result 2 seconds after the hand has left. All must be true:
- Both lower cups stand upright on the table, within 1.5 cm of the right height.
- They're side by side, within 2 cm of the intended gap.
- The top cup is at the height of resting on them, centred over both within 2 cm, and tilted under 10 degrees.

### Results
- **30 of 30** on new random setups.
- **10 of 10** with very light 15 g cups.

This went better than I expected. You thought it was probably possible, and you were right.

### From a rendered video
A raised cup breaks the marker conversion, which only works on the table. So `triangle_observer.py` recognises it in the image instead: the cup's base appears above the line joining the two lower cups, and between them. The build position comes from the two lower cups, which are on the table.

It worked on a rendered video: it found the order and built the triangle.

**The camera matters.** From high above, the top cup hid a lower one and tracking failed. From the side, with the lower cups side by side in the picture, all three stayed visible. To run it on your video: `git checkout cup-task`, then `PYTHONPATH=. .venv/bin/python scripts/cup_from_video.py --video data/raw/cups_01.mp4`.

### What this doesn't show
- **Most results use a hand-written plan,** and none use real video, so the hard rule isn't met yet for this task. It's now been shown working from a rendered video, as below, but not a real one.
- **The simulated hand has an advantage:** it knows each cup's exact position. A real robot wouldn't.
- **Real cups differ:** they're hollow, flexible and much lighter, and the simulator's top cup rests on a solid surface.
- **Pelvis is pinned,** as in Task 1.

---

## 8D. Task 3: flipping a fourth cup

**This lives on its own git branch, `cup-flip`.** `main` and `cup-task` aren't affected.

**The task:** build the triangle, then stand a fourth cup on top with its **opening up**. The cup starts opening-down, so the robot has to turn it over with one hand.

```
        \___/     cup 4: opening up
        /   \     cup 3
     /   \ /   \  cups 1, 2
```

### The main idea
Flipping with the palm above the cup puts the palm **underneath** at the end, where it would hit the tower. So the robot grips with its palm **beside** the cup, and rotates around the line between thumb and fingers, like food on a spit. The palm ends on the cup's other side, so nothing is in the way.

### What it took
Each failure taught something. It took about 12 attempts at the design:

| Problem | Cause | Fix | Lesson |
|---|---|---|---|
| Cup thrown to the floor | Inverse kinematics made a big shoulder swing | Turn **only the wrist joint** | Sometimes direct control beats a solver |
| Flip ended 45 degrees short | The forearm wasn't level | Search for an arm pose with a level forearm | Wrist turning flips cleanly only about a level axis |
| Cup dropped however hard it gripped | It sat past the fingertips, so squeezing pushed it out, like a slippery seed | Hold it deeper, with fingertips past its centre | More force doesn't fix bad geometry |
| Arm flailed wildly | The solver reset the waist, and later a fallen cup caused a huge "correction" | Keep the pose steady, and never correct if the cup isn't held | Check assumptions before acting on a measurement |
| Tilted about 12 degrees | Grip and arm alignment | Measure the tilt, and correct it slowly | Closed-loop correction again |
| Carrying it to the tower failed | Reaching far with the hand turned over | **Build the tower where the twist ends** | Change the task layout, not just the motion |
| Cup crept out over seconds | Soft friction model | Higher `impratio` | Simulation settings can matter as much as code |
| Palm knocked the cup over | Coming straight down hit its wide base | Come down beside it, then slide in | |
| Fingers carried the cup away | It's widest at the top, so lifting caught the rim | Centre the hand, back out sideways, then lift | |
| Cup shoved 1.2 cm on release | The thumb stays loaded like a spring after squeezing | Set finger targets to their real positions first | The same cause as a Task 1 bug |
| Tower knocked over at the end | My own final arm swing | Remove the swing | |

One of my diagnoses was wrong along the way: I blamed the fingertip paths, calculated them, and found they moved outward as they should. So it's worth checking a theory before building on it.

### Results
Scores improved as bugs were fixed, on new setups each time:

| Stage | Result |
|---|---|
| First random test | 1 of 9 |
| After the higher `impratio` | 1 of 12, a new problem appeared |
| After backing out and removing the swing | 5 of 12 |
| After sliding in and centring | 9 of 15 |
| Before the aiming fix | 21 of 24 (87.5%) |
| **Final version, new setups** | **50 of 50 (100%)** |

The 3 failures at 21 of 24 had the cup on top and flipped, but more than 2 cm off-centre. I kept that check as it was. Tracing the cup through each failed run showed two small errors adding up:

1. **Aiming at the plan.** The hand aimed at where the top cup should be, but that cup had settled up to 1.5 cm away.
2. **A shift on release.** Opening the fingers moved the cup about 0.4 cm forward and 0.3 cm sideways, in a similar direction every time.

**The fix:** aim at the top cup's measured position, aim slightly the opposite way to cancel the shift, and correct once more just above the tower. The cup in the 3 failed setups went from just over 2 cm off-centre to under 0.3 cm. Then 50 setups the robot had never seen all passed, and the original 24 passed 24 of 24.

**100% here doesn't mean it can never fail.** With 50 of 50, a true failure rate of a few percent could still go unseen. The new test also has the same limits listed below.

### What this doesn't show
- **The robot doesn't learn the flip from a video.** It's programmed. Nothing connects this task to video yet, not even a rendered one, so **the hard rule isn't met for it**. Your README must say so.
- **The layout is restricted.** The tower has a fixed spot, and the fourth cup must start 40-44 cm forward. It isn't as flexible as Tasks 1 and 2.
- **Simulation only:** rigid solid cups, a dark disc that represents the opening, a pinned pelvis, and a friction setting chosen because it worked.
- **A real robot may behave differently.** Exact knowledge of the cup's position and tilt wouldn't be available, and the final fix relies on it: the robot reads both cups' exact positions from the simulator.

---

## 9. Worked example: putting the apple in the box

Here's what happens for `{ "object": "apple", "reference": "box", "relation": "in" }`, using the default positions. Runs also randomise them by up to 3 cm.

**Starting state**
- Hand at (0.30, −0.10, 0.95), palm down.
- Apple at (0.35, −0.28), centre 75.5 cm high. That's the 72 cm table plus half its 7 cm height.
- Box at (0.50, −0.21).

**Pick**

| Stage | Hand target | What happens |
|---|---|---|
| Open | No movement | Finger gap goes to 10 cm |
| Hover | (0.35, −0.28, 0.89) | Torso turns about 26 degrees right; the arm follows a smooth path |
| Lower | (0.35, −0.28, 0.76) | Moves at 10 cm/s, and the fingers straddle the apple |
| Close | No movement | Grip target is 6 − 1.5 = 4.5 cm, narrower than the object, so the fingers press until the force limit |
| Lift | (0.35, −0.28, 0.89) | Friction holds the apple |
| Check | | Rose at least 5 cm, within 6 cm of the hand: `picked = True` |

**Place**

1. **Find a slot.** The box's half-width is 9 cm, the apple's 3.5 cm, and the wall margin is 0.8 cm. So the slot is 9 − 3.5 − 0.8 = 4.7 cm from the centre, giving y = −0.21 − 0.047 = −0.257.
2. **Find the height.** Target centre is 72 + 3.5 = 75.5 cm.
3. **Allow for the grip.** If the apple sits slightly off-centre in the hand, the hand target shifts by that much.
4. **Move:** hover above the slot, then lower until the apple's target height is 1.5 cm above its final position.
5. **Release:** the gap opens to 6 + 3.5 = 9.5 cm. The apple drops the last few millimetres onto the tray floor.
6. **Retreat:** straight up, so nothing gets knocked over.

**Check**

`classify_relation` asks whether the apple is inside the box's outline and below its rim. It returns `in`, which matches the plan, so the step succeeds.

Total: about 10 simulated seconds per object, and the 3-step plan takes about 39 seconds. You can watch a passing run in `data/processed/phase1/pinned_trial_500.mp4`.

---

## 10. What went wrong and what it taught us

The first full test scored **0 out of 10**. Here's how that became 80%.

| # | Symptom | How it was found | Cause | Fix | Lesson |
|---|---|---|---|---|---|
| 1 | Lemon knocked over during the apple's pick | Printed positions | I first blamed the left arm, which was wrong | See #2 | A plausible theory isn't proof |
| 2 | Hand slammed into the table at 0.5 s | Logged every contact with a timestamp | The start pose wasn't palm-down, so the arm snapped to it | Start palm-down | Measure instead of guessing |
| 3 | Hand stopped 21 cm short, reported as success | Compared the target with the actual position | "Reached" only meant the maths found an answer | Measure the real hand position | **Silent failures inflate results** |
| 4 | Shoulder hit the torso | Contact log | Waist twisted away from the target | Turn the torso toward the target | Copy how humans move |
| 5 | Objects slid 2–3 cm on release | Measured landing errors | Thumb commanded far past the object kept pushing | Calibrated finger poses by width | Command close to reality |
| 6 | Object landed at 0.751 m instead of 0.761 m | Height exactly matched a block on its side | It was tipping over | Led to #7 and #8 | Odd numbers can carry meaning |
| 7 | Object held tilted 15–18 degrees | Measured tilt at each stage; a sweep showed the height mismatch wasn't the cause | Angled finger pads | Tip the hand forward 0.2 rad | Test hypotheses with experiments |
| 8 | Object pivoting in the hand | Contact log showed only one finger touching | The 5 cm block was narrower than the 5.8 cm finger spacing | Use realistic 7 cm objects | Check that test objects are realistic |
| 9 | Waist drifted to its backward limit | Logged joint values | No posture preference, so small drift built up | Prefer a slight forward lean | Small errors accumulate |
| 10 | Failures close to the body, then at the far corner | Failures clustered by position | Limits of reach | Moved the layout | Fit the task to the robot |

**One earlier decision also changed.** I'd expected to need a shortcut that attaches objects to the hand. Grasping through friction alone worked, so no shortcut is used.

---

## 11. Results and how to read them

*This chapter covers the first version, with coloured blocks. For your real glasses, see chapter 21.*

| Test | Setups | Passed | Notes |
|---|---|---|---|
| Used for tuning, old check | 10 | 9 | Flattering, because I adjusted the layout using these |
| Unseen, old check | 20 | 16 | My check was flawed; see below |
| Same setups, corrected check | 20 | 20 | Shows the effect of the fix |
| **New setups, corrected check** | **20** | **20** | **The number to quote** |
| Unpinned, old check | 10 | 0 | Hand missed due to sway, and later testing showed it also falls on far reaches |

**20 of 20 doesn't mean 100%.** With 20 runs, the true rate could still be somewhat lower, roughly above 85%. It also only covers shifts of up to 3 cm, simple blocks and a pinned pelvis.

### Why "unseen" matters
If you tune using the same setups you test on, the score flatters you, like a student who sees the exam in advance. Every setup here is generated from a random seed, so running new seeds gives setups the system hasn't been tuned on.

### What a pass means
All of these must be true for a pass:
- Every pick was really held.
- Each object measured as `in` its container.
- All objects were still in place at the end.

### The scoring mistake
The old check required the hand to arrive within 2 cm. In all 4 failures of the 16-of-20 test, the hand stopped 2-4 cm short at the far tray, yet **the apple landed correctly in the box**. The check still called that failure and stopped the plan, so the other objects were never tried.

That judged the wrong thing, because the goal is the object's position, not the hand's. So I changed it, and retested on new setups in case the fix merely flattered the score. A score rising because a check changed should always make you suspicious, so asking why it changed is the right habit.

### Limitations of the block version
*Since then, item 4 is solved for the glasses (the plan comes from your video), and item 5 too (two arms are optional). See chapter 21 for today's limits.*

1. **Pelvis pinned**, so there's no real balance.
2. **Three-finger hand**, where the challenge suggests five.
3. **Simple blocks**, not real shapes.
4. **Hand-written plan**: your video isn't connected yet, so **the challenge's hard rule isn't met yet**.
5. **One arm only.**
6. **Small test size:** 20 runs give a rough estimate, not a precise rate.
7. **The trays don't move**, so they can't be knocked around.
8. **The physics isn't validated against reality**, because no real robot is involved.

---

### Does the retry loop actually help?
To find out, I made the task harder on purpose. Each condition ran 8 setups (seeds 800-807).

| Condition | One attempt per step | With up to 4 attempts | Average attempts when it worked (3 is perfect) |
|---|---|---|---|
| Normal | 8 of 8 | 8 of 8 | 3.0 |
| Heavy, 300 g | 8 of 8 | 8 of 8 | 3.0 |
| Rotated 30 degrees | 8 of 8 | 8 of 8 | 3.0 |
| Slippery, friction 0.3 | 6 of 8 | 6 of 8 | 3.0 |
| Very slippery, friction 0.15 | 5 of 8 | 6 of 8 | 3.2 |
| Weak grip, 0.5 N·m | 8 of 8 | 8 of 8 | 3.0 |
| **Weak grip, 0.3 N·m** | **0 of 8** | **8 of 8** | 6.0 |
| **Very weak grip, 0.15 N·m** | **0 of 8** | **8 of 8** | 9.0 |
| Object knocked out after placing | 8 of 8* | 8 of 8 | 4.0 |

*Even with one attempt, the final recheck still runs. With no recheck at all, that object would just stay out of place.

**How to read it:**
- **Where retries matter:** a weak grip, because each retry truly squeezes harder. It starts too weak, measures the failed lift, and grips harder until it holds.
- **Where they don't:** when the first attempt already works, it costs nothing extra.
- **Slipperiness mostly isn't fixed.** Trying again with the same approach only gained one run out of 16.

**The first version of this test was wrong twice, and that's instructive:**
1. **Retries had no effect**, as "grip harder" only changed the target finger position. With a force limit, the fingers couldn't push any harder. Now it raises the force limit too, up to a maximum of 2 N·m.
2. **"Slippery" wasn't slippery.** MuJoCo uses the higher friction of two touching surfaces, so the grippy fingers cancelled the setting. It only counted once I gave the object priority.

A test that passes can be measuring nothing. So when a result looks good, it's worth asking what would have made it fail.

**These conditions are chosen by me to expose failures. They aren't real-world measurements,** and the sample is small.

---

## 12. What comes next

*This was the plan at the start, kept to show how the project grew. Where things stand now:*

| Phase | Status |
|---|---|
| 2. Key moments from the hand | Built. For your glasses, rest spots of the glasses decide the timing instead, because the hand gave too many moments |
| 3. Objects and real positions | Done for your glasses, without SAM 2 or a marker: bright tops and the table's corners (chapter 14) |
| 4. Writing the plan | Done without AI; Claude is optional (chapter 17) |
| 5. Closed loop | The robot's result is checked against your video; step checks say where a run failed (chapter 20) |
| 6. Scoring | Done; results in chapter 21 |
| 7. Submission | In progress |

### Phase 2: finding key moments [Code built, not yet tried on real video]
**You:** record about 10 videos using `docs/RECORDING_GUIDE.md`, with rough labels.

**Code:** MediaPipe finds your hand in each frame. We compute its speed: distance moved between frames, divided by time. Then we smooth that curve and find the valleys.

```
speed
  |   /\      /\      /\
  |  /  \    /  \    /  \
  |_/    \__/    \__/    \_
        pick   place   pick
```

**Pass if** at least 90% of labelled moments are found within 0.5 seconds.

### Phase 3: finding objects and real positions [Code built, not yet tried on real video]
- **Grounding DINO or Florence-2** find objects in the first frame from text descriptions.
- **SAM 2** follows each object through the video, keeping the same ID. It needs a GPU.
- **Homography** converts pixels to table centimetres.

**Homography, explained:** when you photograph a flat table from an angle, the table looks like a trapezoid. There's a mathematical mapping that undoes this, and it's fully determined by four known points. Our printed marker provides exactly that: four corners of a square known to be 10 cm wide. So if the apple's base is at pixel (640, 400), the mapping might tell us it's at table position (35 cm, −28 cm).

**Where the marker goes:** at the phone's end of the table, 48 cm from the robot's edge. I first put it near the robot, but drawing the view from a 75 cm tripod showed the blocks standing in front of it. Nothing can stand between the phone and something right beside it. Blocks still measured within about 1.1 cm in a rendered test, and trays within about 2.7 cm.

**A tracker bug that this view exposed:** to find an object's base, the tracker used only the lowest row of its coloured pixels. If the bottom edge looks slightly tilted, that row is a single corner, so the position jumped sideways by up to half a block. A test block read 3 cm off. It now uses the lowest pixel in every column.

**Testing against rough footage:** I damaged a rendered video the way a phone might, with dim light, noise, compression, colour-tinted light, shake and a passing hand. Blocks jumped about 7 cm off. A block's shaded front face fell below the tracker's brightness limit, so it mistook the top face for the whole block, and the colour-picking tool made this worse by setting the limit from the bright top face. With the limit lowered and colours picked per video, the worst block in those conditions was about 1.2 cm off, and trays about 1 cm. These are imitations of phone problems, so real footage can still surprise us. Two limits remain: moving the camera about 60 pixels during a video gave a wrong plan, and at 35% brightness a block was still 7 cm off. So keep the phone still and the room bright.

It only works for points **on the table's surface**. A point at the top of an object would be misplaced, so we use the point where the object touches the table.

**Pass if** the median error is 3 cm or less against a ruler.

### Phase 4: writing the plan [Built without AI; AI version needs a model choice]
**First, decide which AI model to use.** You chose to leave this until now. It needs an API key: a private password that lets programs use an online service.

We then adapt SeeDo's prompts and compare two approaches: the AI choosing the relation alone, and the AI choosing the objects while measured positions choose the relation.

**Pass if** SSR is at least 70%.

### Phase 5: closed loop, the main idea [Code built, not yet tried on real video]
1. The robot runs the plan.
2. We compute the final arrangement in simulation.
3. We compute the final arrangement from the last video frame.
4. If they differ, we classify the error and respond:
   - **Execution error:** adjust the grip settings and retry.
   - **Plan error:** ask the AI again, telling it what went wrong.

**Important honesty rule:** this check can use only information from the video, never your hand-written labels. Otherwise it would be cheating.

### Phase 6: scoring [Code built, not yet tried on real video]

| Metric | Question it answers |
|---|---|
| **TSR** (task success rate) | Was the whole plan exactly right, in order? |
| **FSR** (final-state success rate) | Was the final arrangement right, in any order? |
| **SSR** (step success rate) | How many steps were right, in order? |
| **Execution success** | Did the robot physically succeed? |
| **Attempts to success** | How many tries did it need? |

**Example.** The true plan is A, B, C. The AI says A, C, B, and all three end up in the right containers:
- **TSR = 0**, because the order is wrong.
- **FSR = 1**, because the final arrangement is right.
- **SSR = 2/3**: A matches, C matches later in the true list, and B can't match after C without going backwards.

A detail: the paper prints a pointer method, but SeeDo's released code uses the longest common subsequence, meaning the longest run of correct steps kept in order. We use that.

Our dataset is small and self-recorded, so our numbers can't be fairly compared with the paper's.

### Phase 7: submission [Planned]
A public GitHub repo, a README, example videos, a sample of your recorded data and a note on how you collected it. The README must be honest about what worked and what didn't.

**Things to fix before going public:**
- Put the data sample in `data/raw/sample/`.
- Keep the session transcripts out, as they contain other participants' names (already ignored).
- Keep the papers out (already ignored).

### Optional: adding training
The simulator could generate training data:

```
Scripted robot runs many times in simulation -> saves camera images and motions as episodes
    -> train ACT on a GPU -> the trained policy replaces the scripted arm motion
    -> score both on the same tests
```

- **When:** after Phase 5, before the benchmark.
- **What's kept:** everything. The scripted robot creates the data, and the same checks score the result.
- **What's needed:** a simulated camera, GPU access and time.
- **Your data's role:** this path trains on simulated data, so your recordings wouldn't directly teach the motion. A stronger link to your data would need retargeting, which is more work.

**This is an open decision, and the official text makes it more important.**

---

## 13. How the project is run

### Files that guide the work

| File | Purpose |
|---|---|
| `CLAUDE.md` | The project's "brain", which Claude Code reads automatically at each session start |
| `PLAN.md` | Phases, pass conditions and status |
| `CONVERSATION_LOG.md` | Short record of discussions and decisions |
| `notes.md` | This file |
| `task.md` | The official challenge. Don't change it |

### Git
**Git** saves snapshots of the project, called **commits**, so you can see the history and go back. **GitHub** is a website that hosts it publicly. Nothing leaves your Mac until it's pushed.

`.gitignore` lists files git must ignore, such as secrets and downloads.

**Current state:** all work is committed on your Mac, on the `overnight` branch. Nothing is pushed until you say so, and the GitHub repo stays private until you decide.

### Commands to try
Run these from the project folder. Lines starting with `#` are comments.

```bash
# Run all tests (under a minute when the Mac is idle, up to about 8 minutes when busy)
.venv/bin/pytest

# Tests with a report of how much code they cover
.venv/bin/pytest --cov=src --cov-report=term-missing

# Print the map of where the hand can reach
PYTHONPATH=. .venv/bin/python scripts/reach_map.py

# Turn a video into a plan and robot run (needs your video and colour settings)
PYTHONPATH=. .venv/bin/python main.py --video data/raw/video_01.mp4 --step full

# Watch the task live in the MuJoCo window
PYTHONPATH=. .venv/bin/mjpython scripts/watch_live.py

# Run the plan once on a new random setup (saves a video)
PYTHONPATH=. .venv/bin/python scripts/phase1_gate.py --trials 1 --first-seed 900

# Watch a passing run
open data/processed/phase1/pinned_trial_500.mp4

# Your real video to the robot (Task 2 take 1)
PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task2_triangle_right_take1.mov

# The same, with Claude finding the glasses and the model choosing arms (uses the API key)
PYTHONPATH=. .venv/bin/python scripts/glass_from_video.py --video data/raw/task2_triangle_take3.mov --find-glasses --choose-arms

# Live control with your webcam (c resets, q quits)
PYTHONPATH=. .venv/bin/python scripts/live_teleop.py

# Live control matched to a recorded video's glasses
PYTHONPATH=. .venv/bin/python scripts/live_teleop.py --source data/raw/task2_triangle_right_take1.mov --match-video
```

`--first-seed` chooses which random setups to use. The same seed always gives the same setup.

### Try changing something
Open `configs/default.yaml`, change `hover_height` from `0.17` to `0.25`, and run the one-trial command above. The robot will lift higher. Change it back afterwards.

---

---

# Part Two: your real videos and real glasses

Chapters 1 to 13 told the story of the first version, which sorted coloured blocks into trays using a printed marker. Then you recorded the real tasks with steel glasses and no marker, and the project changed a lot. Part Two explains everything built for your videos, from zero, in the order a video flows through the system.

```
your video ──► read the table ──► find the glasses ──► work out the moves ──► choose arm(s)
                                                                                  │
   check each step ◄── robot builds it in MuJoCo ◄── place glasses where they stood ◄┘
```

---

## 14. Reading your videos: steel glasses, no marker

### The problem
The first version found objects by colour and measured positions with a printed marker. Your glasses are steel, so they have almost no colour, and you recorded without a marker. Both old tools were useless.

### Finding glasses with no colour
Each glass stands upside down, so its closed end faces up. Seen from above, that closed end is a **bright circle** against the dark table. So the reader (`src/perception/glass_tracker.py`) looks, in every frame, for round patches much brighter than that frame's table. It compares within one frame, so when the camera changes its brightness by itself, nothing breaks.

**Example.** In `task2_triangle_right_take1`, three bright circles sit still for the first 3 seconds. At 3.0 s one of them disappears from its spot, and a new one appears beside the others a few seconds later. That's one move.

### Rest spots and moves
- A **rest spot** is a bright circle seen in the same place for a while: a glass standing still.
- Reflections are thrown out, because they land off the table or flicker.
- A glass being carried doesn't stay anywhere long enough to count.
- A **move** is one rest spot ending, and a new one starting soon after.

In take 1 the moves happen at **3.0 s, 12.6 s and 20.8 s**, and they come out in the right order.

### Using the table instead of a marker
A rectangle photographed from an angle looks like a squashed shape. Because we know your table is **80 × 50 cm**, the way it's squashed reveals two things: the camera's lens, and where the camera stands (`src/perception/table_view.py`). After that, any pixel can be turned into a real position on the table.

**How the corners are found:**
1. You clicked four rough corners once (`data/labels/`).
2. The code fits a straight line along each table edge.
3. The real corners are where those lines cross. This matters because a real table's corners are rounded.

Moving your rough clicks by up to 20 pixels changed glass positions by only about 0.1 cm.

### Correcting for height
A glass's top is 11.5 cm above the table, or 23 cm for the glass on top of the tower. Seen at an angle, a raised point appears farther away than the glass's base really is. Since we know the camera's position, we can correct for this.

**Example check:** after the correction, the top glass in take 1 lands within **0.2 cm** of the midpoint of the two lower glasses, which is exactly where it should be.

### Recognising the flip (Task 3)
A flipped glass shows its **open end**, which is dark, not bright. So Task 3 shows four starting spots but only three bright new spots. To confirm the flip, the code looks where the tower's top should be. At hand-picked spots, closed tops measured 80–88% bright pixels and the opening measured 0%.

### When the top glass isn't seen
Against the bright floor, the top glass can go unnoticed. If the reader sees three glasses leave and two end side by side with nothing above them, it concludes the third must be on top. The report then says `top_glass_seen_in_video: false`.

### What this doesn't show
The reader's settings were chosen by looking at your first videos, so later takes are the fairer test. Of the eight later Task 2 takes, four were refused because a starting glass was missed. That's the reader's main weakness, and chapter 17 shows how Claude fixes it.

---

## 15. Task 3: choosing the arm and where to stand

### The problem
The flip only works if the robot holds its forearm level and turns its wrist. It can only do that towards the working arm's **own side**: reaching across its chest, the right arm hits the robot's own body. In your first Task 3 video, the tower is on the robot's left of the glasses, so the right arm can't do it from the middle of the table.

### The fix: choose the arm, then where to stand
`plan_stand` in `src/reasoning/fit_layout.py`:
1. **Choose the arm.** If the tower is on the robot's left of the glasses, use the left arm.
2. **Choose where to stand.** Stand so the tower is 28.5 cm to the working arm's side, the middle of the area where the flip was tested.

**Example.** For `task3_flip_right_take1`, this means the left arm, standing 23.4 cm to the right of the table's middle. The robot can't walk, so the simulation simply starts it there.

### Why the left arm needed no new tuning
The G1's arms are mirror images. `src/simulation/mirror.py` imagines a mirror along the robot's centre line:
- a point 10 cm to the right becomes 10 cm to the left;
- a joint that swings forward and back keeps its angle;
- a joint that turns sideways changes sign.

So settings are written once, for the right arm, and the left arm gets their mirror image.

### What later testing showed (4 Nov)
The step checks (chapter 20) showed that on later takes, **Task 3 fails because a glass is out of reach**:
- In takes 2, 3 and 7, your glasses stood 20–39 cm from your edge of the table. The flip was only tested with starting glasses 13–19 cm from that edge.
- The robot's hand stopped **11, 31 and 41 cm** short of a glass.

**Experiment: stand closer.** The robot normally stands 16 cm back from the table's edge. Moving it closer, with the glasses unchanged:

| Take | Normal | 6 cm closer | 10 cm closer |
|---|---|---|---|
| 2 | missed a glass by 11 cm | missed by 8 cm | **built and flipped** |
| 7 | missed by 41 cm | missed by 4 cm | reached that glass, then missed the fourth by 5 cm |
| 3 | missed by 31 cm | missed by 45 cm | not tried: its glass is 47 cm to the side, not too deep |

So standing closer rescued one take (one run). It isn't part of the system yet: it would need a rule for how close to stand, and a check that takes 1 and 9, which work now, still work.

**Lesson:** a robot that can't walk, standing in one spot, can't reach glasses spread that widely. A real robot would step between glasses. This is a real limit of our setup, and the results say so.

---

## 16. Choosing arms for Task 2, and the logistic regression model

### Three ways to use the arms
1. **One arm for everything (the nearest arm).** `nearest_arm` averages how far the glasses and tower are to the robot's side and picks that side's arm. On layouts toward the robot's left, this built 24 of 30, against 10 of 30 for the right arm alone.
2. **An arm per glass (`--two-arms`).** `arm_for_each_glass` gives each glass the arm on its own side. When arms swap, the old hand rises, moves out beside the body and hangs back, and the new one comes in the same way. A direct swap swept glasses off the table.
3. **Let a small learned model choose (`--choose-arms`).** This is the logistic regression model.

### What a logistic regression model is, from zero
Imagine scoring a layout like a judge with a checklist:
- start with a base score;
- add or subtract points for each thing you notice, such as "the tower is far to the side";
- a big positive total means "likely to succeed", a big negative total means "likely to fail".

The points for each item are the **weights**. Logistic regression learns those weights from examples, then squeezes the total into a chance between 0% and 100% using the **sigmoid**, an S-shaped curve:

```
chance
100% |                 ______
     |               /
 50% |-------------/----------
     |           /
  0% |______ /
     +-------------------------- total score
        negative   0   positive
```

**Ours** has about 16 weights and is written in plain NumPy. It looks at 7 numbers describing the layout (in centimetres), for example:
- how far forward the tower is;
- how far to the side the tower is;
- how far the glasses are from the tower.

It asks: "with one arm, what's the chance? With an arm per glass, what's the chance?" Then it picks the higher one.

### How it learned
It was **trained on 90 simulator layouts, each run both ways (180 runs)**. Training means adjusting the weights, step by step, until its chances match what really happened. This method is called gradient descent.

**What it learned, as weights** (bigger number = stronger pull):

| As this grows... | Weight | Pulls toward |
|---|---|---|
| Tower farther to the side | −1.24 | one arm |
| Glasses farther from the tower | −0.77 | one arm |
| Glasses farther to the side | +0.42 | an arm per glass |
| Tower farther forward | +0.39 | an arm per glass |
| More glasses on the far side of the tower | +0.01 | (hardly matters) |

**In plain words:** an arm per glass works when the tower is near the middle. One arm works better when the tower is far to one side, because otherwise the second arm must reach across the body.

### Worked example
Say the tower is 23 cm to the robot's right and the glasses are close to it. "Tower far to the side" gets a big negative weight, so the "arm per glass" score drops below the "one arm" score. The model picks one arm. In the real test this exact kind of layout built with one arm and failed with an arm per glass.

### How well it works
Tested on **60 new layouts it never saw**:

| Choice | Triangles built |
|---|---|
| Always one arm | 43 of 60 |
| Always an arm per glass | 44 of 60 |
| Hand-written rule (arm per glass if the tower is within 14 cm of the middle) | 49 of 60 |
| **The logistic regression model** | **50 of 60** |
| Best possible, if you knew every answer in advance | 51 of 60 |

It beats always choosing the same way, but only by one layout over a simple rule.

**Are its chances honest?** Roughly, and a little cautious. Runs it gave about a 77% chance really succeeded 87% of the time, and runs it gave about 88% all succeeded (14 of 14).

The picture `docs/results/logistic_regression_explained.png` shows all of this: maps of the table coloured by its choice, the weights, and these results.

### Swapping arms without knocking anything (your idea, 4 Nov)
**The problem.** In hard take 1, while handing over from the right arm to the left, the robot twisted its whole body. That shoved the table, and a glass ended up off it.

**Your idea:** keep the arms clear of the table while swapping. The swap now goes:
1. The old hand rises **straight up**, about 30 cm above the table, higher than even the finished tower.
2. It moves, still high, to a point **beside the body, just off the table's edge**, and hangs back.
3. The new hand comes to the same kind of point on its own side, moves **high over to the table's middle**, and only then comes down.
4. **The waist stays still** throughout, so the body can't twist into the table.

**Result:** hard take 1 went from a knocked glass to **built, 0.3 cm** from yours.

**Copying which side you put the first glass on (tried, not kept).** You noticed the robot put its first lower glass on the opposite side to you. That's deliberate: with one arm, it fills its far side first, so it never reaches past a glass it just placed. Copying your order instead broke take 4 (reach) and hard take 1 (place). So it's an option, `--copy-side-order`, off by default.

### What this doesn't show
- All of it is simulator layouts with the robot's pelvis pinned.
- On your 9 real Task 2 videos it chose an arm per glass every time, so on real videos it never had to make a hard choice.
- **It isn't a world model.** It predicts only "will this succeed?", not how the scene changes step by step.

---

## 17. The vision language model (Claude) helper

### What it is
A **vision language model (VLM)** reads pictures and text and answers in text. We use **Claude Sonnet 5.5** through Anthropic's API (`src/reasoning/vlm_client.py`). Claude never moves the robot and never gives the final positions. It **finds and decides**, and measured positions stay in charge.

It does three jobs, each switched on by an option:

| Option | Job | Example |
|---|---|---|
| `--find-glasses` | Points to every glass top in an early frame. Glasses the reader missed are added (`src/perception/vlm_rescue.py`) | Take 3: the reader saw only two starting glasses; Claude pointed to the third, and the triangle was built |
| `--instruction "..."` | Turns a typed instruction into an allowed task and order (`src/reasoning/instruction.py`) | "Start with the glass nearest the right edge" gives order: right glass first |
| (clutter test) | Tells glasses apart from other bright things | In simulated scenes with other bright objects on the table, it found exactly the three glasses |

### Why it's safe
Claude's answer is checked against a short list of allowed tasks and orders. If the answer doesn't fit, the run stops with a clear message instead of guessing.

### Results
| Test | Without Claude | With Claude |
|---|---|---|
| Task 2, your 9 real videos | 3 of 9 | **8 of 9** (tower within 0.0–0.5 cm of yours) |
| Task 3, your 9 real videos | 2 of 9 | 2 of 9 (no change: Task 3's problem is reach, chapter 15) |
| 10 cluttered simulator scenes | a brightness rule saw 20 bright patches | **10 of 10** found exactly the glasses, about 1 cm off |
| 8 typed instructions (Task 3 take 9) | — | 8 of 8 put the glasses in the right order |

**Not a fair test:** the reader was tuned on these same videos.

### Which Claude model
Haiku, Sonnet and Opus were compared on 10 cases. All three got 9 of 10 fully right. Sonnet was the most accurate (0.33 cm average error) and far cheaper than Opus, so the project uses Sonnet.

### Version 3: Claude reads the start *and* the end (4 Nov)
**The problem you spotted:** Claude only looked at the first frame. Where the glasses *finished* was still read from shiny tops, and shine depends on the light.

**Who does what now:**

| Job | Tool | Why |
|---|---|---|
| Which things are glasses at the start | Claude, one early picture | It recognises glasses and ignores clutter |
| Where the glasses are at the end | Claude, one picture near the end | Same reason, and it doesn't need a shiny top |
| *When* each glass moved, so the order | The frame-by-frame reader | It needs every frame, and costs nothing; Claude on every frame would cost about $5–7 a video |
| Pixels to centimetres | The table's corners | Exact geometry |

Claude now costs about 1–2 cents a video (2 pictures).

**How the two are combined.** A Claude point and a reader spot within 30 pixels (about 1–2 cm) are the same glass:

| Case | Meaning | Used |
|---|---|---|
| Both see it | Same glass | The reader's position, the more exact one |
| Only Claude | The reader missed it (a dull top) | Claude's point |
| Only the reader | The reader misread something (clutter, a glint, an arm) | Dropped |

**Worked example: cluttered take 2.** At the end, the reader saw the two lower glasses but not the dull top glass. Claude saw all three, so the top glass came from Claude's point. The plug and the box, which the reader saw, matched no Claude glass and were dropped. The robot built the triangle 0.4 cm from yours.

**More rules found by your hard videos.** Each one first failed a fair test, and the rule came from its failure:

| Video | What went wrong | Rule |
|---|---|---|
| Cluttered take 1 | One dull glass's departure couldn't be seen | Work it out from the order: a glass leaves before it lands |
| Hard take 1 | A 4th glass never moved | A glass Claude sees in the same place at both ends, which never left, takes no part |
| Hard take 1 | A white tube beside a glass stayed bright after the glass left | No glass there at the end, per Claude, means it left |
| Hard take 3 | The steel laptop stand glinted mid-video | Every landing must be where Claude sees a glass at the end |
| Hard take 4 | A hand hid a lower glass until after the top was placed | Choose the top glass by position, not arrival time |

**Sunlight.** Your video lit only by sunlight was read correctly on its fair test. Claude added one starting and two end glasses the shiny-top reader missed in the new light. The robot then failed for a different reason: the tower was too close to its chest.

**What these don't show:** each rule came from the video that failed, so those videos are "not a fair test" once fixed. A fresh video is the honest check.

### Version 4: SAM 2 follows each glass by its shape (optional)
**SAM 2** (Segment Anything 2, from Meta) outlines any object you click on, and keeps outlining it through a video. We use its smallest model, "tiny", 156 MB.

**How it's used:**
1. Claude points at each glass in the first frame. Those points are SAM 2's clicks.
2. SAM 2 outlines each glass in every third frame, all the way to the end, even while it's in your hand.
3. Each outline's track gives that glass's own start, **when it left** (the outline moved away and stayed away), and **where it ended**.
4. These become ordinary resting spots, so all the usual checks still run: the order of moves, which glass is on top, the triangle's shape.

**Worked example: your sunlight video.** The shiny-top reader, confused by the new light, timed the moves at 2.6, 3.4 and 8.2 s. Watching the video, the glasses really left at about 2.7, 9 and 15 s. SAM 2 said **2.7, 8.9 and 14.5 s**. It never took the laptop stand, tube or plug for a glass.

**Its catch: speed.** On this Mac it took about 70 minutes for a 21-second video, though other jobs were running. On a proper GPU it would take about a minute. So it's an option: run `scripts/sam_track_glasses.py` once (it saves the tracks), then add `--sam-tracks`.

**Why it's better than shiny tops:** each track keeps its glass's identity, so it knows directly which start became which end. The reader had to guess that from the order of events.

### The API key, and an expensive lesson
The key lives only in `.env`, a file git ignores, under the name **`VIDEO2ROBOT_CLAUDE_KEY`**.

It used to be called `ANTHROPIC_API_KEY`. VS Code quietly loads `.env` into new terminals, and Claude Code (the assistant writing this project) also looks for `ANTHROPIC_API_KEY`. So for a few hours, the assistant's own coding work was billed to your API balance instead of your Claude Pro subscription: $15 in three hours. Renaming the key fixed it.

**Lesson:** a secret's name can matter as much as where it's stored.

---

## 18. Copying your hand's carry (`--hand-path`)

### What it does
By default the robot carries each glass along its own smooth path. With `--hand-path`, it copies **how high you lifted the glass, and how fast you moved, during the carry**.

### How
1. MediaPipe finds your hand in each frame. `follow_moving_hand` follows the hand that's moving.
2. `carry_window` finds when the carry starts and ends.
3. `profile_from_pixels` turns the hand's pixels into height above the table and progress along the carry. It assumes your hand moves in the upright plane between the glass's start and end.
4. `robot_carry_path` fits that profile between the robot's own start and end points:
   - heights are kept between 20 and 38 cm;
   - the speed is slowed down if it's too fast for the robot.

**Example.** One copied carry from take 2 came out 60 cm high, which isn't believable for a glass carried across a table. So `is_believable` dropped it, and the robot used its normal carry for that glass.

### What it doesn't copy
- The sideways route: the robot still carries in a straight line.
- The reach toward the glass.
- Your fingers.

Takes 1, 2, 5 and 6 succeed with it. That's not a fair test, because these takes were used while building it.

---

## 19. Live hand control, and matching a recorded video

### What it is
`scripts/live_teleop.py` lets a webcam (or a recorded video) drive the robot directly: you move, it moves. This is called **teleoperation**. Nothing is learned; the robot copies you frame by frame. It's a separate demo from the video-to-task pipeline, and it doesn't change those results.

### The 21 hand points
MediaPipe gives **21 points** (landmarks) on your hand in every frame: the wrist, four points on each finger, and the thumb. It gives them twice:
- **in the picture**, in pixels;
- **in metres**, around the hand's centre, as if measured with a ruler.

### What the robot copies
- **Fingers.** For your index and middle finger, the code adds up how much each finger bends at its knuckles. Straight is about 0.6 radians, fully curled about 3.2. That becomes "0% to 100% closed", and the robot closes its matching finger the same amount.
  - **Example:** holding a glass, your index read **74% closed**.
  - **Known problem:** the thumb is measured by how close its tip comes to your palm. Wrapped around a wide glass, the thumb stays far from the palm, so it reads 0% even while gripping.
- **Wrist.** `hand_frame` builds a direction from your wrist and knuckles, and the robot's wrist turns by however much yours has turned since the start, capped at about 69°.
- **Position.** In plain live mode, moving your hand moves the robot's hand by a set amount, relative to where you started.

### Safety, and lessons from the first live try
The first try with your Logitech webcam failed:

| What went wrong | Why | Fix |
|---|---|---|
| The arm shot up and stayed "out of reach" | Control started on your hand while you walked into view | It now starts only once your hand is **held still** for 15 frames, and starts over after 2 s without a hand |
| The robot's body spun and leaned | The arm solver also moved the waist | Only the arm moves; the waist is **locked** |
| The window closed at once | The video saver needs an even width, and the camera gave 1493 pixels | The width is rounded to an even number |
| Only 8 frames a second | Full-HD pictures are slow to process | The camera now uses 1280 × 720 |

After the fixes, the robot followed your hand in **941 of 1460 frames**.

### Matching a recorded video (`--match-video`)
**Your point:** copying a recorded video means nothing if the simulated glasses stand somewhere else. So this mode does two things.

**1. Place the glasses where they stood.** It reads the video's glasses (chapter 14) and puts the simulated glasses there.

**2. Place your hand on the real table, in 3D.** One camera can't see depth directly, but similar triangles help:

> A thing looks **half as big** when it's **twice as far away**.
> distance = lens focal length × real size ÷ size in pixels

**Worked example.** Your camera's focal length is about 1900 pixels. Suppose your palm is 8 cm across and looks 120 pixels wide:
distance = 1900 × 0.08 ÷ 120 ≈ **1.27 m** from the camera.

The pixel tells us the *direction* from the camera, and the size tells us *how far*. Together they give a point in 3D. The table's corners say where the camera is, so that point becomes a position on the table, the same way glasses are placed.

**The correction factor.** MediaPipe assumes an average hand size. If yours is bigger, your hand looks closer than it is. So one factor per video corrects the distance. It's chosen so that, at the moment you took each glass, your palm lands on that glass. This uses only the video.

**Results, Task 2 take 1 (one run):**
- Factor 0.91. Your palm landed **0.9, 1.3 and 0.6 cm** from each glass as you took it.
- The robot's hand was **2.6, 15.3 and 5.9 cm** from each glass at those moments. Glass 2 had already been nudged 13 cm by the robot's arm before you reached it.
- The robot keeps its palm facing down in this mode. Also copying your wrist asked for poses its arm couldn't reach, and the arm pointed straight up.

### What this doesn't show
- On other takes, the hand's distance is less reliable: 1 to 30 cm off. A hand turned sideways looks smaller, so it seems farther away.
- The robot's hand is bigger than yours, so passing near a glass can knock it where your hand cleared it.
- "Out of reach" in the window means the arm couldn't match the exact pose within 5 mm while chasing a moving target. That's stricter than it sounds.

---

## 20. Step checks: finding where a run fails

### Why
The final check only says "the tower is right" or "the tower is wrong". When a run fails, that doesn't tell you **where** it went wrong, so you can't fix the right thing.

### What is checked
After every step, `src/verification/steps.py` measures exact simulator positions:

| Step | Passes if | Tolerance |
|---|---|---|
| **reach** | the hand got to the gripping point | within 2 cm |
| **grip** | the glass is in the hand | within 6 cm of the hand |
| **lift** | the glass rose with the hand | at least 5 cm |
| **carry** | the glass is still in the hand over the target | within 6 cm |
| **twist** (flip only) | the glass is still in the hand after turning over | within 6 cm |
| **place** | the glass is over its target just before letting go | within 1 cm |
| **release** | the glass stands straight after letting go (upside down for the flip) | leaning 10° or less |
| **others still** | the other glasses weren't knocked | each moved 1 cm or less |

The report's **`failed_at`** names the first step that went wrong.

### Worked example
A failed flip from the simulator test reports:
```
failed_at: cup_d: release
  leaning 86 degrees
  cup_a moved 11.2 cm, cup_b moved 10.7 cm, cup_c moved 19.1 cm
```
In plain words: everything went fine until the robot let go of the flipped glass. Then it fell over, and the whole tower got knocked.

### Proof that checking changes nothing
The checks only measure. On the 30-layout flip test:
- same result, **28 of 30**;
- the same two runs failed;
- no step was ever flagged in a run that succeeded.

### What it found
**Task 3 fails mostly at reach** (chapter 15):
- your real takes 2, 3 and 7: all at reach;
- the simulator's random layouts: 19 of 22 passed, with 2 failures at reach and 1 at grip.

---

## 21. All results on your real videos, and how to read them

These are the numbers to quote. "Real video" means the robot's plan and glass positions came from your recording. The robot itself is always simulated, with its pelvis pinned.

### Task 2: the triangle (9 real videos)
| Setup | Built |
|---|---|
| No AI, nearest arm | 3 of 9 |
| With Claude finding glasses | **8 of 9** |
| With Claude and the logistic regression model choosing arms | 8 of 9 |
| Copying your hand's carry (`--hand-path`) | takes 1, 2, 5, 6 |

### Task 3: the flip (9 real videos)
| | Result |
|---|---|
| Built and flipped | **2 of 9** (takes 1 and 9) |
| Refused while reading the video | 4 (takes 4, 5, 6, 8). With Claude, take 6 is read but then fails |
| Failed at reach (a glass too far for a robot that can't walk) | takes 2, 3, 7. Standing 10 cm closer, take 2 succeeded (one run, not yet part of the system) |

### What the robot can do, with exact simulator positions
| Test | Result |
|---|---|
| Task 2, 30 random layouts in its working area | 28 of 30 |
| Task 3, 30 random layouts in its working area | 28 of 30 |
| Task 3, choosing where to stand, random layouts | 25 of 30 |

### How to read these honestly
- **"Not a fair test"** means the code was adjusted while looking at those same videos, so new videos would probably score lower.
- **Simulator layouts** test the robot's body, not the video reading.
- **9 videos is small.** 8 of 9 could be anywhere from about 55% to 98% in the long run.
- **The pelvis is pinned:** the robot can't fall, and it can't walk.

### Where to see it
- `data/processed/showcase/`:
  - your recordings with the tracked hand drawn on, beside the robot, for both tasks;
  - the live-control replay.
- `docs/results/`: every saved test and picture.

---

## 22. How the project got here: one idea at a time

Real projects don't go in a straight line. This one went in a loop, again and again:

```
try the simplest thing ──► test it on stated data ──► count the failures
        ▲                                                    │
        │                                                    ▼
 keep it only if it helped ◄── retest on NEW data ◄── one idea aimed at the cause
```

This chapter tells the whole project through that loop, in seven stages. Each stage says:
- what we started with;
- what we tested it on;
- how often it failed, and what the failures showed;
- the idea that came from them;
- what happened after.

**Why "new data" each time?** If you test on the same setups you tuned on, the score flatters you, like a student who has seen the exam. Where that wasn't possible, as with your 9 real videos, the result is marked **not a fair test**.

The same story, row by row, is in `docs/JOURNEY.md`.

---

### Stage 1: teaching the robot to pick and place (simulator, coloured blocks)

**Starting point.** The G1 sorts three blocks into trays. It uses inverse kinematics (chapter 3) and a plan written by hand.
**Tested on:** 10 random setups. **Result: 0 of 10.** Nothing worked.

**What the failures showed, and what each led to:**

| Failure seen | What it told us | Idea | After |
|---|---|---|---|
| The hand slammed into the table half a second in | The arm started in a twisted pose and snapped to the target | Start palm-down; log every contact with its time | The slam stopped |
| The hand stopped 21 cm short, but the code said "reached" | "Reached" only meant the maths found an answer, not that the hand arrived | Measure where the hand really is | Silent failures were caught |
| Blocks slid 2–3 cm when let go, and were held tilted 15–18° | The fingers pushed past the block, and the finger pads are angled | Finger poses set from the block's width; tip the hand forward 0.2 rad | Sliding and tilting fixed |
| 4 of 20 "failures" had put the object exactly right | The check judged the hand's position, not the object's | Judge where the **object** ends up | **20 of 20** on new setups |

**Then a stress test.** With a deliberately weak grip, objects slipped every time: **0 of 8**. The idea was to measure the failed lift and retry with a stronger grip. Result: **8 of 8**.

**Lesson:** measure what you care about (where the object lands), not a stand-in for it (where the hand went).

---

### Stage 2: the real tasks in the simulator (cups)

**Task 2, the triangle.** The idea: place two cups side by side, then the third across their **measured** tops. **Tested on 30 setups: 30 of 30.** It worked first time, because stage 1's lessons carried over.

**Task 3, the flip.** Turning a cup over with the palm above it would leave the palm underneath at the end, where it would hit the tower. So the idea was to grip the cup **beside** it and turn it like food on a spit.

**First test, 9 new setups: 1 of 9.** Then one fix at a time, each tested on new setups:

| Failure seen | Idea | Result |
|---|---|---|
| The cup crept out of the hand over a few seconds | Stiffer friction setting (`impratio`) | 1 of 12: a new problem showed up |
| Fingers carried the cup away when lifting; the robot's last arm swing knocked the tower | Back out sideways before lifting; remove the swing | 5 of 12 |
| The palm hit the cup's wide base on the way down | Come down beside it, slide in, then centre the hand | 9 of 15 |
| Cups thrown, slipping out of the fingertips, flips ending short | Turn only the wrist; hold the cup deeper; keep the forearm level | 21 of 24 |
| The last 3 failures were flipped, but over 2 cm off-centre | Aim at the top cup's **measured** position, and cancel a small shift on release | **50 of 50** |

**Lesson:** one fix at a time shows which fix helped. A fix can also reveal the next problem, as the 1 of 12 did.

---

### Stage 3: your real videos and real glasses

**Starting point.** The block version: colour tracking and a printed marker.
**Tested on:** your first Task 2 video.
**Result:** unusable. Your glasses are steel, with almost no colour, and there was no marker.

| Failure seen | Idea | Result |
|---|---|---|
| No colour to track | Follow the **bright closed tops** of the upside-down glasses | Every move in take 1 found, in order |
| No marker to measure with | Use the **table's corners** and its measured size (80 × 50 cm) as the ruler | Check: the top glass measured within 0.2 cm of the midpoint of the two below it, where it must be |
| Taller real glasses broke the robot's motion: **0 of 6** | Grip higher, level the glass, aim from measurements, lift clear after letting go | **19 of 20**, then **28 of 30** for each task once its working area was found |
| Task 3 take 1: the right arm can't flip across its own chest | Move the glasses to fit the robot | Worked, but moved glasses up to 33 cm, a poor copy. **Dropped** |
| Same problem | Choose the **arm and where to stand** from the layout, moving nothing | Take 1 builds; 25 of 30 random layouts |

**A fair test on new videos.** You recorded 8 more Task 2 takes:
- **0 of 8** succeeded, and only 2 were even read.
- The top glass was often lost against the bright floor, so I added a rule that works out an unseen top glass from the others. Result: **2 of 8** (takes 2 and 5). That's not a fair test, because the rule came from those videos.
- 4 were still refused, because a starting glass was missed. That led to stage 5.

You also recorded 6 Task 3 takes in varied layouts: **0 of 6**. Then take 9, in a layout chosen to suit the robot, built with nothing changed. That's one run, and when nudged by 1.5 cm in the simulator, it passed only 3 of 6.

**Lesson:** a method tuned on one video can fail on the next. That's why new videos matter.

---

### Stage 4: which arm to use

| Failure seen | Idea | Tested on | Result |
|---|---|---|---|
| Layouts on the robot's left failed with the right arm: **10 of 30** | Use the **nearest arm** for the whole task | 30 left-side layouts | **24 of 30** |
| Layouts spread across both sides: the nearest arm gave no gain (21 vs 20 of 30) | **An arm per glass** | 30 spread layouts | First version **9 of 30**: the swapping arm knocked glasses |
| The old arm swept glasses during the swap | The old hand rises and goes out beside the body; the new one comes in the same way | Two sets of 30 | A tie with one arm (20 vs 20, 22 vs 21), but real take 6 now builds |
| Neither choice wins everywhere | A **logistic regression model** (chapter 16) learns from 180 saved runs which choice will work | 60 new layouts | Model **50**, hand rule 49, always one way 43–44, best possible 51 |

**Lesson:** a smarter choice can only help as much as the gap between the options. The model beat the simple rule by just one layout.

---

### Stage 5: when the reader misses glasses

**Problem.** After stage 3, Task 2 built **4 of your 9 videos**. Most failures were a starting glass the reader never saw.

**Idea.** Ask Claude (chapter 17) to point at every glass in an early frame, and add the ones the reader missed. The measured positions stay in charge.

| Test | Before | After |
|---|---|---|
| Your 9 real Task 2 videos, same options for every video | 4 of 9 | **8 of 9**: Claude rescued takes 3, 4, 7 and 8, each refused for a missed glass (not a fair test) |
| The one failure, `task2_triangle_check` | — | Read correctly, but its first glass was beyond the robot's reach (37 cm from your edge, 30 cm to the side). You judged it a poor example and removed it, so later tests use 8 videos. Counting it, 8 of 9 |
| 10 cluttered simulated tables | a brightness rule saw 20 bright patches | **10 of 10** found exactly the glasses |
| Which Claude model? (10 cases) | — | All 9 of 10; Sonnet most accurate and far cheaper than Opus, so Sonnet is used |
| Your 9 real Task 3 videos | 2 of 9 | 2 of 9: no change, so Task 3's problem isn't reading (stage 7) |
| A fresh video on a cluttered table (fair test) | refused | **refused**: the reader took an earbud case and a laptop stand for glasses and missed the dim starting glasses. Claude can add missed glasses but not remove false ones |

---

### Stage 6: copying how you move, not just what you did

| Problem | Idea | Result | Kept? |
|---|---|---|---|
| The robot needs the key moments of a video | Find them from where the hand slows down | 13 moments for 6 real ones | **Dropped**: glass rest spots decide timing |
| The robot copied *what* you did, not *how* you moved | Copy your hand's **height and timing** while carrying | 4 of 4 videos built (not a fair test) | Optional |
| Copy your grip too? | Measure the thumb–finger gap around each pick | No reliable pattern: the glass hides your fingers | **Dropped** |
| Could the robot copy you live? | Webcam hand tracking drives the arm, wrist and fingers | First live try failed: the arm flew up, the body spun | Fixed below |
| Control started on a moving hand; the waist turned | Start only on a still hand; lock the waist; start over after 2 s without a hand | Followed your hand in 941 of 1460 frames | Yes |
| Copying a video means little if the simulated glasses stand elsewhere | Place the video's glasses; put your hand on the table in 3D (chapter 19) | Your palm within 0.9–1.3 cm of each glass you took | Yes |

---

### Stage 7: finding where Task 3 fails, and fixing it

**Problem.** A failed run only said "the tower is wrong". That doesn't tell you what to fix.

**Idea 1: step checks** (chapter 20). After each step, measure whether it worked. Tested on 30 flip layouts: the same **28 of 30**, the same failures, and no false alarms. So the checks only measure, as intended.

**What they found.** Your real Task 3 takes 2, 3 and 7 all fail at **reach**. The hand stopped 11, 31 and 41 cm short. Your glasses stood 20–39 cm from your edge of the table, but the flip was only tested with starting glasses 13–19 cm from it.

**Idea 2: stand closer to the table.**

| Take | Normal | 6 cm closer | 10 cm closer |
|---|---|---|---|
| 2 | 11 cm short | 8 cm short | **built and flipped** |
| 7 | 41 cm short | 4 cm short | reached it, then missed the fourth glass by 5 cm |
| 3 | 31 cm short | 45 cm short | its glass is 47 cm to the side, not too deep |

**Idea 3: retry from the ready pose.** In take 2, the third glass stood untouched, and the solver says it's reachable, yet the hand stopped 11 cm short. The solver seemed stuck after the previous move. So when a move ended short, the arm went back to its ready pose and tried again. Take 2 then **reached** the glass, but the retry motion knocked it off the table, and takes 7 and 3 were unchanged. **0 of 3, so this idea was dropped.**

**Where this stands.** One take is rescued (by standing closer), in one run. Next comes a rule for how close to stand, then retesting:
- takes 1 and 9, which work now, must still work;
- the simulator's working-area tests must not get worse.

Take 3 shows the real limit: a robot that can't walk can't reach glasses spread that widely.

---

### Bigger AI methods: what they are, and what was realistic here
You asked about several well-known methods. Here is each one in plain words, what it would need, and what was done with it.

| Method | What it is | What it needs | What happened here |
|---|---|---|---|
| **VLA** (vision-language-action model, such as OpenVLA or π0) | One big network that reads camera images and an instruction, and outputs robot motor commands directly | A GPU with lots of memory, and thousands of robot demonstrations to fine-tune on | **Not built.** This Mac has no suitable GPU, and there are no robot demonstrations to train on. Our split (Claude decides *what*, inverse kinematics decides *how*) gets the same "watch and do" result without training |
| **RL** (reinforcement learning, such as PPO) | The robot learns by trial and error, rewarded when the triangle stands | Millions of simulated tries, normally run in parallel on a GPU for hours or days | **Not built.** On this CPU, even a simple grasp would take days to learn. The step checks (chapter 20) would make a good reward signal if a GPU were available |
| **ACT** (Action Chunking Transformer) | Imitation learning: a network copies demonstrations and predicts short bursts of motion | 50 or more demonstrations of the robot itself doing the task, usually teleoperated, plus GPU training | **Not built.** The live hand control (chapter 19) could record such demonstrations; training would need a GPU. Copying our own scripted robot would only learn what is already programmed |
| **A small RNN/GRU** trained on your demonstrations | A small network that learns timing from sequences | Many recorded demonstrations | **Not built,** for the same reason: too few demonstrations |
| **SAM 2 / SAM 3** (Segment Anything) | Outlines any object in a picture from a click, and follows that outline through a video | Runs on CPU slowly; its smallest model is 156 MB | **Tried on 4 Nov:** given Claude's points as the clicks, it follows each glass by its *shape*, not its shine. See chapter 17 for how it compared with the bright-top reader |
| **Dex-retargeting** | Maps a human hand's finger joints onto a robot hand with different fingers | A hand model and the robot's finger layout | **Not needed:** the live demo maps three fingers by bend angle (chapter 19), and the task grips are programmed |

**The only model trained in this project** is the 16-weight logistic regression model (chapter 16). Everything else either is programmed, or uses a model someone else trained: MediaPipe for hands, Claude for finding glasses, and SAM 2 for outlines.

### The journey in one table

| What | Where it started | Where it is now |
|---|---|---|
| Block sorting, simulator | 0 of 10 | 20 of 20 |
| Task 3 flip, simulator | 1 of 9 | 50 of 50 |
| Real glasses, robot body | 0 of 6 | 28 of 30 |
| Task 2, your real videos | 1 video | 4 of 9, and **8 of 9** with Claude |
| Task 3, your real videos | 1 video, then 0 of 6 varied | 2 of 9, and take 2 promising with the robot closer |

---

## 23. Glossary

| Term | Meaning |
|---|---|
| **ACT** | A popular imitation-learning model that predicts short bursts of motion |
| **Actuator** | A motor |
| **API key** | Private password that lets a program use an online service |
| **ArUco marker** | A printed black-and-white square that software can detect precisely |
| **Branch** | A separate line of a project's history in git |
| **Calibration (live control)** | The moment your still hand is matched to the robot's starting pose |
| **Closed loop** | Measuring results and correcting; the opposite of open loop |
| **Collision / contact** | Two bodies touching in the simulator |
| **Config** | File of settings |
| **Damping** | Force that resists speed and reduces wobble |
| **Depth factor** | One number per video that corrects how far away the hand seems, because MediaPipe assumes an average hand size |
| **Digital twin** | A simulated copy of a real setup |
| **DOF** | Degree of freedom: one independent way of moving |
| **Egocentric** | Filmed from your own point of view |
| **Elliptic cone** | A more accurate friction model in MuJoCo |
| **`.env`** | A private file of settings, such as the API key, that git ignores |
| **Environment variable** | A named setting a program can read from the terminal it runs in |
| **Episode** | One complete recorded attempt at a task |
| **`failed_at`** | In a report, the first step that went wrong, such as `cup_d: reach` |
| **FK** | Forward kinematics: joint angles to hand position |
| **Focal length** | How strongly a camera's lens magnifies, in pixels; it links real size, distance and size in the picture |
| **Free joint** | Lets a body move freely in space |
| **Friction** | Resistance to sliding, which is what makes gripping possible |
| **FSR** | Final-state success rate |
| **Gate** | A measurable test a phase must pass |
| **GPU** | Graphics chip that speeds up AI work |
| **Gradient descent** | Learning by nudging the weights a little at a time in the direction that reduces mistakes |
| **Grasp site** | Our invisible point at the middle of the robot's grip |
| **Gravity compensation** | Extra motor effort that cancels the arm's weight |
| **Grounding DINO** | AI that finds objects from text descriptions |
| **Homography** | Mapping between a flat surface and its image |
| **HSV** | A colour format separating hue from brightness |
| **IK** | Inverse kinematics: hand position to joint angles |
| **Impratio** | A MuJoCo setting; higher values stop objects slowly creeping in a grip |
| **Keyframe** | An important frame chosen from a video |
| **Landmark** | One of the 21 points MediaPipe finds on a hand |
| **Logistic regression** | A small model that adds weighted numbers and turns the total into a chance from 0% to 100% |
| **Longest common subsequence** | The longest run of matching items kept in order |
| **MediaPipe** | Library that finds hand positions in images |
| **Menagerie** | Free collection of MuJoCo robot models |
| **Minimum jerk** | A smooth motion curve that starts and stops gently |
| **mink** | The IK library we use |
| **MuJoCo** | The physics simulator we use |
| **Open loop** | Acting without checking the result |
| **Pinned** | Fixed in place so it can't fall |
| **Pitch, roll, yaw** | Nodding, tilting sideways, and turning |
| **Policy** | A trained model that decides robot motions |
| **Pose** | Position plus orientation |
| **Post-training** | Specialising a pre-trained model using your own data |
| **PPO** | A common reinforcement-learning method |
| **Primitive** | A basic reusable action, such as `pick` |
| **Prominence** | How much a dip stands out from its surroundings |
| **Pydantic** | Library that checks data matches a format |
| **Radian** | Unit of angle; 1 rad is about 57 degrees |
| **Real-to-sim** | Copying real-world data and scenes into simulation |
| **Rest spot** | A place where a glass's bright top stays still for a while |
| **Retargeting** | Mapping human motion onto a robot's different body |
| **RL** | Reinforcement learning: learning through trial and reward |
| **SAM 2** | AI that outlines and follows objects through video |
| **Savitzky-Golay filter** | Smooths a curve while keeping its dips in place |
| **Seed** | Number that makes "random" results repeatable |
| **Sigmoid** | The S-shaped curve that turns any score into a chance between 0% and 100% |
| **Sim-to-real gap** | Difference between simulated and real behaviour |
| **Singularity** | An arm pose where tiny hand motions demand huge joint motions |
| **SSR** | Step success rate |
| **Step check** | A measurement after one step of a task, such as "is the glass still in the hand after the carry?" |
| **Stiffness** | How hard a motor pushes per unit of error |
| **Teleoperation** | Driving a robot directly with your own movements |
| **Torque** | Turning force, measured in N·m |
| **TSR** | Task success rate |
| **VLA** | Vision-language-action model: images and text to robot motion |
| **VLM** | Vision language model: images and text to text |
| **Waypoint** | An intermediate target along a path |
| **Weight** | In a model, how many points one feature adds or subtracts |
| **World model** | A model that predicts what happens next |
