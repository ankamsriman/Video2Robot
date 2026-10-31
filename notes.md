# Video2Robot: complete project notes

These notes explain the whole project from zero. You don't need any Python or robotics background. Read them in order the first time, then use the glossary at the end as a reference.

Throughout, **[Built]** means the code exists and has been tested. **[Planned]** means it doesn't exist yet.

## Contents
1. [The project in one story](#1-the-project-in-one-story)
2. [What the challenge asks for](#2-what-the-challenge-asks-for)
3. [Robotics ideas from zero](#3-robotics-ideas-from-zero)
4. [The Unitree G1 robot](#4-the-unitree-g1-robot)
5. [Types of AI models](#5-types-of-ai-models)
6. [What the mentors and papers said](#6-what-the-mentors-and-papers-said)
7. [Our design and why](#7-our-design-and-why)
8. [The code, file by file](#8-the-code-file-by-file)
9. [Worked example: putting the apple in the box](#9-worked-example-putting-the-apple-in-the-box)
10. [What went wrong and what it taught us](#10-what-went-wrong-and-what-it-taught-us)
11. [Results and how to read them](#11-results-and-how-to-read-them)
12. [What comes next](#12-what-comes-next)
13. [How the project is run](#13-how-the-project-is-run)
14. [Glossary](#14-glossary)

---

## 1. The project in one story

Imagine you sit at a table and film yourself putting an apple in a box. You hand the video to a robot that has never seen this task. It should be able to:

1. **Watch** the video and notice the moments that matter, such as "the hand slowed down here, so something was probably picked up".
2. **Identify** what's on the table: an apple, a lemon, a box.
3. **Understand** what you did and write it down: "put the apple in the box".
4. **Do it** with its own body, which is nothing like yours.
5. **Check** whether it worked, and try again if it didn't.

Doing this on a real humanoid would be expensive and risky, so we use a **simulator**: a program that acts like a video game with accurate physics. If the robot's fingers don't squeeze hard enough in the simulator, the object really does slip.

**Where we are:** steps 4 and part of 5 are built, using a plan I wrote by hand. Steps 1 to 3 are not built yet, and they need your videos.

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
| Phone recording | Planned | Fixed tripod. Egocentric "works well" but isn't required |
| Your data plays a real role | Planned | Your video would decide the task, the order and the object positions |
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
| AI decides *what*; measurement decides *where* | Targets SeeDo's biggest failure source |
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

The `keyframe_selector`, `perception`, `reasoning` and `evaluation` folders under `src/` are empty placeholders, because those phases aren't built yet.

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
| `watch_live.py` | Runs the task live in the MuJoCo window |

### `tests/`
There are 34 test functions. Some run several cases, so `pytest` reports 40 tests.

| File | What it checks | Example |
|---|---|---|
| `test_g1_model.py` | Model loads and renders | Image isn't blank |
| `test_plan_schema.py` | Plan format | Rejects `"beside"` |
| `test_relations.py` | Geometry | Hovering isn't `in` |
| `test_robot_env.py` | Motion and full physics | Hand blocked by the table reports failure |

We usually write the test **first**, watch it fail, then write the code. That proves the test can actually catch the problem.

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

### Current limitations
1. **Pelvis pinned**, so there's no real balance.
2. **Three-finger hand**, where the challenge suggests five.
3. **Simple blocks**, not real shapes.
4. **Hand-written plan**: your video isn't connected yet, so **the challenge's hard rule isn't met yet**.
5. **One arm only.**
6. **Small test size:** 20 runs give a rough estimate, not a precise rate.
7. **The trays don't move**, so they can't be knocked around.
8. **The physics isn't validated against reality**, because no real robot is involved.

---

## 12. What comes next

### Phase 2: finding key moments [Planned]
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

### Phase 3: finding objects and real positions [Planned]
- **Grounding DINO or Florence-2** find objects in the first frame from text descriptions.
- **SAM 2** follows each object through the video, keeping the same ID. It needs a GPU.
- **Homography** converts pixels to table centimetres.

**Homography, explained:** when you photograph a flat table from an angle, the table looks like a trapezoid. There's a mathematical mapping that undoes this, and it's fully determined by four known points. Our printed marker provides exactly that: four corners of a square known to be 10 cm wide. So if the apple's base is at pixel (640, 400), the mapping might tell us it's at table position (35 cm, −28 cm).

It only works for points **on the table's surface**. A point at the top of an object would be misplaced, so we use the point where the object touches the table.

**Pass if** the median error is 3 cm or less against a ruler.

### Phase 4: writing the plan [Planned]
**First, decide which AI model to use.** You chose to leave this until now. It needs an API key: a private password that lets programs use an online service.

We then adapt SeeDo's prompts and compare two approaches: the AI choosing the relation alone, and the AI choosing the objects while measured positions choose the relation.

**Pass if** SSR is at least 70%.

### Phase 5: closed loop, the main idea [Planned]
1. The robot runs the plan.
2. We compute the final arrangement in simulation.
3. We compute the final arrangement from the last video frame.
4. If they differ, we classify the error and respond:
   - **Execution error:** adjust the grip settings and retry.
   - **Plan error:** ask the AI again, telling it what went wrong.

**Important honesty rule:** this check can use only information from the video, never your hand-written labels. Otherwise it would be cheating.

### Phase 6: scoring [Planned]

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

**Current state:** Phases 0 and 1 are committed on your Mac, but not on GitHub yet.

### Commands to try
Run these from the project folder. Lines starting with `#` are comments.

```bash
# Run all tests (under a minute when the Mac is idle, up to about 8 minutes when busy)
.venv/bin/pytest

# Tests with a report of how much code they cover
.venv/bin/pytest --cov=src --cov-report=term-missing

# Print the map of where the hand can reach
PYTHONPATH=. .venv/bin/python scripts/reach_map.py

# Run the plan once on a new random setup (saves a video)
PYTHONPATH=. .venv/bin/python scripts/phase1_gate.py --trials 1 --first-seed 900

# Watch a passing run
open data/processed/phase1/pinned_trial_500.mp4
```

`--first-seed` chooses which random setups to use. The same seed always gives the same setup.

### Try changing something
Open `configs/default.yaml`, change `hover_height` from `0.17` to `0.25`, and run the one-trial command above. The robot will lift higher. Change it back afterwards.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **ACT** | A popular imitation-learning model that predicts short bursts of motion |
| **Actuator** | A motor |
| **API key** | Private password that lets a program use an online service |
| **ArUco marker** | A printed black-and-white square that software can detect precisely |
| **Closed loop** | Measuring results and correcting; the opposite of open loop |
| **Collision / contact** | Two bodies touching in the simulator |
| **Config** | File of settings |
| **Damping** | Force that resists speed and reduces wobble |
| **Digital twin** | A simulated copy of a real setup |
| **DOF** | Degree of freedom: one independent way of moving |
| **Egocentric** | Filmed from your own point of view |
| **Episode** | One complete recorded attempt at a task |
| **FK** | Forward kinematics: joint angles to hand position |
| **Free joint** | Lets a body move freely in space |
| **Friction** | Resistance to sliding, which is what makes gripping possible |
| **FSR** | Final-state success rate |
| **Gate** | A measurable test a phase must pass |
| **GPU** | Graphics chip that speeds up AI work |
| **Grasp site** | Our invisible point at the middle of the robot's grip |
| **Gravity compensation** | Extra motor effort that cancels the arm's weight |
| **Grounding DINO** | AI that finds objects from text descriptions |
| **Homography** | Mapping between a flat surface and its image |
| **IK** | Inverse kinematics: hand position to joint angles |
| **Keyframe** | An important frame chosen from a video |
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
| **Pydantic** | Library that checks data matches a format |
| **Radian** | Unit of angle; 1 rad is about 57 degrees |
| **Real-to-sim** | Copying real-world data and scenes into simulation |
| **Retargeting** | Mapping human motion onto a robot's different body |
| **RL** | Reinforcement learning: learning through trial and reward |
| **SAM 2** | AI that outlines and follows objects through video |
| **Seed** | Number that makes "random" results repeatable |
| **Sim-to-real gap** | Difference between simulated and real behaviour |
| **SSR** | Step success rate |
| **Stiffness** | How hard a motor pushes per unit of error |
| **TSR** | Task success rate |
| **Torque** | Turning force, measured in N·m |
| **VLA** | Vision-language-action model: images and text to robot motion |
| **VLM** | Vision language model: images and text to text |
| **Waypoint** | An intermediate target along a path |
| **World model** | A model that predicts what happens next |
