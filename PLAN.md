# Video2Robot plan

The simulator comes first, so there is always a working robot to show. A phase is done when its gate passes. Gate numbers are initial targets; misses are reported, not hidden.

```
video -> keyframes -> tracked objects + table positions (cm) -> plan.json -> G1 in MuJoCo -> check result
                                                                   ^                            |
                                                                   +---- diagnose and retry ----+
```

## Status

| Phase | Name | Target (if deadline is Nov 4) | Status |
|---|---|---|---|
| 0 | Foundations | Sat Oct 31 | Done |
| 1 | Digital twin, G1 runs a hand-written plan | Sat Oct 31 - Sun Nov 1 | Next |
| 2 | Keyframes | Sun Nov 1 | Not started |
| 3 | Perception and real positions | Sun Nov 1 | Not started |
| 4 | Planner | Mon Nov 2 | Not started |
| 5 | Closed loop | Mon Nov 2 - Tue Nov 3 | Not started |
| 6 | Benchmark | Tue Nov 3 | Not started |
| 7 | Demo and submission | Wed Nov 4 | Not started |

If time runs short, cut in this order: stretch, viewer polish, free-base standing, hybrid comparison depth. The closed loop and an honest demo video stay.

## Phase 0: Foundations (done)
- [x] Git, `.gitignore`, virtual environment, dependencies, setup script
- [x] `CLAUDE.md`, `PLAN.md`, `CONVERSATION_LOG.md`, config file
- [x] Secret and video ignore rules checked before the first commit
- [x] **Gate:** G1 renders and `pytest` runs (2 tests pass)

## Phase 1: Digital twin and G1 executes a hand-written plan
- [ ] Table, objects and containers in MuJoCo
- [ ] Pin the pelvis first, then try a free base with legs held by joint control
- [ ] `plan.json` schema and arm IK with `mink`
- [ ] `pick(object)` and `place(object, reference, relation)`
- [ ] Measure arm reach and use it to set the real demo area
- [ ] Try friction grasping, time-boxed; otherwise use a clearly labelled assisted grasp
- [ ] Record rollout videos
- [ ] **Gate:** a 3-step plan succeeds in at least 8 of 10 runs with positions randomised by about 3 cm

## Phase 2: Keyframes
- [ ] User records about 10 videos and labels pick and place times and the correct plan
- [ ] Hand-speed curve from MediaPipe, smoothing, valley detection, merging, saved plot
- [ ] **Gate:** at least 90% of labelled moments found within 0.5 s

## Phase 3: Perception and real positions
- [ ] Object detection in the first frame, falling back to clicking objects by hand
- [ ] SAM 2 tracking on the GPU, with outlines and IDs drawn on saved keyframes
- [ ] Pixels to table centimetres from the marker, using each object's base contact point
- [ ] Fallback if SAM 2 is unavailable: colour tracking, reported as such
- [ ] **Gate:** median position error of 3 cm or less, with IDs stable across keyframes

## Phase 4: Planner
- [ ] **Decide the vision AI model and key with the user first**
- [ ] Adapt SeeDo's prompts (filter, pick, reference, relation)
- [ ] Pair keyframes into steps, classify the 6 relations, validate with Pydantic
- [ ] Compare AI-only relations with AI plus measured geometry
- [ ] **Gate:** SSR of at least 70%, with the comparison recorded either way

## Phase 5: Closed loop
- [ ] Compute the final relation state from the simulation and from the demo's last frame
- [ ] Classify failures as vision, spatial, temporal or execution, and respond to each
- [ ] Use only information from the video, never the hand-written labels
- [ ] **Gate:** full pipeline on 3 videos not used for tuning, reporting attempts to success with and without the loop

## Phase 6: Benchmark
- [ ] TSR, FSR, SSR, execution success, attempts to success, error types; Markdown and CSV reports
- [ ] Check SSR against SeeDo's `evaluation_script.py`, because the paper's pointer update looks wrong for matches found further along
- [ ] Do not present results as directly comparable to the paper's
- [ ] **Gate:** one command regenerates every reported number

## Phase 7: Demo and submission
- [ ] `main.py` CLI, simple viewer, side-by-side demo video, README
- [ ] Confirm the submission format and rules with the organisers
- [ ] **Gate:** full run works from a clean checkout following the README

## Stretch (pick one, only after Phase 7's core is safe)
- Torso lean or squat reach with a centre-of-mass constraint
- A two-handed task
- An ACT policy trained on simulated episodes using the GPU

## Needed from the user
- Confirm the deadline and the submission rules
- Record and label videos in Phase 2
- GPU access details by Phase 3
- Decide the AI model at the start of Phase 4
- Mentor's GitHub links, if they arrive
