# Video2Robot plan

**Deadline: Wed 4 Nov 2026, 23:59 IST.** `task.md` holds the official challenge text.
**Hard rule:** the data recorded by the user must play a real role. This is not met until Phase 4.
**Submit:** public GitHub repo, README or deck (how to run, example videos, design choices, what worked and what didn't), a sample of recorded data, and a note on how it was collected.
**Judged on:** creativity, results, and clarity ("No AI slop").

**Order of work (each builds on the last; each must be complete before starting the next):**
- **A:** video -> plan -> scripted robot (Phases 2-6).
- **B:** also retarget the recorded hand path to the G1. Start only if A is solid by about Tue Nov 3 midday.
- **C:** train ACT or a VLA on the retargeted motion, using a GPU. Start only if B finishes early.

Save the full hand path in Phase 2, not just its speed, so B needs no rework.

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
| 1 | Digital twin, G1 runs a hand-written plan | Sat Oct 31 - Sun Nov 1 | Done, 16 of 20 unseen, pinned pelvis |
| 2 | Keyframes | Sun Nov 1 | Next, needs videos |
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

## Phase 1: Digital twin and G1 executes a hand-written plan (done)
- [x] Table, objects and containers in MuJoCo (sizes are placeholders until real measurements arrive)
- [x] `plan.json` schema and arm IK with `mink`, plus torso turning and hand correction
- [x] `pick(object)` and `place(object, reference, relation)`, using friction only, with no assisted grasp
- [x] Reach measured; layout fitted to it
- [x] Rollout videos, and a live viewer (`scripts/watch_live.py`)
- [x] **Gate:** target was at least 8 of 10. Result: **16 of 20 on unseen setups** (seeds 500-519). All 4 failures stopped 2-4 cm short at the far tray.
- **Limitations to state in results:** pelvis pinned (unpinned, it stands idle but falls on far reaches); three-finger hand; simple blocks.

## Phase 2: Keyframes
- [ ] User records about 10 videos and labels pick and place times and the correct plan
- [ ] Hand-speed curve from MediaPipe, smoothing, valley detection, merging, saved plot
- [ ] Save the full hand path for option B
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

## Stretch (only after A, B and C decisions are settled)
- Whole-body balance with a centre-of-mass constraint, so the pelvis need not be pinned
- A two-handed task, or a five-finger hand

## Needed from the user
- Record and label videos in Phase 2
- GPU access details by Phase 3
- Decide the AI model at the start of Phase 4
- Mentor's GitHub links, if they arrive
