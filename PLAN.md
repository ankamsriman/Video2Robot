# Video2Robot plan

**Deadline: Wed 4 Nov 2026, 23:59 IST.** `task.md` holds the official challenge text.
**Hard rule:** the data recorded by the user must play a real role. The real-video pipeline meets it. The phases below are the original plan, and the status table shows what really happened: the vision AI model in Phase 4 was replaced by measured positions and rules.
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

**The plan changed on Mon Nov 2, once real videos arrived:** steel glasses, no marker, Task A and Task B, with real-video sorting dropped. Phases 2-6 below describe the original sorting plan, and were replaced as follows.

| Step | Status |
|---|---|
| Real video to positions, without a marker | Done on 2 right-side videos; 2 old left-side videos are refused |
| Task A, video to robot | 3 videos succeed (takes 1, 2, 5); of 8 later takes, 4 read and 2 succeed, not a fair test of the top-glass rule |
| Task B, video to robot | Done with no glass moved: the layout picks the arm and standing spot. One video, one run |
| Outcome model, logistic regression (optional) | Chooses one or two arms; 26 of 30 fresh layouts, hand rule 25 |
| VLM helper (optional) | Instructions and missed-glass finding work; Task A real videos 8 of 9; clutter 10 of 10 found |
| Task A hand path (optional) | Copies height and timing; takes 1, 2, 5 and 6 succeed, not a fair test |
| Task A two arms (optional) | Tie on random layouts; real take 6 succeeds |
| Task A nearest arm | 24 of 30 against 10 of 30 on left-side layouts; 20 against 21 of 30 when spread across both sides |
| Choosing arm and standing spot | 19 of 21 on the video's layout wobbled; 25 of 30 on random layouts |
| What the robot can do, in chosen areas | 28 of 30 for each task, pinned pelvis |
| Test on new takes kept apart from tuning | Task B: **0 of 6** in varied layouts. In a chosen layout, 1 of 2 (take 9; 3 of 6 when nudged) |
| README, data sample, example videos | Done, data note needs the user's details |
| Live hand control (separate demo) | **Removed 4 Nov.** Worked on recorded video and in one live try (64% of frames), but later live tries followed only 14-22% of frames. Code in git history; finger closure kept for videos |
| Step checks | Every pick and place reports its first failed step (reach, grip, lift, carry, twist, place, release, others still). They only measure: same 28 of 30 flips, same failures |
| Task B reach | Real takes 2, 3 and 7 fail at reach: glasses 20-39 cm from the person's edge of the table, beyond the tested 13-19 cm for starting glasses. Standing 10 cm closer, take 2 succeeds (one run); take 7 still misses the fourth glass by 5 cm; take 3's glass is 47 cm to the side. Not yet a rule in the system |
| notes.md | Rebuilt Part Two (chapters 14-22) as a from-scratch book for the real videos, with worked examples |
| Journey | `docs/JOURNEY.md`: about 40 ideas in seven stages, each with problem, test, before → after, kept or dropped; summarised in the README. Every new experiment is added there |
| Claude reads both ends (version 3) | Claude points at glasses in the first and last frames; the frame-by-frame reader times the moves. Rules for clutter, unused glasses, glints and hidden glasses, each from a fresh video that first failed. Task A: 20 of the 28 videos kept build, 20 of 30 counting 2 deleted as out of reach (not a fair test); 7 fresh videos were 0 of 7 on their fair first run |
| Arm swap | Hands rise, pass beside the body off the table, waist held still (the user's idea): hard takes 1 and 2 build, take 4 (a glass at the edge of reach) lost |
| SAM 2 (version 4, optional) | SAM 2.1 tiny in a separate environment follows each glass by shape; in sunlight it timed the moves correctly where shiny tops didn't. About 70 minutes per video on this Mac |
| Real-world reel | `docs/videos/reel_real_world.mp4`, 3.5 minutes |
| Repo public and submission | **The user decides** |

Options B and C (retargeting the hand, training a policy) were not started. A works, but it isn't yet proven on new takes.

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
- [x] **Gate:** target was at least 8 of 10. Result: **20 of 20 on new setups** (seeds 600-619), positions shifted by up to 3 cm.
- **Scoring history, to report honestly:** an earlier check scored seeds 500-519 as 16 of 20. In all 4 failures the object had landed correctly, but the hand stopped 2-4 cm short, so the check failed the run and stopped the plan. Steps are now judged on where the object ends up, with hand arrival kept as a diagnostic. Seeds 500-519 now score 20 of 20. Seeds 600-619 were first run after the change.
- **Measured area:** one full run at each of 78 positions passed 70, with the corrected check. Object starts passed across 26-38 cm forward and 42 cm right to 14 cm left. Tray positions failed at the far right corner. See `data/processed/workspace_map.json`.
- **Limitations to state in results:** pelvis pinned (unpinned, it stands idle but falls on far reaches); three-finger hand; simple blocks.

## Phase 2: Keyframes (code built, untested on real video)
- [ ] User records about 10 videos and labels pick and place times and the correct plan
- [ ] Hand-speed curve from MediaPipe, smoothing, valley detection, merging, saved plot
- [ ] Save the full hand path for option B
- [ ] **Gate:** at least 90% of labelled moments found within 0.5 s

## Phase 3: Perception and real positions (colour tracking and marker built, tested on a synthetic video only)
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

## Phase 5: Closed loop (retry logic built and stress-tested, not yet run on real video)
Stress test, `data/processed/stress_retry.md`: weak grip went from 0 of 8 to 8 of 8 with retries; knocked objects were recovered; slipperiness barely improved.

- [ ] Compute the final relation state from the simulation and from the demo's last frame
- [ ] Classify failures as vision, spatial, temporal or execution, and respond to each
- [ ] Use only information from the video, never the hand-written labels
- [ ] **Gate:** full pipeline on 3 videos not used for tuning, reporting attempts to success with and without the loop

## Phase 6: Benchmark
- [ ] TSR, FSR, SSR, execution success, attempts to success, error types; Markdown and CSV reports
- [x] SSR uses the longest common subsequence, as SeeDo's released script does, rather than the pointer method in its paper
- [ ] Do not present results as directly comparable to the paper's
- [ ] **Gate:** one command regenerates every reported number

## Phase 7: Demo and submission
- [ ] `main.py` CLI, simple viewer, side-by-side demo video, README
- [ ] Confirm the submission format and rules with the organisers
- [ ] **Gate:** full run works from a clean checkout following the README

## Stretch (only after A, B and C decisions are settled)
- Whole-body balance with a centre-of-mass constraint, so the pelvis need not be pinned
- A two-handed task, or a five-finger hand

## Task A: cup triangle (branch `cup-task`, not merged)
Physics: 30 of 30 on new setups, 10 of 10 with 15 g cups, hand-written plan. Recognised from a rendered video and rebuilt by the G1. Real video not tried. `scripts/cup_from_video.py` runs it. Merge only if the user asks.

## Task B: flipped fourth cup (branch `cup-flip`, not merged)
Physics: 50 of 50 on new setups (seeds 300-349; 21 of 24 before the aiming fix), hand-written plan, programmed flip, pinned pelvis, restricted layout. No video link yet, so the hard rule isn't met for this task. `scripts/flip_trial.py` runs it.

## Needed from the user
- Record and label videos in Phase 2
- GPU access details by Phase 3
- Decide the AI model at the start of Phase 4
- Mentor's GitHub links, if they arrive
