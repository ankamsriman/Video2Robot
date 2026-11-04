# How the project got here: one idea at a time

Every step below follows the same pattern:
1. Start from what works.
2. Test it on stated data.
3. Count the failures.
4. Find the cause in those failures.
5. Try one idea that targets it.
6. Retest, and keep the idea only if it helped.

Ideas that didn't help are listed too, because they shaped the design as much as the ones that worked.

**How to read the numbers:**
- "Simulator layouts" are random setups with exact positions. They test the robot's body, not the video reading.
- "Real videos" are my phone recordings.
- "Not a fair test" means the idea was built while looking at those same videos.
- The robot is always simulated, with its pelvis pinned. It can't fall, and it can't walk.

---

## Stage 1: teaching the robot to pick and place (simulator, coloured blocks)

| # | Problem seen | Idea | Test | Before → after | Kept? |
|---|---|---|---|---|---|
| 1.1 | First full run of the block-sorting task | Baseline: inverse kinematics plus a written plan | 10 setups | **0 of 10** | Starting point |
| 1.2 | Hand slammed into the table at 0.5 s; objects knocked over | Start the arm palm-down, and log every contact with a timestamp | Same setups | The slam stopped | Yes |
| 1.3 | Hand stopped 21 cm short but was reported as "reached" | Measure where the hand really is, not whether the maths found an answer | Same setups | Silent failures caught | Yes |
| 1.4 | Objects slid 2–3 cm on release; held tilted 15–18° | Finger poses calibrated to object width; tip the hand forward 0.2 rad | Same setups | Sliding and tilt fixed | Yes |
| 1.5 | 4 of 20 "failures" had actually placed the object correctly | Judge where the **object** ends up, not where the hand is | 20 new setups | 16 of 20 → **20 of 20** | Yes |
| 1.6 | A weak grip drops objects | Retry with a stronger grip after a failed lift | 8 setups, weak grip | **0 of 8 → 8 of 8** | Yes, for blocks |

## Stage 2: the real tasks, in the simulator (cups)

| # | Problem seen | Idea | Test | Before → after | Kept? |
|---|---|---|---|---|---|
| 2.1 | Task 2, a triangle of cups | Place two side by side, then the third across their measured tops | 30 setups | **30 of 30** | Yes |
| 2.2 | Task 3, flip a fourth cup on top. Flipping with the palm above would end with the palm under the cup, hitting the tower | Grip with the palm **beside** the cup and turn it like food on a spit | 9 new setups | **1 of 9** | Starting point |
| 2.3 | The cup crept out of the hand over seconds | A stiffer friction setting (`impratio`) | 12 new setups | **1 of 12**: a new problem appeared | Yes |
| 2.4 | The fingers carried the cup away, and the robot's final arm swing knocked the tower | Back out sideways before lifting; remove the swing | 12 new setups | **5 of 12** | Yes |
| 2.5 | The palm hit the cup's wide base on the way down | Come down beside it, slide in, then centre the hand | 15 new setups | **9 of 15** | Yes |
| 2.6 | Remaining misses: thrown cups, cups slipping out of the fingertips, flips ending short | Turn only the wrist; hold the cup deeper; keep the forearm level | 24 new setups | **21 of 24** | Yes |
| 2.7 | The 3 failures were flipped but over 2 cm off-centre: aimed at the plan, plus a small shift on release | Aim at the top cup's **measured** position and cancel the release shift | 50 new setups | **50 of 50** | Yes |

## Stage 3: my real videos and real glasses

| # | Problem seen | Idea | Test | Before → after | Kept? |
|---|---|---|---|---|---|
| 3.1 | My glasses are steel (no colour), and I recorded with no marker | Follow the **bright closed tops**; use the **table's corners** (80 × 50 cm) as the ruler | Task 2 take 1 | Colour tracking: unusable → every move found, in order | Yes |
| 3.2 | Taller real glasses broke the robot's motion | Grip higher, level the glass, aim from measured positions, lift clear after letting go | Simulator, glass size | **0 of 6 → 19 of 20** | Yes |
| 3.3 | Which areas of the table the robot can work in | Try wide areas, narrow them where failures cluster, then retest fresh | 30 new layouts per task | **28 of 30** each | Yes |
| 3.4 | Task 3 take 1: the right arm can't flip across its own chest | Move the glasses to fit the robot | Take 1 | Worked, but moved glasses up to 33 cm, a poor copy | Dropped |
| 3.5 | Same problem | Choose the **arm and standing spot** from the layout, moving no glass | Take 1; 30 random layouts | Take 1 builds; **25 of 30** random | Yes |
| 3.6 | 8 new Task 2 takes, fair test | Run the reader as it was | 8 real videos | **0 of 8** (only 2 even read) | Starting point |
| 3.7 | The top glass is often unseen against the bright floor | Infer an unseen top glass from the other glasses | Same 8 videos | 0 → **2** (takes 2, 5; not a fair test) | Yes |
| 3.8 | 6 new Task 3 takes in varied layouts, fair test | Run as it was | 6 real videos | **0 of 6** (3 refused, 3 robot failures) | Diagnosed in stage 7 |
| 3.9 | — | One more Task 3 video in a layout chosen to suit the robot | Take 9, nothing changed | **Built** (one run). Nudged by 1.5 cm: 3 of 6 | Yes |

## Stage 4: which arm to use

| # | Problem seen | Idea | Test | Before → after | Kept? |
|---|---|---|---|---|---|
| 4.1 | Layouts on the robot's left failed with the right arm | Use the **nearest arm** for the whole task | 30 left-side layouts | **10 of 30 → 24 of 30** | Yes |
| 4.2 | Layouts spread across both sides: no gain from 4.1 (21 vs 20 of 30) | **An arm per glass** (`--two-arms`) | 30 spread layouts | First version **9 of 30**: the swap knocked glasses | Fixed in 4.3 |
| 4.3 | The old arm swept glasses during the swap | The old hand rises, goes out beside the body and hangs back; the new one comes in the same way | Two sets of 30 | Tie with one arm: 20 vs 20, 22 vs 21. Real take 6 now builds | Optional |
| 4.4 | Neither choice wins everywhere | A **logistic regression model** (16 weights) learns from 180 saved runs which choice will work | 60 new layouts | Always one arm 43, always per glass 44, hand rule 49, **model 50**, best possible 51 | Optional |

## Stage 5: when the reader misses glasses

| # | Problem seen | Idea | Test | Before → after | Kept? |
|---|---|---|---|---|---|
| 5.1 | Task 2: 4 of 9 real videos refused or wrong, mostly a missed starting glass | **Claude (Sonnet) points to every glass**; missed ones are added, measured positions stay in charge | 9 real Task 2 videos | **4 of 9 → 8 of 9** (not a fair test) | Optional |
| 5.1b | The 4 → 8 above mixed different options per video | Rerun every video with the same options: no Claude (an arm per glass) against Claude + the logistic regression model | 9 real Task 2 videos | **4 of 9 → 8 of 9**. Claude rescued takes 3, 4, 7, 8, all refused for a missed glass. Towers within 0.2–0.6 cm of mine. The model chose an arm per glass on all 9. The "check" video fails at reach either way | Yes |
| 5.1c | The one failure, `task2_triangle_check`, was read correctly but its first glass stood 37 cm from my edge and 30 cm to the side: the left arm stopped 4.4 cm short | I judged the layout a poor example, beyond the robot's reach, and removed the video | — | Later tests use 8 videos. **Counting it, 8 of 9** | Excluded, by my choice |
| 5.2 | Would Claude be fooled by other bright things? | Cluttered simulated tables | 10 scenes | **10 of 10** found exactly the glasses, about 1 cm off | Yes |
| 5.3 | Which Claude model? | Compare Haiku, Sonnet, Opus | 10 cases | All 9 of 10; Sonnet most accurate (0.33 cm) and far cheaper than Opus | Sonnet |
| 5.4 | Task 3 still 2 of 9 with Claude | — | — | No change: Task 3's problem isn't reading | See stage 7 |
| 5.5 | A fresh Task 2 video on a cluttered table (phone, earbud case, card, charger, laptop stand) | Fair test, nothing tuned on it | 1 new real video | **Refused**, with and without Claude. The reader took the earbud case and laptop stand for glasses, and missed all three starting glasses, whose tops looked dim from this camera angle. Claude added 2 missed glasses but can't remove false ones (`scripts/explain_reading.py` shows it frame by frame) | Next: drop spots far from every glass Claude finds |

## Stage 6: copying how I move, not just what I did

| # | Problem seen | Idea | Test | Before → after | Kept? |
|---|---|---|---|---|---|
| 6.1 | Key moments were needed | Find them from the hand's speed | 2 real videos | Found 13 moments for 6 real ones, 17 for 8 | Dropped; glass rest spots decide timing |
| 6.2 | The robot copied *what* I did, not *how* I moved | Copy the hand's **height and timing** during each carry (`--hand-path`) | 4 real videos | **4 of 4** built (not a fair test) | Optional |
| 6.3 | Copy my grip too? | Measure thumb–finger gap around each pick | 3 real videos | No reliable pattern: the glass hides my fingers | Dropped |
| 6.4 | Could the robot copy me live? | Webcam hand tracking drives arm, wrist and fingers | Take 6 as a stand-in camera | Followed 476 of 561 frames | Separate demo |
| 6.5 | First live webcam try: the arm flew up, the body spun | Start only on a still hand; lock the waist; start over after 2 s without a hand | Live webcam | Failed → followed **941 of 1460** frames | Yes |
| 6.6 | Copying a video means little if the glasses stand elsewhere | `--match-video`: place the video's glasses, and put the hand on the table in 3D from its apparent size | Task 2 take 1 | Palm within **0.9–1.3 cm** of each glass; robot hand 2.6 / 15.3 / 5.9 cm | Yes |

## Stage 7: finding where Task 3 fails, and fixing it

| # | Problem seen | Idea | Test | Before → after | Kept? |
|---|---|---|---|---|---|
| 7.1 | A failed run only said "tower wrong", not where | **Step checks** after reach, grip, lift, carry, twist, place, release, others still | 30 flip layouts | Same **28 of 30**, same failures, no false alarms (only measures) | Yes |
| 7.2 | Why do real Task 3 takes fail? | Rerun takes 2, 3, 7 with step checks | 3 real videos | All three fail at **reach**: hand 11, 31, 41 cm short; glasses 20–39 cm from the person's edge of the table, where the flip was tested on starting glasses 13–19 cm cm | Diagnosis |
| 7.3 | Glasses too deep for a robot that can't walk | **Stand closer** to the table | Takes 2, 3, 7 | 6 cm closer: 0 of 3 (misses 8, 45, 4 cm). **10 cm closer: take 2 builds**; take 7 misses its fourth glass by 5 cm | Promising, not yet a rule |
| 7.4 | Take 2's third glass stood untouched and solver-reachable, yet the hand stopped 11 cm short: the solver seemed stuck after the previous move | When a move ends short, return to the ready pose and try once more | Takes 2, 7, 3 | Take 2 now **reached** the glass, but the retry motion knocked it off the table; 7 and 3 unchanged. 0 of 3 | Dropped |
| 7.5 | Next | A rule for how close to stand, then retest takes 1 and 9 (must still work) and the simulator areas | — | — | Planned |

---

## Ideas discussed but not built
- **A recurrent network (RNN/GRU) trained on demos.** Suggested in a plan, but not built: there was no time to collect training data. Nothing in this project is trained except the 16-weight logistic regression model.
- **SAM 2 and dex-retargeting.** Not needed. The glasses are identical, and rest spots keep their order.
- **Retrying a missed glass grip.** Tried, and rescued none of the failed setups, so it was removed for glasses.
- **Flipping in open space, then carrying the glass across.** Destroyed the tower, so it was removed.

## The journey in numbers

| Milestone | Result |
|---|---|
| Block sorting, simulator | 0 of 10 → 20 of 20 |
| Task 3 flip, simulator | 1 of 9 → 1 of 12 → 5 of 12 → 9 of 15 → 21 of 24 → 50 of 50 |
| Real glasses, robot body | 0 of 6 → 28 of 30 |
| Task 2, real videos | 1 video → 4 of 9 → **8 of 9** with Claude |
| Task 3, real videos | 1 video → 0 of 6 varied → **2 of 9**, and take 2 promising with the robot closer |
