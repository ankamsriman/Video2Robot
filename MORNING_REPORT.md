# Morning report

**In short: everything planned got done, and the cup triangle worked far better than I expected. But none of it has run on real video yet, so the challenge's hard rule still isn't met. Your videos are the next step.**

## Results

| Work | Result | Caveat |
|---|---|---|
| **Sorting** | 20 of 20 on new setups | Hand-written plan, shifts up to 3 cm, pinned pelvis |
| **Cup triangle** | 30 of 30 on new setups, 10 of 10 with 15 g cups | Hand-written plan, rigid simulated cups |
| **Flipped fourth cup** | 21 of 24 on new setups | Programmed flip, restricted layout, no video link, branch `cup-flip` |
| **Video to robot, sorting** | Correct plan, task completed | Rendered video only |
| **Video to robot, cups** | Recognised the triangle and rebuilt it | Rendered video only, branch `cup-task` |
| **Retry loop** | Weak grip: 0 of 8 to 8 of 8 | Conditions I chose; slipperiness barely improved |
| **Working area** | 70 of 78 positions passed | One run per position |
| **Tests** | All pass, about 97% coverage | Hand tracking has only been tried on a still photo |

**Your instinct about the cups was right.** I worried the third cup wouldn't balance, but after fixing slip, it did every time.

## Mistakes I found and fixed
- **My 16-of-20 score was wrong.** In all 4 failures the object had landed correctly, but the check judged the hand. I retested on new setups.
- **My first retry test measured nothing.** "Grip harder" didn't change the force limit, and "slippery" had no effect. It's fixed and rerun.
- **I overwrote a results file.** It's regenerated, and files can't overwrite each other now.

## What can't be known yet
Rendered video is much easier than real video, which has shadows, blur and a real hand. **Expect problems with tracking and the marker on your first real video.** That's why I built a checker for it.

## Where things are

| Location | Contents |
|---|---|
| `main` branch | Sorting, the video chain, retry loop, docs |
| `cup-task` branch | Everything above, plus cups. **Not merged** |
| `cup-flip` branch | Cups, plus the flipped fourth cup. **Not merged** |
| GitHub | All are pushed to your **private** repo |

## Videos to watch
From `data/processed/`:
- `phase1/pinned_trial_600.mp4`: sorting
- `cups/cups_seed_0.mp4`: the cup triangle
- `flip/flip_seed_200.mp4`: the flipped fourth cup
- `cups/rendered_demo_video.mp4` then `cups/from_rendered_video_rollout.mp4`: a rendered demo, then the robot copying it

## What I need from you, in order

**1. Today: prepare and record 1 test video.**
- Print `docs/marker.pdf` at 100% scale, and check the square measures 10 cm.
- Follow `docs/RECORDING_GUIDE.md`.
- Send me the file path and each object's size, and I'll set up colours and run `scripts/check_video.py`.

**2. Then record the rest:** about 10 sorting videos. If you decide to show the cups, also record 3-4 of those. Use 3 opaque cups in different colours, filmed from the side.

**3. Decide:**
- **Which tasks to submit?** I'd show both: sorting as the base, cups as the highlight, and say the cups are simplified.
- **Option B or C (recorded motion, or training)?** The challenge's main example is post-training. I haven't built either. I'd consider B or C only once real video works.
- **An AI model?** No longer essential, since the no-AI planner can drive the robot. It would let us compare against SeeDo's approach. Without a model, the AI planner stays unused.
- **GPU?** Only needed if colour tracking fails on real video, or for training.

## Before submitting by Wed 4 Nov, 23:59 IST
- Make the repo public. I haven't, and will only do so when you ask.
- Add your video sample to `data/raw/sample/`, and a note on how you collected it.
- Finish the README with real results. A draft is in place.
- Revoke the temporary GitHub login afterwards: GitHub → Settings → Applications.
- Read `notes.md`, so you can explain every part yourself. New sections cover everything from tonight.
